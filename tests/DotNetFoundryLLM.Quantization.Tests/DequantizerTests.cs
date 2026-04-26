using System.Buffers.Binary;
using DotNetFoundryLLM.Quantization;
using Xunit;

namespace DotNetFoundryLLM.Quantization.Tests;

/// <summary>Tests for <see cref="Dequantizer"/>.</summary>
public sealed class DequantizerTests
{
    // ── HalfToFloat ──────────────────────────────────────────────────────────

    [Fact]
    public void HalfToFloat_Zero_ReturnsZero()
    {
        Assert.Equal(0f, Dequantizer.HalfToFloat(0x0000));
    }

    [Fact]
    public void HalfToFloat_One_ReturnsOne()
    {
        // IEEE 754 half: 0 01111 0000000000 = 0x3C00
        Assert.Equal(1.0f, Dequantizer.HalfToFloat(0x3C00), precision: 6);
    }

    [Fact]
    public void HalfToFloat_NegativeOne_ReturnsNegativeOne()
    {
        // IEEE 754 half: 1 01111 0000000000 = 0xBC00
        Assert.Equal(-1.0f, Dequantizer.HalfToFloat(0xBC00), precision: 6);
    }

    // ── BFloat16ToFloat ──────────────────────────────────────────────────────

    [Fact]
    public void BFloat16ToFloat_Zero_ReturnsZero()
    {
        Assert.Equal(0f, Dequantizer.BFloat16ToFloat(0x0000));
    }

    [Fact]
    public void BFloat16ToFloat_One_ReturnsOne()
    {
        // BF16 1.0 = 0 01111111 0000000 = 0x3F80
        Assert.Equal(1.0f, Dequantizer.BFloat16ToFloat(0x3F80), precision: 6);
    }

    [Fact]
    public void BFloat16ToFloat_Two_ReturnsTwo()
    {
        // BF16 2.0 = 0 10000000 0000000 = 0x4000
        Assert.Equal(2.0f, Dequantizer.BFloat16ToFloat(0x4000), precision: 6);
    }

    // ── DequantizeF32 ─────────────────────────────────────────────────────────

    [Fact]
    public void DequantizeF32_RoundTrip()
    {
        float[] original = [1.5f, -2.25f, 0f, 3.14f];
        var bytes = new byte[original.Length * 4];
        for (int i = 0; i < original.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(i * 4, 4),
                BitConverter.SingleToInt32Bits(original[i]));
        }

        var dst = new float[original.Length];
        Dequantizer.DequantizeF32(bytes, dst);

        Assert.Equal(original, dst);
    }

    [Fact]
    public void DequantizeF32_WrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() => Dequantizer.DequantizeF32(new byte[5], new float[1]));
    }

    // ── DequantizeF16 ─────────────────────────────────────────────────────────

    [Fact]
    public void DequantizeF16_KnownValues()
    {
        // Encode 1.0 and -1.0 as F16 little-endian bytes.
        var bytes = new byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(0, 2), 0x3C00); // 1.0
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), 0xBC00); // -1.0

        var dst = new float[2];
        Dequantizer.DequantizeF16(bytes, dst);

        Assert.Equal(1.0f,  dst[0], precision: 5);
        Assert.Equal(-1.0f, dst[1], precision: 5);
    }

    // ── DequantizeBF16 ────────────────────────────────────────────────────────

    [Fact]
    public void DequantizeBF16_KnownValues()
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(0, 2), 0x3F80); // 1.0
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), 0x4000); // 2.0

        var dst = new float[2];
        Dequantizer.DequantizeBF16(bytes, dst);

        Assert.Equal(1.0f, dst[0], precision: 5);
        Assert.Equal(2.0f, dst[1], precision: 5);
    }

    // ── DequantizeQ4_0 ────────────────────────────────────────────────────────

    [Fact]
    public void DequantizeQ4_0_ZeroBlock_AllZeros()
    {
        // A Q4_0 block with delta=0 (f16 0x0000) and all nibbles=8 produces 0*scale=0.
        var block = new byte[18];
        BinaryPrimitives.WriteUInt16LittleEndian(block.AsSpan(0, 2), 0x0000); // delta=0
        // qs bytes 0..15: 0x88 → lower nibble=8, upper nibble=8 → (8-8)*0 = 0
        for (int i = 2; i < 18; i++) block[i] = 0x88;

        var dst = new float[32];
        Dequantizer.DequantizeQ4_0(block, dst);

        foreach (var v in dst) Assert.Equal(0f, v);
    }

    [Fact]
    public void DequantizeQ4_0_KnownBlock()
    {
        // delta = 1.0 (f16 0x3C00), all nibbles = 8 → (8-8)*1 = 0.
        // nibble = 9 → (9-8)*1 = 1; nibble = 7 → (7-8)*1 = -1.
        var block = new byte[18];
        BinaryPrimitives.WriteUInt16LittleEndian(block.AsSpan(0, 2), 0x3C00); // delta=1.0
        // qs[0] = 0x79: lower=9, upper=7 → element[0]=1, element[16]=-1
        block[2] = 0x79;
        // remaining nibbles = 8 → 0
        for (int i = 3; i < 18; i++) block[i] = 0x88;

        var dst = new float[32];
        Dequantizer.DequantizeQ4_0(block, dst);

        Assert.Equal(1f,  dst[0],  precision: 5);
        Assert.Equal(-1f, dst[16], precision: 5);
        Assert.Equal(0f,  dst[1],  precision: 5);
    }

    [Fact]
    public void DequantizeQ4_0_WrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            Dequantizer.DequantizeQ4_0(new byte[17], new float[32]));
    }

    // ── DequantizeQ8_0 ────────────────────────────────────────────────────────

    [Fact]
    public void DequantizeQ8_0_KnownBlock()
    {
        // delta = 1.0 (f16 0x3C00), quants are [-1, 0, 1, 0, ..., 0]
        var block = new byte[34];
        BinaryPrimitives.WriteUInt16LittleEndian(block.AsSpan(0, 2), 0x3C00); // delta=1.0
        block[2] = unchecked((byte)(sbyte)-1); // -1 * 1.0 = -1
        block[3] = 0;                           //  0 * 1.0 =  0
        block[4] = 1;                           //  1 * 1.0 =  1

        var dst = new float[32];
        Dequantizer.DequantizeQ8_0(block, dst);

        Assert.Equal(-1f, dst[0], precision: 5);
        Assert.Equal(0f,  dst[1], precision: 5);
        Assert.Equal(1f,  dst[2], precision: 5);
    }

    [Fact]
    public void DequantizeQ8_0_WrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            Dequantizer.DequantizeQ8_0(new byte[33], new float[32]));
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(99)]
    public void Dequantize_UnknownType_Throws(int tensorType)
    {
        Assert.Throws<NotSupportedException>(() =>
            Dequantizer.Dequantize(tensorType, new byte[4], new float[1]));
    }

    [Fact]
    public void ElementCount_F32()
    {
        Assert.Equal(4L, Dequantizer.ElementCount(0, 16));
    }

    [Fact]
    public void ElementCount_Q4_0()
    {
        // 18 bytes = 1 block = 32 elements
        Assert.Equal(32L, Dequantizer.ElementCount(2, 18));
    }

    [Fact]
    public void ElementCount_Q8_0()
    {
        // 34 bytes = 1 block = 32 elements
        Assert.Equal(32L, Dequantizer.ElementCount(8, 34));
    }
}
