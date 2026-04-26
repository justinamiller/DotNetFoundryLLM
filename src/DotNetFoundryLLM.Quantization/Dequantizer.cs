using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace DotNetFoundryLLM.Quantization;

/// <summary>
/// Dequantizes GGUF/GGML quantized weight blocks to 32-bit floating-point values.
/// <para>
/// Supported formats:
/// <list type="bullet">
///   <item><description><b>F32</b> — identity copy.</description></item>
///   <item><description><b>F16</b> — IEEE 754 half-precision to float32.</description></item>
///   <item><description><b>BF16</b> — Google Brain float16 to float32.</description></item>
///   <item><description><b>Q4_0</b> — 4-bit symmetric quantisation, block size 32.</description></item>
///   <item><description><b>Q8_0</b> — 8-bit symmetric quantisation, block size 32.</description></item>
/// </list>
/// </para>
/// </summary>
public static class Dequantizer
{
    // Q4_0: { f16 scale; uint8 qs[16]; }  → 32 float values per 18-byte block.
    private const int Q4BlockSize = 32;
    private const int Q4BlockBytes = 18;

    // Q8_0: { f16 scale; int8 qs[32]; }   → 32 float values per 34-byte block.
    private const int Q8BlockSize = 32;
    private const int Q8BlockBytes = 34;

    /// <summary>
    /// Converts an IEEE 754 half-precision value (stored as a <see cref="ushort"/>) to a
    /// single-precision float.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HalfToFloat(ushort bits) =>
        (float)BitConverter.UInt16BitsToHalf(bits);

    /// <summary>
    /// Converts a bfloat16 value (stored as a <see cref="ushort"/>) to a single-precision float.
    /// BF16 shares the same exponent field as F32; the mantissa is zero-extended.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float BFloat16ToFloat(ushort bits) =>
        BitConverter.Int32BitsToSingle((int)bits << 16);

    /// <summary>
    /// Copies F32 bytes from <paramref name="src"/> to <paramref name="dst"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 4 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
    public static void DequantizeF32(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (src.Length % 4 != 0)
        {
            throw new ArgumentException("Source length must be a multiple of 4 for F32.", nameof(src));
        }

        int count = src.Length / 4;
        if (dst.Length < count)
        {
            throw new ArgumentException($"Destination must hold at least {count} floats.", nameof(dst));
        }

        for (int i = 0; i < count; i++)
        {
            dst[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(src.Slice(i * 4, 4)));
        }
    }

    /// <summary>
    /// Dequantizes an F16 (half-precision) buffer into <paramref name="dst"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 2 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
    public static void DequantizeF16(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (src.Length % 2 != 0)
        {
            throw new ArgumentException("Source length must be a multiple of 2 for F16.", nameof(src));
        }

        int count = src.Length / 2;
        if (dst.Length < count)
        {
            throw new ArgumentException($"Destination must hold at least {count} floats.", nameof(dst));
        }

        for (int i = 0; i < count; i++)
        {
            dst[i] = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2)));
        }
    }

    /// <summary>
    /// Dequantizes a BF16 (bfloat16) buffer into <paramref name="dst"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 2 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
    public static void DequantizeBF16(ReadOnlySpan<byte> src, Span<float> dst)
    {
        if (src.Length % 2 != 0)
        {
            throw new ArgumentException("Source length must be a multiple of 2 for BF16.", nameof(src));
        }

        int count = src.Length / 2;
        if (dst.Length < count)
        {
            throw new ArgumentException($"Destination must hold at least {count} floats.", nameof(dst));
        }

        for (int i = 0; i < count; i++)
        {
            dst[i] = BFloat16ToFloat(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2)));
        }
    }

    /// <summary>
    /// Dequantizes a Q4_0-encoded buffer into <paramref name="dst"/>.
    /// <para>
    /// Block layout (18 bytes per block of 32 elements):
    /// <list type="bullet">
    ///   <item><description>bytes [0..1] — f16 delta (scale)</description></item>
    ///   <item><description>bytes [2..17] — 16 nibble-packed bytes; lower nibble is element[j], upper nibble is element[j+16] for j in [0..15]</description></item>
    /// </list>
    /// Dequantized value: <c>(nibble − 8) × delta</c>.
    /// </para>
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 18 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ4_0(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q4BlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q4BlockBytes} for Q4_0.", nameof(src));
        }

        int blockCount = src.Length / Q4BlockBytes;
        int elementCount = blockCount * Q4BlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q4BlockBytes, Q4BlockBytes);
            float d = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block));
            int dstBase = b * Q4BlockSize;

            for (int j = 0; j < 16; j++)
            {
                byte qs = block[2 + j];
                dst[dstBase + j]      = ((qs & 0x0F) - 8) * d;
                dst[dstBase + j + 16] = ((qs >> 4)   - 8) * d;
            }
        }
    }

    /// <summary>
    /// Dequantizes a Q8_0-encoded buffer into <paramref name="dst"/>.
    /// <para>
    /// Block layout (34 bytes per block of 32 elements):
    /// <list type="bullet">
    ///   <item><description>bytes [0..1] — f16 delta (scale)</description></item>
    ///   <item><description>bytes [2..33] — 32 signed int8 quantized values</description></item>
    /// </list>
    /// Dequantized value: <c>int8_value × delta</c>.
    /// </para>
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 34 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ8_0(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q8BlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q8BlockBytes} for Q8_0.", nameof(src));
        }

        int blockCount = src.Length / Q8BlockBytes;
        int elementCount = blockCount * Q8BlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q8BlockBytes, Q8BlockBytes);
            float d = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block));
            int dstBase = b * Q8BlockSize;

            for (int j = 0; j < Q8BlockSize; j++)
            {
                dst[dstBase + j] = (sbyte)block[2 + j] * d;
            }
        }
    }

    /// <summary>
    /// Returns the number of float32 elements encoded in a buffer of the given byte length
    /// for the specified GGML tensor type.
    /// </summary>
    /// <param name="tensorType">GGML tensor type identifier (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 30=BF16).</param>
    /// <param name="byteLength">Total byte length of the raw tensor data.</param>
    /// <returns>Number of float32 elements after dequantization.</returns>
    /// <exception cref="NotSupportedException">Thrown for unsupported tensor types.</exception>
    public static long ElementCount(int tensorType, long byteLength) =>
        tensorType switch
        {
            0  => byteLength / 4,       // F32
            1  => byteLength / 2,       // F16
            2  => byteLength / Q4BlockBytes * Q4BlockSize, // Q4_0
            8  => byteLength / Q8BlockBytes * Q8BlockSize, // Q8_0
            30 => byteLength / 2,       // BF16
            _  => throw new NotSupportedException($"Tensor type {tensorType} is not supported for dequantization.")
        };

    /// <summary>
    /// Dequantizes raw tensor bytes to float32 using the specified GGML tensor type.
    /// </summary>
    /// <param name="tensorType">GGML tensor type identifier.</param>
    /// <param name="src">Raw tensor bytes.</param>
    /// <param name="dst">Destination float32 buffer. Must be pre-allocated with sufficient capacity.</param>
    /// <exception cref="NotSupportedException">Thrown for unsupported tensor types.</exception>
    public static void Dequantize(int tensorType, ReadOnlySpan<byte> src, Span<float> dst)
    {
        switch (tensorType)
        {
            case 0:  DequantizeF32(src, dst); break;
            case 1:  DequantizeF16(src, dst); break;
            case 2:  DequantizeQ4_0(src, dst); break;
            case 8:  DequantizeQ8_0(src, dst); break;
            case 30: DequantizeBF16(src, dst); break;
            default: throw new NotSupportedException(
                $"Tensor type {tensorType} is not supported for dequantization.");
        }
    }
}
