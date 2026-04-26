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

    // Q4_K: { f16 d, f16 dmin, scales[12], qs[128] } -> 256 float values per 144-byte block.
    private const int Q4KBlockSize = 256;
    private const int Q4KBlockBytes = 144;

    // Q5_K: { f16 d, f16 dmin, scales[12], qh[32], qs[128] } -> 256 float values per 176-byte block.
    private const int Q5KBlockSize = 256;
    private const int Q5KBlockBytes = 176;

    // Q6_K: { ql[128], qh[64], scales[16], f16 d } -> 256 float values per 210-byte block.
    private const int Q6KBlockSize = 256;
    private const int Q6KBlockBytes = 210;

    // Q8_K: { f32 d, i8 qs[256], i16 bsums[16] } -> 256 float values per 292-byte block.
    private const int Q8KBlockSize = 256;
    private const int Q8KBlockBytes = 292;

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
    /// Dequantizes a Q4_K-encoded buffer into <paramref name="dst"/>.
    /// <para>
    /// Block layout (144 bytes per block of 256 elements):
    /// <list type="bullet">
    ///   <item><description>bytes [0..1] — f16 delta (scale)</description></item>
    ///   <item><description>bytes [2..3] — f16 delta min</description></item>
    ///   <item><description>bytes [4..15] — 12 bytes for scales</description></item>
    ///   <item><description>bytes [16..143] — 128 bytes for quantized values</description></item>
    /// </list>
    /// Dequantized value: <c>dg * q - mg</c>, where <c>dg = delta × scale</c> and <c>mg = delta_min × min</c>.
    /// </para>
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 144 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ4_K(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q4KBlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q4KBlockBytes} for Q4_K.", nameof(src));
        }

        int blockCount = src.Length / Q4KBlockBytes;
        int elementCount = blockCount * Q4KBlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q4KBlockBytes, Q4KBlockBytes);

            float d = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(0, 2)));
            float dmin = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(2, 2)));
            var scales = block.Slice(4, 12);
            var qs = block.Slice(16, 128);

            int dstBase = b * Q4KBlockSize;
            for (int g = 0; g < 8; g++)
            {
                DecodeKScaleMin(scales, g, out int scaleRaw, out int minRaw);
                float dg = d * scaleRaw;
                float mg = dmin * minRaw;

                int groupBase = dstBase + g * 32;
                for (int i = 0; i < 32; i++)
                {
                    int qIndex = g * 16 + (i >> 1);
                    byte packed = qs[qIndex];
                    int q = (i & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
                    dst[groupBase + i] = dg * q - mg;
                }
            }
        }
    }

    /// <summary>
    /// Dequantizes a Q5_K-encoded buffer into <paramref name="dst"/>.
    /// <para>
    /// Block layout (176 bytes per block of 256 elements):
    /// <list type="bullet">
    ///   <item><description>bytes [0..1] — f16 delta (scale)</description></item>
    ///   <item><description>bytes [2..3] — f16 delta min</description></item>
    ///   <item><description>bytes [4..15] — 12 bytes for scales</description></item>
    ///   <item><description>bytes [16..47] — 32 bytes for upper 5 bits of quantized values</description></item>
    ///   <item><description>bytes [48..175] — 128 bytes for lower 4 bits of quantized values</description></item>
    /// </list>
    /// Dequantized value: <c>dg * q - mg</c>, where <c>dg = delta × scale</c> and <c>mg = delta_min × min</c>.
    /// This format packs the upper 5 bits of each value in an 8-bit byte, with the msb as sign bit.
    /// </para>
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 176 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ5_K(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q5KBlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q5KBlockBytes} for Q5_K.", nameof(src));
        }

        int blockCount = src.Length / Q5KBlockBytes;
        int elementCount = blockCount * Q5KBlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q5KBlockBytes, Q5KBlockBytes);

            float d = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(0, 2)));
            float dmin = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(2, 2)));
            var scales = block.Slice(4, 12);
            var qh = block.Slice(16, 32);
            var qs = block.Slice(48, 128);

            int dstBase = b * Q5KBlockSize;
            for (int g = 0; g < 8; g++)
            {
                DecodeKScaleMin(scales, g, out int scaleRaw, out int minRaw);
                float dg = d * scaleRaw;
                float mg = dmin * minRaw;

                int groupBase = dstBase + g * 32;
                for (int i = 0; i < 32; i++)
                {
                    int tokenIndex = g * 32 + i;

                    int qLowIndex = tokenIndex >> 1;
                    byte packedLow = qs[qLowIndex];
                    int qLow = (tokenIndex & 1) == 0 ? (packedLow & 0x0F) : (packedLow >> 4);

                    int qhByteIndex = tokenIndex >> 3;
                    int qhBit = tokenIndex & 0x07;
                    int qHigh = (qh[qhByteIndex] >> qhBit) & 0x01;

                    int q = (qHigh << 4) | qLow;
                    dst[groupBase + i] = dg * q - mg;
                }
            }
        }
    }

    /// <summary>
    /// Dequantizes a Q6_K-encoded buffer into <paramref name="dst"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 210 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ6_K(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q6KBlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q6KBlockBytes} for Q6_K.", nameof(src));
        }

        int blockCount = src.Length / Q6KBlockBytes;
        int elementCount = blockCount * Q6KBlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q6KBlockBytes, Q6KBlockBytes);
            var ql = block.Slice(0, 128);
            var qh = block.Slice(128, 64);
            var scales = block.Slice(192, 16);
            float d = HalfToFloat(BinaryPrimitives.ReadUInt16LittleEndian(block.Slice(208, 2)));

            int dstBase = b * Q6KBlockSize;
            for (int i = 0; i < Q6KBlockSize; i++)
            {
                byte qlByte = ql[i >> 1];
                int low = (i & 1) == 0 ? (qlByte & 0x0F) : (qlByte >> 4);

                byte qhByte = qh[i >> 2];
                int high = (qhByte >> ((i & 0x03) * 2)) & 0x03;

                int q = ((high << 4) | low) - 32;
                int scale = (sbyte)scales[i >> 4];
                dst[dstBase + i] = d * scale * q;
            }
        }
    }

    /// <summary>
    /// Dequantizes a Q8_K-encoded buffer into <paramref name="dst"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="src"/> length is not a multiple of 292 or
    /// <paramref name="dst"/> does not have the required capacity.
    /// </exception>
#pragma warning disable CA1707 // Standard GGML/ML naming convention
    public static void DequantizeQ8_K(ReadOnlySpan<byte> src, Span<float> dst)
#pragma warning restore CA1707
    {
        if (src.Length % Q8KBlockBytes != 0)
        {
            throw new ArgumentException(
                $"Source length must be a multiple of {Q8KBlockBytes} for Q8_K.", nameof(src));
        }

        int blockCount = src.Length / Q8KBlockBytes;
        int elementCount = blockCount * Q8KBlockSize;
        if (dst.Length < elementCount)
        {
            throw new ArgumentException(
                $"Destination must hold at least {elementCount} floats.", nameof(dst));
        }

        for (int b = 0; b < blockCount; b++)
        {
            var block = src.Slice(b * Q8KBlockBytes, Q8KBlockBytes);
            float d = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(block.Slice(0, 4)));
            var qs = block.Slice(4, 256);

            int dstBase = b * Q8KBlockSize;
            for (int i = 0; i < Q8KBlockSize; i++)
            {
                dst[dstBase + i] = (sbyte)qs[i] * d;
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
            0  => byteLength / 4,
            1  => byteLength / 2,
            2  => byteLength / Q4BlockBytes * Q4BlockSize,
            8  => byteLength / Q8BlockBytes * Q8BlockSize,
            12 => byteLength / Q4KBlockBytes * Q4KBlockSize,
            13 => byteLength / Q5KBlockBytes * Q5KBlockSize,
            14 => byteLength / Q6KBlockBytes * Q6KBlockSize,
            15 => byteLength / Q8KBlockBytes * Q8KBlockSize,
            30 => byteLength / 2,
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
            case 12: DequantizeQ4_K(src, dst); break;
            case 13: DequantizeQ5_K(src, dst); break;
            case 14: DequantizeQ6_K(src, dst); break;
            case 15: DequantizeQ8_K(src, dst); break;
            case 30: DequantizeBF16(src, dst); break;
            default: throw new NotSupportedException(
                $"Tensor type {tensorType} is not supported for dequantization.");
        }
    }

    private static void DecodeKScaleMin(ReadOnlySpan<byte> packed, int groupIndex, out int scale, out int min)
    {
        System.Diagnostics.Debug.Assert((uint)groupIndex < 8,
            "DecodeKScaleMin: groupIndex must be in [0,7].");

        if (groupIndex < 4)
        {
            scale = packed[groupIndex] & 0x3F;
            min = packed[groupIndex + 4] & 0x3F;
            return;
        }

        int i = groupIndex - 4;
        scale = (packed[groupIndex + 4] & 0x0F) | ((packed[i] >> 6) << 4);
        min = (packed[groupIndex + 4] >> 4) | ((packed[i + 4] >> 6) << 4);
    }
}
