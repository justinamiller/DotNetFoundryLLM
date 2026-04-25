using Xunit;

namespace DotNetFoundryLLM.Tokenization.Tests;

/// <summary>Tests for <see cref="ByteEncoder"/>.</summary>
public sealed class ByteEncoderTests
{
    [Fact]
    public void ByteToChar_PrintableAscii_ReturnsSameChar()
    {
        // Printable ASCII bytes (33-126) must map to themselves.
        for (int b = 33; b <= 126; b++)
        {
            Assert.Equal((char)b, ByteEncoder.ByteToChar((byte)b));
        }
    }

    [Fact]
    public void ByteToChar_ExtendedLatin_ReturnsSameChar()
    {
        // Extended Latin bytes (161-172 and 174-255) must also map to themselves.
        for (int b = 161; b <= 172; b++)
        {
            Assert.Equal((char)b, ByteEncoder.ByteToChar((byte)b));
        }

        for (int b = 174; b <= 255; b++)
        {
            Assert.Equal((char)b, ByteEncoder.ByteToChar((byte)b));
        }
    }

    [Fact]
    public void ByteToChar_ControlBytes_MappedToHighCodepoints()
    {
        // Control characters (0-32) must NOT map to themselves.
        for (int b = 0; b <= 32; b++)
        {
            Assert.NotEqual((char)b, ByteEncoder.ByteToChar((byte)b));
        }
    }

    [Fact]
    public void AllBytesMappedToDistinctChars()
    {
        // The mapping must be a bijection — every byte produces a unique character.
        var chars = new HashSet<char>();
        for (int b = 0; b < 256; b++)
        {
            Assert.True(chars.Add(ByteEncoder.ByteToChar((byte)b)),
                $"Byte {b} produced a duplicate character '{ByteEncoder.ByteToChar((byte)b)}'.");
        }
    }

    [Fact]
    public void RoundTrip_AllBytes()
    {
        // ByteToChar followed by CharToByte must recover the original byte for all values.
        for (int b = 0; b < 256; b++)
        {
            var c = ByteEncoder.ByteToChar((byte)b);
            Assert.Equal((byte)b, ByteEncoder.CharToByte(c));
        }
    }

    [Fact]
    public void CharToByte_InvalidChar_ThrowsArgumentOutOfRangeException()
    {
        // A character well outside the encoded range should throw.
        Assert.Throws<ArgumentOutOfRangeException>(() => ByteEncoder.CharToByte('\uFFFF'));
    }
}
