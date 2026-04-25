namespace DotNetFoundryLLM.Tokenization;

/// <summary>
/// Provides the GPT-2 byte-to-Unicode encoding used by BPE tokenizers.
/// Maps all 256 possible byte values to unique, printable Unicode characters so that any byte
/// sequence can be represented as a string without loss of information or unknown characters.
/// Bytes that are already printable ASCII or extended Latin map to themselves; the remaining
/// 68 bytes (control characters and soft hyphen) map to Unicode code points starting at U+0100.
/// </summary>
public static class ByteEncoder
{
    private static readonly char[] s_byteToChar = BuildByteToChar();
    private static readonly byte[] s_charToByteTable = BuildCharToByteTable(s_byteToChar);

    private static char[] BuildByteToChar()
    {
        // "Nice" bytes: printable ASCII (33-126) and extended Latin (161-172, 174-255)
        // All other byte values (0-32, 127-160, 173) get mapped to U+0100 and above.
        var isNice = new bool[256];
        for (int i = 33; i <= 126; i++) isNice[i] = true;   // '!' through '~'
        for (int i = 161; i <= 172; i++) isNice[i] = true;  // '¡' through '¬'
        for (int i = 174; i <= 255; i++) isNice[i] = true;  // '®' through 'ÿ'

        var table = new char[256];
        int extra = 256;
        for (int b = 0; b < 256; b++)
        {
            table[b] = isNice[b] ? (char)b : (char)extra++;
        }

        return table;
    }

    private static byte[] BuildCharToByteTable(char[] byteToChar)
    {
        // Determine the highest char value in the mapping (max is 323 = 256 + 67).
        int maxChar = 0;
        foreach (var c in byteToChar)
        {
            if (c > maxChar) maxChar = c;
        }

        var table = new byte[maxChar + 1];
        for (int b = 0; b < 256; b++)
        {
            table[byteToChar[b]] = (byte)b;
        }

        return table;
    }

    /// <summary>Converts a byte value to its unique BPE-encoded Unicode character.</summary>
    public static char ByteToChar(byte b) => s_byteToChar[b];

    /// <summary>
    /// Converts a BPE-encoded Unicode character back to its original byte value.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown if <paramref name="c"/> is not a valid BPE-encoded character.
    /// </exception>
    public static byte CharToByte(char c)
    {
        if (c >= s_charToByteTable.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(c), c,
                "Character is not a valid BPE-encoded character.");
        }

        return s_charToByteTable[c];
    }
}
