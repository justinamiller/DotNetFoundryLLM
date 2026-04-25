using System.Text.RegularExpressions;
using Xunit;

namespace DotNetFoundryLLM.Tokenization.Tests;

/// <summary>Tests for <see cref="BpeTokenizer"/>.</summary>
public sealed class BpeTokenizerTests
{
    // -------------------------------------------------------------------------
    // Shared helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Builds a minimal BPE vocabulary for the ASCII letters used in "hello" and "world".
    /// Layout:
    ///   0=&lt;unk&gt;, 1=&lt;s&gt;, 2=&lt;/s&gt;
    ///   Single-char tokens (byte-encoded via ByteEncoder, which maps printable ASCII
    ///   to itself, so 'h'→'h', etc.):
    ///   3=h, 4=e, 5=l, 6=o, 7=w, 8=r, 9=d, 10=Ġ (space — 0x20 maps to Ġ = U+0120)
    ///   Merged tokens:
    ///   11=he (h+e, rank 0)
    ///   12=ll (l+l, rank 1)
    ///   13=Ġw (space+w, rank 2)
    ///   14=or (o+r, rank 3)
    ///   15=ld (l+d, rank 4)
    ///   16=Ġworld (Ġw+or+ld composed: Ġwor then Ġworl then Ġworld — needs extra merges)
    /// Rather than a full chain, we keep merges limited so tests remain readable.
    /// </summary>
    private static BpeTokenizer BuildTokenizer()
    {
        // Space (0x20) is not printable-ASCII in the GPT-2 scheme (only 33-126 are).
        // ByteEncoder.ByteToChar(0x20) returns the char at position 256+n where n counts
        // the non-printable bytes before 0x20 (bytes 0-31 → 32 entries, so space=byte 32
        // maps to char 256+32 = U+0108). Let us use a fixed single-word test instead
        // so we avoid space-handling complexity.

        // Tokens indexed by ID.
        string[] tokens =
        [
            "<unk>",   // 0
            "<s>",     // 1
            "</s>",    // 2
            "h",       // 3
            "e",       // 4
            "l",       // 5
            "o",       // 6
            "he",      // 7  — merge("h","e") rank 0
            "ll",      // 8  — merge("l","l") rank 1
            "hel",     // 9  — merge("he","l") rank 2
            "hell",    // 10 — merge("hel","l") rank 3  (uses "l" from id=5)
            "hello",   // 11 — merge("hell","o") rank 4
        ];

        (string, string)[] merges =
        [
            ("h", "e"),     // rank 0 → "he"
            ("l", "l"),     // rank 1 → "ll"
            ("he", "l"),    // rank 2 → "hel"
            ("hel", "l"),   // rank 3 → "hell"
            ("hell", "o"),  // rank 4 → "hello"
        ];

        var vocab = new TokenizerVocab(tokens, merges);
        return new BpeTokenizer(vocab);
    }

    // Use a simple identity split pattern so "hello" is treated as one chunk.
    private static readonly Regex WholeWordPattern = new(@"\S+", RegexOptions.CultureInvariant);

    // -------------------------------------------------------------------------
    // VocabSize
    // -------------------------------------------------------------------------

    [Fact]
    public void VocabSize_ReturnsVocabTokenCount()
    {
        var tokenizer = BuildTokenizer();
        Assert.Equal(12, tokenizer.VocabSize);
    }

    // -------------------------------------------------------------------------
    // Encode
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_EmptyText_ReturnsBosOnly()
    {
        var tokenizer = BuildTokenizer();
        var ids = tokenizer.Encode("".AsSpan(), addBos: true, addEos: false);
        Assert.Equal([1], ids.Span.ToArray());
    }

    [Fact]
    public void Encode_EmptyText_NoBosNoEos_ReturnsEmpty()
    {
        var tokenizer = BuildTokenizer();
        var ids = tokenizer.Encode("".AsSpan(), addBos: false, addEos: false);
        Assert.Empty(ids.Span.ToArray());
    }

    [Fact]
    public void Encode_Hello_FullyMergedToSingleToken()
    {
        // Merge chain that fully collapses "hello" into one token:
        //   ['h','e','l','l','o']
        //   → rank 0: ("h","e")   → ['he','l','l','o']
        //   → rank 1: ("l","l")   → ['he','ll','o']
        //   → rank 2: ("he","ll") → ['hell','o']
        //   → rank 3: ("hell","o")→ ['hello']  → id 10
        string[] tokens =
        [
            "<unk>", "<s>", "</s>",
            "h", "e", "l", "o",   // 3-6
            "he",                  // 7
            "ll",                  // 8
            "hell",                // 9
            "hello",               // 10
        ];

        (string, string)[] merges =
        [
            ("h", "e"),      // rank 0
            ("l", "l"),      // rank 1
            ("he", "ll"),    // rank 2
            ("hell", "o"),   // rank 3
        ];

        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, splitPattern: WholeWordPattern);

        var ids = tokenizer.Encode("hello".AsSpan(), addBos: false, addEos: false);
        Assert.Equal([10], ids.Span.ToArray());
    }

    [Fact]
    public void Encode_Hello_PartialMerge()
    {
        // Vocab with only the first two merge rules: "h"+"e" and "l"+"l".
        // "hello" → ['h','e','l','l','o'] → merge h+e → ['he','l','l','o']
        //                                  → merge l+l → ['he','ll','o']
        // IDs: he=7, ll=8, o=6
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        var ids = tokenizer.Encode("hello".AsSpan(), addBos: false, addEos: false);
        Assert.Equal([7, 8, 6], ids.Span.ToArray());
    }

    [Fact]
    public void Encode_AddsBosAndEos()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        var ids = tokenizer.Encode("hello".AsSpan(), addBos: true, addEos: true).Span.ToArray();

        // [BOS=1, he=7, ll=8, o=6, EOS=2]
        Assert.Equal(1, ids[0]);
        Assert.Equal(2, ids[^1]);
        Assert.Equal(5, ids.Length);
    }

    [Fact]
    public void Encode_UnknownByte_UsesUnknownTokenId()
    {
        // Vocabulary with no real tokens (only specials), so everything falls back to id=0.
        string[] tokens = ["<unk>", "<s>", "</s>"];
        (string, string)[] merges = [];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        var ids = tokenizer.Encode("ab".AsSpan(), addBos: false, addEos: false);
        Assert.All(ids.Span.ToArray(), id => Assert.Equal(0, id));
    }

    // -------------------------------------------------------------------------
    // Decode
    // -------------------------------------------------------------------------

    [Fact]
    public void Decode_SkipsBosAndEos()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        // Explicitly include BOS and EOS token IDs in the sequence.
        var text = tokenizer.Decode([1, 7, 8, 6, 2]);
        Assert.Equal("hello", text);
    }

    [Fact]
    public void Decode_EmptySpan_ReturnsEmptyString()
    {
        var tokenizer = BuildTokenizer();
        Assert.Equal(string.Empty, tokenizer.Decode([]));
    }

    // -------------------------------------------------------------------------
    // DecodeToken
    // -------------------------------------------------------------------------

    [Fact]
    public void DecodeToken_SingleChar_ReturnsCorrectUtf8()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        Assert.Equal("h", tokenizer.DecodeToken(3));
        Assert.Equal("e", tokenizer.DecodeToken(4));
    }

    [Fact]
    public void DecodeToken_MergedToken_ReturnsFullString()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        Assert.Equal("he", tokenizer.DecodeToken(7));
        Assert.Equal("ll", tokenizer.DecodeToken(8));
    }

    [Fact]
    public void DecodeToken_InvalidId_ThrowsArgumentOutOfRangeException()
    {
        var tokenizer = BuildTokenizer();
        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.DecodeToken(9999));
    }

    // -------------------------------------------------------------------------
    // Round-trip
    // -------------------------------------------------------------------------

    [Fact]
    public void RoundTrip_EncodeDecodeIsIdentity()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll"];
        (string, string)[] merges = [("h", "e"), ("l", "l")];
        var vocab = new TokenizerVocab(tokens, merges);
        var tokenizer = new BpeTokenizer(vocab, WholeWordPattern);

        var ids = tokenizer.Encode("hello".AsSpan(), addBos: false, addEos: false);
        var decoded = tokenizer.Decode(ids.Span);

        Assert.Equal("hello", decoded);
    }
}
