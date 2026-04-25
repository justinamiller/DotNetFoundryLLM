using Xunit;

namespace DotNetFoundryLLM.Tokenization.Tests;

/// <summary>Tests for <see cref="TokenizerVocab"/>.</summary>
public sealed class TokenizerVocabTests
{
    private static TokenizerVocab BuildSmallVocab()
    {
        // Minimal vocab: <unk>=0, <s>=1, </s>=2, h=3, e=4, l=5, o=6, he=7, ll=8
        string[] tokens =
        [
            "<unk>", "<s>", "</s>",
            "h", "e", "l", "o",
            "he", "ll"
        ];

        (string, string)[] merges =
        [
            ("h", "e"),  // rank 0
            ("l", "l"),  // rank 1
        ];

        return new TokenizerVocab(tokens, merges);
    }

    [Fact]
    public void VocabSize_ReturnsTokenCount()
    {
        var vocab = BuildSmallVocab();
        Assert.Equal(9, vocab.VocabSize);
    }

    [Fact]
    public void DefaultSpecialTokenIds_AreCorrect()
    {
        var vocab = BuildSmallVocab();
        Assert.Equal(1, vocab.BosTokenId);
        Assert.Equal(2, vocab.EosTokenId);
        Assert.Equal(0, vocab.UnknownTokenId);
    }

    [Fact]
    public void TryGetId_KnownToken_ReturnsTrueAndCorrectId()
    {
        var vocab = BuildSmallVocab();
        Assert.True(vocab.TryGetId("he", out var id));
        Assert.Equal(7, id);
    }

    [Fact]
    public void TryGetId_UnknownToken_ReturnsFalse()
    {
        var vocab = BuildSmallVocab();
        Assert.False(vocab.TryGetId("xyz", out _));
    }

    [Fact]
    public void GetToken_ValidId_ReturnsCorrectString()
    {
        var vocab = BuildSmallVocab();
        Assert.Equal("ll", vocab.GetToken(8));
    }

    [Fact]
    public void GetToken_OutOfRange_ThrowsArgumentOutOfRangeException()
    {
        var vocab = BuildSmallVocab();
        Assert.Throws<ArgumentOutOfRangeException>(() => vocab.GetToken(999));
    }

    [Fact]
    public void MergeRanks_ContainsCorrectRanks()
    {
        var vocab = BuildSmallVocab();
        Assert.Equal(0, vocab.MergeRanks[("h", "e")]);
        Assert.Equal(1, vocab.MergeRanks[("l", "l")]);
    }

    [Fact]
    public void AddedTokens_OverrideVocab()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "a"];
        (string, string)[] merges = [];
        var added = new Dictionary<string, int> { ["<custom>"] = 99 };

        var vocab = new TokenizerVocab(tokens, merges, addedTokens: added);

        Assert.True(vocab.TryGetId("<custom>", out var id));
        Assert.Equal(99, id);
    }
}
