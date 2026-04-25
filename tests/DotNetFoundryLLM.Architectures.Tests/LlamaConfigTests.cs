using DotNetFoundryLLM.Architectures;
using Xunit;

namespace DotNetFoundryLLM.Architectures.Tests;

/// <summary>Tests for <see cref="LlamaConfig"/>.</summary>
public sealed class LlamaConfigTests
{
    private static LlamaConfig MakeConfig(
        int layers = 8,
        int hidden = 512,
        int ffn = 1376,
        int heads = 8,
        int kvHeads = 4,
        int ctx = 2048,
        int vocab = 32000) =>
        new()
        {
            LayerCount       = layers,
            HiddenSize       = hidden,
            IntermediateSize = ffn,
            NumHeads         = heads,
            NumKvHeads       = kvHeads,
            MaxContextLength = ctx,
            VocabSize        = vocab,
        };

    [Fact]
    public void HeadDim_IsHiddenDivHeads()
    {
        var cfg = MakeConfig(hidden: 512, heads: 8);
        Assert.Equal(64, cfg.HeadDim);
    }

    [Fact]
    public void QueryDim_IsNumHeadsTimesHeadDim()
    {
        var cfg = MakeConfig(hidden: 512, heads: 8);
        Assert.Equal(512, cfg.QueryDim);
    }

    [Fact]
    public void KvDim_IsNumKvHeadsTimesHeadDim()
    {
        var cfg = MakeConfig(hidden: 512, heads: 8, kvHeads: 4);
        Assert.Equal(256, cfg.KvDim);
    }

    [Fact]
    public void KvDim_MatchesQueryDim_WhenKvHeadsEqualsNumHeads()
    {
        var cfg = MakeConfig(hidden: 512, heads: 8, kvHeads: 8);
        Assert.Equal(cfg.QueryDim, cfg.KvDim);
    }

    [Fact]
    public void ToString_ContainsArchitecture()
    {
        var cfg = MakeConfig();
        Assert.Contains("llama", cfg.ToString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DefaultValues_AreReasonable()
    {
        var cfg = MakeConfig();
        Assert.Equal(10000f, cfg.RopeBaseFreq);
        Assert.Equal(1, cfg.BosTokenId);
        Assert.Equal(2, cfg.EosTokenId);
    }
}
