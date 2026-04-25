using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="TopKSampler"/>.</summary>
public sealed class TopKSamplerTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_InvalidK_Throws(int k)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopKSampler(k));
    }

    [Fact]
    public void Sample_AlwaysReturnsValidIndex()
    {
        var sampler = new TopKSampler(k: 2, seed: 7);
        float[] logits = [1f, 5f, 0.5f, 3f];
        for (int i = 0; i < 100; i++)
        {
            var copy = (float[])logits.Clone();
            int result = sampler.Sample(copy);
            Assert.InRange(result, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Sample_K1_AlwaysReturnsArgmax()
    {
        // k=1 means only the highest-logit token can be sampled.
        var sampler = new TopKSampler(k: 1, seed: 5);
        float[] logits = [1f, 5f, 0.5f, 3f]; // max is index 1

        for (int i = 0; i < 20; i++)
        {
            var copy = (float[])logits.Clone();
            Assert.Equal(1, sampler.Sample(copy));
        }
    }

    [Fact]
    public void ApplyTopK_SetsLowLogitsToNegInfinity()
    {
        float[] logits = [1f, 5f, 0.5f, 3f];
        TopKSampler.ApplyTopK(logits, k: 2);
        // Only the top-2 (indices 1 and 3) should remain; others → -inf.
        Assert.Equal(float.NegativeInfinity, logits[0]);
        Assert.Equal(float.NegativeInfinity, logits[2]);
        Assert.True(logits[1] > float.NegativeInfinity);
        Assert.True(logits[3] > float.NegativeInfinity);
    }

    [Fact]
    public void ApplyTopK_KGreaterThanOrEqualLength_LeavesAllLogits()
    {
        float[] logits = [1f, 2f, 3f];
        var original = (float[])logits.Clone();
        TopKSampler.ApplyTopK(logits, k: 10);
        Assert.Equal(original, logits);
    }

    [Fact]
    public void Sample_OnlyTopKTokensCanBeChosen()
    {
        // logits: index 0=0.1, 1=10, 2=0.2, 3=0.1. With k=1 only index 1 can be chosen.
        float[] logits = [0.1f, 10f, 0.2f, 0.1f];
        var sampler = new TopKSampler(k: 1, seed: 42);
        for (int i = 0; i < 30; i++)
        {
            var copy = (float[])logits.Clone();
            Assert.Equal(1, sampler.Sample(copy));
        }
    }
}
