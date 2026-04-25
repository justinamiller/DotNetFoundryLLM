using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="TopPSampler"/>.</summary>
public sealed class TopPSamplerTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_InvalidP_Throws(float p)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampler(p));
    }

    [Fact]
    public void Sample_AlwaysReturnsValidIndex()
    {
        var sampler = new TopPSampler(p: 0.9f, seed: 11);
        float[] logits = [0.1f, 0.5f, 0.3f, 0.1f];
        for (int i = 0; i < 100; i++)
        {
            var copy = (float[])logits.Clone();
            int result = sampler.Sample(copy);
            Assert.InRange(result, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void ApplyTopP_P1_LeavesDistributionUnchanged()
    {
        float[] probs = [0.1f, 0.6f, 0.2f, 0.1f];
        var original = (float[])probs.Clone();
        TopPSampler.ApplyTopP(probs, 1.0f);
        // All probabilities should be unchanged (still sum to 1, same shape).
        for (int i = 0; i < probs.Length; i++)
        {
            Assert.Equal(original[i], probs[i], precision: 5);
        }
    }

    [Fact]
    public void ApplyTopP_SmallP_ZeroesOutLowProbTokens()
    {
        // Distribution: [0.02, 0.90, 0.05, 0.03], p=0.91.
        // Sorted descending: 0.90 (idx1), 0.05 (idx2), 0.03 (idx3), 0.02 (idx0).
        // Cumulative walk: 0.90 → not yet >= 0.91; 0.90+0.05=0.95 → nucleus reached (include idx2).
        // After: idx1 (0.90) and idx2 (0.05) remain; idx3 and idx0 zeroed.
        float[] probs = [0.02f, 0.90f, 0.05f, 0.03f];
        TopPSampler.ApplyTopP(probs, p: 0.91f);

        Assert.Equal(0f, probs[0]); // 0.02 → zeroed
        Assert.True(probs[1] > 0f); // 0.90 → kept
        Assert.True(probs[2] > 0f); // 0.05 → kept (pushed cumulative over threshold)
        Assert.Equal(0f, probs[3]); // 0.03 → zeroed
    }

    [Fact]
    public void ApplyTopP_RenormalizesRemainingProbabilities()
    {
        float[] probs = [0.1f, 0.6f, 0.2f, 0.1f];
        TopPSampler.ApplyTopP(probs, p: 0.8f);
        float sum = probs.Sum();
        Assert.True(MathF.Abs(sum - 1f) < 1e-5f, $"Expected sum ~1, got {sum}");
    }

    [Fact]
    public void Sample_P1_CanReturnAnyToken()
    {
        // With p=1 no filtering; all tokens remain eligible.
        var sampler = new TopPSampler(p: 1.0f, seed: 999);
        float[] logits = [1f, 1f, 1f, 1f];
        bool[] seen = new bool[logits.Length];
        for (int i = 0; i < 200; i++)
        {
            var copy = (float[])logits.Clone();
            seen[sampler.Sample(copy)] = true;
        }

        // All tokens should be seen eventually.
        Assert.All(seen, s => Assert.True(s));
    }
}
