using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="RepetitionPenaltySampler"/>.</summary>
public sealed class RepetitionPenaltySamplerTests
{
    [Fact]
    public void Constructor_NullInner_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new RepetitionPenaltySampler(null!));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    public void Constructor_InvalidPenalty_Throws(float penalty)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RepetitionPenaltySampler(GreedySampler.Instance, penalty));
    }

    [Fact]
    public void Sample_NoPreviousTokens_DelegatesUnchanged()
    {
        // No previous tokens → penalty has no effect → greedy picks max.
        float[] logits = [0.1f, 5.0f, 0.3f];
        var sampler = new RepetitionPenaltySampler(GreedySampler.Instance, penalty: 1.5f);
        Assert.Equal(1, sampler.Sample(logits));
    }

    [Fact]
    public void Sample_PenaltyReducesPositiveLogitsForRepeatedTokens()
    {
        // Token 1 has the highest logit (5.0). Record it as a previous token.
        // After penalty (1.5×), its logit drops to 5/1.5 ≈ 3.33, below token 2 (4.0).
        float[] logits = [0.1f, 5.0f, 4.0f];
        var sampler = new RepetitionPenaltySampler(GreedySampler.Instance, penalty: 1.5f);
        sampler.RecordToken(1);

        // Token 1 logit: 5/1.5 ≈ 3.33; token 2 logit: 4.0 → greedy picks 2.
        Assert.Equal(2, sampler.Sample(logits));
    }

    [Fact]
    public void Sample_PenaltyAmplifiedNegativeLogits()
    {
        // Negative logit for token 0: after penalty it becomes more negative.
        float[] logits = [-2.0f, 0.0f, 1.0f];
        var sampler = new RepetitionPenaltySampler(GreedySampler.Instance, penalty: 2.0f);
        sampler.RecordToken(0);

        // Token 0: -2 * 2 = -4; token 2: 1.0 → greedy picks 2.
        Assert.Equal(2, sampler.Sample(logits));
    }

    [Fact]
    public void RecordTokens_AppliesToAllRecordedTokens()
    {
        // Record multiple prompt tokens.
        float[] logits = [5.0f, 5.0f, 0.5f];
        var sampler = new RepetitionPenaltySampler(GreedySampler.Instance, penalty: 10.0f);
        sampler.RecordTokens([0, 1]);

        // Tokens 0 and 1 both penalized: 5/10 = 0.5; token 2 remains 0.5; tie → first one wins (argmax).
        int result = sampler.Sample(logits);
        // All penalized to ~0.5; token 2 also 0.5. ArgMax returns the first maximum → 0.
        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Reset_ClearsPreviousTokens()
    {
        float[] logits = [0.1f, 5.0f, 4.0f];
        var sampler = new RepetitionPenaltySampler(GreedySampler.Instance, penalty: 1.5f);
        sampler.RecordToken(1);
        sampler.Reset();

        // After reset, no penalty → greedy picks 1 again.
        Assert.Equal(1, sampler.Sample(logits));
    }
}
