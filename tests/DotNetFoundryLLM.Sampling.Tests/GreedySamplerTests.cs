using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="GreedySampler"/>.</summary>
public sealed class GreedySamplerTests
{
    [Fact]
    public void Sample_ReturnsMostProbableToken()
    {
        float[] logits = [0.1f, 5.0f, 0.3f, 0.2f];
        var sampler = new GreedySampler();
        Assert.Equal(1, sampler.Sample(logits));
    }

    [Fact]
    public void Sample_FirstElementIsMax()
    {
        float[] logits = [100f, 0f, -1f];
        Assert.Equal(0, GreedySampler.Instance.Sample(logits));
    }

    [Fact]
    public void Sample_LastElementIsMax()
    {
        float[] logits = [-1f, 0f, 99f];
        Assert.Equal(2, GreedySampler.Instance.Sample(logits));
    }

    [Fact]
    public void Sample_SingleElement_ReturnsZero()
    {
        float[] logits = [42f];
        Assert.Equal(0, GreedySampler.Instance.Sample(logits));
    }

    [Fact]
    public void Sample_IsDeterministic_SameInputSameOutput()
    {
        float[] logits = [0.5f, 0.3f, 0.9f, 0.1f];
        var result1 = GreedySampler.Instance.Sample(logits);
        var result2 = GreedySampler.Instance.Sample(logits);
        Assert.Equal(result1, result2);
    }
}
