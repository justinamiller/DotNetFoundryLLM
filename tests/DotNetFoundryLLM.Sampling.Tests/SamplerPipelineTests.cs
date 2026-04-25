using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="SamplerPipeline"/>.</summary>
public sealed class SamplerPipelineTests
{
    [Fact]
    public void Constructor_NegativeTemperature_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SamplerPipeline(temperature: -1f));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1.1f)]
    public void Constructor_InvalidTopP_Throws(float p)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SamplerPipeline(topP: p));
    }

    [Fact]
    public void Sample_ZeroTemperature_Greedy()
    {
        // Temperature=0 → deterministic argmax.
        var pipeline = new SamplerPipeline(temperature: 0f);
        float[] logits = [0.1f, 5.0f, 3.0f];
        Assert.Equal(1, pipeline.Sample(logits));
    }

    [Fact]
    public void Sample_AlwaysReturnsValidIndex()
    {
        var pipeline = new SamplerPipeline(temperature: 1.0f, topK: 2, topP: 0.9f, seed: 7);
        float[] logits = [0.1f, 0.5f, 0.3f, 0.1f];
        for (int i = 0; i < 100; i++)
        {
            var copy = (float[])logits.Clone();
            int result = pipeline.Sample(copy);
            Assert.InRange(result, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Sample_TopK1_AlwaysReturnsArgmax()
    {
        var pipeline = new SamplerPipeline(temperature: 1.0f, topK: 1, seed: 42);
        float[] logits = [1f, 10f, 0.5f, 3f];
        for (int i = 0; i < 20; i++)
        {
            var copy = (float[])logits.Clone();
            Assert.Equal(1, pipeline.Sample(copy));
        }
    }

    [Fact]
    public void Sample_IsDeterministicWithSameSeed()
    {
        float[] logits = [0.5f, 0.3f, 0.9f, 0.1f];
        var pipeline1 = new SamplerPipeline(temperature: 1.0f, seed: 123);
        var pipeline2 = new SamplerPipeline(temperature: 1.0f, seed: 123);

        for (int i = 0; i < 50; i++)
        {
            var copy1 = (float[])logits.Clone();
            var copy2 = (float[])logits.Clone();
            Assert.Equal(pipeline1.Sample(copy1), pipeline2.Sample(copy2));
        }
    }
}
