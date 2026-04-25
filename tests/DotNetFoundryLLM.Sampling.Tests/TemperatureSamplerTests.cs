using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="TemperatureSampler"/>.</summary>
public sealed class TemperatureSamplerTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    public void Constructor_InvalidTemperature_Throws(float temp)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TemperatureSampler(temp));
    }

    [Fact]
    public void Sample_AlwaysReturnsValidIndex()
    {
        var sampler = new TemperatureSampler(temperature: 1.0f, seed: 42);
        float[] logits = [0.1f, 0.5f, 0.3f, 0.1f];
        for (int i = 0; i < 100; i++)
        {
            var copy = (float[])logits.Clone();
            int result = sampler.Sample(copy);
            Assert.InRange(result, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Sample_UniformLogits_DistributionIsApproximatelyFlat()
    {
        // With uniform logits and temperature=1, each token should be equally likely.
        const int vocabSize = 4;
        const int iterations = 10000;
        float[] logits = [1f, 1f, 1f, 1f];
        var sampler = new TemperatureSampler(temperature: 1.0f, seed: 123);
        int[] counts = new int[vocabSize];
        for (int i = 0; i < iterations; i++)
        {
            var copy = (float[])logits.Clone();
            counts[sampler.Sample(copy)]++;
        }

        // Each token should appear roughly 25% of the time; allow ±8% tolerance.
        foreach (var count in counts)
        {
            Assert.InRange(count / (float)iterations, 0.17f, 0.33f);
        }
    }

    [Fact]
    public void Sample_HighTemperature_ProducesMoreVariation()
    {
        // High temperature (e.g. 2.0) should distribute samples more evenly than low temperature.
        float[] logits = [5f, 0.1f, 0.1f, 0.1f];
        const int iterations = 1000;

        var highTempSampler = new TemperatureSampler(temperature: 2.0f, seed: 1);
        var lowTempSampler = new TemperatureSampler(temperature: 0.1f, seed: 1);

        int highTempNonZero = 0, lowTempNonZero = 0;
        for (int i = 0; i < iterations; i++)
        {
            var copy = (float[])logits.Clone();
            if (highTempSampler.Sample(copy) != 0) highTempNonZero++;

            copy = (float[])logits.Clone();
            if (lowTempSampler.Sample(copy) != 0) lowTempNonZero++;
        }

        // High temperature should produce more non-zero index samples than low temperature.
        Assert.True(highTempNonZero > lowTempNonZero,
            $"High temp non-zero: {highTempNonZero}, low temp non-zero: {lowTempNonZero}");
    }
}
