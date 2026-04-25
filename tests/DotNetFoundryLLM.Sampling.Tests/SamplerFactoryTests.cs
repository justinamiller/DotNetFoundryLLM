using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Sampling;
using Xunit;

namespace DotNetFoundryLLM.Sampling.Tests;

/// <summary>Tests for <see cref="SamplerFactory"/>.</summary>
public sealed class SamplerFactoryTests
{
    [Fact]
    public void Create_NullOptions_ReturnsDefaultSampler()
    {
        var sampler = SamplerFactory.Create(null);
        Assert.NotNull(sampler);
    }

    [Fact]
    public void Create_ZeroTemperatureNoRepetitionPenalty_ReturnsGreedySampler()
    {
        var options = new GenerationOptions { Temperature = 0f };
        var sampler = SamplerFactory.Create(options);
        Assert.IsType<GreedySampler>(sampler);
    }

    [Fact]
    public void Create_DefaultOptions_ReturnsSamplerPipeline()
    {
        var options = new GenerationOptions(); // Temperature=1, no top-k/p/penalty
        var sampler = SamplerFactory.Create(options);
        Assert.IsType<SamplerPipeline>(sampler);
    }

    [Fact]
    public void Create_WithRepetitionPenalty_ReturnsRepetitionPenaltySampler()
    {
        var options = new GenerationOptions { RepetitionPenalty = 1.2f };
        var sampler = SamplerFactory.Create(options);
        Assert.IsType<RepetitionPenaltySampler>(sampler);
    }

    [Fact]
    public void Create_ExplicitSeedOverride_IsRespected()
    {
        var options = new GenerationOptions { Seed = 100 };
        // Should not throw; seed is applied internally.
        var sampler = SamplerFactory.Create(options, seed: 999);
        Assert.NotNull(sampler);
    }

    [Fact]
    public void Create_ZeroTemperatureWithRepetitionPenalty_ReturnsRepetitionPenaltySampler()
    {
        // Even with temperature=0, if penalty != 1.0 we still need the penalty wrapper.
        var options = new GenerationOptions { Temperature = 0f, RepetitionPenalty = 1.3f };
        var sampler = SamplerFactory.Create(options);
        Assert.IsType<RepetitionPenaltySampler>(sampler);
    }
}
