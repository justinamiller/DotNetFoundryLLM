using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// Creates <see cref="ISampler"/> instances from <see cref="GenerationOptions"/>.
/// </summary>
public static class SamplerFactory
{
    /// <summary>
    /// Creates the appropriate sampler for the given generation options.
    /// <list type="bullet">
    ///   <item><description>
    ///     If <see cref="GenerationOptions.Temperature"/> is 0 and no repetition penalty is
    ///     configured, returns a singleton <see cref="GreedySampler"/>.
    ///   </description></item>
    ///   <item><description>
    ///     Otherwise, builds a <see cref="SamplerPipeline"/> (temperature + top-K + top-P)
    ///     and optionally wraps it in a <see cref="RepetitionPenaltySampler"/>.
    ///   </description></item>
    /// </list>
    /// </summary>
    /// <param name="options">The generation options. May be <see langword="null"/>, in which case defaults are used.</param>
    /// <param name="seed">
    /// Optional explicit seed override. When <c>null</c>, <see cref="GenerationOptions.Seed"/>
    /// is used; if that is also <see langword="null"/> a non-deterministic seed is chosen.
    /// </param>
    public static ISampler Create(GenerationOptions? options, int? seed = null)
    {
        options ??= new GenerationOptions();

        ulong rngSeed = seed.HasValue
            ? (ulong)seed.Value
            : options.Seed.HasValue
                ? (ulong)options.Seed.Value
                : 0UL;

        bool hasRepetitionPenalty = options.RepetitionPenalty != 1.0f;

        // Pure greedy: temperature=0 with no repetition penalty.
        if (options.Temperature == 0f && !hasRepetitionPenalty)
        {
            return GreedySampler.Instance;
        }

        ISampler core = new SamplerPipeline(
            temperature: options.Temperature,
            topK: options.TopK,
            topP: options.TopP,
            seed: rngSeed);

        if (hasRepetitionPenalty)
        {
            return new RepetitionPenaltySampler(core, options.RepetitionPenalty);
        }

        return core;
    }
}
