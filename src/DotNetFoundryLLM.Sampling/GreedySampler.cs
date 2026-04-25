using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// Deterministic greedy sampler that always returns the token with the highest logit.
/// Equivalent to temperature = 0; no randomness.
/// </summary>
public sealed class GreedySampler : ISampler
{
    /// <summary>Singleton instance for convenience.</summary>
    public static readonly GreedySampler Instance = new();

    /// <inheritdoc />
    /// <remarks>The logits span is not modified.</remarks>
    public int Sample(Span<float> logits) => TensorOperations.ArgMax(logits);
}
