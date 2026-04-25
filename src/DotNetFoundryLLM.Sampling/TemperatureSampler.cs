using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// Samples tokens from a categorical distribution formed by applying softmax to the logits.
/// An optional temperature parameter scales the logits before softmax — higher temperature
/// yields a flatter (more random) distribution; lower temperature sharpens it.
/// </summary>
public sealed class TemperatureSampler : ISampler
{
    private readonly float _temperature;
    private readonly XoshiroRandom _random;

    /// <summary>
    /// Initializes a new temperature sampler.
    /// </summary>
    /// <param name="temperature">
    /// Scaling factor applied to logits before softmax.
    /// Must be greater than zero. Values close to 0 approach greedy; values above 1 increase randomness.
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility. <c>0</c> = random seed.</param>
    public TemperatureSampler(float temperature = 1.0f, ulong seed = 0)
    {
        if (temperature <= 0f) throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be greater than zero.");
        _temperature = temperature;
        _random = seed == 0 ? new XoshiroRandom((ulong)Environment.TickCount64) : new XoshiroRandom(seed);
    }

    /// <inheritdoc />
    /// <remarks>The logits span is modified in-place (scaled and converted to probabilities).</remarks>
    public int Sample(Span<float> logits)
    {
        // Scale by inverse temperature.
        if (_temperature != 1.0f)
        {
            float invTemp = 1.0f / _temperature;
            TensorOperations.Scale(logits, invTemp, logits);
        }

        // Convert to probability distribution.
        TensorOperations.Softmax(logits);

        return SampleFromProbabilities(logits, _random);
    }

    /// <summary>
    /// Samples a token index using weighted random selection from a probability distribution.
    /// Uses Gumbel-max / inverse CDF method: draw a uniform random number and walk the CDF.
    /// </summary>
    public static int SampleFromProbabilities(ReadOnlySpan<float> probs, XoshiroRandom rng)
    {
        float u = rng.NextFloat();
        float cumulative = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (u < cumulative) return i;
        }

        // Fallback in case of floating-point rounding: return last non-zero.
        for (int i = probs.Length - 1; i >= 0; i--)
        {
            if (probs[i] > 0f) return i;
        }

        return probs.Length - 1;
    }
}
