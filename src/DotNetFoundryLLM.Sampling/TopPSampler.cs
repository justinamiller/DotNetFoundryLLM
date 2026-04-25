using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// Top-P (nucleus) sampler: after applying softmax, retains the smallest set of tokens whose
/// cumulative probability mass reaches at least <c>p</c>, then samples from that nucleus.
/// Adapts the effective vocabulary size dynamically based on the distribution shape.
/// </summary>
public sealed class TopPSampler : ISampler
{
    private readonly float _p;
    private readonly XoshiroRandom _random;

    /// <summary>
    /// Initializes a new nucleus sampler.
    /// </summary>
    /// <param name="p">
    /// Cumulative probability cutoff in (0, 1]. A value of 1.0 is equivalent to unrestricted
    /// sampling; lower values restrict sampling to the most probable tokens.
    /// </param>
    /// <param name="seed">Optional random seed. <c>0</c> = random seed.</param>
    public TopPSampler(float p = 0.9f, ulong seed = 0)
    {
        if (p <= 0f || p > 1f) throw new ArgumentOutOfRangeException(nameof(p), "p must be in (0, 1].");
        _p = p;
        _random = seed == 0 ? new XoshiroRandom((ulong)Environment.TickCount64) : new XoshiroRandom(seed);
    }

    /// <inheritdoc />
    /// <remarks>The logits span is modified in-place (converted to probabilities, then filtered).</remarks>
    public int Sample(Span<float> logits)
    {
        TensorOperations.Softmax(logits);
        ApplyTopP(logits, _p);
        return TemperatureSampler.SampleFromProbabilities(logits, _random);
    }

    /// <summary>
    /// Filters <paramref name="probs"/> (already a probability distribution) in-place so that
    /// only the nucleus — the smallest set of tokens whose cumulative probability sums to at
    /// least <paramref name="p"/> — retains non-zero probability.  All other positions are set
    /// to zero. The distribution is then re-normalized so probabilities sum to 1.
    /// </summary>
    public static void ApplyTopP(Span<float> probs, float p)
    {
        if (p >= 1.0f) return; // No filtering needed.

        // Build an index array sorted by probability (descending).
        int[] indices = new int[probs.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        // Copy probabilities to an array so the lambda can capture it.
        float[] probsArray = new float[probs.Length];
        probs.CopyTo(probsArray);
        Array.Sort(indices, (a, b) => probsArray[b].CompareTo(probsArray[a]));

        // Walk sorted indices and zero out anything beyond the nucleus.
        float cumulative = 0f;
        bool nucleusReached = false;
        for (int rank = 0; rank < indices.Length; rank++)
        {
            if (nucleusReached)
            {
                probs[indices[rank]] = 0f;
            }
            else
            {
                cumulative += probs[indices[rank]];
                if (cumulative >= p)
                {
                    // Include this token, then start zeroing out the rest.
                    nucleusReached = true;
                }
            }
        }

        // Re-normalize so the remaining probabilities sum to 1.
        float sum = 0f;
        foreach (var v in probs) sum += v;
        if (sum > 0f)
        {
            for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        }
    }
}
