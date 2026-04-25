using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// Top-K sampler: zeroes out all logits except the K highest, applies softmax, then
/// samples from the resulting truncated distribution.
/// Ensures the model cannot sample from the long tail of low-probability tokens.
/// </summary>
public sealed class TopKSampler : ISampler
{
    private readonly int _k;
    private readonly XoshiroRandom _random;

    /// <summary>
    /// Initializes a new top-K sampler.
    /// </summary>
    /// <param name="k">Number of top tokens to keep. Must be at least 1.</param>
    /// <param name="seed">Optional random seed. <c>0</c> = random seed.</param>
    public TopKSampler(int k, ulong seed = 0)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), "k must be at least 1.");
        _k = k;
        _random = seed == 0 ? new XoshiroRandom((ulong)Environment.TickCount64) : new XoshiroRandom(seed);
    }

    /// <inheritdoc />
    /// <remarks>The logits span is modified in-place (filtered logits → probabilities).</remarks>
    public int Sample(Span<float> logits)
    {
        ApplyTopK(logits, _k);
        TensorOperations.Softmax(logits);
        return TemperatureSampler.SampleFromProbabilities(logits, _random);
    }

    /// <summary>
    /// Zeroes out all positions in <paramref name="logits"/> except the top-<paramref name="k"/>
    /// values, setting eliminated positions to <see cref="float.NegativeInfinity"/>.
    /// </summary>
    public static void ApplyTopK(Span<float> logits, int k)
    {
        if (k >= logits.Length) return; // Nothing to filter.

        // Find the k-th largest value using a partial selection.
        // Copy logits to find the threshold without allocating a separate sorted array.
        // We rent from ArrayPool to avoid heap pressure.
        float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(logits.Length);
        try
        {
            logits.CopyTo(scratch.AsSpan(0, logits.Length));
            // Partial sort: bring top-k to the front.
            var scratchSpan = scratch.AsSpan(0, logits.Length);
            PartialSortDescending(scratchSpan, k);
            float threshold = scratchSpan[k - 1];

            // Zero out (set to -inf) everything below the threshold.
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] < threshold)
                {
                    logits[i] = float.NegativeInfinity;
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>
    /// Partially sorts the span in-place so that the first <paramref name="k"/> elements are
    /// the k largest (in any order among themselves, but all >= remaining elements).
    /// Uses an introselect-style partition (quickselect) for O(n) average time.
    /// </summary>
    private static void PartialSortDescending(Span<float> span, int k)
    {
        // Use nth_element equivalent: partition so that span[k-1] is the k-th largest.
        int left = 0, right = span.Length - 1;
        int target = k - 1; // 0-based index of the k-th largest element.

        while (left < right)
        {
            int pivotIdx = Partition(span, left, right);
            if (pivotIdx == target) break;
            if (pivotIdx < target) left = pivotIdx + 1;
            else right = pivotIdx - 1;
        }
    }

    /// <summary>Lomuto partition scheme for descending order.</summary>
    private static int Partition(Span<float> span, int left, int right)
    {
        // Median-of-three pivot for better average-case performance.
        int mid = left + (right - left) / 2;
        if (span[left] < span[mid]) (span[left], span[mid]) = (span[mid], span[left]);
        if (span[left] < span[right]) (span[left], span[right]) = (span[right], span[left]);
        if (span[mid] < span[right]) (span[mid], span[right]) = (span[right], span[mid]);
        // span[left] is now the median — use as pivot.
        float pivot = span[left];
        int i = left + 1;
        for (int j = left + 1; j <= right; j++)
        {
            if (span[j] > pivot) // descending: keep larger elements on the left
            {
                (span[i], span[j]) = (span[j], span[i]);
                i++;
            }
        }

        (span[left], span[i - 1]) = (span[i - 1], span[left]);
        return i - 1;
    }
}
