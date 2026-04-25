using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// A stateful sampler decorator that applies a repetition penalty to logits based on
/// tokens seen in the current generation context, then delegates sampling to an inner sampler.
/// <para>
/// For each previously seen token, if its logit is positive the logit is divided by
/// the penalty value; if it is negative the logit is multiplied by the penalty value.
/// This discourages repetition without completely excluding already-used tokens from consideration.
/// </para>
/// </summary>
public sealed class RepetitionPenaltySampler : ISampler
{
    private readonly ISampler _inner;
    private readonly float _penalty;
    private readonly List<int> _previousTokens = [];

    /// <summary>
    /// Initializes a new repetition penalty sampler.
    /// </summary>
    /// <param name="inner">The underlying sampler to delegate to after applying the penalty.</param>
    /// <param name="penalty">
    /// Penalty multiplier. Values greater than 1.0 suppress repetition; 1.0 disables the penalty.
    /// Must be greater than zero.
    /// </param>
    public RepetitionPenaltySampler(ISampler inner, float penalty = 1.1f)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (penalty <= 0f) throw new ArgumentOutOfRangeException(nameof(penalty), "Penalty must be greater than zero.");
        _inner = inner;
        _penalty = penalty;
    }

    /// <summary>
    /// Records a generated token so that it will be penalized in future calls to <see cref="Sample"/>.
    /// Call this after each token is committed to the generation context.
    /// </summary>
    public void RecordToken(int tokenId) => _previousTokens.Add(tokenId);

    /// <summary>
    /// Records a range of prompt tokens so they are included in repetition tracking from the start.
    /// </summary>
    public void RecordTokens(ReadOnlySpan<int> tokenIds)
    {
        foreach (var id in tokenIds) _previousTokens.Add(id);
    }

    /// <summary>Clears all previously recorded tokens.</summary>
    public void Reset() => _previousTokens.Clear();

    /// <inheritdoc />
    /// <remarks>
    /// The logits span is modified in-place: penalty is applied to previously-seen token positions
    /// before delegating to the inner sampler.
    /// </remarks>
    public int Sample(Span<float> logits)
    {
        if (_penalty != 1.0f)
        {
            foreach (var id in _previousTokens)
            {
                if ((uint)id >= (uint)logits.Length) continue;
                logits[id] = logits[id] > 0f
                    ? logits[id] / _penalty
                    : logits[id] * _penalty;
            }
        }

        return _inner.Sample(logits);
    }
}
