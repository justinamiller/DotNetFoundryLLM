using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Represents a single-token forward-pass engine for an autoregressive language model.
/// </summary>
public interface IForwardPass
{
    /// <summary>Gets the logits produced by the most recent <see cref="Forward"/> call.</summary>
    ReadOnlySpan<float> Logits { get; }

    /// <summary>Runs one forward pass for the specified token and position using the supplied KV cache.</summary>
    void Forward(int tokenId, int position, IKvCache kvCache);
}
