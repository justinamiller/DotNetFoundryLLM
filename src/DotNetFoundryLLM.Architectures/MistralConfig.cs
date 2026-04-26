namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Hyperparameters for a Mistral-family transformer model, extracted from GGUF metadata.
/// </summary>
public sealed class MistralConfig
{
    /// <summary>Number of transformer layers (blocks).</summary>
    public required int LayerCount { get; init; }

    /// <summary>Hidden dimension size (model embedding dimension).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Intermediate (feed-forward) dimension size.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of query attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of key-value attention heads.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Dimension per attention head.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Total dimension of the concatenated query vectors.</summary>
    public int QueryDim => NumHeads * HeadDim;

    /// <summary>Total dimension of concatenated key or value vectors.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Maximum supported context length (sequence length).</summary>
    public required int MaxContextLength { get; init; }

    /// <summary>Vocabulary size.</summary>
    public required int VocabSize { get; init; }

    /// <summary>RoPE base frequency (default 10000).</summary>
    public float RopeBaseFreq { get; init; } = 10000f;

    /// <summary>RoPE scaling factor used for extended-context models.</summary>
    public float RopeScalingFactor { get; init; } = 1.0f;

    /// <summary>RoPE scaling type reported by GGUF metadata.</summary>
    public string RopeScalingType { get; init; } = "none";

    /// <summary>Token ID for the beginning-of-sequence special token.</summary>
    public int BosTokenId { get; init; } = 1;

    /// <summary>Token ID for the end-of-sequence special token.</summary>
    public int EosTokenId { get; init; } = 2;

    /// <summary>Architecture string as stored in GGUF metadata.</summary>
    public string Architecture { get; init; } = "mistral";

    /// <summary>Model family name.</summary>
    public string ModelFamily { get; init; } = "mistral";

    /// <summary>Sliding-window attention size. <see cref="int.MaxValue"/> disables windowing.</summary>
    public int SlidingWindowSize { get; init; } = int.MaxValue;
}
