namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Hyperparameters for a LLaMA-family transformer model, extracted from GGUF metadata.
/// </summary>
public sealed class LlamaConfig
{
    /// <summary>Number of transformer layers (blocks).</summary>
    public required int LayerCount { get; init; }

    /// <summary>Hidden dimension size (model embedding dimension).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Intermediate (feed-forward) dimension size.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of query attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>
    /// Number of key-value attention heads.
    /// Equal to <see cref="NumHeads"/> for standard MHA; smaller for GQA/MQA.
    /// </summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Dimension per attention head (<see cref="HiddenSize"/> / <see cref="NumHeads"/>).</summary>
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

    /// <summary>Token ID for the beginning-of-sequence special token.</summary>
    public int BosTokenId { get; init; } = 1;

    /// <summary>Token ID for the end-of-sequence special token.</summary>
    public int EosTokenId { get; init; } = 2;

    /// <summary>Architecture string as stored in GGUF metadata (e.g., "llama").</summary>
    public string Architecture { get; init; } = "llama";

    /// <summary>Model family name (e.g., "llama3", "mistral").</summary>
    public string ModelFamily { get; init; } = "llama";

    /// <summary>Returns a human-readable description of the configuration.</summary>
    public override string ToString() =>
        $"{Architecture} layers={LayerCount} hidden={HiddenSize} heads={NumHeads}/{NumKvHeads} ffn={IntermediateSize} vocab={VocabSize}";
}
