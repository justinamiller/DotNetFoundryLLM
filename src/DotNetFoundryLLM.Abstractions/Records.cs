namespace DotNetFoundryLLM.Abstractions;

/// <summary>A single chat conversation turn.</summary>
public sealed record ChatMessage(Role Role, string Content);

/// <summary>A request for a chat completion.</summary>
public sealed record ChatRequest(
    IReadOnlyList<ChatMessage> Messages,
    GenerationOptions? Options = null);

/// <summary>A request for a text completion.</summary>
public sealed record CompletionRequest(
    string Prompt,
    GenerationOptions? Options = null);

/// <summary>Options controlling text generation.</summary>
public sealed record GenerationOptions
{
    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 512;

    /// <summary>Sampling temperature; higher = more random.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Nucleus sampling probability mass cutoff.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Top-K sampling cutoff; 0 = disabled.</summary>
    public int TopK { get; init; }

    /// <summary>Penalty applied to repeated tokens.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Optional random seed for reproducibility.</summary>
    public int? Seed { get; init; }

    /// <summary>Stop sequences that terminate generation.</summary>
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>
    /// When true, each TokenStreamChunk includes a TokenBreakdown with the
    /// token's log-probability and elapsed decode time.
    /// Adds one extra softmax per decode step.
    /// </summary>
    public bool ReturnLogProbs { get; init; }

    /// <summary>
    /// Number of top-alternative tokens to return per step when ReturnLogProbs
    /// is true. Clamped to [0, 20]. Zero disables top-alternative collection.
    /// </summary>
    public int TopLogProbsCount { get; init; }
}

/// <summary>A single generated token.</summary>
public readonly record struct Token(int Id, string Text, float Logprob = 0f);

/// <summary>A chunk of a streaming token generation response.</summary>
public sealed record TokenStreamChunk(
    Token Token,
    bool IsFinished,
    string? FinishReason = null,
    Usage? Usage = null,
    TokenBreakdown? Breakdown = null);

/// <summary>Token usage statistics for a request.</summary>
public sealed record Usage(
    int PromptTokens,
    int CompletionTokens,
    int TotalTokens);

/// <summary>Metadata about a loaded model.</summary>
public sealed record ModelMetadata(
    string Architecture,
    string ModelFamily,
    ulong ParameterCount,
    int ContextLength,
    int EmbeddingDimension,
    int VocabSize,
    DType WeightDtype,
    IReadOnlyDictionary<string, object> RawMetadata);
