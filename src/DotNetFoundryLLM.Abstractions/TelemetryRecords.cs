#pragma warning disable CS1591
namespace DotNetFoundryLLM.Abstractions;

/// <summary>Represents a token candidate and its log-probability.</summary>
public readonly record struct LogProbEntry
{
    /// <summary>Creates a new <see cref="LogProbEntry"/>.</summary>
    public LogProbEntry(int tokenId, string tokenText, float logProb)
    {
        TokenId = tokenId;
        TokenText = tokenText;
        LogProb = logProb;
    }

    /// <summary>The token identifier.</summary>
    public int TokenId { get; init; }

    /// <summary>The decoded token text.</summary>
    public string TokenText { get; init; }

    /// <summary>The natural-log probability for the token.</summary>
    public float LogProb { get; init; }
}

/// <summary>Per-token breakdown payload emitted on stream chunks.</summary>
public sealed record TokenBreakdown
{
    /// <summary>Creates a new <see cref="TokenBreakdown"/>.</summary>
    public TokenBreakdown(int position, long elapsedMs, IReadOnlyList<LogProbEntry>? topAlternatives = null)
    {
        Position = position;
        ElapsedMs = elapsedMs;
        TopAlternatives = topAlternatives;
    }

    /// <summary>Zero-based generated token position.</summary>
    public int Position { get; init; }

    /// <summary>Elapsed decode time in milliseconds when this token was produced.</summary>
    public long ElapsedMs { get; init; }

    /// <summary>Optional top alternative token candidates for this decode step.</summary>
    public IReadOnlyList<LogProbEntry>? TopAlternatives { get; init; }
}

/// <summary>Detailed per-token insight captured for generation telemetry.</summary>
public sealed record TokenInsight
{
    /// <summary>Creates a new <see cref="TokenInsight"/>.</summary>
    public TokenInsight(
        int position,
        int tokenId,
        string tokenText,
        float logProb,
        long elapsedMs,
        IReadOnlyList<LogProbEntry>? topAlternatives = null)
    {
        Position = position;
        TokenId = tokenId;
        TokenText = tokenText;
        LogProb = logProb;
        ElapsedMs = elapsedMs;
        TopAlternatives = topAlternatives;
    }

    /// <summary>Zero-based generated token position.</summary>
    public int Position { get; init; }

    /// <summary>The emitted token identifier.</summary>
    public int TokenId { get; init; }

    /// <summary>The emitted token text.</summary>
    public string TokenText { get; init; }

    /// <summary>The natural-log probability of the emitted token.</summary>
    public float LogProb { get; init; }

    /// <summary>Elapsed decode time in milliseconds when this token was produced.</summary>
    public long ElapsedMs { get; init; }

    /// <summary>Optional top alternative token candidates for this decode step.</summary>
    public IReadOnlyList<LogProbEntry>? TopAlternatives { get; init; }
}

/// <summary>Aggregate telemetry for a completed generation request.</summary>
public sealed record GenerationTelemetry
{
    /// <summary>The model family used for generation.</summary>
    public required string ModelFamily { get; init; }

    /// <summary>Number of prompt tokens provided to the model.</summary>
    public required int PromptTokenCount { get; init; }

    /// <summary>Number of completion tokens produced by the model.</summary>
    public required int CompletionTokenCount { get; init; }

    /// <summary>Request queue time in milliseconds, if applicable.</summary>
    public long QueueTimeMs { get; init; }

    /// <summary>Prefill phase duration in milliseconds.</summary>
    public required long PrefillTimeMs { get; init; }

    /// <summary>Time to first generated token in milliseconds.</summary>
    public required long TimeToFirstTokenMs { get; init; }

    /// <summary>Decode phase duration in milliseconds.</summary>
    public required long DecodeTimeMs { get; init; }

    /// <summary>Total end-to-end latency in milliseconds.</summary>
    public required long TotalLatencyMs { get; init; }

    /// <summary>Generated decode throughput in tokens per second.</summary>
    public required double TokensPerSecond { get; init; }

    /// <summary>Final completion reason.</summary>
    public required string FinishReason { get; init; }

    /// <summary>Optional per-token insight breakdown.</summary>
    public IReadOnlyList<TokenInsight>? TokenBreakdown { get; init; }
}

/// <summary>Telemetry payload emitted after model load completion.</summary>
public sealed record ModelLoadTelemetry
{
    /// <summary>Creates a new <see cref="ModelLoadTelemetry"/> payload.</summary>
    public ModelLoadTelemetry(
        string modelPath,
        string architecture,
        long loadTimeMs,
        long fileSizeBytes,
        ulong parameterCount)
    {
        ModelPath = modelPath;
        Architecture = architecture;
        LoadTimeMs = loadTimeMs;
        FileSizeBytes = fileSizeBytes;
        ParameterCount = parameterCount;
    }

    /// <summary>Loaded model file path.</summary>
    public string ModelPath { get; init; }

    /// <summary>Resolved model architecture name.</summary>
    public string Architecture { get; init; }

    /// <summary>Model load duration in milliseconds.</summary>
    public long LoadTimeMs { get; init; }

    /// <summary>Model file size in bytes, when available.</summary>
    public long FileSizeBytes { get; init; }

    /// <summary>Estimated parameter count for the loaded model.</summary>
    public ulong ParameterCount { get; init; }
}
#pragma warning restore CS1591
