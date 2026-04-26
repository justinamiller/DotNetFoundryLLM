using Microsoft.Extensions.Logging;

namespace DotNetFoundryLLM.Core;

/// <summary>High-performance logger message definitions using source-generated delegates.</summary>
public static partial class LoggerMessages
{
    /// <summary>Logs that a model is being loaded from a path.</summary>
    [LoggerMessage(Level = LogLevel.Information, Message = "Loading model from {Path}")]
    public static partial void LogModelLoading(this ILogger logger, string path);

    /// <summary>Logs that a model has been loaded successfully.</summary>
    [LoggerMessage(Level = LogLevel.Information, Message = "Model loaded: {Architecture} with {ParameterCount} parameters")]
    public static partial void LogModelLoaded(this ILogger logger, string architecture, ulong parameterCount);

    /// <summary>Logs that token generation has started.</summary>
    [LoggerMessage(Level = LogLevel.Debug, Message = "Starting generation: maxTokens={MaxTokens}, temperature={Temperature}")]
    public static partial void LogGenerationStarted(this ILogger logger, int maxTokens, float temperature);

    /// <summary>Logs that token generation has completed.</summary>
    [LoggerMessage(Level = LogLevel.Debug, Message = "Generation complete: {TokenCount} tokens in {ElapsedMs}ms")]
    public static partial void LogGenerationCompleted(this ILogger logger, int tokenCount, long elapsedMs);

    /// <summary>Logs detailed generation telemetry for a completed request.</summary>
    [LoggerMessage(Level = LogLevel.Debug,
        Message = "Generation telemetry: prompt={PromptTokens} completion={CompletionTokens} ttft={TimeToFirstTokenMs}ms decode={DecodeTimeMs}ms throughput={TokensPerSecond:F1}tok/s finish={FinishReason}")]
    public static partial void LogGenerationTelemetry(
        this ILogger logger,
        int promptTokens, int completionTokens,
        long timeToFirstTokenMs, long decodeTimeMs,
        double tokensPerSecond, string finishReason);

    /// <summary>Logs a structured summary when model loading completes.</summary>
    [LoggerMessage(Level = LogLevel.Information,
        Message = "Model load complete: architecture={Architecture} params={ParameterCount} loadTime={LoadTimeMs}ms fileSize={FileSizeBytes}B")]
    public static partial void LogModelLoadComplete(
        this ILogger logger,
        string architecture, ulong parameterCount,
        long loadTimeMs, long fileSizeBytes);

    /// <summary>Logs a tensor operation.</summary>
    [LoggerMessage(Level = LogLevel.Trace, Message = "Tensor op: {Operation} shape={Shape}")]
    public static partial void LogTensorOperation(this ILogger logger, string operation, string shape);
}
