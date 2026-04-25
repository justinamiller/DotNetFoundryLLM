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

    /// <summary>Logs a tensor operation.</summary>
    [LoggerMessage(Level = LogLevel.Trace, Message = "Tensor op: {Operation} shape={Shape}")]
    public static partial void LogTensorOperation(this ILogger logger, string operation, string shape);
}
