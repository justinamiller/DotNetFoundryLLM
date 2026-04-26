using Microsoft.Extensions.Logging;

namespace DotNetFoundryLLM.Api;

static partial class AppLogs
{
    [LoggerMessage(Level = LogLevel.Information, Message = "{Endpoint} stream={Stream} promptLength={PromptLength}")]
    public static partial void CompletionRequestReceived(ILogger logger, string endpoint, bool stream, int promptLength);

    [LoggerMessage(Level = LogLevel.Information, Message = "{Endpoint} stream={Stream} messageCount={MessageCount}")]
    public static partial void ChatCompletionRequestReceived(ILogger logger, string endpoint, bool stream, int messageCount);

    [LoggerMessage(Level = LogLevel.Information, Message = "{Endpoint} tokenCount={TokenCount} finishReason={FinishReason}")]
    public static partial void GenerationCompleted(ILogger logger, string endpoint, int tokenCount, string finishReason);

    [LoggerMessage(Level = LogLevel.Warning, Message = "{Endpoint} validation failed: {Reason}")]
    public static partial void ValidationFailed(ILogger logger, string endpoint, string reason);

    [LoggerMessage(Level = LogLevel.Warning, Message = "Model load failed: {Message}")]
    public static partial void ModelLoadFailed(ILogger logger, string message);

    [LoggerMessage(Level = LogLevel.Warning, Message = "Model path does not have .gguf extension: {Path}")]
    public static partial void ModelPathInvalidExtension(ILogger logger, string path);

    [LoggerMessage(Level = LogLevel.Warning, Message = "Model path does not exist: {Path}")]
    public static partial void ModelPathFileNotFound(ILogger logger, string path);
}
