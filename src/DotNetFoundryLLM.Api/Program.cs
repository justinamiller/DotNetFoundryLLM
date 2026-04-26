using System.Text;
using System.Text.Json.Serialization;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Api;
using DotNetFoundryLLM.ChatTemplates;
using DotNetFoundryLLM.ModelFormats.Gguf;
using DotNetFoundryLLM.Telemetry;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.ClearProviders().AddConsole();
builder.Services.AddSingleton<IInferenceTelemetry, InferenceTelemetry>();
builder.Services.AddSingleton<GgufModelLoader>();
builder.Services.AddSingleton<ChatTemplateRegistry>();
builder.Services.AddSingleton<ModelState>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<Program>>();
    var path = Environment.GetEnvironmentVariable("FOUNDRY_MODEL_PATH");
    if (string.IsNullOrWhiteSpace(path))
    {
        return new(null, null);
    }

    if (!path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
    {
        AppLogs.ModelPathInvalidExtension(logger, path);
        return new(path, null);
    }

    if (!File.Exists(path))
    {
        AppLogs.ModelPathFileNotFound(logger, path);
        return new(path, null);
    }

    try
    {
        return new(path, sp.GetRequiredService<GgufModelLoader>().LoadAsync(path).GetAwaiter().GetResult());
    }
    catch (Exception ex)
    {
        AppLogs.ModelLoadFailed(logger, ex.Message);
        return new(path, null);
    }
});
var app = builder.Build();
app.MapGet("/health", () => "ok");

app.MapPost("/v1/completions", async Task<IResult> (CompletionApiRequest body, ModelState state, HttpContext http, ILogger<Program> logger, CancellationToken ct) =>
{
    const string endpoint = "/v1/completions";

    if (string.IsNullOrEmpty(body.Prompt))
    {
        return ValidationError(logger, endpoint, "prompt is required");
    }
    if (body.Prompt.Length > 65536)
    {
        return ValidationError(logger, endpoint, "prompt exceeds maximum length of 65536 characters");
    }
    if (body.MaxTokens is < 1 or > 4096)
    {
        return ValidationError(logger, endpoint, "max_tokens must be between 1 and 4096");
    }
    if (body.Temperature is < 0.0f or > 2.0f)
    {
        return ValidationError(logger, endpoint, "temperature must be between 0 and 2");
    }

    AppLogs.CompletionRequestReceived(logger, endpoint, body.Stream ?? false, body.Prompt.Length);

    if (state.Model is null)
    {
        return state.Path is null ? ModelNotLoadedError() : ModelLoadFailedError();
    }

    var request = new CompletionRequest(body.Prompt, body.ToOptions());
    if (body.Stream == true)
    {
        http.Response.Headers.Append("Content-Type", "text/event-stream");
        await foreach (var chunk in state.Model.GenerateAsync(request, ct))
        {
            var payload = new { choices = new[] { new { text = chunk.Token.Text, finish_reason = chunk.FinishReason } }, usage = chunk.Usage };
            await http.Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(payload)}\n\n", ct);
            await http.Response.Body.FlushAsync(ct);
        }
        await http.Response.WriteAsync("data: [DONE]\n\n", ct);
        return Results.Empty;
    }

    var sb = new StringBuilder();
    string finishReason = "stop";
    Usage? usage = null;
    await foreach (var chunk in state.Model.GenerateAsync(request, ct))
    {
        sb.Append(chunk.Token.Text);
        finishReason = chunk.FinishReason ?? finishReason;
        usage = chunk.Usage ?? usage;
    }
    var output = sb.ToString();
    AppLogs.GenerationCompleted(logger, endpoint, usage?.CompletionTokens ?? 0, finishReason);
    return Results.Ok(new { choices = new[] { new { text = output, finish_reason = finishReason } }, usage = usage ?? new Usage(0, 0, 0) });
});

app.MapPost("/v1/chat/completions", async Task<IResult> (ChatCompletionApiRequest body, ModelState state, ChatTemplateRegistry templates, HttpContext http, ILogger<Program> logger, CancellationToken ct) =>
{
    const string endpoint = "/v1/chat/completions";

    if (body.Messages is null or { Length: 0 })
    {
        return ValidationError(logger, endpoint, "messages is required");
    }
    if (body.Messages.Length > 100)
    {
        return ValidationError(logger, endpoint, "messages array exceeds maximum length of 100");
    }
    foreach (var m in body.Messages)
    {
        if (string.IsNullOrEmpty(m.Content))
        {
            return ValidationError(logger, endpoint, "message content must not be empty");
        }
    }
    if (body.MaxTokens is < 1 or > 4096)
    {
        return ValidationError(logger, endpoint, "max_tokens must be between 1 and 4096");
    }
    if (body.Temperature is < 0.0f or > 2.0f)
    {
        return ValidationError(logger, endpoint, "temperature must be between 0 and 2");
    }

    AppLogs.ChatCompletionRequestReceived(logger, endpoint, body.Stream ?? false, body.Messages.Length);

    if (state.Model is null)
    {
        return state.Path is null ? ModelNotLoadedError() : ModelLoadFailedError();
    }

    var template = templates.ForFamily(state.Model.Metadata.ModelFamily);
    var request = new ChatRequest(body.Messages.Select(m => new ChatMessage(ParseRole(m.Role), m.Content ?? string.Empty)).ToArray(), body.ToOptions());
    if (body.Stream == true)
    {
        http.Response.Headers.Append("Content-Type", "text/event-stream");
        await foreach (var chunk in state.Model.GenerateAsync(request, template, ct))
        {
            var payload = new { choices = new[] { new { message = new { role = "assistant", content = chunk.Token.Text }, finish_reason = chunk.FinishReason } }, usage = chunk.Usage };
            await http.Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(payload)}\n\n", ct);
            await http.Response.Body.FlushAsync(ct);
        }
        await http.Response.WriteAsync("data: [DONE]\n\n", ct);
        return Results.Empty;
    }

    var sb = new StringBuilder();
    string finishReason = "stop";
    Usage? usage = null;
    await foreach (var chunk in state.Model.GenerateAsync(request, template, ct))
    {
        sb.Append(chunk.Token.Text);
        finishReason = chunk.FinishReason ?? finishReason;
        usage = chunk.Usage ?? usage;
    }
    var output = sb.ToString();
    AppLogs.GenerationCompleted(logger, endpoint, usage?.CompletionTokens ?? 0, finishReason);
    return Results.Ok(new { choices = new[] { new { message = new { role = "assistant", content = output }, finish_reason = finishReason } }, usage = usage ?? new Usage(0, 0, 0) });
});

app.Run();

static IResult ValidationError(ILogger logger, string endpoint, string reason)
{
    AppLogs.ValidationFailed(logger, endpoint, reason);
    return Results.Json(new ErrorBody(new ErrorDetail(reason, "invalid_request_error", "invalid_value")), statusCode: 400);
}

static IResult ModelNotLoadedError() =>
    Results.Json(new ErrorBody(new ErrorDetail("No model is loaded. Set the FOUNDRY_MODEL_PATH environment variable.", "service_unavailable", "model_not_loaded")), statusCode: 503);

static IResult ModelLoadFailedError() =>
    Results.Json(new ErrorBody(new ErrorDetail("Model failed to load. Check server logs.", "service_unavailable", "model_load_failed")), statusCode: 503);

static Role ParseRole(string? role) => role?.ToLowerInvariant() switch
{
    "system" => Role.System,
    "assistant" => Role.Assistant,
    "tool" => Role.Tool,
    _ => Role.User
};

sealed record ModelState(string? Path, ILanguageModel? Model);
sealed record ApiMessage(string? Role, string? Content);
sealed record CompletionApiRequest(string? Prompt, [property: JsonPropertyName("max_tokens")] int? MaxTokens, float? Temperature, [property: JsonPropertyName("top_p")] float? TopP, [property: JsonPropertyName("top_k")] int? TopK, int? Seed, string[]? Stop, bool? Stream)
{
    public GenerationOptions ToOptions() => new() { MaxTokens = MaxTokens ?? 512, Temperature = Temperature ?? 1.0f, TopP = TopP ?? 1.0f, TopK = TopK ?? 0, Seed = Seed, StopSequences = Stop };
}
sealed record ChatCompletionApiRequest(ApiMessage[]? Messages, string? Model, [property: JsonPropertyName("max_tokens")] int? MaxTokens, float? Temperature, [property: JsonPropertyName("top_p")] float? TopP, [property: JsonPropertyName("top_k")] int? TopK, int? Seed, string[]? Stop, bool? Stream)
{
    public GenerationOptions ToOptions() => new() { MaxTokens = MaxTokens ?? 512, Temperature = Temperature ?? 1.0f, TopP = TopP ?? 1.0f, TopK = TopK ?? 0, Seed = Seed, StopSequences = Stop };
}
sealed record ErrorDetail(string Message, string Type, string Code);
sealed record ErrorBody([property: JsonPropertyName("error")] ErrorDetail Error);
