using System.Text.Json.Serialization;
using DotNetFoundryLLM.Abstractions;
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
    var path = Environment.GetEnvironmentVariable("FOUNDRY_MODEL_PATH");
    if (string.IsNullOrWhiteSpace(path))
    {
        return new(null, null);
    }

    try
    {
        return new(path, sp.GetRequiredService<GgufModelLoader>().LoadAsync(path).GetAwaiter().GetResult());
    }
    catch
    {
        return new(path, null);
    }
});
var app = builder.Build();
app.MapGet("/health", () => "ok");
app.MapPost("/v1/completions", async Task<IResult> (CompletionApiRequest body, ModelState state, HttpContext http, CancellationToken ct) =>
{
    if (state.Model is null)
    {
        return Results.StatusCode(StatusCodes.Status503ServiceUnavailable);
    }

    var request = new CompletionRequest(body.Prompt ?? string.Empty, body.ToOptions());
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

    string output = string.Empty;
    string finishReason = "stop";
    Usage? usage = null;
    await foreach (var chunk in state.Model.GenerateAsync(request, ct))
    {
        output += chunk.Token.Text;
        finishReason = chunk.FinishReason ?? finishReason;
        usage = chunk.Usage ?? usage;
    }

    return Results.Ok(new { choices = new[] { new { text = output, finish_reason = finishReason } }, usage = usage ?? new Usage(0, 0, 0) });
});
app.MapPost("/v1/chat/completions", async Task<IResult> (ChatCompletionApiRequest body, ModelState state, ChatTemplateRegistry templates, HttpContext http, CancellationToken ct) =>
{
    if (state.Model is null)
    {
        return Results.StatusCode(StatusCodes.Status503ServiceUnavailable);
    }

    var template = templates.ForFamily(state.Model.Metadata.ModelFamily);
    var request = new ChatRequest((body.Messages ?? []).Select(m => new ChatMessage(ParseRole(m.Role), m.Content ?? string.Empty)).ToArray(), body.ToOptions());
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

    string output = string.Empty;
    string finishReason = "stop";
    Usage? usage = null;
    await foreach (var chunk in state.Model.GenerateAsync(request, template, ct))
    {
        output += chunk.Token.Text;
        finishReason = chunk.FinishReason ?? finishReason;
        usage = chunk.Usage ?? usage;
    }

    return Results.Ok(new { choices = new[] { new { message = new { role = "assistant", content = output }, finish_reason = finishReason } }, usage = usage ?? new Usage(0, 0, 0) });
});
app.Run();

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
