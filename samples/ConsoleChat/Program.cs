using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ModelFormats.Gguf;

var modelPath = args.FirstOrDefault() ?? Environment.GetEnvironmentVariable("FOUNDRY_MODEL_PATH");
if (string.IsNullOrWhiteSpace(modelPath))
{
    Console.Error.WriteLine("Usage: dotnet run --project samples/ConsoleChat -- /path/to/model.gguf");
    return 1;
}

var loader = new GgufModelLoader();
using var model = await loader.LoadAsync(modelPath);
await foreach (var chunk in model.GenerateAsync(new CompletionRequest("Hello, world!")))
{
    Console.Write(chunk.Token.Text);
}

return 0;
