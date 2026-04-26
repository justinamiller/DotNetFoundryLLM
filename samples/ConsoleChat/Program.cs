using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ModelFormats.Gguf;

var loader = new GgufModelLoader();
using var model = await loader.LoadAsync(@"C:\model\gguf-TINYLLAMA-dqg-v3a-q8_0-unsloth.Q8_0.gguf");

await foreach (var chunk in model.GenerateAsync(new CompletionRequest("Hello, world!")))
{
    Console.Write(chunk.Token.Text);
}
