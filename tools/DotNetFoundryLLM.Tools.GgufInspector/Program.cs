using DotNetFoundryLLM.ModelFormats.Gguf;

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: GgufInspector <path-to-model.gguf>");
    return 1;
}

string path = args[0];

if (!File.Exists(path))
{
    Console.Error.WriteLine($"File not found: {path}");
    return 1;
}

Console.WriteLine($"Inspecting: {path}");
Console.WriteLine();

GgufFile gguf;
try
{
    gguf = await GgufReader.ReadAsync(path);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error reading GGUF file: {ex.Message}");
    return 1;
}

Console.WriteLine($"GGUF version : {gguf.Version}");
Console.WriteLine($"Tensors      : {gguf.Tensors.Count}");
Console.WriteLine($"Metadata keys: {gguf.Metadata.Count}");
Console.WriteLine();

// Print key metadata.
var interestingKeys = new[]
{
    "general.architecture",
    "general.name",
    "llama.block_count",
    "llama.embedding_length",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.context_length",
    "llama.vocab_size",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
};

Console.WriteLine("── Key metadata ─────────────────────────────────────────");
foreach (var key in interestingKeys)
{
    if (gguf.Metadata.TryGetValue(key, out var val))
    {
        Console.WriteLine($"  {key,-42} = {val}");
    }
}
Console.WriteLine();

// Print tensor list (truncated to 30).
Console.WriteLine("── Tensors ──────────────────────────────────────────────");
int shown = 0;
foreach (var t in gguf.Tensors)
{
    if (shown++ >= 30)
    {
        Console.WriteLine($"  ... and {gguf.Tensors.Count - 30} more");
        break;
    }

    var dims = string.Join(" × ", t.Dims);
    Console.WriteLine($"  {t.Name,-45} [{dims}]  {t.TensorType}");
}

return 0;
