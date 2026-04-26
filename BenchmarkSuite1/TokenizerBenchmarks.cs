using BenchmarkDotNet.Attributes;
using DotNetFoundryLLM.Tokenization;

namespace DotNetFoundryLLM.Bench.Tokenization;

[MemoryDiagnoser]
public class TokenizerBenchmarks
{
    private BpeTokenizer _tokenizer = null!;
    private string _shortText = null!;
    private string _mediumText = null!;
    private string _longText = null!;
    private ReadOnlyMemory<int> _encodedShort;
    private ReadOnlyMemory<int> _encodedMedium;
    private ReadOnlyMemory<int> _encodedLong;
    [GlobalSetup]
    public void Setup()
    {
        // Create a realistic synthetic vocabulary with 32K tokens (typical for LLMs)
        var tokens = new List<string>
        {
            "<unk>",
            "<s>",
            "</s>"
        };
        // Add single-byte tokens (256 bytes mapped via ByteEncoder)
        for (int b = 0; b < 256; b++)
        {
            tokens.Add(new string (ByteEncoder.ByteToChar((byte)b), 1));
        }

        // Add common 2-gram merges (simulate realistic BPE merges)
        var merges = new List<(string, string)>();
        for (int i = 0; i < 30000; i++)
        {
            // Create synthetic merge tokens
            int left = 3 + (i % 256);
            int right = 3 + ((i + 1) % 256);
            tokens.Add(tokens[left] + tokens[right]);
            merges.Add((tokens[left], tokens[right]));
        }

        var vocab = new TokenizerVocab(tokens.ToArray(), merges);
        _tokenizer = new BpeTokenizer(vocab);
        // Real-world prompts of varying lengths
        _shortText = "Hello, world! How are you doing today?";
        _mediumText = @"The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. 
Machine learning models, particularly large language models, have revolutionized natural language processing.
They can understand context, generate coherent text, and perform complex reasoning tasks.
Tokenization is a critical preprocessing step that breaks text into subword units for efficient processing.";
        _longText = @"Artificial intelligence has transformed nearly every aspect of modern technology. From voice assistants 
to recommendation systems, AI models power the applications we use daily. Large language models represent a 
significant breakthrough in natural language understanding. These models are trained on vast amounts of text data, 
learning patterns and relationships in human language. The tokenization process is fundamental to how these models 
work. By breaking text into smaller units called tokens, the model can process and understand language more efficiently.
Each token represents a piece of text, which could be a word, part of a word, or even punctuation. The Byte Pair Encoding 
algorithm is a popular tokenization method that balances vocabulary size with representation efficiency. It starts with 
individual bytes and iteratively merges the most frequent pairs, building a vocabulary of common subword units. This 
approach allows the model to handle rare words and out-of-vocabulary terms gracefully. The tokenization step directly 
impacts model performance, inference speed, and memory usage. Optimizing this critical path can yield significant 
improvements in overall system throughput.";
        // Pre-encode for decode benchmarks
        _encodedShort = _tokenizer.Encode(_shortText.AsSpan(), addBos: false, addEos: false);
        _encodedMedium = _tokenizer.Encode(_mediumText.AsSpan(), addBos: false, addEos: false);
        _encodedLong = _tokenizer.Encode(_longText.AsSpan(), addBos: false, addEos: false);
    }

    [Benchmark]
    public ReadOnlyMemory<int> Encode_Short()
    {
        return _tokenizer.Encode(_shortText.AsSpan(), addBos: false, addEos: false);
    }

    [Benchmark]
    public ReadOnlyMemory<int> Encode_Medium()
    {
        return _tokenizer.Encode(_mediumText.AsSpan(), addBos: false, addEos: false);
    }

    [Benchmark]
    public ReadOnlyMemory<int> Encode_Long()
    {
        return _tokenizer.Encode(_longText.AsSpan(), addBos: false, addEos: false);
    }

    [Benchmark]
    public string Decode_Short()
    {
        return _tokenizer.Decode(_encodedShort.Span);
    }

    [Benchmark]
    public string Decode_Medium()
    {
        return _tokenizer.Decode(_encodedMedium.Span);
    }

    [Benchmark]
    public string Decode_Long()
    {
        return _tokenizer.Decode(_encodedLong.Span);
    }
}
