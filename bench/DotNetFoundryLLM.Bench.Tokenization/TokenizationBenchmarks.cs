using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using DotNetFoundryLLM.Tokenization;

namespace DotNetFoundryLLM.Bench.Tokenization;

/// <summary>
/// Benchmarks for tokenization operations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class TokenizationBenchmarks
{
    private BpeTokenizer _tok = null!;

    private const string Short = "Hello, world! How are you?";
    private const string Long =
        "The quick brown fox jumps over the lazy dog. " +
        "Pack my box with five dozen liquor jugs. " +
        "How vexingly quick daft zebras jump! " +
        "The five boxing wizards jump quickly.";

    /// <summary>
    /// Initializes benchmark data.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        // 128 single-byte tokens (GPT-2 byte encoding for ASCII printables)
        var tokens = Enumerable.Range(0, 128)
            .Select(i => ByteEncoder.ByteToChar((byte)i).ToString())
            .ToArray();

        // 50 synthetic merge pairs: consecutive pairs of single chars
        var merges = Enumerable.Range(0, 50)
            .Select(i => (tokens[i * 2 % 128], tokens[(i * 2 + 1) % 128]))
            .ToList();

        var vocab = new TokenizerVocab(tokens, merges, bosTokenId: 1, eosTokenId: 2);
        _tok = new BpeTokenizer(vocab);
    }

    /// <summary>
    /// Benchmark encoding short text.
    /// </summary>
    [Benchmark]
    public ReadOnlyMemory<int> EncodeShort() => _tok.Encode(Short.AsSpan());

    /// <summary>
    /// Benchmark encoding long text.
    /// </summary>
    [Benchmark]
    public ReadOnlyMemory<int> EncodeLong() => _tok.Encode(Long.AsSpan());

    /// <summary>
    /// Benchmark decoding a single token.
    /// </summary>
    [Benchmark]
    public string DecodeToken42() => _tok.DecodeToken(42);
}
