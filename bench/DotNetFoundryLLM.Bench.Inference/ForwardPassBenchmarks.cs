using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using DotNetFoundryLLM.Architectures;
using DotNetFoundryLLM.Inference;

namespace DotNetFoundryLLM.Bench.Inference;

/// <summary>
/// Benchmarks for forward pass operations on the LLaMA model.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
[System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1001:Types that own disposable fields should be disposable")]
public class ForwardPassBenchmarks
{
    private LlamaForwardPass _fwd = null!;
    private LlamaWeights _w = null!;
    private KvCache _kv = null!;

    /// <summary>
    /// Initializes benchmark model and caches.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        // Tiny synthetic model: 4 layers, hidden=64, heads=4, kv_heads=2, ffn=128, vocab=256
        var cfg = new LlamaConfig
        {
            LayerCount = 4,
            HiddenSize = 64,
            IntermediateSize = 128,
            NumHeads = 4,
            NumKvHeads = 2,
            MaxContextLength = 512,
            VocabSize = 256,
            RopeBaseFreq = 10000f
        };

        static float[] R(long n)
        {
            var a = new float[n];
            var r = new Random(42);
            for (int i = 0; i < n; i++)
            {
                a[i] = (float)(r.NextDouble() - 0.5);
            }
            return a;
        }

        static float[][] RL(int layers, long n) =>
            Enumerable.Range(0, layers).Select(_ => R(n)).ToArray();

        _w = new LlamaWeights(
            cfg,
            R(256L * 64),
            RL(4, 64),
            RL(4, 64L * 64),
            RL(4, 32L * 64),
            RL(4, 32L * 64),
            RL(4, 64L * 64),
            RL(4, 64),
            RL(4, 128L * 64),
            RL(4, 128L * 64),
            RL(4, 64L * 128),
            R(64),
            R(256L * 64));

        _fwd = new LlamaForwardPass(_w);
        _kv = new KvCache(4, 512, 32);
    }

    /// <summary>
    /// Cleans up model resources.
    /// </summary>
    [GlobalCleanup]
    public void Cleanup() => _w.Dispose();

    /// <summary>
    /// Resets KV cache for each iteration.
    /// </summary>
    [IterationSetup]
    public void Reset() => _kv.Clear();

    /// <summary>
    /// Benchmark single token forward pass.
    /// </summary>
    [Benchmark]
    public void SingleToken() => _fwd.Forward(42, 0, _kv);

    /// <summary>
    /// Benchmark multiple token forward passes.
    /// </summary>
    [Benchmark]
    [Arguments(16)]
    [Arguments(64)]
    [Arguments(128)]
    public void NTokens(int n)
    {
        _kv.Clear();
        for (int i = 0; i < n; i++)
        {
            _fwd.Forward(i % 256, i, _kv);
        }
    }
}
