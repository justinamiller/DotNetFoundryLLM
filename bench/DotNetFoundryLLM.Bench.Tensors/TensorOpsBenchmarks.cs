using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Bench.Tensors;

/// <summary>
/// Benchmarks for core tensor operations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class TensorOpsBenchmarks
{
    private float[] _a = null!;
    private float[] _b = null!;
    private float[] _dst = null!;
    private float[] _w = null!;

    /// <summary>
    /// Vector size parameter.
    /// </summary>
    [Params(512, 4096)]
    public int N { get; set; }

    /// <summary>
    /// Initializes benchmark data.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        _a = new float[N];
        _b = new float[N];
        _dst = new float[N];
        _w = new float[N];

        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            _a[i] = (float)(rng.NextDouble() - 0.5);
            _b[i] = (float)(rng.NextDouble() - 0.5);
            _w[i] = (float)(rng.NextDouble() * 0.5 + 0.5);
        }
    }

    /// <summary>
    /// Benchmark RmsNorm operation.
    /// </summary>
    [Benchmark]
    public void RmsNorm() => TensorOperations.RmsNorm(_a, _w, _dst);

    /// <summary>
    /// Benchmark Softmax operation.
    /// </summary>
    [Benchmark]
    public void Softmax()
    {
        _a.CopyTo(_dst, 0);
        TensorOperations.Softmax(_dst);
    }

    /// <summary>
    /// Benchmark ArgMax operation.
    /// </summary>
    [Benchmark]
    public int ArgMax() => TensorOperations.ArgMax(_a);

    /// <summary>
    /// Benchmark Silu activation.
    /// </summary>
    [Benchmark]
    public void Silu()
    {
        _a.CopyTo(_dst, 0);
        TensorOperations.Silu(_dst);
    }

    /// <summary>
    /// Benchmark Dot product.
    /// </summary>
    [Benchmark]
    public float Dot() => TensorOperations.Dot(_a, _b);

    /// <summary>
    /// Benchmark RoPE application.
    /// </summary>
    [Benchmark]
    public void ApplyRope()
    {
        _a.AsSpan(0, 128).CopyTo(_dst);
        TensorOperations.ApplyRope(_dst.AsSpan(0, 128), 42, 128, 10000f);
    }
}
