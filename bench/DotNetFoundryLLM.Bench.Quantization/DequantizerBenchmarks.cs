using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using DotNetFoundryLLM.Quantization;

namespace DotNetFoundryLLM.Bench.Quantization;

/// <summary>
/// Benchmarks for quantization dequantization operations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class DequantizerBenchmarks
{
    private byte[] _q4Src = null!;
    private byte[] _q8Src = null!;
    private byte[] _f16Src = null!;
    private float[] _dst = null!;

    /// <summary>
    /// Initializes benchmark data.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        // 256 blocks = 8192 F32 elements
        _q4Src = new byte[256 * 18];   // Q4_0: 256 blocks × 18 bytes
        _q8Src = new byte[256 * 34];   // Q8_0: 256 blocks × 34 bytes
        _f16Src = new byte[8192 * 2];  // F16:  8192 elements × 2 bytes
        _dst = new float[8192];

        var rng = new Random(42);
        rng.NextBytes(_q4Src);
        rng.NextBytes(_q8Src);
        rng.NextBytes(_f16Src);
    }

    /// <summary>
    /// Benchmark F16 dequantization.
    /// </summary>
    [Benchmark]
    public void DequantF16() => Dequantizer.DequantizeF16(_f16Src, _dst);

    /// <summary>
    /// Benchmark Q4_0 dequantization.
    /// </summary>
    [Benchmark]
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "CA1707:Identifiers should not contain underscores")]
    public void DequantQ40() => Dequantizer.DequantizeQ4_0(_q4Src, _dst);

    /// <summary>
    /// Benchmark Q8_0 dequantization.
    /// </summary>
    [Benchmark]
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "CA1707:Identifiers should not contain underscores")]
    public void DequantQ80() => Dequantizer.DequantizeQ8_0(_q8Src, _dst);
}
