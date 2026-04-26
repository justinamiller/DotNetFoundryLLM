namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Holds all dequantized (float32) weight tensors for a Qwen2-family model.
/// </summary>
public sealed class Qwen2Weights : IDisposable
{
    private bool _disposed;
    private readonly long _estimatedBytes;

    /// <summary>Initializes a new <see cref="Qwen2Weights"/> instance with per-layer arrays.</summary>
    public Qwen2Weights(
        LlamaConfig config,
        float[] tokenEmbedding,
        float[][] attnNorm,
        float[][] wq,
        float[][] wk,
        float[][] wv,
        float[][] wo,
        float[][] bq,
        float[][] bk,
        float[][] bv,
        float[][] ffnNorm,
        float[][] ffnGate,
        float[][] ffnUp,
        float[][] ffnDown,
        float[] outputNorm,
        float[] outputWeight)
    {
        ArgumentNullException.ThrowIfNull(config);
        Config = config;
        TokenEmbedding = tokenEmbedding;
        AttnNorm = attnNorm;
        Wq = wq;
        Wk = wk;
        Wv = wv;
        Wo = wo;
        Bq = bq;
        Bk = bk;
        Bv = bv;
        FfnNorm = ffnNorm;
        FfnGate = ffnGate;
        FfnUp = ffnUp;
        FfnDown = ffnDown;
        OutputNorm = outputNorm;
        OutputWeight = outputWeight;

        _estimatedBytes = EstimateTotalBytes();
        if (_estimatedBytes > 0)
        {
            GC.AddMemoryPressure(_estimatedBytes);
        }
    }

    /// <summary>Model configuration.</summary>
    public LlamaConfig Config { get; }

    /// <summary>Token embedding table. Shape: [vocab_size, hidden_size].</summary>
    public float[] TokenEmbedding { get; }

    /// <summary>Attention RMSNorm weights per layer.</summary>
    public float[][] AttnNorm { get; }

    /// <summary>Query projection weights per layer.</summary>
    public float[][] Wq { get; }

    /// <summary>Key projection weights per layer.</summary>
    public float[][] Wk { get; }

    /// <summary>Value projection weights per layer.</summary>
    public float[][] Wv { get; }

    /// <summary>Output projection weights per layer.</summary>
    public float[][] Wo { get; }

    /// <summary>Query projection biases per layer.</summary>
    public float[][] Bq { get; }

    /// <summary>Key projection biases per layer.</summary>
    public float[][] Bk { get; }

    /// <summary>Value projection biases per layer.</summary>
    public float[][] Bv { get; }

    /// <summary>FFN RMSNorm weights per layer.</summary>
    public float[][] FfnNorm { get; }

    /// <summary>FFN gate projection weights per layer.</summary>
    public float[][] FfnGate { get; }

    /// <summary>FFN up projection weights per layer.</summary>
    public float[][] FfnUp { get; }

    /// <summary>FFN down projection weights per layer.</summary>
    public float[][] FfnDown { get; }

    /// <summary>Final RMSNorm weights.</summary>
    public float[] OutputNorm { get; }

    /// <summary>Language model head projection weights.</summary>
    public float[] OutputWeight { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_estimatedBytes > 0)
        {
            GC.RemoveMemoryPressure(_estimatedBytes);
        }

        GC.SuppressFinalize(this);
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private long EstimateTotalBytes()
    {
        long total = 0;
        total += (long)TokenEmbedding.Length * sizeof(float);
        total += SumJaggedBytes(AttnNorm);
        total += SumJaggedBytes(Wq);
        total += SumJaggedBytes(Wk);
        total += SumJaggedBytes(Wv);
        total += SumJaggedBytes(Wo);
        total += SumJaggedBytes(Bq);
        total += SumJaggedBytes(Bk);
        total += SumJaggedBytes(Bv);
        total += SumJaggedBytes(FfnNorm);
        total += SumJaggedBytes(FfnGate);
        total += SumJaggedBytes(FfnUp);
        total += SumJaggedBytes(FfnDown);
        total += (long)OutputNorm.Length * sizeof(float);
        total += (long)OutputWeight.Length * sizeof(float);
        return total;
    }

    private static long SumJaggedBytes(float[][] data)
    {
        long total = 0;
        for (int i = 0; i < data.Length; i++)
        {
            total += (long)data[i].Length * sizeof(float);
        }

        return total;
    }
}
