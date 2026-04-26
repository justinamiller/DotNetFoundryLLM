namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Holds all dequantized (float32) weight tensors for a Gemma 2-family model.
/// </summary>
public sealed class Gemma2Weights : IDisposable
{
    private bool _disposed;
    private readonly long _estimatedBytes;

    /// <summary>Initializes a new <see cref="Gemma2Weights"/> instance with per-layer arrays.</summary>
    public Gemma2Weights(
        Gemma2Config config,
        float[] tokenEmbedding,
        float[][] attnNorm,
        float[][] postAttnNorm,
        float[][] wq,
        float[][] wk,
        float[][] wv,
        float[][] wo,
        float[][] ffnNorm,
        float[][] postFfnNorm,
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
        PostAttnNorm = postAttnNorm;
        Wq = wq;
        Wk = wk;
        Wv = wv;
        Wo = wo;
        FfnNorm = ffnNorm;
        PostFfnNorm = postFfnNorm;
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
    public Gemma2Config Config { get; }

    /// <summary>Token embedding table. Shape: [vocab_size, hidden_size].</summary>
    public float[] TokenEmbedding { get; }

    /// <summary>Attention RMSNorm weights per layer.</summary>
    public float[][] AttnNorm { get; }

    /// <summary>Post-attention RMSNorm weights per layer.</summary>
    public float[][] PostAttnNorm { get; }

    /// <summary>Query projection weights per layer.</summary>
    public float[][] Wq { get; }

    /// <summary>Key projection weights per layer.</summary>
    public float[][] Wk { get; }

    /// <summary>Value projection weights per layer.</summary>
    public float[][] Wv { get; }

    /// <summary>Output projection weights per layer.</summary>
    public float[][] Wo { get; }

    /// <summary>FFN RMSNorm weights per layer.</summary>
    public float[][] FfnNorm { get; }

    /// <summary>Post-FFN RMSNorm weights per layer.</summary>
    public float[][] PostFfnNorm { get; }

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
        total += SumJaggedBytes(PostAttnNorm);
        total += SumJaggedBytes(Wq);
        total += SumJaggedBytes(Wk);
        total += SumJaggedBytes(Wv);
        total += SumJaggedBytes(Wo);
        total += SumJaggedBytes(FfnNorm);
        total += SumJaggedBytes(PostFfnNorm);
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
