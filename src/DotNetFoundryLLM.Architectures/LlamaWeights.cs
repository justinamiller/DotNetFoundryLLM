namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Holds all dequantized (float32) weight tensors for a LLaMA-family model.
/// Weights are stored as flat row-major arrays following the convention:
/// <c>weight[row, col] = weight[row * cols + col]</c>.
/// </summary>
public sealed class LlamaWeights : IDisposable
{
    private bool _disposed;
    private readonly long _estimatedBytes;

    /// <summary>
    /// Initializes a new <see cref="LlamaWeights"/> instance with per-layer arrays.
    /// </summary>
    public LlamaWeights(
        LlamaConfig config,
        float[] tokenEmbedding,
        float[][] attnNorm,
        float[][] wq,
        float[][] wk,
        float[][] wv,
        float[][] wo,
        float[][] ffnNorm,
        float[][] ffnGate,
        float[][] ffnUp,
        float[][] ffnDown,
        float[] outputNorm,
        float[] outputWeight)
    {
        ArgumentNullException.ThrowIfNull(config);
        Config       = config;
        TokenEmbedding = tokenEmbedding;
        AttnNorm     = attnNorm;
        Wq           = wq;
        Wk           = wk;
        Wv           = wv;
        Wo           = wo;
        FfnNorm      = ffnNorm;
        FfnGate      = ffnGate;
        FfnUp        = ffnUp;
        FfnDown      = ffnDown;
        OutputNorm   = outputNorm;
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

    /// <summary>Attention RMSNorm weights per layer. Shape per layer: [hidden_size].</summary>
    public float[][] AttnNorm { get; }

    /// <summary>Query projection weights per layer. Shape per layer: [q_dim, hidden_size].</summary>
    public float[][] Wq { get; }

    /// <summary>Key projection weights per layer. Shape per layer: [kv_dim, hidden_size].</summary>
    public float[][] Wk { get; }

    /// <summary>Value projection weights per layer. Shape per layer: [kv_dim, hidden_size].</summary>
    public float[][] Wv { get; }

    /// <summary>Output (attention) projection weights per layer. Shape per layer: [hidden_size, q_dim].</summary>
    public float[][] Wo { get; }

    /// <summary>FFN RMSNorm weights per layer. Shape per layer: [hidden_size].</summary>
    public float[][] FfnNorm { get; }

    /// <summary>FFN gate projection weights per layer. Shape per layer: [intermediate_size, hidden_size].</summary>
    public float[][] FfnGate { get; }

    /// <summary>FFN up projection weights per layer. Shape per layer: [intermediate_size, hidden_size].</summary>
    public float[][] FfnUp { get; }

    /// <summary>FFN down projection weights per layer. Shape per layer: [hidden_size, intermediate_size].</summary>
    public float[][] FfnDown { get; }

    /// <summary>Final RMSNorm weights. Shape: [hidden_size].</summary>
    public float[] OutputNorm { get; }

    /// <summary>Language model head (output projection). Shape: [vocab_size, hidden_size].</summary>
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

    private long EstimateTotalBytes()
    {
        long total = 0;
        total += (long)TokenEmbedding.Length * sizeof(float);
        total += SumJaggedBytes(AttnNorm);
        total += SumJaggedBytes(Wq);
        total += SumJaggedBytes(Wk);
        total += SumJaggedBytes(Wv);
        total += SumJaggedBytes(Wo);
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

    internal void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(_disposed, this);
}
