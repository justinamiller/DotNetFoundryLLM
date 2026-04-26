using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace DotNetFoundryLLM.Tensors;

/// <summary>SIMD-accelerated tensor math operations using <see cref="TensorPrimitives"/>.</summary>
public static class TensorOperations
{
    /// <summary>Element-wise addition: <c>dst = left + right</c>.</summary>
    public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> dst) =>
        TensorPrimitives.Add(left, right, dst);

    /// <summary>Element-wise multiplication: <c>dst = left * right</c>.</summary>
    public static void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> dst) =>
        TensorPrimitives.Multiply(left, right, dst);

    /// <summary>Scales a span: <c>dst = x * scalar</c>.</summary>
    public static void Scale(ReadOnlySpan<float> x, float scalar, Span<float> dst) =>
        TensorPrimitives.Multiply(x, scalar, dst);

    /// <summary>Computes in-place softmax over a span.</summary>
    public static void Softmax(Span<float> x)
    {
        if (x.IsEmpty)
        {
            return;
        }

        float max = TensorPrimitives.Max(x);
        TensorPrimitives.Add(x, -max, x);
        TensorPrimitives.Exp(x, x);

        float sum = TensorPrimitives.Sum(x);
        if (sum != 0f)
        {
            TensorPrimitives.Divide(x, sum, x);
        }
    }

    /// <summary>Computes RMS norm over <paramref name="x"/> using scale <paramref name="w"/>, writing to <paramref name="dst"/>.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> dst, float eps = 1e-5f)
    {
        if (x.Length != w.Length || x.Length != dst.Length)
        {
            throw new ArgumentException("All spans must have the same length.");
        }

        float sumSq = TensorPrimitives.Dot(x, x);
        float rms = 1.0f / MathF.Sqrt(sumSq / x.Length + eps);

        for (int i = 0; i < x.Length; i++)
        {
            dst[i] = x[i] * rms * w[i];
        }
    }

    /// <summary>Applies the SiLU activation (x * sigmoid(x)) element-wise in-place.</summary>
    public static void Silu(Span<float> x)
    {
        if (x.IsEmpty)
        {
            return;
        }

        float[] scratch = ArrayPool<float>.Shared.Rent(x.Length);
        try
        {
            var sig = scratch.AsSpan(0, x.Length);
            TensorPrimitives.Sigmoid(x, sig);
            TensorPrimitives.Multiply(x, sig, x);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>Applies the SiLU activation (x * sigmoid(x)) element-wise in-place using provided scratch buffer.</summary>
    public static void Silu(Span<float> x, Span<float> scratch)
    {
        if (x.IsEmpty)
        {
            return;
        }

        TensorPrimitives.Sigmoid(x, scratch);
        TensorPrimitives.Multiply(x, scratch, x);
    }

    /// <summary>Computes Gemma-style RMS norm over <paramref name="x"/> using scale <c>1 + weight[i]</c>, writing to <paramref name="dst"/>.</summary>
    public static void GemmaRmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> dst)
    {
        float sumSq = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            sumSq += x[i] * x[i];
        }

        float rms = 1f / MathF.Sqrt(sumSq / x.Length + 1e-6f);
        for (int i = 0; i < x.Length; i++)
        {
            dst[i] = x[i] * rms * (1f + weight[i]);
        }
    }

    /// <summary>Applies the GELU activation element-wise in-place.</summary>
    public static void Gelu(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
    }

    /// <summary>Applies tanh-based softcapping to logits in-place.</summary>
    public static void SoftCap(Span<float> x, float cap)
    {
        float inv = 1f / cap;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Tanh(x[i] * inv) * cap;
        }
    }

    /// <summary>
    /// Matrix multiply: C (m×n) = A (m×k) × B (k×n).
    /// Rows-major layout assumed.
    /// </summary>
    public static void MatMul(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n)
    {
        if (a.Length != m * k)
        {
            throw new ArgumentException("a length mismatch");
        }

        if (b.Length != k * n)
        {
            throw new ArgumentException("b length mismatch");
        }

        if (c.Length != m * n)
        {
            throw new ArgumentException("c length mismatch");
        }

        c.Clear();
        for (var row = 0; row < m; row++)
        {
            var cRow = c.Slice(row * n, n);
            var aRow = a.Slice(row * k, k);
            for (var col = 0; col < n; col++)
            {
                float acc = 0f;
                for (var i = 0; i < k; i++)
                {
                    acc += aRow[i] * b[i * n + col];
                }

                cRow[col] = acc;
            }
        }
    }

    /// <summary>Dot product of two spans of equal length.</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        TensorPrimitives.Dot(a, b);

    /// <summary>Applies a causal mask to attention scores: sets future positions to <c>-infinity</c>.</summary>
    public static void ApplyCausalMask(Span<float> scores, int seqLen)
    {
        for (var row = 0; row < seqLen; row++)
        {
            for (var col = row + 1; col < seqLen; col++)
            {
                scores[row * seqLen + col] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>Returns the index of the maximum element in the span.</summary>
    /// <exception cref="ArgumentException">Thrown when the span is empty.</exception>
    public static int ArgMax(ReadOnlySpan<float> x)
    {
        if (x.IsEmpty)
        {
            throw new ArgumentException("Span must not be empty.", nameof(x));
        }

        return TensorPrimitives.IndexOfMax(x);
    }

    /// <summary>
    /// Applies Rotary Position Embeddings (RoPE) to query/key vectors in-place.
    /// </summary>
    /// <param name="x">The vector to rotate (headDim elements).</param>
    /// <param name="position">The token position.</param>
    /// <param name="headDim">Number of dimensions per head (must be even).</param>
    /// <param name="baseFreq">Base frequency (default 10000).</param>
    /// <param name="ropeScalingFactor">Scaling factor for the rope position (default 1.0).</param>
    public static void ApplyRope(Span<float> x, int position, int headDim, float baseFreq = 10000f, float ropeScalingFactor = 1.0f)
    {
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("headDim must be even.");
        }

        float scaledPosition = ropeScalingFactor > 0f ? position / ropeScalingFactor : position;

        for (var i = 0; i < headDim / 2; i++)
        {
            var theta = scaledPosition / MathF.Pow(baseFreq, 2f * i / headDim);
            var cos = MathF.Cos(theta);
            var sin = MathF.Sin(theta);
            var x0 = x[i * 2];
            var x1 = x[i * 2 + 1];
            x[i * 2] = x0 * cos - x1 * sin;
            x[i * 2 + 1] = x0 * sin + x1 * cos;
        }
    }

    /// <summary>
    /// Applies Rotary Position Embeddings (RoPE) to query/key vectors in-place using precomputed frequencies.
    /// </summary>
    /// <param name="x">The vector to rotate (headDim elements).</param>
    /// <param name="position">The token position.</param>
    /// <param name="freqs">Precomputed base frequencies (length = headDim / 2).</param>
    public static void ApplyRope(Span<float> x, int position, ReadOnlySpan<float> freqs)
    {
        for (int i = 0; i < freqs.Length; i++)
        {
            float theta = position * freqs[i];
            float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
            float x0 = x[i * 2], x1 = x[i * 2 + 1];
            x[i * 2]     = x0 * cos - x1 * sin;
            x[i * 2 + 1] = x0 * sin + x1 * cos;
        }
    }
}
