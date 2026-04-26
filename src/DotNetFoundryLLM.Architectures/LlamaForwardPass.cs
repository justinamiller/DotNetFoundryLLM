using System.Buffers;
using System.Numerics.Tensors;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Implements a single forward pass of the LLaMA transformer architecture.
/// <para>
/// Given a token ID and its position in the sequence, this class computes the
/// unnormalized output logits over the vocabulary, using a KV-cache to avoid
/// recomputing attention keys/values for earlier tokens.
/// </para>
/// <para>
/// The forward pass follows the standard LLaMA-2/3 structure:
/// <list type="number">
///   <item><description>Token embedding lookup</description></item>
///   <item><description>For each layer: attention (with RoPE + KV cache) + residual, then SiLU-FFN + residual</description></item>
///   <item><description>Final RMSNorm + LM head projection</description></item>
/// </list>
/// </para>
/// </summary>
public sealed class LlamaForwardPass
{
    private readonly LlamaWeights _weights;
    private readonly LlamaConfig _cfg;

    // Scratch buffers reused across forward calls (thread-unsafe by design).
    private readonly float[] _x;       // [hidden_size]
    private readonly float[] _xNorm;   // [hidden_size]
    private readonly float[] _q;       // [q_dim]
    private readonly float[] _k;       // [kv_dim]
    private readonly float[] _v;       // [kv_dim]
    private readonly float[] _attnOut; // [q_dim]
    private readonly float[] _ffnBuf;  // [intermediate_size]
    private readonly float[] _ffnUp;   // [intermediate_size]
    private readonly float[] _logits;  // [vocab_size]

    /// <summary>Initializes a forward-pass engine for the given weights.</summary>
    public LlamaForwardPass(LlamaWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
        _cfg     = weights.Config;

        if (_cfg.NumKvHeads <= 0 || _cfg.NumHeads <= 0 || _cfg.NumKvHeads > _cfg.NumHeads || (_cfg.NumHeads % _cfg.NumKvHeads) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weights),
                $"Invalid attention head configuration: NumHeads={_cfg.NumHeads}, NumKvHeads={_cfg.NumKvHeads}. Expected NumKvHeads > 0, NumKvHeads <= NumHeads, and NumHeads % NumKvHeads == 0.");
        }

        _x       = new float[_cfg.HiddenSize];
        _xNorm   = new float[_cfg.HiddenSize];
        _q       = new float[_cfg.QueryDim];
        _k       = new float[_cfg.KvDim];
        _v       = new float[_cfg.KvDim];
        _attnOut = new float[_cfg.QueryDim];
        _ffnBuf  = new float[_cfg.IntermediateSize];
        _ffnUp   = new float[_cfg.IntermediateSize];
        _logits  = new float[_cfg.VocabSize];
    }

    /// <summary>Logits buffer populated after each call to <see cref="Forward"/>.</summary>
    public ReadOnlySpan<float> Logits => _logits;

    /// <summary>
    /// Runs one forward pass for the token at <paramref name="position"/> and returns the logits.
    /// </summary>
    /// <param name="tokenId">Input token ID.</param>
    /// <param name="position">Zero-based position in the sequence.</param>
    /// <param name="kvCache">KV cache that accumulates keys/values across forward calls.</param>
    public void Forward(int tokenId, int position, IKvCache kvCache)
    {
        ArgumentNullException.ThrowIfNull(kvCache);
        _weights.ThrowIfDisposed();

        int h = _cfg.HiddenSize;

        if ((uint)tokenId >= (uint)_cfg.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId,
                $"Token id {tokenId} is outside the valid vocabulary range [0, {_cfg.VocabSize - 1}].");
        }

        // 1. Token embedding lookup.
        _weights.TokenEmbedding.AsSpan(tokenId * h, h).CopyTo(_x);

        // 2. Transformer layers.
        for (int layer = 0; layer < _cfg.LayerCount; layer++)
        {
            // Attention sub-layer.
            TensorOperations.RmsNorm(_x, _weights.AttnNorm[layer], _xNorm);
            LinearProjection(_xNorm, _weights.Wq[layer], _q, _cfg.QueryDim);
            LinearProjection(_xNorm, _weights.Wk[layer], _k, _cfg.KvDim);
            LinearProjection(_xNorm, _weights.Wv[layer], _v, _cfg.KvDim);

            // Apply RoPE to Q and K (per-head).
            ApplyRopeAllHeads(_q, position, _cfg.NumHeads,   _cfg.HeadDim);
            ApplyRopeAllHeads(_k, position, _cfg.NumKvHeads, _cfg.HeadDim);

            // Append current K, V to cache.
            kvCache.Append(layer, _k, _v);

            // Compute multi-head attention using the full cached context.
            ComputeAttention(layer, position + 1, kvCache);

            // Output projection + residual.
            LinearProjection(_attnOut, _weights.Wo[layer], _xNorm, h);
            TensorPrimitives.Add(_x, _xNorm, _x);

            // FFN sub-layer (SiLU gated).
            TensorOperations.RmsNorm(_x, _weights.FfnNorm[layer], _xNorm);
            ComputeSiluFfn(_xNorm, layer);
            TensorPrimitives.Add(_x, _xNorm, _x);
        }

        // 3. Final norm + LM head.
        TensorOperations.RmsNorm(_x, _weights.OutputNorm, _xNorm);
        LinearProjection(_xNorm, _weights.OutputWeight, _logits, _cfg.VocabSize);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>Applies RoPE independently to each head's slice of a [numHeads × headDim] vector.</summary>
    private void ApplyRopeAllHeads(float[] vec, int position, int numHeads, int headDim)
    {
        for (int h = 0; h < numHeads; h++)
        {
            TensorOperations.ApplyRope(
                vec.AsSpan(h * headDim, headDim),
                position,
                headDim,
                _cfg.RopeBaseFreq);
        }
    }

    /// <summary>
    /// Computes grouped-query attention for the current token position
    /// and writes the result to <see cref="_attnOut"/>.
    /// </summary>
    private void ComputeAttention(int layer, int seqLen, IKvCache kvCache)
    {
        int headDim  = _cfg.HeadDim;
        int numHeads = _cfg.NumHeads;
        int numKvH   = _cfg.NumKvHeads;
        int groupSize = numHeads / numKvH; // queries per KV head (GQA grouping)
        float scale   = 1.0f / MathF.Sqrt(headDim);

        // Keys and values for all past tokens: [seqLen, kv_dim]
        var allKeys   = kvCache.GetKeys(layer);
        var allValues = kvCache.GetValues(layer);
        int kvDim     = _cfg.KvDim;

        float[]? rentedScores = null;
        Span<float> scores = seqLen <= 512
            ? stackalloc float[seqLen]
            : (rentedScores = ArrayPool<float>.Shared.Rent(seqLen)).AsSpan(0, seqLen);

        try
        {
            _attnOut.AsSpan().Clear();

            for (int qHead = 0; qHead < numHeads; qHead++)
            {
                int kvHead = qHead / groupSize;
                var qSlice = _q.AsSpan(qHead * headDim, headDim);

                // Compute attention scores: q · k_t for all t.
                for (int t = 0; t < seqLen; t++)
                {
                    var kSlice = allKeys.Slice(t * kvDim + kvHead * headDim, headDim);
                    scores[t]  = TensorOperations.Dot(qSlice, kSlice) * scale;
                }

                // Softmax over scores.
                TensorOperations.Softmax(scores.Slice(0, seqLen));

                // Weighted sum of values.
                var outSlice = _attnOut.AsSpan(qHead * headDim, headDim);
                for (int t = 0; t < seqLen; t++)
                {
                    var vSlice = allValues.Slice(t * kvDim + kvHead * headDim, headDim);
                    float w    = scores[t];
                    for (int d = 0; d < headDim; d++)
                    {
                        outSlice[d] += w * vSlice[d];
                    }
                }
            }
        }
        finally
        {
            if (rentedScores is not null)
            {
                ArrayPool<float>.Shared.Return(rentedScores);
            }
        }
    }

    /// <summary>
    /// Computes the SiLU-gated FFN: out = W_down(SiLU(W_gate(x)) * W_up(x)).
    /// Writes the result back to <see cref="_xNorm"/>.
    /// </summary>
    private void ComputeSiluFfn(float[] xNorm, int layer)
    {
        int ffnDim = _cfg.IntermediateSize;

        LinearProjection(xNorm, _weights.FfnGate[layer], _ffnBuf, ffnDim);
        TensorOperations.Silu(_ffnBuf);

        LinearProjection(xNorm, _weights.FfnUp[layer], _ffnUp, ffnDim);

        for (int i = 0; i < ffnDim; i++)
        {
            _ffnBuf[i] *= _ffnUp[i];
        }

        LinearProjection(_ffnBuf, _weights.FfnDown[layer], _xNorm, _cfg.HiddenSize);
    }

    /// <summary>
    /// Computes <c>dst[i] = Σ_j weight[i*inDim + j] * src[j]</c> for <c>i</c> in [0, outDim).
    /// </summary>
    private static void LinearProjection(
        ReadOnlySpan<float> src,
        float[] weight,
        Span<float> dst,
        int outDim)
    {
        int inDim = src.Length;
        for (int i = 0; i < outDim; i++)
        {
            dst[i] = TensorOperations.Dot(src, weight.AsSpan(i * inDim, inDim));
        }
    }
}
