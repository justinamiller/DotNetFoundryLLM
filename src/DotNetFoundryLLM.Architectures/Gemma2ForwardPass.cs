using System.Buffers;
using System.Numerics.Tensors;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Architectures;

/// <summary>
/// Implements a single forward pass of the Gemma 2 transformer architecture.
/// </summary>
public sealed class Gemma2ForwardPass : IForwardPass
{
    private readonly Gemma2Weights _weights;
    private readonly Gemma2Config _cfg;
    private readonly float[] _x;
    private readonly float[] _xNorm;
    private readonly float[] _q;
    private readonly float[] _k;
    private readonly float[] _v;
    private readonly float[] _attnOut;
    private readonly float[] _ffnBuf;
    private readonly float[] _ffnUp;
    private readonly float[] _logits;
    private readonly float[] _ropeInvFreqs;

    /// <summary>Initializes a forward-pass engine for the given weights.</summary>
    public Gemma2ForwardPass(Gemma2Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
        _cfg = weights.Config;

        if (_cfg.NumKvHeads <= 0 || _cfg.NumHeads <= 0 || _cfg.NumKvHeads > _cfg.NumHeads || (_cfg.NumHeads % _cfg.NumKvHeads) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weights),
                $"Invalid attention head configuration: NumHeads={_cfg.NumHeads}, NumKvHeads={_cfg.NumKvHeads}. Expected NumKvHeads > 0, NumKvHeads <= NumHeads, and NumHeads % NumKvHeads == 0.");
        }

        _x = new float[_cfg.HiddenSize];
        _xNorm = new float[_cfg.HiddenSize];
        _q = new float[_cfg.QueryDim];
        _k = new float[_cfg.KvDim];
        _v = new float[_cfg.KvDim];
        _attnOut = new float[_cfg.QueryDim];
        _ffnBuf = new float[_cfg.IntermediateSize];
        _ffnUp = new float[_cfg.IntermediateSize];
        _logits = new float[_cfg.VocabSize];

        // Precompute RoPE inverse frequencies to eliminate per-token Pow calls
        int half = _cfg.HeadDim / 2;
        _ropeInvFreqs = new float[half];
        for (int i = 0; i < half; i++)
        {
            _ropeInvFreqs[i] = 1f / MathF.Pow(_cfg.RopeBaseFreq, 2f * i / _cfg.HeadDim);
        }
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> Logits => _logits;

    /// <inheritdoc />
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

        _weights.TokenEmbedding.AsSpan(tokenId * h, h).CopyTo(_x);

        for (int layer = 0; layer < _cfg.LayerCount; layer++)
        {
            TensorOperations.GemmaRmsNorm(_x, _weights.AttnNorm[layer], _xNorm);
            LinearProjection(_xNorm, _weights.Wq[layer], _q, _cfg.QueryDim);
            LinearProjection(_xNorm, _weights.Wk[layer], _k, _cfg.KvDim);
            LinearProjection(_xNorm, _weights.Wv[layer], _v, _cfg.KvDim);

            ApplyRopeAllHeads(_q, position, _cfg.NumHeads, _cfg.HeadDim);
            ApplyRopeAllHeads(_k, position, _cfg.NumKvHeads, _cfg.HeadDim);

            kvCache.Append(layer, _k, _v);
            ComputeAttention(layer, position + 1, kvCache, layer % 2 == 0 ? _cfg.SlidingWindowSize : int.MaxValue);

            LinearProjection(_attnOut, _weights.Wo[layer], _xNorm, h);
            TensorOperations.GemmaRmsNorm(_xNorm, _weights.PostAttnNorm[layer], _xNorm);
            TensorPrimitives.Add(_x, _xNorm, _x);

            TensorOperations.GemmaRmsNorm(_x, _weights.FfnNorm[layer], _xNorm);
            ComputeGeluFfn(_xNorm, layer);
            TensorOperations.GemmaRmsNorm(_xNorm, _weights.PostFfnNorm[layer], _xNorm);
            TensorPrimitives.Add(_x, _xNorm, _x);
        }

        TensorOperations.GemmaRmsNorm(_x, _weights.OutputNorm, _xNorm);
        LinearProjection(_xNorm, _weights.OutputWeight, _logits, _cfg.VocabSize);
        if (_cfg.FinalLogitSoftcap > 0f)
        {
            TensorOperations.SoftCap(_logits, _cfg.FinalLogitSoftcap);
        }
    }

    private void ApplyRopeAllHeads(float[] vec, int position, int numHeads, int headDim)
    {
        float scaledPos = ShouldScaleRope(_cfg.RopeScalingType) && _cfg.RopeScalingFactor > 0f
            ? position / _cfg.RopeScalingFactor
            : (float)position;

        for (int h = 0; h < numHeads; h++)
        {
            TensorOperations.ApplyRope(
                vec.AsSpan(h * headDim, headDim),
                scaledPos,
                _ropeInvFreqs);
        }
    }

    private static bool ShouldScaleRope(string ropeScalingType)
        => ropeScalingType.Equals("linear", StringComparison.OrdinalIgnoreCase)
            || ropeScalingType.Equals("yarn", StringComparison.OrdinalIgnoreCase);

    private void ComputeAttention(int layer, int seqLen, IKvCache kvCache, int slidingWindowSize)
    {
        int headDim = _cfg.HeadDim;
        int numHeads = _cfg.NumHeads;
        int numKvH = _cfg.NumKvHeads;
        int groupSize = numHeads / numKvH;
        float scale = _cfg.QueryPreAttnScalar;
        var allKeys = kvCache.GetKeys(layer);
        var allValues = kvCache.GetValues(layer);
        int kvDim = _cfg.KvDim;
        int windowStart = Math.Max(0, seqLen - slidingWindowSize);
        int attendedLength = seqLen - windowStart;

        float[]? rentedScores = null;
        Span<float> scores = attendedLength <= 512
            ? stackalloc float[attendedLength]
            : (rentedScores = ArrayPool<float>.Shared.Rent(attendedLength)).AsSpan(0, attendedLength);

        try
        {
            _attnOut.AsSpan().Clear();

            for (int qHead = 0; qHead < numHeads; qHead++)
            {
                int kvHead = qHead / groupSize;
                var qSlice = _q.AsSpan(qHead * headDim, headDim);

                for (int t = windowStart; t < seqLen; t++)
                {
                    var kSlice = allKeys.Slice(t * kvDim + kvHead * headDim, headDim);
                    scores[t - windowStart] = TensorOperations.Dot(qSlice, kSlice) * scale;
                }

                if (_cfg.AttnLogitSoftcap > 0f)
                {
                    TensorOperations.SoftCap(scores.Slice(0, attendedLength), _cfg.AttnLogitSoftcap);
                }

                TensorOperations.Softmax(scores.Slice(0, attendedLength));

                var outSlice = _attnOut.AsSpan(qHead * headDim, headDim);
                for (int t = windowStart; t < seqLen; t++)
                {
                    var vSlice = allValues.Slice(t * kvDim + kvHead * headDim, headDim);
                    float w = scores[t - windowStart];
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

    private void ComputeGeluFfn(float[] xNorm, int layer)
    {
        int ffnDim = _cfg.IntermediateSize;
        LinearProjection(xNorm, _weights.FfnGate[layer], _ffnBuf, ffnDim);
        TensorOperations.Gelu(_ffnBuf);
        LinearProjection(xNorm, _weights.FfnUp[layer], _ffnUp, ffnDim);
        for (int i = 0; i < ffnDim; i++)
        {
            _ffnBuf[i] *= _ffnUp[i];
        }

        LinearProjection(_ffnBuf, _weights.FfnDown[layer], _xNorm, _cfg.HiddenSize);
    }

    private static void LinearProjection(ReadOnlySpan<float> src, float[] weight, Span<float> dst, int outDim)
    {
        int inDim = src.Length;
        for (int i = 0; i < outDim; i++)
        {
            dst[i] = TensorOperations.Dot(src, weight.AsSpan(i * inDim, inDim));
        }
    }
}
