using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Architectures;
using Xunit;

namespace DotNetFoundryLLM.Architectures.Tests;

/// <summary>Tests for <see cref="Gemma2ForwardPass"/>.</summary>
public sealed class Gemma2ForwardPassTests
{
    [Fact]
    public void Forward_ProducesFiniteLogitsWithExpectedShape()
    {
        var cfg = new Gemma2Config
        {
            Architecture = "gemma2",
            ModelFamily = "gemma2",
            LayerCount = 2,
            HiddenSize = 8,
            IntermediateSize = 8,
            NumHeads = 2,
            NumKvHeads = 2,
            MaxContextLength = 8,
            VocabSize = 16,
            SlidingWindowSize = 2,
            AttnLogitSoftcap = 4f,
            FinalLogitSoftcap = 6f,
            QueryPreAttnScalar = 0.5f,
        };

        using var weights = new Gemma2Weights(
            cfg,
            MakeArray(cfg.VocabSize * cfg.HiddenSize, 0.01f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize, 0.02f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize, 0.025f),
            MakeJagged(cfg.LayerCount, cfg.QueryDim * cfg.HiddenSize, 0.03f),
            MakeJagged(cfg.LayerCount, cfg.KvDim * cfg.HiddenSize, 0.04f),
            MakeJagged(cfg.LayerCount, cfg.KvDim * cfg.HiddenSize, 0.05f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize * cfg.QueryDim, 0.06f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize, 0.07f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize, 0.075f),
            MakeJagged(cfg.LayerCount, cfg.IntermediateSize * cfg.HiddenSize, 0.08f),
            MakeJagged(cfg.LayerCount, cfg.IntermediateSize * cfg.HiddenSize, 0.09f),
            MakeJagged(cfg.LayerCount, cfg.HiddenSize * cfg.IntermediateSize, 0.10f),
            MakeArray(cfg.HiddenSize, 0.11f),
            MakeArray(cfg.VocabSize * cfg.HiddenSize, 0.12f));

        var forward = new Gemma2ForwardPass(weights);
        using var kv = new TestKvCache(cfg.LayerCount, cfg.MaxContextLength, cfg.KvDim);

        forward.Forward(1, 0, kv);
        forward.Forward(2, 1, kv);
        forward.Forward(3, 2, kv);

        var logits = forward.Logits;
        Assert.Equal(cfg.VocabSize, logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.True(float.IsFinite(logits[i]));
        }
    }

    private static float[] MakeArray(int length, float scale)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = ((i % 7) - 3) * scale;
        }

        return data;
    }

    private static float[][] MakeJagged(int outer, int inner, float scale)
    {
        var data = new float[outer][];
        for (int i = 0; i < outer; i++)
        {
            data[i] = MakeArray(inner, scale + i * 0.01f);
        }

        return data;
    }

    private sealed class TestKvCache(int layerCount, int maxSequenceLength, int kvDim) : IKvCache
    {
        private readonly float[][] _keys = Enumerable.Range(0, layerCount).Select(_ => new float[maxSequenceLength * kvDim]).ToArray();
        private readonly float[][] _values = Enumerable.Range(0, layerCount).Select(_ => new float[maxSequenceLength * kvDim]).ToArray();
        private int _currentLength;
        private int _nextLayerToAppend;

        public int MaxSequenceLength { get; } = maxSequenceLength;
        public int CurrentLength => _currentLength;

        public void Append(int layer, ReadOnlySpan<float> keySlice, ReadOnlySpan<float> valueSlice)
        {
            int offset = _currentLength * kvDim;
            keySlice.CopyTo(_keys[layer].AsSpan(offset, kvDim));
            valueSlice.CopyTo(_values[layer].AsSpan(offset, kvDim));
            _nextLayerToAppend++;
            if (_nextLayerToAppend == layerCount)
            {
                _nextLayerToAppend = 0;
                _currentLength++;
            }
        }

        public ReadOnlySpan<float> GetKeys(int layer)
        {
            int effectiveLength = _currentLength;
            if (_nextLayerToAppend > 0 && layer < _nextLayerToAppend)
            {
                effectiveLength++;
            }

            return _keys[layer].AsSpan(0, effectiveLength * kvDim);
        }

        public ReadOnlySpan<float> GetValues(int layer)
        {
            int effectiveLength = _currentLength;
            if (_nextLayerToAppend > 0 && layer < _nextLayerToAppend)
            {
                effectiveLength++;
            }

            return _values[layer].AsSpan(0, effectiveLength * kvDim);
        }

        public void Clear()
        {
            _currentLength = 0;
            _nextLayerToAppend = 0;
        }

        public void Dispose()
        {
        }
    }
}
