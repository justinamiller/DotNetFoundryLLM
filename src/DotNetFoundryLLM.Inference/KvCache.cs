using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Inference;

/// <summary>
/// A flat-array KV cache that stores attention keys and values for all transformer
/// layers across the sequence generated so far.
/// <para>
/// Storage layout per layer:
/// <c>keys[layer][position * kvDim .. (position+1) * kvDim - 1]</c> —
/// one contiguous slice of <c>kvDim</c> floats per token position.
/// </para>
/// </summary>
public sealed class KvCache : IKvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _kvDim;
    private readonly int _layerCount;
    private int _currentLength;
    private int _nextLayerToAppend;
    private bool _disposed;

    /// <summary>
    /// Allocates a new KV cache.
    /// </summary>
    /// <param name="layerCount">Number of transformer layers.</param>
    /// <param name="maxSequenceLength">Maximum number of token positions to cache.</param>
    /// <param name="kvDim">Dimension of each key/value slice (numKvHeads × headDim).</param>
    public KvCache(int layerCount, int maxSequenceLength, int kvDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);

        MaxSequenceLength = maxSequenceLength;
        _kvDim = kvDim;
        _layerCount = layerCount;

        _keys = new float[layerCount][];
        _values = new float[layerCount][];
        for (int i = 0; i < layerCount; i++)
        {
            _keys[i] = new float[maxSequenceLength * kvDim];
            _values[i] = new float[maxSequenceLength * kvDim];
        }
    }

    /// <inheritdoc />
    public int MaxSequenceLength { get; }

    /// <inheritdoc />
    public int CurrentLength => _currentLength;

    /// <inheritdoc />
    public void Append(int layer, ReadOnlySpan<float> keySlice, ReadOnlySpan<float> valueSlice)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if ((uint)layer >= (uint)_layerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), layer,
                $"Layer index must be in [0, {_layerCount - 1}].");
        }

        if (layer != _nextLayerToAppend)
        {
            throw new InvalidOperationException(
                $"KV cache append order violation: expected layer {_nextLayerToAppend} but received layer {layer}. " +
                "Append layers sequentially for each token position.");
        }

        if (_currentLength >= MaxSequenceLength)
        {
            throw new InvalidOperationException(
                $"KV cache is full (capacity {MaxSequenceLength} tokens). Clear the cache or reduce input length.");
        }

        if (keySlice.Length != _kvDim || valueSlice.Length != _kvDim)
        {
            throw new ArgumentException(
                $"Key/value slice length must be {_kvDim} but got key={keySlice.Length}, value={valueSlice.Length}.");
        }

        int offset = _currentLength * _kvDim;
        keySlice.CopyTo(_keys[layer].AsSpan(offset, _kvDim));
        valueSlice.CopyTo(_values[layer].AsSpan(offset, _kvDim));

        _nextLayerToAppend++;
        if (_nextLayerToAppend == _layerCount)
        {
            _nextLayerToAppend = 0;
            _currentLength++;
        }
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> GetKeys(int layer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layer >= (uint)_layerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), layer,
                $"Layer index must be in [0, {_layerCount - 1}].");
        }

        int effectiveLength = _currentLength;
        if (_nextLayerToAppend > 0 && layer < _nextLayerToAppend)
        {
            effectiveLength++;
        }

        return _keys[layer].AsSpan(0, effectiveLength * _kvDim);
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> GetValues(int layer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layer >= (uint)_layerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), layer,
                $"Layer index must be in [0, {_layerCount - 1}].");
        }

        int effectiveLength = _currentLength;
        if (_nextLayerToAppend > 0 && layer < _nextLayerToAppend)
        {
            effectiveLength++;
        }

        return _values[layer].AsSpan(0, effectiveLength * _kvDim);
    }

    /// <inheritdoc />
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _currentLength = 0;
        _nextLayerToAppend = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _disposed = true;
    }
}
