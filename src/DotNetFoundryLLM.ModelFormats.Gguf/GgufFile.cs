namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>
/// Represents a fully parsed GGUF file, exposing its metadata and tensor descriptors.
/// Raw tensor bytes are available via <see cref="GetTensorBytes"/>.
/// </summary>
public sealed class GgufFile : IDisposable
{
    private readonly byte[] _fileBytes;
    private readonly long _dataStart;
    private bool _disposed;

    internal GgufFile(
        uint version,
        IReadOnlyDictionary<string, GgufMetadataValue> metadata,
        IReadOnlyList<GgufTensorInfo> tensors,
        byte[] fileBytes,
        long dataStart)
    {
        Version = version;
        Metadata = metadata;
        Tensors = tensors;
        _fileBytes = fileBytes;
        _dataStart = dataStart;
    }

    /// <summary>GGUF format version (2 or 3).</summary>
    public uint Version { get; }

    /// <summary>All key-value metadata pairs parsed from the file header.</summary>
    public IReadOnlyDictionary<string, GgufMetadataValue> Metadata { get; }

    /// <summary>Descriptors for every tensor stored in the file.</summary>
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    /// <summary>Looks up a tensor descriptor by name.</summary>
    /// <param name="name">Exact tensor name.</param>
    /// <returns>The matching <see cref="GgufTensorInfo"/>, or <see langword="null"/> if not found.</returns>
    public GgufTensorInfo? FindTensor(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        foreach (var t in Tensors)
        {
            if (t.Name == name) return t;
        }

        return null;
    }

    /// <summary>
    /// Returns a read-only span over the raw bytes of the specified tensor's data.
    /// </summary>
    /// <param name="info">Tensor descriptor obtained from <see cref="Tensors"/>.</param>
    /// <returns>Raw (potentially quantized) tensor bytes.</returns>
    /// <exception cref="ObjectDisposedException">Thrown after the file has been disposed.</exception>
    public ReadOnlySpan<byte> GetTensorBytes(GgufTensorInfo info)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(info);

        long byteSize = ComputeByteSize(info);
        long absoluteOffset = _dataStart + (long)info.Offset;
        return _fileBytes.AsSpan((int)absoluteOffset, (int)byteSize);
    }

    /// <summary>Returns the size in bytes of the tensor data, computed from its type and element count.</summary>
    public static long ComputeByteSize(GgufTensorInfo info)
    {
        ArgumentNullException.ThrowIfNull(info);
        long elems = info.ElementCount;
        return info.TensorType switch
        {
            GgufTensorType.F32  => elems * 4,
            GgufTensorType.F16  => elems * 2,
            GgufTensorType.BF16 => elems * 2,
#pragma warning disable CA1707
            GgufTensorType.Q4_0 => elems / 32 * 18,
            GgufTensorType.Q4_1 => elems / 32 * 20,
            GgufTensorType.Q5_0 => elems / 32 * 22,
            GgufTensorType.Q5_1 => elems / 32 * 24,
            GgufTensorType.Q8_0 => elems / 32 * 34,
            GgufTensorType.Q8_1 => elems / 32 * 36,
#pragma warning restore CA1707
            _ => throw new NotSupportedException($"Cannot compute byte size for tensor type {info.TensorType}.")
        };
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _disposed = true;
    }
}
