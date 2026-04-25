using System.Buffers.Binary;
using System.Text;
using DotNetFoundryLLM.Core;

namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>
/// Parses GGUF binary files (versions 2 and 3) into a <see cref="GgufFile"/> instance.
/// <para>
/// The GGUF format layout:
/// <list type="number">
///   <item><description>4-byte magic: <c>GGUF</c></description></item>
///   <item><description>uint32 version</description></item>
///   <item><description>uint64 tensor_count</description></item>
///   <item><description>uint64 metadata_kv_count</description></item>
///   <item><description>metadata key-value pairs (metadata_kv_count entries)</description></item>
///   <item><description>tensor info entries (tensor_count entries)</description></item>
///   <item><description>alignment padding</description></item>
///   <item><description>raw tensor data</description></item>
/// </list>
/// </para>
/// </summary>
public sealed class GgufReader
{
    private static readonly byte[] s_magic = [(byte)'G', (byte)'G', (byte)'U', (byte)'F'];

    // Default alignment used when the metadata does not specify general.alignment.
    private const int DefaultAlignment = 32;

    /// <summary>
    /// Reads a GGUF file from <paramref name="path"/> into a <see cref="GgufFile"/>.
    /// The entire file is loaded into memory to allow zero-copy tensor data access.
    /// </summary>
    /// <param name="path">Absolute or relative path to the GGUF file.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A parsed <see cref="GgufFile"/>.</returns>
    /// <exception cref="ModelLoadException">Thrown when the file is not a valid GGUF file.</exception>
    public static async Task<GgufFile> ReadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        byte[] data;
        try
        {
            data = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new ModelLoadException(path, $"Failed to read GGUF file: {ex.Message}", ex);
        }

        try
        {
            return Parse(data);
        }
        catch (ModelLoadException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new ModelLoadException(path, $"Failed to parse GGUF file: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Parses a GGUF file from a raw byte array.
    /// The array is retained by the returned <see cref="GgufFile"/> for zero-copy tensor access.
    /// </summary>
    /// <param name="data">Complete GGUF file contents.</param>
    /// <returns>A parsed <see cref="GgufFile"/>.</returns>
    /// <exception cref="ModelLoadException">Thrown when the data is not a valid GGUF file.</exception>
    public static GgufFile Parse(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);

        var reader = new SpanReader(data);

        // ── Magic ────────────────────────────────────────────────────────────
        var magic = reader.ReadBytes(4);
        if (!magic.SequenceEqual(s_magic))
        {
            throw new ModelLoadException("(memory)", "Not a GGUF file: invalid magic bytes.");
        }

        // ── Version ──────────────────────────────────────────────────────────
        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
        {
            throw new ModelLoadException("(memory)", $"Unsupported GGUF version {version}. Expected 2 or 3.");
        }

        // ── Counts ───────────────────────────────────────────────────────────
        ulong tensorCount  = reader.ReadUInt64();
        ulong kvCount      = reader.ReadUInt64();

        // ── Metadata ─────────────────────────────────────────────────────────
        var metadata = new Dictionary<string, GgufMetadataValue>((int)kvCount, StringComparer.Ordinal);
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = reader.ReadGgufString();
            var value  = ReadMetadataValue(ref reader, version);
            metadata[key] = value;
        }

        // ── Tensor infos ──────────────────────────────────────────────────────
        var tensors = new List<GgufTensorInfo>((int)tensorCount);
        for (ulong i = 0; i < tensorCount; i++)
        {
            tensors.Add(ReadTensorInfo(ref reader));
        }

        // ── Alignment + data start ────────────────────────────────────────────
        int alignment = DefaultAlignment;
        if (metadata.TryGetValue("general.alignment", out var alignMeta))
        {
            alignment = alignMeta.ValueType switch
            {
                GgufValueType.Uint32 => (int)(alignMeta.Uint32Value ?? DefaultAlignment),
                GgufValueType.Uint64 => (int)(alignMeta.Uint64Value ?? DefaultAlignment),
                _                    => DefaultAlignment
            };
        }

        long headerEnd   = reader.Position;
        long alignedStart = ((headerEnd + alignment - 1) / alignment) * alignment;

        return new GgufFile(version, metadata, tensors, data, alignedStart);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static GgufMetadataValue ReadMetadataValue(ref SpanReader reader, uint version)
    {
        var valueType = (GgufValueType)reader.ReadUInt32();

        switch (valueType)
        {
            case GgufValueType.Uint8:   return GgufMetadataValue.FromUint8(reader.ReadByte());
            case GgufValueType.Int8:    return GgufMetadataValue.FromInt8((sbyte)reader.ReadByte());
            case GgufValueType.Uint16:  return GgufMetadataValue.FromUint16(reader.ReadUInt16());
            case GgufValueType.Int16:   return GgufMetadataValue.FromInt16(reader.ReadInt16());
            case GgufValueType.Uint32:  return GgufMetadataValue.FromUint32(reader.ReadUInt32());
            case GgufValueType.Int32:   return GgufMetadataValue.FromInt32(reader.ReadInt32());
            case GgufValueType.Float32: return GgufMetadataValue.FromFloat32(reader.ReadFloat32());
            case GgufValueType.Bool:    return GgufMetadataValue.FromBool(reader.ReadByte() != 0);
            case GgufValueType.String:  return GgufMetadataValue.FromString(reader.ReadGgufString());
            case GgufValueType.Uint64:  return GgufMetadataValue.FromUint64(reader.ReadUInt64());
            case GgufValueType.Int64:   return GgufMetadataValue.FromInt64(reader.ReadInt64());
            case GgufValueType.Float64: return GgufMetadataValue.FromFloat64(reader.ReadFloat64());

            case GgufValueType.Array:
            {
                var elemType  = (GgufValueType)reader.ReadUInt32();
                // v2 uses uint32 count; v3 uses uint64 count.
                ulong count = version >= 3 ? reader.ReadUInt64() : reader.ReadUInt32();
                var elements = new List<GgufMetadataValue>((int)count);
                for (ulong j = 0; j < count; j++)
                {
                    elements.Add(ReadScalarValue(ref reader, elemType));
                }

                return GgufMetadataValue.FromArray(elemType, elements);
            }

            default:
                throw new NotSupportedException($"Unknown GGUF metadata value type: {valueType}");
        }
    }

    private static GgufMetadataValue ReadScalarValue(ref SpanReader reader, GgufValueType valueType) =>
        valueType switch
        {
            GgufValueType.Uint8   => GgufMetadataValue.FromUint8(reader.ReadByte()),
            GgufValueType.Int8    => GgufMetadataValue.FromInt8((sbyte)reader.ReadByte()),
            GgufValueType.Uint16  => GgufMetadataValue.FromUint16(reader.ReadUInt16()),
            GgufValueType.Int16   => GgufMetadataValue.FromInt16(reader.ReadInt16()),
            GgufValueType.Uint32  => GgufMetadataValue.FromUint32(reader.ReadUInt32()),
            GgufValueType.Int32   => GgufMetadataValue.FromInt32(reader.ReadInt32()),
            GgufValueType.Float32 => GgufMetadataValue.FromFloat32(reader.ReadFloat32()),
            GgufValueType.Bool    => GgufMetadataValue.FromBool(reader.ReadByte() != 0),
            GgufValueType.String  => GgufMetadataValue.FromString(reader.ReadGgufString()),
            GgufValueType.Uint64  => GgufMetadataValue.FromUint64(reader.ReadUInt64()),
            GgufValueType.Int64   => GgufMetadataValue.FromInt64(reader.ReadInt64()),
            GgufValueType.Float64 => GgufMetadataValue.FromFloat64(reader.ReadFloat64()),
            _                     => throw new NotSupportedException($"Unsupported array element type: {valueType}")
        };

    private static GgufTensorInfo ReadTensorInfo(ref SpanReader reader)
    {
        string name   = reader.ReadGgufString();
        uint nDims    = reader.ReadUInt32();
        var dims      = new ulong[nDims];
        for (uint d = 0; d < nDims; d++)
        {
            dims[d] = reader.ReadUInt64();
        }

        var tensorType = (GgufTensorType)reader.ReadUInt32();
        ulong offset   = reader.ReadUInt64();

        return new GgufTensorInfo
        {
            Name       = name,
            Dims       = dims,
            TensorType = tensorType,
            Offset     = offset,
        };
    }

    // ── Lightweight span-based reader ─────────────────────────────────────────

    private ref struct SpanReader
    {
        private readonly ReadOnlySpan<byte> _data;
        private int _pos;

        public SpanReader(byte[] data)
        {
            _data = data;
            _pos  = 0;
        }

        public readonly long Position => _pos;

        public ReadOnlySpan<byte> ReadBytes(int count)
        {
            var slice = _data.Slice(_pos, count);
            _pos += count;
            return slice;
        }

        public byte ReadByte()          => _data[_pos++];
        public ushort ReadUInt16()      { var v = BinaryPrimitives.ReadUInt16LittleEndian(_data.Slice(_pos, 2)); _pos += 2; return v; }
        public short  ReadInt16()       { var v = BinaryPrimitives.ReadInt16LittleEndian(_data.Slice(_pos, 2));  _pos += 2; return v; }
        public uint   ReadUInt32()      { var v = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos, 4)); _pos += 4; return v; }
        public int    ReadInt32()       { var v = BinaryPrimitives.ReadInt32LittleEndian(_data.Slice(_pos, 4));  _pos += 4; return v; }
        public ulong  ReadUInt64()      { var v = BinaryPrimitives.ReadUInt64LittleEndian(_data.Slice(_pos, 8)); _pos += 8; return v; }
        public long   ReadInt64()       { var v = BinaryPrimitives.ReadInt64LittleEndian(_data.Slice(_pos, 8));  _pos += 8; return v; }
        public float  ReadFloat32()     { var v = BitConverter.Int32BitsToSingle(ReadInt32()); return v; }
        public double ReadFloat64()     { var v = BitConverter.Int64BitsToDouble(ReadInt64());  return v; }

        public string ReadGgufString()
        {
            ulong len = ReadUInt64();
            var bytes = ReadBytes((int)len);
            return Encoding.UTF8.GetString(bytes);
        }
    }
}
