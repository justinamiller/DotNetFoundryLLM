namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>
/// Holds the parsed value of a GGUF metadata entry.
/// Only one of the typed properties will be non-null; the <see cref="ValueType"/> property
/// indicates which one.
/// </summary>
public sealed class GgufMetadataValue
{
    private GgufMetadataValue() { }

    /// <summary>The GGUF type of this value.</summary>
    public required GgufValueType ValueType { get; init; }

    // ── Scalar payloads ──────────────────────────────────────────────────────

    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Uint8"/>.</summary>
    public byte? Uint8Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Int8"/>.</summary>
    public sbyte? Int8Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Uint16"/>.</summary>
    public ushort? Uint16Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Int16"/>.</summary>
    public short? Int16Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Uint32"/>.</summary>
    public uint? Uint32Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Int32"/>.</summary>
    public int? Int32Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Float32"/>.</summary>
    public float? Float32Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Bool"/>.</summary>
    public bool? BoolValue { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.String"/>.</summary>
    public string? StringValue { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Uint64"/>.</summary>
    public ulong? Uint64Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Int64"/>.</summary>
    public long? Int64Value { get; init; }
    /// <summary>Value when <see cref="ValueType"/> is <see cref="GgufValueType.Float64"/>.</summary>
    public double? Float64Value { get; init; }

    // ── Array payload ─────────────────────────────────────────────────────────

    /// <summary>Element type when <see cref="ValueType"/> is <see cref="GgufValueType.Array"/>.</summary>
    public GgufValueType? ArrayElementType { get; init; }

    /// <summary>Array elements when <see cref="ValueType"/> is <see cref="GgufValueType.Array"/>.</summary>
    public IReadOnlyList<GgufMetadataValue>? ArrayValue { get; init; }

    // ── Convenience factory methods ───────────────────────────────────────────

    internal static GgufMetadataValue FromUint8(byte v)     => new() { ValueType = GgufValueType.Uint8,   Uint8Value   = v };
    internal static GgufMetadataValue FromInt8(sbyte v)     => new() { ValueType = GgufValueType.Int8,    Int8Value    = v };
    internal static GgufMetadataValue FromUint16(ushort v)  => new() { ValueType = GgufValueType.Uint16,  Uint16Value  = v };
    internal static GgufMetadataValue FromInt16(short v)    => new() { ValueType = GgufValueType.Int16,   Int16Value   = v };
    internal static GgufMetadataValue FromUint32(uint v)    => new() { ValueType = GgufValueType.Uint32,  Uint32Value  = v };
    internal static GgufMetadataValue FromInt32(int v)      => new() { ValueType = GgufValueType.Int32,   Int32Value   = v };
    internal static GgufMetadataValue FromFloat32(float v)  => new() { ValueType = GgufValueType.Float32, Float32Value = v };
    internal static GgufMetadataValue FromBool(bool v)      => new() { ValueType = GgufValueType.Bool,    BoolValue    = v };
    internal static GgufMetadataValue FromString(string v)  => new() { ValueType = GgufValueType.String,  StringValue  = v };
    internal static GgufMetadataValue FromUint64(ulong v)   => new() { ValueType = GgufValueType.Uint64,  Uint64Value  = v };
    internal static GgufMetadataValue FromInt64(long v)     => new() { ValueType = GgufValueType.Int64,   Int64Value   = v };
    internal static GgufMetadataValue FromFloat64(double v) => new() { ValueType = GgufValueType.Float64, Float64Value = v };

    internal static GgufMetadataValue FromArray(GgufValueType elemType, IReadOnlyList<GgufMetadataValue> elems) =>
        new() { ValueType = GgufValueType.Array, ArrayElementType = elemType, ArrayValue = elems };

    // ── Typed accessors ───────────────────────────────────────────────────────

    /// <summary>Returns the value as a boxed object for diagnostic purposes.</summary>
    public object? AsObject() => ValueType switch
    {
        GgufValueType.Uint8   => Uint8Value,
        GgufValueType.Int8    => Int8Value,
        GgufValueType.Uint16  => Uint16Value,
        GgufValueType.Int16   => Int16Value,
        GgufValueType.Uint32  => Uint32Value,
        GgufValueType.Int32   => Int32Value,
        GgufValueType.Float32 => Float32Value,
        GgufValueType.Bool    => BoolValue,
        GgufValueType.String  => StringValue,
        GgufValueType.Uint64  => Uint64Value,
        GgufValueType.Int64   => Int64Value,
        GgufValueType.Float64 => Float64Value,
        GgufValueType.Array   => ArrayValue,
        _                     => null
    };

    /// <inheritdoc />
    public override string ToString() =>
        ValueType == GgufValueType.Array
            ? $"Array<{ArrayElementType}>[{ArrayValue?.Count}]"
            : AsObject()?.ToString() ?? "(null)";
}
