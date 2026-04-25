namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>Value types stored in GGUF metadata key-value pairs.</summary>
// These enum member names match the GGUF specification exactly.
#pragma warning disable CA1720 // Identifier contains type name — required to match GGUF spec
public enum GgufValueType : uint
{
    /// <summary>Unsigned 8-bit integer.</summary>
    Uint8 = 0,
    /// <summary>Signed 8-bit integer.</summary>
    Int8 = 1,
    /// <summary>Unsigned 16-bit integer.</summary>
    Uint16 = 2,
    /// <summary>Signed 16-bit integer.</summary>
    Int16 = 3,
    /// <summary>Unsigned 32-bit integer.</summary>
    Uint32 = 4,
    /// <summary>Signed 32-bit integer.</summary>
    Int32 = 5,
    /// <summary>32-bit floating-point.</summary>
    Float32 = 6,
    /// <summary>Boolean (stored as a single byte).</summary>
    Bool = 7,
    /// <summary>UTF-8 string prefixed by a uint64 length.</summary>
    String = 8,
    /// <summary>Array of a homogeneous value type.</summary>
    Array = 9,
    /// <summary>Unsigned 64-bit integer.</summary>
    Uint64 = 10,
    /// <summary>Signed 64-bit integer.</summary>
    Int64 = 11,
    /// <summary>64-bit floating-point.</summary>
    Float64 = 12,
}
#pragma warning restore CA1720
