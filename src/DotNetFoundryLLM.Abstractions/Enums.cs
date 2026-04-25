namespace DotNetFoundryLLM.Abstractions;

/// <summary>Data type of tensor elements.</summary>
public enum DType
{
    /// <summary>32-bit floating point.</summary>
    F32,
    /// <summary>16-bit floating point.</summary>
    F16,
    /// <summary>Brain 16-bit floating point.</summary>
    BF16,
#pragma warning disable CA1707 // Identifiers should not contain underscores - standard ML/GGUF naming
    /// <summary>8-bit quantized, block size 32.</summary>
    Q8_0,
    /// <summary>5-bit K-quantized.</summary>
    Q5_K,
    /// <summary>4-bit K-quantized, medium.</summary>
    Q4_K_M,
    /// <summary>4-bit quantized, block size 32.</summary>
    Q4_0,
    /// <summary>3-bit K-quantized.</summary>
    Q3_K,
    /// <summary>2-bit K-quantized.</summary>
    Q2_K,
#pragma warning restore CA1707
    /// <summary>8-bit integer.</summary>
    I8,
    /// <summary>32-bit integer.</summary>
    I32
}

/// <summary>Role of a chat message participant.</summary>
public enum Role
{
    /// <summary>System prompt role.</summary>
    System,
    /// <summary>User turn role.</summary>
    User,
    /// <summary>Assistant turn role.</summary>
    Assistant,
    /// <summary>Tool call result role.</summary>
    Tool
}

/// <summary>Type of rotary positional embedding.</summary>
public enum RopeType
{
    /// <summary>Standard LLaMA-style RoPE.</summary>
    Llama,
    /// <summary>NeoX-style RoPE.</summary>
    NeoX,
    /// <summary>Long-context RoPE scaling.</summary>
    LongRope,
    /// <summary>YaRN context extension.</summary>
    Yarn
}

/// <summary>Attention mechanism variant.</summary>
public enum AttentionKind
{
    /// <summary>Multi-head attention.</summary>
    Mha,
    /// <summary>Multi-query attention.</summary>
    Mqa,
    /// <summary>Grouped-query attention.</summary>
    Gqa,
    /// <summary>Sliding-window attention.</summary>
    SlidingWindow
}
