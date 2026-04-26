namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>GGML tensor data types (used in tensor info sections of GGUF files).</summary>
public enum GgufTensorType : uint
{
    /// <summary>32-bit float.</summary>
    F32 = 0,
    /// <summary>16-bit IEEE 754 half-precision float.</summary>
    F16 = 1,
    /// <summary>4-bit symmetric quantisation, block size 32 (Q4_0).</summary>
#pragma warning disable CA1707 // underscores are standard GGML naming
    Q4_0 = 2,
    /// <summary>4-bit affine quantisation, block size 32 (Q4_1).</summary>
    Q4_1 = 3,
    /// <summary>5-bit symmetric quantisation, block size 32 (Q5_0).</summary>
    Q5_0 = 6,
    /// <summary>5-bit affine quantisation, block size 32 (Q5_1).</summary>
    Q5_1 = 7,
    /// <summary>8-bit symmetric quantisation, block size 32 (Q8_0).</summary>
    Q8_0 = 8,
    /// <summary>8-bit affine quantisation, block size 32 (Q8_1).</summary>
    Q8_1 = 9,
    /// <summary>2-bit K-quantisation (Q2_K).</summary>
    Q2_K = 10,
    /// <summary>3-bit K-quantisation (Q3_K).</summary>
    Q3_K = 11,
    /// <summary>4-bit K-quantisation (Q4_K).</summary>
    Q4_K = 12,
    /// <summary>5-bit K-quantisation (Q5_K).</summary>
    Q5_K = 13,
    /// <summary>6-bit K-quantisation (Q6_K).</summary>
    Q6_K = 14,
    /// <summary>8-bit K-quantisation (Q8_K).</summary>
    Q8_K = 15,
#pragma warning restore CA1707
    /// <summary>16-bit bfloat16.</summary>
    BF16 = 30,
}
