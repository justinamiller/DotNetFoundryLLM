using System.Numerics;
using System.Runtime.InteropServices;

namespace DotNetFoundryLLM.Tensors;

/// <summary>Reports available hardware acceleration capabilities.</summary>
public static class HardwareCapabilities
{
    /// <summary>Returns true if AVX2 is supported.</summary>
    public static bool HasAvx2 { get; } = System.Runtime.Intrinsics.X86.Avx2.IsSupported;

    /// <summary>Returns true if AVX-512F is supported.</summary>
    public static bool HasAvx512F { get; } = System.Runtime.Intrinsics.X86.Avx512F.IsSupported;

    /// <summary>Returns true if ARM AdvSIMD is supported.</summary>
    public static bool HasAdvSimd { get; } = System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported;

    /// <summary>Returns true if SSE4.1 is supported.</summary>
    public static bool HasSse41 { get; } = System.Runtime.Intrinsics.X86.Sse41.IsSupported;

    /// <summary>Returns true if hardware vector acceleration is available.</summary>
    public static bool HasVectorAcceleration { get; } = Vector.IsHardwareAccelerated;

    /// <summary>Gets the native vector width for <typeparamref name="T"/> in elements.</summary>
    public static int VectorWidth<T>() where T : struct => Vector<T>.Count;

    /// <summary>Returns a string describing available acceleration.</summary>
    public static string Describe()
    {
        var features = new System.Text.StringBuilder("HardwareCapabilities{");
        if (HasAvx512F) features.Append(" AVX-512F");
        if (HasAvx2) features.Append(" AVX2");
        if (HasSse41) features.Append(" SSE4.1");
        if (HasAdvSimd) features.Append(" AdvSIMD");
        if (HasVectorAcceleration) features.Append(" Vector");
        features.Append(" }");
        return features.ToString();
    }
}
