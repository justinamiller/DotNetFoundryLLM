using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Tensors;

/// <summary>An <see cref="ITensorAllocator"/> backed by <see cref="System.Buffers.ArrayPool{T}"/>.</summary>
public sealed class ArrayPoolTensorAllocator : ITensorAllocator
{
    /// <summary>A shared default instance.</summary>
    public static ArrayPoolTensorAllocator Shared { get; } = new();

    /// <inheritdoc />
    public ITensor Allocate(DType dtype, ReadOnlySpan<int> shape)
    {
        var s = new Shape(shape.ToArray());
        return dtype switch
        {
            DType.F32 => new Tensor<float>(s),
            DType.F16 => new Tensor<ushort>(s, DType.F16),
            DType.BF16 => new Tensor<ushort>(s, DType.BF16),
            DType.I32 => new Tensor<int>(s),
            DType.I8 => new Tensor<sbyte>(s),
            _ => new Tensor<float>(s)
        };
    }

    /// <summary>Allocates a float32 tensor of the given shape.</summary>
    public static Tensor<float> AllocateF32(Shape shape) => new(shape);

    /// <summary>Allocates a float16 (represented as ushort) tensor of the given shape.</summary>
    public static Tensor<ushort> AllocateF16(Shape shape) => new(shape, DType.F16);

    /// <summary>Allocates a bfloat16 (represented as ushort) tensor of the given shape.</summary>
    public static Tensor<ushort> AllocateBF16(Shape shape) => new(shape, DType.BF16);

    /// <summary>Allocates an int32 tensor of the given shape.</summary>
    public static Tensor<int> AllocateI32(Shape shape) => new(shape);

    /// <summary>Allocates an int8 (represented as sbyte) tensor of the given shape.</summary>
    public static Tensor<sbyte> AllocateI8(Shape shape) => new(shape);
}
