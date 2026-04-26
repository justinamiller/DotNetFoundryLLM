using System.Buffers;
using System.Numerics;
using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Tensors;

/// <summary>A dense tensor backed by a rented memory buffer.</summary>
/// <typeparam name="T">The element type. Must be an unmanaged numeric type.</typeparam>
public sealed class Tensor<T> : ITensor, IDisposable
    where T : unmanaged, INumber<T>
{
    private T[]? _buffer;
    private bool _disposed;
    private readonly int[] _strides;
    private readonly DType _dtype;

    /// <summary>Initializes a new tensor with the given shape, renting a buffer from the array pool.</summary>
    public Tensor(Shape shape)
        : this(shape, dtypeOverride: null)
    {
    }

    /// <summary>Initializes a new tensor with the given shape and explicit logical dtype.</summary>
    public Tensor(Shape shape, DType? dtypeOverride)
    {
        ArgumentNullException.ThrowIfNull(shape);
        Shape = shape;
        _strides = ComputeStrides(shape);
        _buffer = ArrayPool<T>.Shared.Rent((int)shape.ElementCount);
        _buffer.AsSpan(0, (int)shape.ElementCount).Clear();
        _dtype = ResolveDType(dtypeOverride);
    }

    /// <summary>Initializes a new tensor from an existing span of data.</summary>
    public Tensor(Shape shape, ReadOnlySpan<T> data)
        : this(shape, data, dtypeOverride: null)
    {
    }

    /// <summary>Initializes a new tensor from an existing span of data and explicit logical dtype.</summary>
    public Tensor(Shape shape, ReadOnlySpan<T> data, DType? dtypeOverride)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (data.Length != shape.ElementCount)
        {
            throw new ArgumentException($"Data length {data.Length} does not match shape element count {shape.ElementCount}.");
        }

        Shape = shape;
        _strides = ComputeStrides(shape);
        _buffer = ArrayPool<T>.Shared.Rent((int)shape.ElementCount);
        data.CopyTo(_buffer);
        _dtype = ResolveDType(dtypeOverride);
    }

    private static int[] ComputeStrides(Shape shape)
    {
        if (shape.Rank == 0)
        {
            return [];
        }

        var strides = new int[shape.Rank];
        strides[shape.Rank - 1] = 1;
        for (var i = shape.Rank - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        return strides;
    }

    /// <summary>The typed shape of this tensor.</summary>
    public Shape Shape { get; }

    /// <inheritdoc />
    public int Rank => Shape.Rank;

    /// <summary>Total number of elements as a 64-bit integer.</summary>
    public long LongElementCount => Shape.ElementCount;

    // Explicit interface implementations to resolve type conflicts
    ReadOnlySpan<int> ITensor.Shape => Shape.Dims;
    ReadOnlySpan<int> ITensor.Strides => _strides;
    int ITensor.ElementCount => (int)Shape.ElementCount;
    DType ITensor.DType => _dtype;

    private static DType GetDefaultDType()
    {
        if (typeof(T) == typeof(float))
        {
            return DType.F32;
        }

        if (typeof(T) == typeof(int))
        {
            return DType.I32;
        }

        if (typeof(T) == typeof(sbyte))
        {
            return DType.I8;
        }

        if (typeof(T) == typeof(ushort))
        {
            return DType.F16;
        }

        return DType.F32;
    }

    private static DType ResolveDType(DType? overrideDType)
    {
        if (overrideDType is null)
        {
            return GetDefaultDType();
        }

        var requested = overrideDType.Value;
        if (typeof(T) == typeof(ushort) && (requested == DType.F16 || requested == DType.BF16))
        {
            return requested;
        }

        if (requested == GetDefaultDType())
        {
            return requested;
        }

        throw new ArgumentException($"DType override {requested} is incompatible with tensor element type {typeof(T).Name}.");
    }

    /// <summary>Returns a span over the tensor's underlying data.</summary>
    public Span<T> Span
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _buffer.AsSpan(0, (int)Shape.ElementCount);
        }
    }

    /// <summary>Returns a read-only span over the tensor's underlying data.</summary>
    public ReadOnlySpan<T> ReadOnlySpan
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _buffer.AsSpan(0, (int)Shape.ElementCount);
        }
    }

    /// <summary>Gets or sets an element by flat index.</summary>
    public T this[int flatIndex]
    {
        get => Span[flatIndex];
        set => Span[flatIndex] = value;
    }

    /// <summary>Gets or sets an element in a 2-D tensor.</summary>
    public T this[int row, int col]
    {
        get => Span[row * Shape[1] + col];
        set => Span[row * Shape[1] + col] = value;
    }

    /// <summary>Creates a new 1-D tensor from an array.</summary>
#pragma warning disable CA1000 // Static factory methods on generic types - required for ergonomic tensor construction
    public static Tensor<T> From(T[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        return new Tensor<T>(Shape.Of(data.Length), data);
    }

    /// <summary>Creates a new tensor filled with zeros.</summary>
    public static Tensor<T> Zeros(Shape shape) => new(shape);

    /// <summary>Creates a new tensor filled with ones.</summary>
    public static Tensor<T> Ones(Shape shape)
    {
        var t = new Tensor<T>(shape);
        t.Span.Fill(T.One);
        return t;
    }
#pragma warning restore CA1000

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_buffer is not null)
            {
                ArrayPool<T>.Shared.Return(_buffer);
                _buffer = null;
            }
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"Tensor<{typeof(T).Name}>{Shape}";
}
