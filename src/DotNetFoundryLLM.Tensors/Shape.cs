namespace DotNetFoundryLLM.Tensors;

/// <summary>Describes the shape of a tensor as an immutable array of dimension sizes.</summary>
public sealed class Shape : IEquatable<Shape>
{
    private readonly int[] _dims;

    /// <summary>Creates a shape from explicit dimension values.</summary>
    public Shape(params int[] dims)
    {
        ArgumentNullException.ThrowIfNull(dims);
        for (var i = 0; i < dims.Length; i++)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dims[i], $"dims[{i}]");
        }

        _dims = (int[])dims.Clone();
        Rank = _dims.Length;
        ElementCount = Rank == 0 ? 1 : ComputeElementCount(_dims);
    }

    private static long ComputeElementCount(int[] dims)
    {
        long count = 1;
        foreach (var d in dims)
        {
            count *= d;
        }

        return count;
    }

    /// <summary>Creates a scalar shape (rank 0).</summary>
    public static Shape Scalar { get; } = new();

    /// <summary>Creates a 1-D shape.</summary>
    public static Shape Of(int d0) => new(d0);

    /// <summary>Creates a 2-D shape.</summary>
    public static Shape Of(int d0, int d1) => new(d0, d1);

    /// <summary>Creates a 3-D shape.</summary>
    public static Shape Of(int d0, int d1, int d2) => new(d0, d1, d2);

    /// <summary>Creates a 4-D shape.</summary>
    public static Shape Of(int d0, int d1, int d2, int d3) => new(d0, d1, d2, d3);

    /// <summary>Number of dimensions.</summary>
    public int Rank { get; }

    /// <summary>Total number of elements.</summary>
    public long ElementCount { get; }

    /// <summary>Gets the size of a specific dimension.</summary>
    public int this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(index, 0);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Rank);
            return _dims[index];
        }
    }

    /// <summary>Gets the dimensions as a read-only span.</summary>
    public ReadOnlySpan<int> Dims => _dims;

    /// <summary>Returns a new shape with the given axis removed.</summary>
    public Shape RemoveAxis(int axis)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(axis, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(axis, Rank);
        var newDims = new int[Rank - 1];
        var j = 0;
        for (var i = 0; i < Rank; i++)
        {
            if (i != axis)
            {
                newDims[j++] = _dims[i];
            }
        }

        return new Shape(newDims);
    }

    /// <summary>Returns a new shape with one dimension replaced.</summary>
    public Shape WithDim(int axis, int newSize)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(axis, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(axis, Rank);
        var newDims = (int[])_dims.Clone();
        newDims[axis] = newSize;
        return new Shape(newDims);
    }

    /// <inheritdoc />
    public bool Equals(Shape? other)
    {
        if (other is null) return false;
        if (Rank != other.Rank) return false;
        for (var i = 0; i < Rank; i++)
        {
            if (_dims[i] != other._dims[i]) return false;
        }

        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Shape s && Equals(s);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var d in _dims)
        {
            hash.Add(d);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString() => $"[{string.Join(", ", _dims)}]";

    /// <summary>Equality operator.</summary>
    public static bool operator ==(Shape? left, Shape? right) =>
        left is null ? right is null : left.Equals(right);

    /// <summary>Inequality operator.</summary>
    public static bool operator !=(Shape? left, Shape? right) => !(left == right);
}
