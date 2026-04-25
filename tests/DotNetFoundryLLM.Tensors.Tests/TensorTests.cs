using DotNetFoundryLLM.Tensors;
using Xunit;

namespace DotNetFoundryLLM.Tensors.Tests;

/// <summary>Tests for <see cref="Tensor{T}"/>.</summary>
public sealed class TensorTests
{
    [Fact]
    public void New_ZeroInitialized()
    {
        using var t = new Tensor<float>(Shape.Of(4));
        foreach (var v in t.ReadOnlySpan)
        {
            Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void From_CopiesData()
    {
        var data = new float[] { 1f, 2f, 3f };
        using var t = Tensor<float>.From(data);
        Assert.Equal(1, t.Rank);
        Assert.Equal(3L, t.Shape.ElementCount);
        Assert.Equal(data, t.ReadOnlySpan.ToArray());
    }

    [Fact]
    public void Ones_FilledWithOne()
    {
        using var t = Tensor<float>.Ones(Shape.Of(5));
        foreach (var v in t.ReadOnlySpan)
        {
            Assert.Equal(1f, v);
        }
    }

    [Fact]
    public void Indexer_ReadsAndWrites()
    {
        using var t = new Tensor<float>(Shape.Of(3));
        t[0] = 42f;
        Assert.Equal(42f, t[0]);
    }

    [Fact]
    public void Indexer2D_ReadsAndWrites()
    {
        using var t = new Tensor<float>(Shape.Of(2, 3));
        t[1, 2] = 99f;
        Assert.Equal(99f, t[1, 2]);
        Assert.Equal(99f, t[5]); // flat index for (1,2) in 2x3
    }

    [Fact]
    public void Dispose_DoubleDispose_DoesNotThrow()
    {
        var t = new Tensor<float>(Shape.Of(4));
        t.Dispose();
        t.Dispose(); // should not throw
    }

    [Fact]
    public void SpanAfterDispose_ThrowsObjectDisposedException()
    {
        var t = new Tensor<float>(Shape.Of(4));
        t.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _ = t.Span);
    }
}
