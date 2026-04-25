using DotNetFoundryLLM.Tensors;
using Xunit;

namespace DotNetFoundryLLM.Tensors.Tests;

/// <summary>Tests for <see cref="Shape"/>.</summary>
public sealed class ShapeTests
{
    [Fact]
    public void Scalar_HasRankZero()
    {
        var s = Shape.Scalar;
        Assert.Equal(0, s.Rank);
        Assert.Equal(1L, s.ElementCount);
    }

    [Fact]
    public void Of1D_CorrectRankAndCount()
    {
        var s = Shape.Of(10);
        Assert.Equal(1, s.Rank);
        Assert.Equal(10L, s.ElementCount);
    }

    [Fact]
    public void Of2D_CorrectRankAndCount()
    {
        var s = Shape.Of(4, 8);
        Assert.Equal(2, s.Rank);
        Assert.Equal(32L, s.ElementCount);
        Assert.Equal(4, s[0]);
        Assert.Equal(8, s[1]);
    }

    [Fact]
    public void Of3D_CorrectRankAndCount()
    {
        var s = Shape.Of(2, 3, 4);
        Assert.Equal(3, s.Rank);
        Assert.Equal(24L, s.ElementCount);
    }

    [Fact]
    public void Equality_SameShapes_Equal()
    {
        Assert.Equal(Shape.Of(2, 3), Shape.Of(2, 3));
    }

    [Fact]
    public void Equality_DifferentShapes_NotEqual()
    {
        Assert.NotEqual(Shape.Of(2, 3), Shape.Of(3, 2));
    }

    [Fact]
    public void RemoveAxis_RemovesCorrectDim()
    {
        var s = Shape.Of(2, 3, 4).RemoveAxis(1);
        Assert.Equal(Shape.Of(2, 4), s);
    }

    [Fact]
    public void WithDim_ReplacesCorrectDim()
    {
        var s = Shape.Of(2, 3).WithDim(1, 10);
        Assert.Equal(Shape.Of(2, 10), s);
    }

    [Fact]
    public void ToString_FormatsCorrectly()
    {
        Assert.Equal("[2, 3]", Shape.Of(2, 3).ToString());
    }

    [Fact]
    public void NegativeDim_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Shape.Of(-1));
    }
}
