using DotNetFoundryLLM.Tensors;
using Xunit;

namespace DotNetFoundryLLM.Tensors.Tests;

/// <summary>Tests for <see cref="TensorOperations"/>.</summary>
public sealed class TensorOperationsTests
{
    [Fact]
    public void Add_ProducesCorrectResult()
    {
        float[] a = [1f, 2f, 3f];
        float[] b = [4f, 5f, 6f];
        float[] dst = new float[3];
        TensorOperations.Add(a, b, dst);
        Assert.Equal([5f, 7f, 9f], dst);
    }

    [Fact]
    public void Multiply_ProducesCorrectResult()
    {
        float[] a = [1f, 2f, 3f];
        float[] b = [4f, 5f, 6f];
        float[] dst = new float[3];
        TensorOperations.Multiply(a, b, dst);
        Assert.Equal([4f, 10f, 18f], dst);
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        float[] x = [1f, 2f, 3f, 4f];
        TensorOperations.Softmax(x);
        var sum = x.Sum();
        Assert.True(MathF.Abs(sum - 1f) < 1e-5f, $"Expected sum ~1, got {sum}");
    }

    [Fact]
    public void Softmax_LargestInputHasLargestProbability()
    {
        float[] x = [1f, 2f, 10f, 3f];
        TensorOperations.Softmax(x);
        Assert.Equal(x[2], x.Max());
    }

    [Fact]
    public void RmsNorm_OutputIsNormalized()
    {
        float[] x = [1f, 2f, 3f, 4f];
        float[] w = [1f, 1f, 1f, 1f];
        float[] dst = new float[4];
        TensorOperations.RmsNorm(x, w, dst);
        // RMS of output should be close to 1
        var sumSq = dst.Sum(v => v * v);
        var rms = MathF.Sqrt(sumSq / dst.Length);
        Assert.True(MathF.Abs(rms - 1f) < 0.01f, $"Expected rms ~1, got {rms}");
    }

    [Fact]
    public void Silu_PositiveLargeInput_ApproachesInput()
    {
        float[] x = [100f];
        TensorOperations.Silu(x);
        // silu(large) ≈ large
        Assert.True(x[0] > 99f, $"silu(100) should be ~100, got {x[0]}");
    }

    [Fact]
    public void Dot_CorrectValue()
    {
        float[] a = [1f, 2f, 3f];
        float[] b = [4f, 5f, 6f];
        var result = TensorOperations.Dot(a, b);
        Assert.Equal(32f, result, precision: 4);
    }

    [Fact]
    public void MatMul_IdentityMatrix()
    {
        // 2x2 identity * 2x2 identity = identity
        float[] I = [1f, 0f, 0f, 1f];
        float[] dst = new float[4];
        TensorOperations.MatMul(I, I, dst, 2, 2, 2);
        Assert.Equal([1f, 0f, 0f, 1f], dst);
    }

    [Fact]
    public void ApplyCausalMask_FuturePositionsAreNegInfinity()
    {
        float[] scores = [1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f]; // 3x3
        TensorOperations.ApplyCausalMask(scores, 3);
        Assert.Equal(float.NegativeInfinity, scores[0 * 3 + 1]);
        Assert.Equal(float.NegativeInfinity, scores[0 * 3 + 2]);
        Assert.Equal(float.NegativeInfinity, scores[1 * 3 + 2]);
        // Past/current positions should be unchanged
        Assert.Equal(1f, scores[1 * 3 + 0]);
    }

    [Fact]
    public void ApplyRope_ChangesVector()
    {
        float[] x = [1f, 0f, 1f, 0f];
        var original = (float[])x.Clone();
        TensorOperations.ApplyRope(x, position: 1, headDim: 4);
        // At least one value should change when position > 0
        Assert.False(x.SequenceEqual(original), "RoPE at position=1 should change the vector.");
    }
}
