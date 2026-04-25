using DotNetFoundryLLM.Inference;
using Xunit;

namespace DotNetFoundryLLM.Inference.Tests;

/// <summary>Tests for <see cref="KvCache"/>.</summary>
public sealed class KvCacheTests
{
    private static KvCache MakeCache(int layers = 2, int maxSeq = 8, int kvDim = 16) =>
        new(layers, maxSeq, kvDim);

    // ── Construction ─────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_InitializesCurrentLengthToZero()
    {
        using var cache = MakeCache();
        Assert.Equal(0, cache.CurrentLength);
    }

    [Fact]
    public void Constructor_NegativeLayer_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KvCache(0, 8, 16));
    }

    [Fact]
    public void Constructor_NegativeMaxSeq_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KvCache(2, 0, 16));
    }

    // ── Append ────────────────────────────────────────────────────────────────

    [Fact]
    public void Append_LastLayer_IncrementsCurrentLength()
    {
        using var cache = MakeCache(layers: 2, kvDim: 4);

        var k = new float[] { 1, 2, 3, 4 };
        var v = new float[] { 5, 6, 7, 8 };

        // Append layer 0 — length should not increment yet.
        cache.Append(0, k, v);
        Assert.Equal(0, cache.CurrentLength);

        // Append layer 1 (last) — length increments.
        cache.Append(1, k, v);
        Assert.Equal(1, cache.CurrentLength);
    }

    [Fact]
    public void Append_WrongSliceLength_Throws()
    {
        using var cache = MakeCache(kvDim: 4);
        Assert.Throws<ArgumentException>(() =>
            cache.Append(0, new float[3], new float[4]));
    }

    [Fact]
    public void Append_BeyondCapacity_Throws()
    {
        using var cache = MakeCache(layers: 1, maxSeq: 2, kvDim: 2);
        var slice = new float[] { 1f, 2f };

        // Fill to capacity.
        cache.Append(0, slice, slice);
        cache.Append(0, slice, slice);

        Assert.Throws<InvalidOperationException>(() =>
            cache.Append(0, slice, slice));
    }

    // ── GetKeys / GetValues ───────────────────────────────────────────────────

    [Fact]
    public void GetKeys_ReturnsAppendedValues()
    {
        using var cache = new KvCache(1, 4, 3);
        var key1 = new float[] { 1f, 2f, 3f };
        var val1 = new float[] { 4f, 5f, 6f };

        cache.Append(0, key1, val1);

        var keys = cache.GetKeys(0);
        Assert.Equal(3, keys.Length);
        Assert.Equal(1f, keys[0]);
        Assert.Equal(2f, keys[1]);
        Assert.Equal(3f, keys[2]);
    }

    [Fact]
    public void GetValues_ReturnsAppendedValues()
    {
        using var cache = new KvCache(1, 4, 3);
        var key1 = new float[] { 1f, 2f, 3f };
        var val1 = new float[] { 7f, 8f, 9f };

        cache.Append(0, key1, val1);

        var vals = cache.GetValues(0);
        Assert.Equal(3, vals.Length);
        Assert.Equal(7f, vals[0]);
    }

    [Fact]
    public void GetKeys_AfterTwoAppends_HasCorrectLength()
    {
        using var cache = new KvCache(1, 8, 2);
        float[] slice = [1f, 2f];

        cache.Append(0, slice, slice);
        cache.Append(0, slice, slice);

        Assert.Equal(4, cache.GetKeys(0).Length); // 2 tokens × 2 kvDim
    }

    // ── Clear ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Clear_ResetsCurrentLength()
    {
        using var cache = new KvCache(1, 4, 2);
        float[] slice = [1f, 2f];
        cache.Append(0, slice, slice);
        Assert.Equal(1, cache.CurrentLength);

        cache.Clear();
        Assert.Equal(0, cache.CurrentLength);
    }

    [Fact]
    public void Clear_AllowsReuseAfterFull()
    {
        using var cache = new KvCache(1, 1, 2);
        float[] slice = [1f, 2f];
        cache.Append(0, slice, slice);

        cache.Clear();

        // Should not throw after clear.
        cache.Append(0, slice, slice);
        Assert.Equal(1, cache.CurrentLength);
    }

    // ── Dispose ───────────────────────────────────────────────────────────────

    [Fact]
    public void Dispose_PreventsFurtherUse()
    {
        var cache = MakeCache();
        cache.Dispose();
        Assert.Throws<ObjectDisposedException>(() => cache.Append(0, new float[16], new float[16]));
    }
}
