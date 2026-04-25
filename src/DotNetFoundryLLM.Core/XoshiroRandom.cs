namespace DotNetFoundryLLM.Core;

/// <summary>xoshiro256** random number generator - very fast with 256-bit state.</summary>
public sealed class XoshiroRandom
{
    private ulong _s0, _s1, _s2, _s3;

    /// <summary>Initializes a new xoshiro256** generator from a 64-bit seed.</summary>
    public XoshiroRandom(ulong seed = 0)
    {
        // Use SplitMix64 to initialize state from seed
        _s0 = SplitMix64(ref seed);
        _s1 = SplitMix64(ref seed);
        _s2 = SplitMix64(ref seed);
        _s3 = SplitMix64(ref seed);
    }

    private static ulong SplitMix64(ref ulong x)
    {
        x += 0x9e3779b97f4a7c15UL;
        var z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
        return z ^ (z >> 31);
    }

    /// <summary>Returns the next uniformly distributed 64-bit unsigned integer.</summary>
    public ulong NextUInt64()
    {
        var result = RotateLeft(_s1 * 5, 7) * 9;
        var t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = RotateLeft(_s3, 45);
        return result;
    }

    private static ulong RotateLeft(ulong x, int k) => (x << k) | (x >> (64 - k));

    /// <summary>Returns a float uniformly distributed in [0, 1).</summary>
    public float NextFloat() => (float)((NextUInt64() >> 11) * (1.0 / (1UL << 53)));

    /// <summary>Returns a double uniformly distributed in [0, 1).</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>Returns an integer uniformly distributed in [0, <paramref name="maxExclusive"/>).</summary>
    public int NextInt(int maxExclusive)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxExclusive);
        return (int)(NextUInt64() % (ulong)maxExclusive);
    }
}
