namespace DotNetFoundryLLM.Core;

/// <summary>PCG32 random number generator - fast with excellent statistical properties.</summary>
public sealed class PcgRandom
{
    private ulong _state;
    private readonly ulong _increment;

    /// <summary>Initializes a new PCG32 generator with the given seed and stream.</summary>
    public PcgRandom(ulong seed = 42, ulong stream = 1)
    {
        _increment = (stream << 1) | 1;
        _state = 0;
        NextUInt32();
        _state += seed;
        NextUInt32();
    }

    /// <summary>Returns the next uniformly distributed 32-bit unsigned integer.</summary>
    public uint NextUInt32()
    {
        var oldState = _state;
        _state = oldState * 6364136223846793005UL + _increment;
        var xorShifted = (uint)(((oldState >> 18) ^ oldState) >> 27);
        var rot = (int)(oldState >> 59);
        return (xorShifted >> rot) | (xorShifted << ((-rot) & 31));
    }

    /// <summary>Returns a float uniformly distributed in [0, 1).</summary>
    public float NextFloat() => (NextUInt32() >> 8) * (1.0f / (1 << 24));

    /// <summary>Returns a double uniformly distributed in [0, 1).</summary>
    public double NextDouble() => NextUInt32() * (1.0 / uint.MaxValue);

    /// <summary>Returns an integer uniformly distributed in [0, <paramref name="maxExclusive"/>).</summary>
    public int NextInt(int maxExclusive)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxExclusive);
        return (int)(NextUInt32() % (uint)maxExclusive);
    }
}
