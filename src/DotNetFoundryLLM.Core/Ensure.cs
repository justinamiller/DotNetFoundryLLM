using System.Runtime.CompilerServices;

namespace DotNetFoundryLLM.Core;

/// <summary>Guard methods for validating arguments and state.</summary>
public static class Ensure
{
    /// <summary>Throws <see cref="ArgumentNullException"/> if <paramref name="value"/> is null.</summary>
    public static T NotNull<T>(
        T? value,
        [CallerArgumentExpression(nameof(value))] string? paramName = null) where T : class
    {
        ArgumentNullException.ThrowIfNull(value, paramName);
        return value;
    }

    /// <summary>Throws <see cref="ArgumentException"/> if <paramref name="value"/> is null or empty.</summary>
    public static string NotNullOrEmpty(
        string? value,
        [CallerArgumentExpression(nameof(value))] string? paramName = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(value, paramName);
        return value;
    }

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is negative.</summary>
    public static int NonNegative(
        int value,
        [CallerArgumentExpression(nameof(value))] string? paramName = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, paramName);
        return value;
    }

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is not positive.</summary>
    public static int Positive(
        int value,
        [CallerArgumentExpression(nameof(value))] string? paramName = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value, paramName);
        return value;
    }

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> if <paramref name="value"/> is outside [<paramref name="min"/>, <paramref name="max"/>].</summary>
    public static int InRange(
        int value,
        int min,
        int max,
        [CallerArgumentExpression(nameof(value))] string? paramName = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, min, paramName);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(value, max, paramName);
        return value;
    }

    /// <summary>Throws <see cref="InvalidOperationException"/> if <paramref name="condition"/> is false.</summary>
    public static void That(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
