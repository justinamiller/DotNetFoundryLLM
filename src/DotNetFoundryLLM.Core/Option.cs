namespace DotNetFoundryLLM.Core;

/// <summary>Represents an optional value - either Some(value) or None.</summary>
/// <typeparam name="T">The value type.</typeparam>
#pragma warning disable CA1716 // 'Option' conflicts with VB reserved keyword - intentional functional type name
public readonly struct Option<T>
#pragma warning restore CA1716
{
    private readonly T? _value;

    private Option(T value)
    {
        _value = value;
        HasValue = true;
    }

    /// <summary>Returns true if this option contains a value.</summary>
    public bool HasValue { get; }

    /// <summary>Returns true if this option has no value.</summary>
    public bool IsNone => !HasValue;

    /// <summary>Gets the contained value. Throws if None.</summary>
    public T Value => HasValue ? _value! : throw new InvalidOperationException("Option has no value.");

#pragma warning disable CA1000 // Static factory methods on generic types - required for monadic construction pattern
    /// <summary>Creates an option containing a value.</summary>
    public static Option<T> Some(T value) => new(value);

    /// <summary>The empty option.</summary>
    public static Option<T> None => default;
#pragma warning restore CA1000

    /// <summary>Returns the value, or <paramref name="defaultValue"/> if None.</summary>
    public T GetValueOrDefault(T defaultValue) => HasValue ? _value! : defaultValue;

    /// <summary>Maps the value using the provided function if present.</summary>
    public Option<TOut> Map<TOut>(Func<T, TOut> mapper) =>
        HasValue ? Option<TOut>.Some(mapper(_value!)) : Option<TOut>.None;

    /// <inheritdoc />
    public override string ToString() => HasValue ? $"Some({_value})" : "None";
}
