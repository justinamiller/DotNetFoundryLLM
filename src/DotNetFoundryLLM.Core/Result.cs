namespace DotNetFoundryLLM.Core;

/// <summary>Represents the outcome of an operation that may succeed with a value or fail with an error.</summary>
/// <typeparam name="T">The success value type.</typeparam>
public readonly struct Result<T>
{
    private readonly T? _value;
    private readonly Exception? _error;

    private Result(T value)
    {
        _value = value;
        _error = null;
        IsSuccess = true;
    }

    private Result(Exception error)
    {
        _value = default;
        _error = error;
        IsSuccess = false;
    }

    /// <summary>Returns true if the result represents a successful outcome.</summary>
    public bool IsSuccess { get; }

    /// <summary>Returns true if the result represents a failure.</summary>
    public bool IsFailure => !IsSuccess;

    /// <summary>Gets the success value. Throws if the result is a failure.</summary>
    public T Value => IsSuccess ? _value! : throw new InvalidOperationException("Result is a failure.", _error);

    /// <summary>Gets the error. Throws if the result is a success.</summary>
    public Exception Error => IsFailure ? _error! : throw new InvalidOperationException("Result is a success.");

    /// <summary>Creates a successful result.</summary>
#pragma warning disable CA1000 // Static factory methods on generic types - required for monadic construction pattern
    public static Result<T> Ok(T value) => new(value);

    /// <summary>Creates a failure result.</summary>
    public static Result<T> Fail(Exception error) => new(error);

    /// <summary>Creates a failure result from a message.</summary>
    public static Result<T> Fail(string message) => new(new InvalidOperationException(message));
#pragma warning restore CA1000

    /// <summary>Maps the success value using the provided function.</summary>
    public Result<TOut> Map<TOut>(Func<T, TOut> mapper)
    {
        if (IsSuccess)
        {
            return Result<TOut>.Ok(mapper(_value!));
        }

        return Result<TOut>.Fail(_error!);
    }

    /// <summary>Returns the value or a default if failure.</summary>
    public T? GetValueOrDefault(T? defaultValue = default) => IsSuccess ? _value : defaultValue;

    /// <inheritdoc />
    public override string ToString() => IsSuccess ? $"Ok({_value})" : $"Fail({_error!.Message})";
}
