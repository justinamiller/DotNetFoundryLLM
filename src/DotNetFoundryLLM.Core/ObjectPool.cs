using System.Collections.Concurrent;

namespace DotNetFoundryLLM.Core;

/// <summary>A thread-safe object pool to reduce allocation pressure.</summary>
/// <typeparam name="T">The pooled object type.</typeparam>
public sealed class ObjectPool<T> where T : class
{
    private readonly ConcurrentBag<T> _items = new();
    private readonly Func<T> _factory;
    private readonly Action<T>? _reset;
    private readonly int _maxSize;

    /// <summary>Initializes a new pool with the given factory and optional reset action.</summary>
    public ObjectPool(Func<T> factory, Action<T>? reset = null, int maxSize = 64)
    {
        ArgumentNullException.ThrowIfNull(factory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSize);
        _factory = factory;
        _reset = reset;
        _maxSize = maxSize;
    }

    /// <summary>Rents an object from the pool, creating a new one if necessary.</summary>
    public T Rent() => _items.TryTake(out var item) ? item : _factory();

    /// <summary>Returns an object to the pool.</summary>
    public void Return(T item)
    {
        ArgumentNullException.ThrowIfNull(item);
        if (_items.Count < _maxSize)
        {
            _reset?.Invoke(item);
            _items.Add(item);
        }
    }
}
