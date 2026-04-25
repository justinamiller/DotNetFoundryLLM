namespace DotNetFoundryLLM.Core;

/// <summary>Base exception for all DotNetFoundryLLM domain errors.</summary>
public class FoundryException : Exception
{
    /// <summary>Initializes a new instance with a message.</summary>
    public FoundryException(string message) : base(message) { }

    /// <summary>Initializes a new instance with a message and inner exception.</summary>
    public FoundryException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Thrown when model loading fails.</summary>
public sealed class ModelLoadException : FoundryException
{
    /// <summary>Path of the model that failed to load.</summary>
    public string Path { get; }

    /// <summary>Initializes a new instance.</summary>
    public ModelLoadException(string path, string message) : base(message) => Path = path;

    /// <summary>Initializes a new instance with an inner exception.</summary>
    public ModelLoadException(string path, string message, Exception inner) : base(message, inner) => Path = path;
}

/// <summary>Thrown when a tensor operation fails due to shape mismatch or invalid input.</summary>
public sealed class TensorException : FoundryException
{
    /// <summary>Initializes a new instance.</summary>
    public TensorException(string message) : base(message) { }

    /// <summary>Initializes a new instance with an inner exception.</summary>
    public TensorException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Thrown when tokenization fails.</summary>
public sealed class TokenizationException : FoundryException
{
    /// <summary>Initializes a new instance.</summary>
    public TokenizationException(string message) : base(message) { }

    /// <summary>Initializes a new instance with an inner exception.</summary>
    public TokenizationException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Thrown when a requested feature is not supported by the current model or configuration.</summary>
public sealed class NotSupportedByModelException : FoundryException
{
    /// <summary>Initializes a new instance.</summary>
    public NotSupportedByModelException(string message) : base(message) { }
}
