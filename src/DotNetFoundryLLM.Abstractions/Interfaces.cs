namespace DotNetFoundryLLM.Abstractions;

/// <summary>Loads models from storage.</summary>
public interface IModelLoader
{
    /// <summary>Returns true if this loader can handle the given path.</summary>
    bool CanLoad(string path);

    /// <summary>Loads a model from the specified path.</summary>
    Task<ILanguageModel> LoadAsync(string path, CancellationToken cancellationToken = default);
}

/// <summary>A loaded language model capable of text generation.</summary>
public interface ILanguageModel : IDisposable
{
    /// <summary>Metadata about this model.</summary>
    ModelMetadata Metadata { get; }

    /// <summary>Generates tokens for the given completion request.</summary>
    IAsyncEnumerable<TokenStreamChunk> GenerateAsync(CompletionRequest request, CancellationToken cancellationToken = default);
}

/// <summary>A loaded embedding model.</summary>
public interface IEmbeddingModel : IDisposable
{
    /// <summary>Metadata about this model.</summary>
    ModelMetadata Metadata { get; }

    /// <summary>Computes an embedding vector for the given text.</summary>
    Task<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);
}

/// <summary>Tokenizes text to/from token IDs.</summary>
public interface ITokenizer
{
    /// <summary>Total number of tokens in the vocabulary.</summary>
    int VocabSize { get; }

    /// <summary>Encodes text into token IDs.</summary>
    ReadOnlyMemory<int> Encode(ReadOnlySpan<char> text, bool addBos = true, bool addEos = false);

    /// <summary>Decodes a sequence of token IDs back to text.</summary>
    string Decode(ReadOnlySpan<int> tokenIds);

    /// <summary>Decodes a single token ID to its text representation.</summary>
    string DecodeToken(int tokenId);
}

/// <summary>Renders a chat template for a model family.</summary>
public interface IChatTemplate
{
    /// <summary>The model family this template targets.</summary>
    string ModelFamily { get; }

    /// <summary>Renders the conversation into a prompt string.</summary>
    string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true);
}

/// <summary>Samples the next token from logits.</summary>
public interface ISampler
{
    /// <summary>Samples a token index from the given logit distribution.</summary>
    int Sample(Span<float> logits);
}

/// <summary>Key-value cache for transformer attention layers.</summary>
public interface IKvCache : IDisposable
{
    /// <summary>Maximum sequence length this cache supports.</summary>
    int MaxSequenceLength { get; }

    /// <summary>Number of tokens currently stored in the cache.</summary>
    int CurrentLength { get; }

    /// <summary>Appends key and value slices for a given layer.</summary>
    void Append(int layer, ReadOnlySpan<float> keySlice, ReadOnlySpan<float> valueSlice);

    /// <summary>Returns all cached keys for the specified layer.</summary>
    ReadOnlySpan<float> GetKeys(int layer);

    /// <summary>Returns all cached values for the specified layer.</summary>
    ReadOnlySpan<float> GetValues(int layer);

    /// <summary>Clears all cached data.</summary>
    void Clear();
}

/// <summary>An N-dimensional tensor.</summary>
public interface ITensor : IDisposable
{
    /// <summary>Data type of this tensor's elements.</summary>
    DType DType { get; }

    /// <summary>Shape of this tensor.</summary>
    ReadOnlySpan<int> Shape { get; }

    /// <summary>Strides for each dimension in elements.</summary>
    ReadOnlySpan<int> Strides { get; }

    /// <summary>Number of dimensions.</summary>
    int Rank { get; }

    /// <summary>Total number of elements.</summary>
    int ElementCount { get; }
}

/// <summary>Allocates tensors from a backing store.</summary>
public interface ITensorAllocator
{
    /// <summary>Allocates a new tensor with the given data type and shape.</summary>
    ITensor Allocate(DType dtype, ReadOnlySpan<int> shape);
}
