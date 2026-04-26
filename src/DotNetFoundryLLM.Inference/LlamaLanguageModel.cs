using System.Diagnostics;
using System.Runtime.CompilerServices;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Architectures;
using DotNetFoundryLLM.Core;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace DotNetFoundryLLM.Inference;

/// <summary>
/// A LLaMA-family language model that implements <see cref="ILanguageModel"/>.
/// Performs autoregressive token generation using a KV cache for efficiency.
/// </summary>
public sealed class LlamaLanguageModel : ILanguageModel
{
    private readonly LlamaWeights _weights;
    private readonly LlamaForwardPass _forward;
    private readonly ITokenizer _tokenizer;
    private readonly ISampler _defaultSampler;
    private readonly ILogger<LlamaLanguageModel> _logger;
    private bool _disposed;

    /// <summary>
    /// Initializes a new <see cref="LlamaLanguageModel"/>.
    /// </summary>
    /// <param name="weights">Dequantized model weights.</param>
    /// <param name="tokenizer">Tokenizer compatible with the model vocabulary.</param>
    /// <param name="defaultSampler">Sampler used when a request does not override sampling.</param>
    /// <param name="metadata">Model metadata (architecture, parameter count, etc.).</param>
    /// <param name="logger">Optional logger; defaults to a no-op logger.</param>
    public LlamaLanguageModel(
        LlamaWeights weights,
        ITokenizer tokenizer,
        ISampler defaultSampler,
        ModelMetadata metadata,
        ILogger<LlamaLanguageModel>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(defaultSampler);
        ArgumentNullException.ThrowIfNull(metadata);

        _weights        = weights;
        _tokenizer      = tokenizer;
        _defaultSampler = defaultSampler;
        Metadata        = metadata;
        _logger         = logger ?? NullLogger<LlamaLanguageModel>.Instance;
        _forward        = new LlamaForwardPass(weights);
    }

    /// <inheritdoc />
    public ModelMetadata Metadata { get; }

    /// <inheritdoc />
    public async IAsyncEnumerable<TokenStreamChunk> GenerateAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        var opts = request.Options ?? new GenerationOptions();

        _logger.LogGenerationStarted(opts.MaxTokens, opts.Temperature);
        var sw = Stopwatch.StartNew();

        // Build sampler from request options.
        var sampler = BuildSampler(opts);

        // Tokenize the prompt — copy to array so it can be used across yield boundaries.
        var promptTokens = _tokenizer.Encode(request.Prompt.AsSpan(), addBos: true, addEos: false);
        int[] promptIds  = promptTokens.ToArray();

        var cfg = _weights.Config;
        using var kvCache = new KvCache(cfg.LayerCount, cfg.MaxContextLength, cfg.KvDim);

        // Prefill: run the prompt through the model.
        int position = 0;
        for (int i = 0; i < promptIds.Length - 1; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _forward.Forward(promptIds[i], position++, kvCache);
        }

        // Start autoregressive generation from the last prompt token.
        int nextToken = promptIds.Length > 0 ? promptIds[^1] : cfg.BosTokenId;

        var stopSeqs = opts.StopSequences;
        var generated = new List<int>(opts.MaxTokens);
        var stopBuffer = new System.Text.StringBuilder();

        for (int step = 0; step < opts.MaxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Forward pass for the current token.
            _forward.Forward(nextToken, position++, kvCache);

            // Copy logits (spans cannot be captured in async iterator).
            var logits = new float[_forward.Logits.Length];
            _forward.Logits.CopyTo(logits);

            // Apply repetition penalty before sampling.
            if (opts.RepetitionPenalty != 1.0f)
            {
                ApplyRepetitionPenalty(logits, generated, opts.RepetitionPenalty);
            }

            nextToken = sampler.Sample(logits);
            generated.Add(nextToken);

            bool isEos = nextToken == cfg.EosTokenId;
            string tokenText = _tokenizer.DecodeToken(nextToken);

            // Check stop sequences.
            bool hitStop = false;
            if (stopSeqs is { Count: > 0 })
            {
                stopBuffer.Append(tokenText);
                string buffered = stopBuffer.ToString();
                foreach (var seq in stopSeqs)
                {
                    if (buffered.Contains(seq, StringComparison.Ordinal))
                    {
                        hitStop = true;
                        break;
                    }
                }
            }

            bool finished  = isEos || hitStop || step == opts.MaxTokens - 1;
            string? reason = isEos   ? "stop"    :
                             hitStop ? "stop_sequence" :
                             step == opts.MaxTokens - 1 ? "length" : null;

            var usageInfo = finished
                ? new Usage(promptIds.Length, generated.Count, promptIds.Length + generated.Count)
                : null;

            yield return new TokenStreamChunk(
                new Token(nextToken, tokenText),
                finished,
                reason,
                usageInfo);

            if (finished) break;

            await Task.Yield(); // allow cooperative cancellation in async context
        }

        _logger.LogGenerationCompleted(generated.Count, sw.ElapsedMilliseconds);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private ISampler BuildSampler(GenerationOptions opts)
    {
        // If generation options match default sampler, reuse it.
        if (opts.Temperature == 1.0f && opts.TopK == 0 && opts.TopP == 1.0f)
        {
            return _defaultSampler;
        }

        return new Sampling.SamplerPipeline(
            opts.Temperature,
            opts.TopK,
            opts.TopP,
            opts.Seed.HasValue ? (ulong)opts.Seed.Value : 0UL);
    }

    private static void ApplyRepetitionPenalty(float[] logits, List<int> generated, float penalty)
    {
        foreach (int id in generated)
        {
            if ((uint)id < (uint)logits.Length)
            {
                if (logits[id] > 0f)
                {
                    logits[id] /= penalty;
                }
                else
                {
                    logits[id] *= penalty;
                }
            }
        }
    }
}
