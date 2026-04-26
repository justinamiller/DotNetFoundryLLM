using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Architectures;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Tensors;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace DotNetFoundryLLM.Inference;

/// <summary>
/// A LLaMA-family language model that implements <see cref="ILanguageModel"/>.
/// Performs autoregressive token generation using a KV cache for efficiency.
/// </summary>
public sealed class LlamaLanguageModel : ILanguageModel
{
    private readonly IDisposable _modelResources;
    private readonly int _layerCount;
    private readonly int _maxContextLength;
    private readonly int _kvDim;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly Func<IForwardPass> _forwardFactory;
    private readonly ITokenizer _tokenizer;
    private readonly ISampler _defaultSampler;
    private readonly ILogger<LlamaLanguageModel> _logger;
    private readonly IInferenceTelemetry _telemetry;
    private bool _disposed;

    /// <summary>
    /// Initializes a new <see cref="LlamaLanguageModel"/>.
    /// </summary>
    /// <param name="modelResources">Owned model resources to dispose with the model.</param>
    /// <param name="layerCount">Number of transformer layers.</param>
    /// <param name="maxContextLength">Maximum supported context length.</param>
    /// <param name="kvDim">Total key/value dimension.</param>
    /// <param name="bosTokenId">Beginning-of-sequence token id.</param>
    /// <param name="eosTokenId">End-of-sequence token id.</param>
    /// <param name="forwardFactory">Factory that creates a fresh forward-pass instance per request.</param>
    /// <param name="tokenizer">Tokenizer compatible with the model vocabulary.</param>
    /// <param name="defaultSampler">Sampler used when a request does not override sampling.</param>
    /// <param name="metadata">Model metadata (architecture, parameter count, etc.).</param>
    /// <param name="logger">Optional logger; defaults to a no-op logger.</param>
    /// <param name="telemetry">Optional telemetry recorder; defaults to <see cref="NullInferenceTelemetry"/>.</param>
    public LlamaLanguageModel(
        IDisposable modelResources,
        int layerCount,
        int maxContextLength,
        int kvDim,
        int bosTokenId,
        int eosTokenId,
        Func<IForwardPass> forwardFactory,
        ITokenizer tokenizer,
        ISampler defaultSampler,
        ModelMetadata metadata,
        ILogger<LlamaLanguageModel>? logger = null,
        IInferenceTelemetry? telemetry = null)
    {
        ArgumentNullException.ThrowIfNull(modelResources);
        ArgumentNullException.ThrowIfNull(forwardFactory);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(defaultSampler);
        ArgumentNullException.ThrowIfNull(metadata);

        _modelResources   = modelResources;
        _layerCount       = layerCount;
        _maxContextLength = maxContextLength;
        _kvDim            = kvDim;
        _bosTokenId       = bosTokenId;
        _eosTokenId       = eosTokenId;
        _forwardFactory   = forwardFactory;
        _tokenizer        = tokenizer;
        _defaultSampler   = defaultSampler;
        Metadata          = metadata;
        _logger           = logger ?? NullLogger<LlamaLanguageModel>.Instance;
        _telemetry        = telemetry ?? NullInferenceTelemetry.Instance;
    }

    /// <inheritdoc />
    public ModelMetadata Metadata { get; }

    /// <inheritdoc />
    public IAsyncEnumerable<TokenStreamChunk> GenerateAsync(
        ChatRequest request,
        IChatTemplate chatTemplate,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(chatTemplate);

        var prompt = chatTemplate.Render(request.Messages, addGenerationPrompt: true);
        return GenerateAsync(new CompletionRequest(prompt, request.Options), cancellationToken);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<TokenStreamChunk> GenerateAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        var opts = request.Options ?? new GenerationOptions();
        var forward = _forwardFactory();

        _logger.LogGenerationStarted(opts.MaxTokens, opts.Temperature);

        var sw = Stopwatch.StartNew();
        int[] promptIds = [];
        var generated = new List<int>(opts.MaxTokens);
        long prefillMs = 0;
        long ttftMs = 0;
        string finishReason = "unknown";
        List<TokenInsight>? tokenInsights = opts.ReturnLogProbs ? new List<TokenInsight>(opts.MaxTokens) : null;

        _telemetry.OnGenerationStarted(Metadata.ModelFamily);

        try
        {
            var sampler = BuildSampler(opts);

            var promptTokens = _tokenizer.Encode(request.Prompt.AsSpan(), addBos: true, addEos: false);
            promptIds = promptTokens.ToArray();

            using var kvCache = new KvCache(_layerCount, _maxContextLength, _kvDim);

            int position = 0;
            var prefillSw = Stopwatch.StartNew();
            for (int i = 0; i < promptIds.Length - 1; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                forward.Forward(promptIds[i], position++, kvCache);
            }

            prefillSw.Stop();
            prefillMs = prefillSw.ElapsedMilliseconds;

            int nextToken = promptIds.Length > 0 ? promptIds[^1] : _bosTokenId;

            var stopSeqs = opts.StopSequences;
            var stopBuffer = new StringBuilder();
            int maxStopLen = stopSeqs is { Count: > 0 }
                ? stopSeqs.Max(s => s.Length)
                : 0;

            int topLogProbsCount = Math.Clamp(opts.TopLogProbsCount, 0, 20);

            for (int step = 0; step < opts.MaxTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                forward.Forward(nextToken, position++, kvCache);

                var logits = new float[forward.Logits.Length];
                forward.Logits.CopyTo(logits);

                if (opts.RepetitionPenalty != 1.0f)
                {
                    ApplyRepetitionPenalty(logits, generated, opts.RepetitionPenalty);
                }

                float logProb = 0f;
                IReadOnlyList<LogProbEntry>? topAlternatives = null;

                if (opts.ReturnLogProbs)
                {
                    float[] rawLogits = ArrayPool<float>.Shared.Rent(logits.Length);
                    try
                    {
                        var rawLogitsSpan = rawLogits.AsSpan(0, logits.Length);
                        logits.CopyTo(rawLogitsSpan);

                        nextToken = sampler.Sample(logits);

                        TensorOperations.Softmax(rawLogitsSpan);
                        logProb = MathF.Log(MathF.Max(rawLogitsSpan[nextToken], 1e-10f));

                        if (topLogProbsCount > 0)
                        {
                            topAlternatives = BuildTopAlternatives(rawLogitsSpan, topLogProbsCount);
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(rawLogits);
                    }
                }
                else
                {
                    nextToken = sampler.Sample(logits);
                }

                if (generated.Count == 0)
                {
                    ttftMs = sw.ElapsedMilliseconds;
                }

                generated.Add(nextToken);

                bool isEos = nextToken == _eosTokenId;
                string tokenText = _tokenizer.DecodeToken(nextToken);

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

                    if (maxStopLen > 0 && stopBuffer.Length > maxStopLen * 2)
                    {
                        stopBuffer.Remove(0, stopBuffer.Length - maxStopLen);
                    }
                }

                bool finished = isEos || hitStop || step == opts.MaxTokens - 1;
                string? reason = isEos ? "stop"
                    : hitStop ? "stop_sequence"
                    : step == opts.MaxTokens - 1 ? "length"
                    : null;

                var usageInfo = finished
                    ? new Usage(promptIds.Length, generated.Count, promptIds.Length + generated.Count)
                    : null;

                long elapsedDecodeMs = Math.Max(0, sw.ElapsedMilliseconds - prefillMs);

                if (opts.ReturnLogProbs)
                {
                    tokenInsights?.Add(new TokenInsight(
                        step,
                        nextToken,
                        tokenText,
                        logProb,
                        elapsedDecodeMs,
                        topAlternatives));
                }

                yield return new TokenStreamChunk(
                    new Token(nextToken, tokenText, logProb),
                    finished,
                    reason,
                    usageInfo,
                    opts.ReturnLogProbs
                        ? new TokenBreakdown(step, elapsedDecodeMs, topAlternatives)
                        : null);

                if (finished)
                {
                    finishReason = reason ?? "stop";
                    break;
                }

                await Task.Yield();
            }

            if (finishReason == "unknown")
            {
                finishReason = opts.MaxTokens <= 0 ? "length" : "stop";
            }
        }
        finally
        {
            long totalMs = sw.ElapsedMilliseconds;
            long decodeMs = Math.Max(0, totalMs - prefillMs);
            double tps = decodeMs > 0 ? generated.Count / (decodeMs / 1000.0) : 0.0;

            var telemetry = new GenerationTelemetry
            {
                ModelFamily = Metadata.ModelFamily,
                PromptTokenCount = promptIds.Length,
                CompletionTokenCount = generated.Count,
                QueueTimeMs = 0,
                PrefillTimeMs = prefillMs,
                TimeToFirstTokenMs = ttftMs,
                DecodeTimeMs = decodeMs,
                TotalLatencyMs = totalMs,
                TokensPerSecond = tps,
                FinishReason = finishReason,
                TokenBreakdown = tokenInsights
            };

            _telemetry.OnGenerationCompleted(telemetry);
            _logger.LogGenerationTelemetry(
                promptIds.Length,
                generated.Count,
                ttftMs,
                decodeMs,
                tps,
                finishReason);
            _logger.LogGenerationCompleted(generated.Count, totalMs);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _modelResources.Dispose();
        }
    }

    private ISampler BuildSampler(GenerationOptions opts)
    {
        if (opts.Temperature == 1.0f && opts.TopK == 0 && opts.TopP == 1.0f && !opts.Seed.HasValue)
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

    private LogProbEntry[] BuildTopAlternatives(ReadOnlySpan<float> probabilities, int topK)
    {
        int vocabSize = probabilities.Length;
        int k = Math.Min(topK, vocabSize);
        if (k <= 0)
        {
            return [];
        }

        int[] indices = ArrayPool<int>.Shared.Rent(vocabSize);
        try
        {
            for (int i = 0; i < vocabSize; i++)
            {
                indices[i] = i;
            }

            PartialSelectTopK(indices, probabilities, k);
            SortTopKDescending(indices, probabilities, k);

            var entries = new LogProbEntry[k];
            for (int i = 0; i < k; i++)
            {
                int tokenId = indices[i];
                float prob = MathF.Max(probabilities[tokenId], 1e-10f);
                entries[i] = new LogProbEntry(tokenId, _tokenizer.DecodeToken(tokenId), MathF.Log(prob));
            }

            return entries;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(indices);
        }
    }

    private static void PartialSelectTopK(int[] indices, ReadOnlySpan<float> probs, int k)
    {
        int left = 0;
        int right = indices.Length - 1;
        int target = k - 1;

        while (left < right)
        {
            int pivotIndex = PartitionDescending(indices, probs, left, right);
            if (pivotIndex == target)
            {
                break;
            }

            if (pivotIndex < target)
            {
                left = pivotIndex + 1;
            }
            else
            {
                right = pivotIndex - 1;
            }
        }
    }

    private static int PartitionDescending(int[] indices, ReadOnlySpan<float> probs, int left, int right)
    {
        int mid = left + ((right - left) / 2);
        if (probs[indices[left]] < probs[indices[mid]])
        {
            (indices[left], indices[mid]) = (indices[mid], indices[left]);
        }

        if (probs[indices[left]] < probs[indices[right]])
        {
            (indices[left], indices[right]) = (indices[right], indices[left]);
        }

        if (probs[indices[mid]] < probs[indices[right]])
        {
            (indices[mid], indices[right]) = (indices[right], indices[mid]);
        }

        int pivotTokenId = indices[left];
        float pivot = probs[pivotTokenId];
        int i = left + 1;

        for (int j = left + 1; j <= right; j++)
        {
            if (probs[indices[j]] > pivot)
            {
                (indices[i], indices[j]) = (indices[j], indices[i]);
                i++;
            }
        }

        (indices[left], indices[i - 1]) = (indices[i - 1], indices[left]);
        return i - 1;
    }

    private static void SortTopKDescending(int[] indices, ReadOnlySpan<float> probs, int k)
    {
        for (int i = 0; i < k - 1; i++)
        {
            int best = i;
            for (int j = i + 1; j < k; j++)
            {
                if (probs[indices[j]] > probs[indices[best]])
                {
                    best = j;
                }
            }

            if (best != i)
            {
                (indices[i], indices[best]) = (indices[best], indices[i]);
            }
        }
    }
}
