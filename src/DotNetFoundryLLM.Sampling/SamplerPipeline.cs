using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Tensors;

namespace DotNetFoundryLLM.Sampling;

/// <summary>
/// A composable token sampling pipeline that applies a sequence of logit transformations
/// (temperature scaling, top-K filtering, top-P filtering) and then draws a token from
/// the resulting distribution using weighted random sampling.
/// <para>
/// Transformation order follows standard practice:
/// <list type="number">
///   <item><description>Temperature scaling — sharpens or flattens the distribution.</description></item>
///   <item><description>Top-K filtering — eliminates the long tail by keeping only the K highest logits.</description></item>
///   <item><description>Softmax — converts logits to a probability distribution.</description></item>
///   <item><description>Top-P filtering — further restricts to the smallest nucleus reaching cumulative probability p.</description></item>
///   <item><description>Categorical sampling — draws a token index proportional to its probability.</description></item>
/// </list>
/// When temperature is zero the pipeline degrades to greedy (argmax) selection.
/// </para>
/// </summary>
public sealed class SamplerPipeline : ISampler
{
    private readonly float _temperature;
    private readonly int _topK;
    private readonly float _topP;
    private readonly XoshiroRandom _random;

    /// <summary>
    /// Initializes a new sampler pipeline.
    /// </summary>
    /// <param name="temperature">
    /// Logit scaling factor. Must be ≥ 0. Zero → greedy (argmax). Default 1.0.
    /// </param>
    /// <param name="topK">
    /// Keep only the top-K logits. Zero or negative → disabled (no filtering). Default 0.
    /// </param>
    /// <param name="topP">
    /// Nucleus probability cutoff in (0, 1]. 1.0 → disabled. Default 1.0.
    /// </param>
    /// <param name="seed">
    /// Random seed. Zero → non-deterministic (uses <see cref="Environment.TickCount64"/>).
    /// </param>
    public SamplerPipeline(float temperature = 1.0f, int topK = 0, float topP = 1.0f, ulong seed = 0)
    {
        if (temperature < 0f) throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be >= 0.");
        if (topP is <= 0f or > 1f) throw new ArgumentOutOfRangeException(nameof(topP), "topP must be in (0, 1].");

        _temperature = temperature;
        _topK = topK;
        _topP = topP;
        _random = seed == 0 ? new XoshiroRandom((ulong)Environment.TickCount64) : new XoshiroRandom(seed);
    }

    /// <inheritdoc />
    /// <remarks>The logits span is modified in-place through each transformation stage.</remarks>
    public int Sample(Span<float> logits)
    {
        // Greedy path: temperature == 0 means always pick the best token.
        if (_temperature == 0f)
        {
            return TensorOperations.ArgMax(logits);
        }

        // Stage 1: Temperature scaling.
        if (_temperature != 1.0f)
        {
            TensorOperations.Scale(logits, 1.0f / _temperature, logits);
        }

        // Stage 2: Top-K filtering (operates on raw logits before softmax).
        if (_topK > 0)
        {
            TopKSampler.ApplyTopK(logits, _topK);
        }

        // Stage 3: Softmax → probability distribution.
        TensorOperations.Softmax(logits);

        // Stage 4: Top-P / nucleus filtering (operates on probabilities).
        if (_topP < 1.0f)
        {
            TopPSampler.ApplyTopP(logits, _topP);
        }

        // Stage 5: Categorical sampling.
        return TemperatureSampler.SampleFromProbabilities(logits, _random);
    }
}
