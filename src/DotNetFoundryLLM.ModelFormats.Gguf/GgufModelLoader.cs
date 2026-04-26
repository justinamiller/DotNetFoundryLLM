using System.Diagnostics;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Architectures;
using DotNetFoundryLLM.ChatTemplates;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Quantization;
using DotNetFoundryLLM.Sampling;
using DotNetFoundryLLM.Tokenization;

namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>
/// Loads LLaMA-family language models from GGUF files and returns an
/// <see cref="ILanguageModel"/> ready for text generation.
/// <para>
/// Supports GGUF versions 2 and 3 with F32, F16, BF16, Q4_0, and Q8_0 weight types.
/// Tokenizer vocabulary and BPE merge rules are extracted from the GGUF metadata.
/// </para>
/// </summary>
public sealed class GgufModelLoader : IModelLoader
{
    private static readonly string[] s_extensions = [".gguf"];
    private readonly IInferenceTelemetry _telemetry;
    private readonly ChatTemplateRegistry _chatTemplates = new();

    /// <summary>Initializes a new GGUF model loader.</summary>
    /// <param name="telemetry">Optional inference telemetry sink.</param>
    public GgufModelLoader(IInferenceTelemetry? telemetry = null)
    {
        _telemetry = telemetry ?? NullInferenceTelemetry.Instance;
    }

    /// <inheritdoc />
    public bool CanLoad(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var ext = Path.GetExtension(path);
        return Array.Exists(s_extensions, e => e.Equals(ext, StringComparison.OrdinalIgnoreCase));
    }

    /// <inheritdoc />
    public async Task<ILanguageModel> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        if (!File.Exists(path))
        {
            throw new ModelLoadException(path, $"GGUF file not found: {path}");
        }

        var sw = Stopwatch.StartNew();

        try
        {
            using var gguf = await GgufReader.ReadAsync(path, cancellationToken).ConfigureAwait(false);

            var model = Build(gguf);

            long fileSize = 0;
            try
            {
                fileSize = new FileInfo(path).Length;
            }
            catch
            {
            }

            _telemetry.RecordModelLoaded(new ModelLoadTelemetry(
                path,
                model.Metadata.Architecture,
                sw.ElapsedMilliseconds,
                fileSize,
                model.Metadata.ParameterCount));

            return model;
        }
        catch (ModelLoadException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new ModelLoadException(path, $"Failed to build model from GGUF: {ex.Message}", ex);
        }
    }

    // ── Private assembly logic ────────────────────────────────────────────────

    private Inference.LlamaLanguageModel Build(GgufFile gguf)
    {
        var meta = gguf.Metadata;
        string arch = GetString(meta, "general.architecture", "llama");
        return arch switch
        {
            "llama" or "llama3" or "llama4" => BuildLlama(gguf),
            "mistral" => BuildMistral(gguf),
            "gemma" or "gemma2" => BuildGemma2(gguf),
            "qwen2" => BuildQwen2(gguf),
            _ => throw new NotSupportedException(
                $"Architecture '{arch}' is not supported. Supported: llama, mistral, gemma2, qwen2.")
        };
    }

    private Inference.LlamaLanguageModel BuildLlama(GgufFile gguf)
    {
        var meta = gguf.Metadata;
        var config = BuildConfig(meta);
        var weights = LoadWeights(gguf, config);
        var tokenizer = BuildTokenizer(meta, config);

        ulong paramCount = EstimateParameterCount(config);
        var rawMetadata = meta.ToDictionary(kv => kv.Key, kv => kv.Value.AsObject() ?? (object)string.Empty);
        rawMetadata["chat_template_family"] = _chatTemplates.ForFamily(config.ModelFamily).ModelFamily;
        var modelMeta = new ModelMetadata(
            Architecture: config.Architecture,
            ModelFamily: config.ModelFamily,
            ParameterCount: paramCount,
            ContextLength: config.MaxContextLength,
            EmbeddingDimension: config.HiddenSize,
            VocabSize: config.VocabSize,
            WeightDtype: DType.F32,
            RawMetadata: rawMetadata);

        var sampler = new SamplerPipeline(temperature: 1.0f, topK: 0, topP: 1.0f, seed: 0);
        return new Inference.LlamaLanguageModel(
            weights,
            config.LayerCount,
            config.MaxContextLength,
            config.KvDim,
            config.BosTokenId,
            config.EosTokenId,
            () => new LlamaForwardPass(weights),
            tokenizer,
            sampler,
            modelMeta,
            telemetry: _telemetry);
    }

    private Inference.LlamaLanguageModel BuildMistral(GgufFile gguf)
    {
        var meta = gguf.Metadata;
        var baseConfig = BuildConfig(meta);
        var config = new MistralConfig
        {
            Architecture = "mistral",
            ModelFamily = baseConfig.ModelFamily,
            LayerCount = baseConfig.LayerCount,
            HiddenSize = baseConfig.HiddenSize,
            IntermediateSize = baseConfig.IntermediateSize,
            NumHeads = baseConfig.NumHeads,
            NumKvHeads = baseConfig.NumKvHeads,
            MaxContextLength = baseConfig.MaxContextLength,
            VocabSize = baseConfig.VocabSize,
            RopeBaseFreq = baseConfig.RopeBaseFreq,
            RopeScalingFactor = baseConfig.RopeScalingFactor,
            RopeScalingType = baseConfig.RopeScalingType,
            BosTokenId = baseConfig.BosTokenId,
            EosTokenId = baseConfig.EosTokenId,
            SlidingWindowSize = (int)GetUInt(meta, "mistral.attention.sliding_window", (ulong)int.MaxValue)
        };

        var llamaConfig = new LlamaConfig
        {
            Architecture = config.Architecture,
            ModelFamily = config.ModelFamily,
            LayerCount = config.LayerCount,
            HiddenSize = config.HiddenSize,
            IntermediateSize = config.IntermediateSize,
            NumHeads = config.NumHeads,
            NumKvHeads = config.NumKvHeads,
            MaxContextLength = config.MaxContextLength,
            VocabSize = config.VocabSize,
            RopeBaseFreq = config.RopeBaseFreq,
            RopeScalingFactor = config.RopeScalingFactor,
            RopeScalingType = config.RopeScalingType,
            BosTokenId = config.BosTokenId,
            EosTokenId = config.EosTokenId,
        };

        var weights = LoadWeights(gguf, llamaConfig);
        var tokenizer = BuildTokenizer(meta, llamaConfig);
        ulong paramCount = EstimateParameterCount(llamaConfig);
        var rawMetadata = meta.ToDictionary(kv => kv.Key, kv => kv.Value.AsObject() ?? (object)string.Empty);
        rawMetadata["chat_template_family"] = _chatTemplates.ForFamily(config.ModelFamily).ModelFamily;
        var modelMeta = new ModelMetadata(
            Architecture: config.Architecture,
            ModelFamily: config.ModelFamily,
            ParameterCount: paramCount,
            ContextLength: config.MaxContextLength,
            EmbeddingDimension: config.HiddenSize,
            VocabSize: config.VocabSize,
            WeightDtype: DType.F32,
            RawMetadata: rawMetadata);

        var sampler = new SamplerPipeline(temperature: 1.0f, topK: 0, topP: 1.0f, seed: 0);
        return new Inference.LlamaLanguageModel(
            weights,
            config.LayerCount,
            config.MaxContextLength,
            config.KvDim,
            config.BosTokenId,
            config.EosTokenId,
            () => new MistralForwardPass(weights, config),
            tokenizer,
            sampler,
            modelMeta,
            telemetry: _telemetry);
    }

    private Inference.LlamaLanguageModel BuildGemma2(GgufFile gguf)
    {
        var meta = gguf.Metadata;
        string arch = GetString(meta, "general.architecture", "gemma2");
        string prefix = meta.ContainsKey("gemma2.block_count") ? "gemma2" : "gemma";
        int headDimOverride = (int)GetUInt(meta, $"{prefix}.attention.key_length", 0);
        int hiddenSize = (int)GetUInt(meta, $"{prefix}.embedding_length");
        int numHeads = (int)GetUInt(meta, $"{prefix}.attention.head_count");
        int headDim = headDimOverride > 0 ? headDimOverride : hiddenSize / numHeads;

        var config = new Gemma2Config
        {
            Architecture = arch,
            ModelFamily = GetString(meta, "general.name", arch),
            LayerCount = (int)GetUInt(meta, $"{prefix}.block_count"),
            HiddenSize = hiddenSize,
            IntermediateSize = (int)GetUInt(meta, $"{prefix}.feed_forward_length"),
            NumHeads = numHeads,
            NumKvHeads = (int)GetUInt(meta, $"{prefix}.attention.head_count_kv", GetUInt(meta, $"{prefix}.attention.head_count")),
            MaxContextLength = (int)GetUInt(meta, $"{prefix}.context_length", 4096),
            VocabSize = (int)GetUInt(meta, $"{prefix}.vocab_size", GetUInt(meta, "tokenizer.ggml.tokens", 32000, arrayLength: true)),
            RopeBaseFreq = GetFloat(meta, $"{prefix}.rope.freq_base", 10000f),
            RopeScalingFactor = GetFloat(meta, $"{prefix}.rope.scaling.factor", GetFloat(meta, $"{prefix}.rope.freq_scale", 1.0f)),
            RopeScalingType = GetString(meta, $"{prefix}.rope.scaling.type", "none"),
            BosTokenId = (int)GetUInt(meta, "tokenizer.ggml.bos_token_id", 1),
            EosTokenId = (int)GetUInt(meta, "tokenizer.ggml.eos_token_id", 2),
            HeadDimOverride = headDimOverride,
            SlidingWindowSize = (int)GetUInt(meta, $"{prefix}.attention.sliding_window", (ulong)int.MaxValue),
            AttnLogitSoftcap = GetFloat(meta, $"{prefix}.attn_logit_softcapping", 0f),
            FinalLogitSoftcap = GetFloat(meta, $"{prefix}.final_logit_softcapping", 0f),
            QueryPreAttnScalar = GetFloat(meta, $"{prefix}.attention.query_pre_attn_scalar", 1.0f / MathF.Sqrt(headDim)),
        };

        var weights = LoadGemma2Weights(gguf, config);
        var tokenizer = BuildTokenizer(meta, ToLlamaConfig(config));
        ulong paramCount = EstimateParameterCount(ToLlamaConfig(config));
        var rawMetadata = meta.ToDictionary(kv => kv.Key, kv => kv.Value.AsObject() ?? (object)string.Empty);
        rawMetadata["chat_template_family"] = _chatTemplates.ForFamily(config.ModelFamily).ModelFamily;
        var modelMeta = new ModelMetadata(
            Architecture: config.Architecture,
            ModelFamily: config.ModelFamily,
            ParameterCount: paramCount,
            ContextLength: config.MaxContextLength,
            EmbeddingDimension: config.HiddenSize,
            VocabSize: config.VocabSize,
            WeightDtype: DType.F32,
            RawMetadata: rawMetadata);

        var sampler = new SamplerPipeline(temperature: 1.0f, topK: 0, topP: 1.0f, seed: 0);
        return new Inference.LlamaLanguageModel(
            weights,
            config.LayerCount,
            config.MaxContextLength,
            config.KvDim,
            config.BosTokenId,
            config.EosTokenId,
            () => new Gemma2ForwardPass(weights),
            tokenizer,
            sampler,
            modelMeta,
            telemetry: _telemetry);
    }

    private Inference.LlamaLanguageModel BuildQwen2(GgufFile gguf)
    {
        var meta = gguf.Metadata;
        var config = BuildConfig(meta);
        var weights = LoadQwen2Weights(gguf, config);
        var tokenizer = BuildTokenizer(meta, config);

        ulong paramCount = EstimateParameterCount(config);
        var rawMetadata = meta.ToDictionary(kv => kv.Key, kv => kv.Value.AsObject() ?? (object)string.Empty);
        rawMetadata["chat_template_family"] = _chatTemplates.ForFamily(config.ModelFamily).ModelFamily;
        var modelMeta = new ModelMetadata(
            Architecture: config.Architecture,
            ModelFamily: config.ModelFamily,
            ParameterCount: paramCount,
            ContextLength: config.MaxContextLength,
            EmbeddingDimension: config.HiddenSize,
            VocabSize: config.VocabSize,
            WeightDtype: DType.F32,
            RawMetadata: rawMetadata);

        var sampler = new SamplerPipeline(temperature: 1.0f, topK: 0, topP: 1.0f, seed: 0);
        return new Inference.LlamaLanguageModel(
            weights,
            config.LayerCount,
            config.MaxContextLength,
            config.KvDim,
            config.BosTokenId,
            config.EosTokenId,
            () => new Qwen2ForwardPass(weights),
            tokenizer,
            sampler,
            modelMeta,
            telemetry: _telemetry);
    }

    private static LlamaConfig BuildConfig(IReadOnlyDictionary<string, GgufMetadataValue> meta)
    {
        // Architecture key e.g. "llama" — used to prefix GGUF metadata keys.
        string arch = GetString(meta, "general.architecture", "llama");

        return new LlamaConfig
        {
            Architecture     = arch,
            ModelFamily      = GetString(meta, "general.name", arch),
            LayerCount       = (int)GetUInt(meta, $"{arch}.block_count"),
            HiddenSize       = (int)GetUInt(meta, $"{arch}.embedding_length"),
            IntermediateSize = (int)GetUInt(meta, $"{arch}.feed_forward_length"),
            NumHeads         = (int)GetUInt(meta, $"{arch}.attention.head_count"),
            NumKvHeads       = (int)GetUInt(meta, $"{arch}.attention.head_count_kv",
                                    defaultVal: GetUInt(meta, $"{arch}.attention.head_count")),
            MaxContextLength = (int)GetUInt(meta, $"{arch}.context_length", defaultVal: 4096),
            VocabSize        = (int)GetUInt(meta, $"{arch}.vocab_size",
                                    defaultVal: GetUInt(meta, "tokenizer.ggml.tokens", defaultVal: 32000,
                                        arrayLength: true)),
            RopeBaseFreq     = GetFloat(meta, $"{arch}.rope.freq_base", 10000f),
            RopeScalingFactor = GetFloat(meta, $"{arch}.rope.scaling.factor",
                                    GetFloat(meta, $"{arch}.rope.freq_scale", 1.0f)),
            RopeScalingType   = GetString(meta, $"{arch}.rope.scaling.type", "none"),
            BosTokenId       = (int)GetUInt(meta, "tokenizer.ggml.bos_token_id", 1),
            EosTokenId       = (int)GetUInt(meta, "tokenizer.ggml.eos_token_id", 2),
        };
    }

    private static LlamaWeights LoadWeights(GgufFile gguf, LlamaConfig cfg)
    {
        float[] tokenEmbed = LoadTensor(gguf, "token_embd.weight",
            (long)cfg.VocabSize * cfg.HiddenSize);

        var attnNorm = new float[cfg.LayerCount][];
        var wq       = new float[cfg.LayerCount][];
        var wk       = new float[cfg.LayerCount][];
        var wv       = new float[cfg.LayerCount][];
        var wo       = new float[cfg.LayerCount][];
        var ffnNorm  = new float[cfg.LayerCount][];
        var ffnGate  = new float[cfg.LayerCount][];
        var ffnUp    = new float[cfg.LayerCount][];
        var ffnDown  = new float[cfg.LayerCount][];

        for (int i = 0; i < cfg.LayerCount; i++)
        {
            attnNorm[i] = LoadTensor(gguf, $"blk.{i}.attn_norm.weight",        cfg.HiddenSize);
            wq[i]       = LoadTensor(gguf, $"blk.{i}.attn_q.weight",    (long)cfg.QueryDim * cfg.HiddenSize);
            wk[i]       = LoadTensor(gguf, $"blk.{i}.attn_k.weight",    (long)cfg.KvDim    * cfg.HiddenSize);
            wv[i]       = LoadTensor(gguf, $"blk.{i}.attn_v.weight",    (long)cfg.KvDim    * cfg.HiddenSize);
            wo[i]       = LoadTensor(gguf, $"blk.{i}.attn_output.weight",(long)cfg.HiddenSize * cfg.QueryDim);
            ffnNorm[i]  = LoadTensor(gguf, $"blk.{i}.ffn_norm.weight",         cfg.HiddenSize);
            ffnGate[i]  = LoadTensor(gguf, $"blk.{i}.ffn_gate.weight",  (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnUp[i]    = LoadTensor(gguf, $"blk.{i}.ffn_up.weight",    (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnDown[i]  = LoadTensor(gguf, $"blk.{i}.ffn_down.weight",  (long)cfg.HiddenSize * cfg.IntermediateSize);
        }

        float[] outputNorm   = LoadTensor(gguf, "output_norm.weight", cfg.HiddenSize);
        float[] outputWeight = LoadTensorOptional(gguf, "output.weight",
            (long)cfg.VocabSize * cfg.HiddenSize) ?? tokenEmbed;

        return new LlamaWeights(cfg, tokenEmbed,
            attnNorm, wq, wk, wv, wo,
            ffnNorm, ffnGate, ffnUp, ffnDown,
            outputNorm, outputWeight);
    }

    private static BpeTokenizer BuildTokenizer(
        IReadOnlyDictionary<string, GgufMetadataValue> meta, LlamaConfig cfg)
    {
        // Extract token strings from metadata.
        string[] tokens = GetStringArray(meta, "tokenizer.ggml.tokens");

        if (tokens.Length == 0)
        {
            throw new InvalidOperationException("GGUF file does not contain tokenizer vocabulary.");
        }

        // Extract BPE merges (optional — SentencePiece models may not have them).
        var mergeStrings = GetStringArray(meta, "tokenizer.ggml.merges");
        var merges = new List<(string, string)>(mergeStrings.Length);
        foreach (var merge in mergeStrings)
        {
            int spaceIdx = merge.IndexOf(' ', StringComparison.Ordinal);
            if (spaceIdx > 0 && spaceIdx < merge.Length - 1)
            {
                merges.Add((merge[..spaceIdx], merge[(spaceIdx + 1)..]));
            }
        }

        var vocab = new TokenizerVocab(
            tokens,
            merges,
            bosTokenId:     cfg.BosTokenId,
            eosTokenId:     cfg.EosTokenId,
            unknownTokenId: (int)GetUInt(meta, "tokenizer.ggml.unknown_token_id", 0));

        return new BpeTokenizer(vocab);
    }

    // ── Tensor helpers ────────────────────────────────────────────────────────

    private static float[] LoadTensor(GgufFile gguf, string name, long expectedElements)
    {
        var info = gguf.FindTensor(name)
            ?? throw new InvalidOperationException($"Required tensor '{name}' not found in GGUF file.");

        var raw  = gguf.GetTensorBytes(info);
        var dst  = new float[expectedElements];
        Dequantizer.Dequantize((int)info.TensorType, raw, dst);
        return dst;
    }

    private static float[]? LoadTensorOptional(GgufFile gguf, string name, long expectedElements)
    {
        var info = gguf.FindTensor(name);
        if (info is null)
        {
            return null;
        }

        var raw = gguf.GetTensorBytes(info);
        var dst = new float[expectedElements];
        Dequantizer.Dequantize((int)info.TensorType, raw, dst);
        return dst;
    }

    // ── Metadata helpers ──────────────────────────────────────────────────────

    private static string GetString(
        IReadOnlyDictionary<string, GgufMetadataValue> meta,
        string key,
        string defaultVal = "")
    {
        return meta.TryGetValue(key, out var v) && v.StringValue is { } s ? s : defaultVal;
    }

    private static ulong GetUInt(
        IReadOnlyDictionary<string, GgufMetadataValue> meta,
        string key,
        ulong defaultVal = 0,
        bool arrayLength = false)
    {
        if (!meta.TryGetValue(key, out var v))
        {
            return defaultVal;
        }

        if (arrayLength && v.ValueType == GgufValueType.Array)
        {
            return (ulong)(v.ArrayValue?.Count ?? 0);
        }

        return v.ValueType switch
        {
            GgufValueType.Uint8  => v.Uint8Value  ?? defaultVal,
            GgufValueType.Uint16 => v.Uint16Value ?? defaultVal,
            GgufValueType.Uint32 => v.Uint32Value ?? defaultVal,
            GgufValueType.Uint64 => v.Uint64Value ?? defaultVal,
            GgufValueType.Int32  => v.Int32Value is { } i ? (ulong)i : defaultVal,
            GgufValueType.Int64  => v.Int64Value  is { } l ? (ulong)l : defaultVal,
            _                    => defaultVal
        };
    }

    private static float GetFloat(
        IReadOnlyDictionary<string, GgufMetadataValue> meta,
        string key,
        float defaultVal = 0f)
    {
        return meta.TryGetValue(key, out var v) && v.Float32Value is { } f ? f : defaultVal;
    }

    private static string[] GetStringArray(
        IReadOnlyDictionary<string, GgufMetadataValue> meta,
        string key)
    {
        if (!meta.TryGetValue(key, out var v) ||
            v.ValueType != GgufValueType.Array ||
            v.ArrayValue is null)
        {
            return [];
        }

        var result = new string[v.ArrayValue.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = v.ArrayValue[i].StringValue ?? string.Empty;
        }

        return result;
    }

    private static ulong EstimateParameterCount(LlamaConfig cfg)
    {
        long p = (long)cfg.VocabSize * cfg.HiddenSize * 2; // embedding + lm_head
        p += cfg.HiddenSize;                                // output_norm
        for (int i = 0; i < cfg.LayerCount; i++)
        {
            p += cfg.HiddenSize;                                             // attn_norm
            p += (long)cfg.QueryDim * cfg.HiddenSize;                       // wq
            p += (long)cfg.KvDim   * cfg.HiddenSize * 2;                   // wk + wv
            p += (long)cfg.HiddenSize * cfg.QueryDim;                       // wo
            p += cfg.HiddenSize;                                             // ffn_norm
            p += (long)cfg.IntermediateSize * cfg.HiddenSize * 2;           // gate + up
            p += (long)cfg.HiddenSize * cfg.IntermediateSize;               // down
        }

        return (ulong)p;
    }

    private static Gemma2Weights LoadGemma2Weights(GgufFile gguf, Gemma2Config cfg)
    {
        float[] tokenEmbed = LoadTensor(gguf, "token_embd.weight", (long)cfg.VocabSize * cfg.HiddenSize);
        var attnNorm = new float[cfg.LayerCount][];
        var postAttnNorm = new float[cfg.LayerCount][];
        var wq = new float[cfg.LayerCount][];
        var wk = new float[cfg.LayerCount][];
        var wv = new float[cfg.LayerCount][];
        var wo = new float[cfg.LayerCount][];
        var ffnNorm = new float[cfg.LayerCount][];
        var postFfnNorm = new float[cfg.LayerCount][];
        var ffnGate = new float[cfg.LayerCount][];
        var ffnUp = new float[cfg.LayerCount][];
        var ffnDown = new float[cfg.LayerCount][];

        for (int i = 0; i < cfg.LayerCount; i++)
        {
            attnNorm[i] = LoadTensor(gguf, $"blk.{i}.attn_norm.weight", cfg.HiddenSize);
            postAttnNorm[i] = LoadTensor(gguf, $"blk.{i}.post_attn_norm.weight", cfg.HiddenSize);
            wq[i] = LoadTensor(gguf, $"blk.{i}.attn_q.weight", (long)cfg.QueryDim * cfg.HiddenSize);
            wk[i] = LoadTensor(gguf, $"blk.{i}.attn_k.weight", (long)cfg.KvDim * cfg.HiddenSize);
            wv[i] = LoadTensor(gguf, $"blk.{i}.attn_v.weight", (long)cfg.KvDim * cfg.HiddenSize);
            wo[i] = LoadTensor(gguf, $"blk.{i}.attn_output.weight", (long)cfg.HiddenSize * cfg.QueryDim);
            ffnNorm[i] = LoadTensor(gguf, $"blk.{i}.ffn_norm.weight", cfg.HiddenSize);
            postFfnNorm[i] = LoadTensor(gguf, $"blk.{i}.post_ffn_norm.weight", cfg.HiddenSize);
            ffnGate[i] = LoadTensor(gguf, $"blk.{i}.ffn_gate.weight", (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnUp[i] = LoadTensor(gguf, $"blk.{i}.ffn_up.weight", (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnDown[i] = LoadTensor(gguf, $"blk.{i}.ffn_down.weight", (long)cfg.HiddenSize * cfg.IntermediateSize);
        }

        float[] outputNorm = LoadTensor(gguf, "output_norm.weight", cfg.HiddenSize);
        float[] outputWeight = LoadTensorOptional(gguf, "output.weight", (long)cfg.VocabSize * cfg.HiddenSize) ?? tokenEmbed;

        return new Gemma2Weights(cfg, tokenEmbed, attnNorm, postAttnNorm, wq, wk, wv, wo, ffnNorm, postFfnNorm, ffnGate, ffnUp, ffnDown, outputNorm, outputWeight);
    }

    private static Qwen2Weights LoadQwen2Weights(GgufFile gguf, LlamaConfig cfg)
    {
        float[] tokenEmbed = LoadTensor(gguf, "token_embd.weight", (long)cfg.VocabSize * cfg.HiddenSize);

        var attnNorm = new float[cfg.LayerCount][];
        var wq = new float[cfg.LayerCount][];
        var wk = new float[cfg.LayerCount][];
        var wv = new float[cfg.LayerCount][];
        var wo = new float[cfg.LayerCount][];
        var bq = new float[cfg.LayerCount][];
        var bk = new float[cfg.LayerCount][];
        var bv = new float[cfg.LayerCount][];
        var ffnNorm = new float[cfg.LayerCount][];
        var ffnGate = new float[cfg.LayerCount][];
        var ffnUp = new float[cfg.LayerCount][];
        var ffnDown = new float[cfg.LayerCount][];

        for (int i = 0; i < cfg.LayerCount; i++)
        {
            attnNorm[i] = LoadTensor(gguf, $"blk.{i}.attn_norm.weight", cfg.HiddenSize);
            wq[i] = LoadTensor(gguf, $"blk.{i}.attn_q.weight", (long)cfg.QueryDim * cfg.HiddenSize);
            wk[i] = LoadTensor(gguf, $"blk.{i}.attn_k.weight", (long)cfg.KvDim * cfg.HiddenSize);
            wv[i] = LoadTensor(gguf, $"blk.{i}.attn_v.weight", (long)cfg.KvDim * cfg.HiddenSize);
            wo[i] = LoadTensor(gguf, $"blk.{i}.attn_output.weight", (long)cfg.HiddenSize * cfg.QueryDim);
            bq[i] = LoadTensor(gguf, $"blk.{i}.attn_q.bias", cfg.QueryDim);
            bk[i] = LoadTensor(gguf, $"blk.{i}.attn_k.bias", cfg.KvDim);
            bv[i] = LoadTensor(gguf, $"blk.{i}.attn_v.bias", cfg.KvDim);
            ffnNorm[i] = LoadTensor(gguf, $"blk.{i}.ffn_norm.weight", cfg.HiddenSize);
            ffnGate[i] = LoadTensor(gguf, $"blk.{i}.ffn_gate.weight", (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnUp[i] = LoadTensor(gguf, $"blk.{i}.ffn_up.weight", (long)cfg.IntermediateSize * cfg.HiddenSize);
            ffnDown[i] = LoadTensor(gguf, $"blk.{i}.ffn_down.weight", (long)cfg.HiddenSize * cfg.IntermediateSize);
        }

        float[] outputNorm = LoadTensor(gguf, "output_norm.weight", cfg.HiddenSize);
        float[] outputWeight = LoadTensorOptional(gguf, "output.weight", (long)cfg.VocabSize * cfg.HiddenSize) ?? tokenEmbed;

        return new Qwen2Weights(cfg, tokenEmbed, attnNorm, wq, wk, wv, wo, bq, bk, bv, ffnNorm, ffnGate, ffnUp, ffnDown, outputNorm, outputWeight);
    }

    private static LlamaConfig ToLlamaConfig(Gemma2Config cfg) => new()
    {
        Architecture = cfg.Architecture,
        ModelFamily = cfg.ModelFamily,
        LayerCount = cfg.LayerCount,
        HiddenSize = cfg.HiddenSize,
        IntermediateSize = cfg.IntermediateSize,
        NumHeads = cfg.NumHeads,
        NumKvHeads = cfg.NumKvHeads,
        MaxContextLength = cfg.MaxContextLength,
        VocabSize = cfg.VocabSize,
        RopeBaseFreq = cfg.RopeBaseFreq,
        RopeScalingFactor = cfg.RopeScalingFactor,
        RopeScalingType = cfg.RopeScalingType,
        BosTokenId = cfg.BosTokenId,
        EosTokenId = cfg.EosTokenId,
    };
}
