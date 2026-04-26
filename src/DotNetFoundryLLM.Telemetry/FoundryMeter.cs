using System.Diagnostics.Metrics;

namespace DotNetFoundryLLM.Telemetry;

/// <summary>Central metrics instruments emitted by the Foundry inference pipeline.</summary>
public static class FoundryMeter
{
    /// <summary>Meter name used for all emitted metrics.</summary>
    public const string MeterName = "DotNetFoundryLLM";

    private static readonly Meter s_meter = new(MeterName, "1.0.0");

    /// <summary>Model load duration histogram in milliseconds.</summary>
    public static readonly Histogram<double> ModelLoadTime =
        s_meter.CreateHistogram<double>("foundry.model.load_time", unit: "ms",
            description: "Time taken to load and initialize a model from disk.");

    /// <summary>Total request latency histogram in milliseconds.</summary>
    public static readonly Histogram<double> TotalLatency =
        s_meter.CreateHistogram<double>("foundry.inference.total_latency", unit: "ms",
            description: "Wall-clock latency for a complete generation request.");

    /// <summary>Time-to-first-token histogram in milliseconds.</summary>
    public static readonly Histogram<double> TimeToFirstToken =
        s_meter.CreateHistogram<double>("foundry.inference.time_to_first_token", unit: "ms",
            description: "Time from request start until the first output token is produced.");

    /// <summary>Prompt prefill time histogram in milliseconds.</summary>
    public static readonly Histogram<double> PrefillTime =
        s_meter.CreateHistogram<double>("foundry.inference.prefill_time", unit: "ms",
            description: "Time spent processing the input prompt (prefill phase).");

    /// <summary>Decode phase time histogram in milliseconds.</summary>
    public static readonly Histogram<double> DecodeTime =
        s_meter.CreateHistogram<double>("foundry.inference.decode_time", unit: "ms",
            description: "Time spent generating output tokens (decode phase).");

    /// <summary>Queue wait time histogram in milliseconds.</summary>
    public static readonly Histogram<double> QueueTime =
        s_meter.CreateHistogram<double>("foundry.inference.queue_time", unit: "ms",
            description: "Time the request spent waiting before processing began.");

    /// <summary>Decode throughput histogram in tokens per second.</summary>
    public static readonly Histogram<double> TokensPerSecond =
        s_meter.CreateHistogram<double>("foundry.inference.tokens_per_second", unit: "tokens/s",
            description: "Decode throughput for a completed generation request.");

    /// <summary>Prompt token count histogram.</summary>
    public static readonly Histogram<int> PromptTokenCount =
        s_meter.CreateHistogram<int>("foundry.inference.prompt_token_count", unit: "tokens",
            description: "Number of tokens in the input prompt.");

    /// <summary>Completion token count histogram.</summary>
    public static readonly Histogram<int> CompletionTokenCount =
        s_meter.CreateHistogram<int>("foundry.inference.completion_token_count", unit: "tokens",
            description: "Number of tokens generated in the response.");

    /// <summary>In-flight request up/down counter.</summary>
    public static readonly UpDownCounter<int> ActiveRequests =
        s_meter.CreateUpDownCounter<int>("foundry.inference.active_requests", unit: "requests",
            description: "Number of generation requests currently in flight.");

    /// <summary>Retry counter for generation/sampling retries.</summary>
    public static readonly Counter<long> RetryCount =
        s_meter.CreateCounter<long>("foundry.inference.retry_count", unit: "retries",
            description: "Number of sampler or generation retries.");
}
