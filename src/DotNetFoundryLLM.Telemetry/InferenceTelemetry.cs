using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Telemetry;

/// <summary>Metrics-backed telemetry implementation using <see cref="FoundryMeter"/> instruments.</summary>
public sealed class InferenceTelemetry : IInferenceTelemetry
{
    /// <inheritdoc />
    public void RecordModelLoaded(ModelLoadTelemetry telemetry)
        => FoundryMeter.ModelLoadTime.Record(telemetry.LoadTimeMs,
               new KeyValuePair<string, object?>("architecture", telemetry.Architecture));

    /// <inheritdoc />
    public void OnGenerationStarted(string modelFamily)
        => FoundryMeter.ActiveRequests.Add(1,
               new KeyValuePair<string, object?>("model_family", modelFamily));

    /// <inheritdoc />
    public void OnGenerationCompleted(GenerationTelemetry telemetry)
    {
        var mfTag = new KeyValuePair<string, object?>("model_family", telemetry.ModelFamily);
        var frTag = new KeyValuePair<string, object?>("finish_reason", telemetry.FinishReason);

        FoundryMeter.ActiveRequests.Add(-1, mfTag);
        FoundryMeter.TotalLatency.Record(telemetry.TotalLatencyMs, mfTag, frTag);
        FoundryMeter.TimeToFirstToken.Record(telemetry.TimeToFirstTokenMs, mfTag);
        FoundryMeter.PrefillTime.Record(telemetry.PrefillTimeMs, mfTag);
        FoundryMeter.DecodeTime.Record(telemetry.DecodeTimeMs, mfTag);
        FoundryMeter.TokensPerSecond.Record(telemetry.TokensPerSecond, mfTag);
        FoundryMeter.PromptTokenCount.Record(telemetry.PromptTokenCount, mfTag);
        FoundryMeter.CompletionTokenCount.Record(telemetry.CompletionTokenCount, mfTag);

        if (telemetry.QueueTimeMs > 0)
        {
            FoundryMeter.QueueTime.Record(telemetry.QueueTimeMs, mfTag);
        }
    }

    /// <inheritdoc />
    public void RecordRetry(string modelFamily, string reason)
        => FoundryMeter.RetryCount.Add(1,
               new KeyValuePair<string, object?>("model_family", modelFamily),
               new KeyValuePair<string, object?>("reason", reason));
}
