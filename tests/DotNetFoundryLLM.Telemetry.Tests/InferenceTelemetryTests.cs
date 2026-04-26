using System.Diagnostics.Metrics;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using DotNetFoundryLLM.Telemetry;
using Xunit;

namespace DotNetFoundryLLM.Telemetry.Tests;

public sealed class InferenceTelemetryTests : IDisposable
{
    private readonly MeterListener _listener;
    private readonly Dictionary<string, List<MeasurementRecord>> _measurements = new(StringComparer.Ordinal);

    public InferenceTelemetryTests()
    {
        _listener = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == FoundryMeter.MeterName)
                {
                    listener.EnableMeasurementEvents(instrument);
                }
            }
        };

        _listener.SetMeasurementEventCallback<double>((instrument, measurement, tags, _) =>
            AddMeasurement(instrument.Name, measurement, tags));
        _listener.SetMeasurementEventCallback<int>((instrument, measurement, tags, _) =>
            AddMeasurement(instrument.Name, measurement, tags));
        _listener.SetMeasurementEventCallback<long>((instrument, measurement, tags, _) =>
            AddMeasurement(instrument.Name, measurement, tags));

        _listener.Start();
    }

    public void Dispose() => _listener.Dispose();

    private void AddMeasurement<T>(string name, T value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        if (!_measurements.TryGetValue(name, out var list))
        {
            list = [];
            _measurements[name] = list;
        }

        var copy = tags.ToArray();
        list.Add(new MeasurementRecord(value!, copy));
    }

    [Fact]
    public void NullInferenceTelemetry_Instance_IsSingleton()
    {
        Assert.Same(NullInferenceTelemetry.Instance, NullInferenceTelemetry.Instance);
    }

    [Fact]
    public void NullInferenceTelemetry_Methods_DoNotThrow()
    {
        var n = NullInferenceTelemetry.Instance;
        n.RecordModelLoaded(new ModelLoadTelemetry("path", "llama", 10, 100, 1));
        n.OnGenerationStarted("llama");
        n.OnGenerationCompleted(new GenerationTelemetry
        {
            ModelFamily = "llama",
            PromptTokenCount = 1,
            CompletionTokenCount = 1,
            PrefillTimeMs = 1,
            TimeToFirstTokenMs = 1,
            DecodeTimeMs = 1,
            TotalLatencyMs = 2,
            TokensPerSecond = 10,
            FinishReason = "stop"
        });
        n.RecordRetry("llama", "test");
    }

    [Fact]
    public void RecordModelLoaded_EmitsLoadTime_WithArchitectureTag()
    {
        var t = new InferenceTelemetry();
        t.RecordModelLoaded(new ModelLoadTelemetry("p", "llama", 123, 1000, 42));

        var ms = AssertSingle("foundry.model.load_time");
        Assert.Equal(123d, (double)ms.Value);
        AssertTag(ms.Tags, "architecture", "llama");
    }

    [Fact]
    public void OnGenerationStarted_EmitsActiveRequestsPlusOne()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationStarted("llama");

        var ms = AssertSingle("foundry.inference.active_requests");
        Assert.Equal(1, (int)ms.Value);
        AssertTag(ms.Tags, "model_family", "llama");
    }

    [Fact]
    public void OnGenerationCompleted_EmitsActiveRequestMinusOne_AndLatencyMetrics()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0));

        Assert.Equal(-1, (int)AssertSingle("foundry.inference.active_requests").Value);
        AssertSingle("foundry.inference.total_latency");
        AssertSingle("foundry.inference.time_to_first_token");
        AssertSingle("foundry.inference.prefill_time");
        AssertSingle("foundry.inference.decode_time");
        AssertSingle("foundry.inference.tokens_per_second");
    }

    [Fact]
    public void TotalLatency_HasFinishReasonTag()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0));

        var ms = AssertSingle("foundry.inference.total_latency");
        AssertTag(ms.Tags, "finish_reason", "stop");
    }

    [Fact]
    public void TimeToFirstToken_EmitsExpectedValue()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, ttft: 77));

        var ms = AssertSingle("foundry.inference.time_to_first_token");
        Assert.Equal(77d, (double)ms.Value);
    }

    [Fact]
    public void PrefillTime_EmitsExpectedValue()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, prefill: 12));

        var ms = AssertSingle("foundry.inference.prefill_time");
        Assert.Equal(12d, (double)ms.Value);
    }

    [Fact]
    public void DecodeTime_EmitsExpectedValue()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, decode: 42));

        var ms = AssertSingle("foundry.inference.decode_time");
        Assert.Equal(42d, (double)ms.Value);
    }

    [Fact]
    public void TokensPerSecond_EmitsExpectedValue()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, tps: 55.5));

        var ms = AssertSingle("foundry.inference.tokens_per_second");
        Assert.Equal(55.5d, (double)ms.Value, 5);
    }

    [Fact]
    public void PromptAndCompletionTokenCounts_AreEmitted()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, promptCount: 8, completionCount: 4));

        Assert.Equal(8, (int)AssertSingle("foundry.inference.prompt_token_count").Value);
        Assert.Equal(4, (int)AssertSingle("foundry.inference.completion_token_count").Value);
    }

    [Fact]
    public void QueueTime_NotEmitted_WhenZero()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0));

        Assert.False(_measurements.ContainsKey("foundry.inference.queue_time"));
    }

    [Fact]
    public void QueueTime_Emitted_WhenPositive()
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 9));

        var ms = AssertSingle("foundry.inference.queue_time");
        Assert.Equal(9d, (double)ms.Value);
    }

    [Fact]
    public void RecordRetry_EmitsRetryCountWithTags()
    {
        var t = new InferenceTelemetry();
        t.RecordRetry("llama", "rate_limit");

        var ms = AssertSingle("foundry.inference.retry_count");
        Assert.Equal(1L, (long)ms.Value);
        AssertTag(ms.Tags, "model_family", "llama");
        AssertTag(ms.Tags, "reason", "rate_limit");
    }

    [Fact]
    public void LogProbEntry_HasValueEquality()
    {
        var a = new LogProbEntry(1, "A", -1.25f);
        var b = new LogProbEntry(1, "A", -1.25f);

        Assert.Equal(a, b);
    }

    [Fact]
    public void TokenBreakdown_RoundTrip()
    {
        IReadOnlyList<LogProbEntry> alternatives =
        [
            new LogProbEntry(2, "B", -0.1f),
            new LogProbEntry(3, "C", -0.2f)
        ];

        var breakdown = new TokenBreakdown(7, 33, alternatives);

        Assert.Equal(7, breakdown.Position);
        Assert.Equal(33, breakdown.ElapsedMs);
        Assert.Equal(alternatives, breakdown.TopAlternatives);
    }

    [Fact]
    public void TokenInsight_RoundTrip()
    {
        IReadOnlyList<LogProbEntry> alternatives =
        [
            new LogProbEntry(2, "B", -0.1f)
        ];

        var insight = new TokenInsight(2, 10, "hello", -0.75f, 50, alternatives);

        Assert.Equal(2, insight.Position);
        Assert.Equal(10, insight.TokenId);
        Assert.Equal("hello", insight.TokenText);
        Assert.Equal(-0.75f, insight.LogProb);
        Assert.Equal(50, insight.ElapsedMs);
        Assert.Equal(alternatives, insight.TopAlternatives);
    }

    [Theory]
    [InlineData("stop")]
    [InlineData("length")]
    [InlineData("stop_sequence")]
    public void TotalLatency_FinishReasonTag_Present_ForExpectedReasons(string finishReason)
    {
        var t = new InferenceTelemetry();
        t.OnGenerationCompleted(CreateTelemetry(queueTimeMs: 0, finishReason: finishReason));

        var ms = AssertSingle("foundry.inference.total_latency");
        AssertTag(ms.Tags, "finish_reason", finishReason);
    }

    private MeasurementRecord AssertSingle(string instrumentName)
    {
        Assert.True(_measurements.TryGetValue(instrumentName, out var list), $"No measurements for {instrumentName}");
        Assert.Single(list);
        return list[0];
    }

    private static void AssertTag(IReadOnlyList<KeyValuePair<string, object?>> tags, string key, string expected)
    {
        var match = tags.FirstOrDefault(t => string.Equals(t.Key, key, StringComparison.Ordinal));
        Assert.Equal(expected, match.Value?.ToString());
    }

    private static GenerationTelemetry CreateTelemetry(
        long queueTimeMs,
        string finishReason = "stop",
        int promptCount = 5,
        int completionCount = 3,
        long prefill = 10,
        long ttft = 20,
        long decode = 30,
        long total = 40,
        double tps = 25.0)
        => new()
        {
            ModelFamily = "llama",
            PromptTokenCount = promptCount,
            CompletionTokenCount = completionCount,
            QueueTimeMs = queueTimeMs,
            PrefillTimeMs = prefill,
            TimeToFirstTokenMs = ttft,
            DecodeTimeMs = decode,
            TotalLatencyMs = total,
            TokensPerSecond = tps,
            FinishReason = finishReason
        };

    private sealed record MeasurementRecord(object Value, IReadOnlyList<KeyValuePair<string, object?>> Tags);
}
