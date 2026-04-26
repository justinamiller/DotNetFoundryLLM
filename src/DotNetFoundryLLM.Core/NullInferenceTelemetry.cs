#pragma warning disable CS1591
using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Core;

/// <summary>No-op telemetry implementation used when telemetry is not configured.</summary>
public sealed class NullInferenceTelemetry : IInferenceTelemetry
{
    /// <summary>Singleton no-op telemetry instance.</summary>
    public static readonly NullInferenceTelemetry Instance = new();

    private NullInferenceTelemetry() { }

    /// <inheritdoc />
    public void RecordModelLoaded(ModelLoadTelemetry telemetry) { }

    /// <inheritdoc />
    public void OnGenerationStarted(string modelFamily) { }

    /// <inheritdoc />
    public void OnGenerationCompleted(GenerationTelemetry telemetry) { }

    /// <inheritdoc />
    public void RecordRetry(string modelFamily, string reason) { }
}
#pragma warning restore CS1591
