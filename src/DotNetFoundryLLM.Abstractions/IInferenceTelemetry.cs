#pragma warning disable CS1591
namespace DotNetFoundryLLM.Abstractions;

/// <summary>
/// Abstraction for recording model-load and generation telemetry events.
/// </summary>
public interface IInferenceTelemetry
{
    /// <summary>Records telemetry after a model has been loaded.</summary>
    void RecordModelLoaded(ModelLoadTelemetry telemetry);

    /// <summary>Signals that a generation request has started.</summary>
    void OnGenerationStarted(string modelFamily);

    /// <summary>Records telemetry after a generation request has completed.</summary>
    void OnGenerationCompleted(GenerationTelemetry telemetry);

    /// <summary>Records a retry event during generation or sampling.</summary>
    void RecordRetry(string modelFamily, string reason);
}
#pragma warning restore CS1591
