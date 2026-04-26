using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;
using Microsoft.Extensions.DependencyInjection;

namespace DotNetFoundryLLM.Telemetry;

/// <summary>Dependency injection helpers for Foundry telemetry services.</summary>
public static class TelemetryServiceExtensions
{
    /// <summary>
    /// Registers <see cref="InferenceTelemetry"/> as the singleton <see cref="IInferenceTelemetry"/>.
    /// Wire up a meter listener after this, e.g.:
    ///   services.AddOpenTelemetry().WithMetrics(b => b.AddMeter(FoundryMeter.MeterName));
    /// </summary>
    public static IServiceCollection AddFoundryTelemetry(this IServiceCollection services)
    {
        services.AddSingleton<IInferenceTelemetry, InferenceTelemetry>();
        return services;
    }

    /// <summary>Registers the no-op <see cref="NullInferenceTelemetry"/> singleton.</summary>
    public static IServiceCollection AddFoundryNullTelemetry(this IServiceCollection services)
    {
        services.AddSingleton<IInferenceTelemetry>(NullInferenceTelemetry.Instance);
        return services;
    }
}
