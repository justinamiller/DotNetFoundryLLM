# DotNetFoundryLLM
Building a dependency-free LLM from first principles in .NET.

## Installation

DotNetFoundryLLM requires .NET 10 SDK or later. Install the packages you need:

```bash
# Core inference engine
dotnet add package DotNetFoundryLLM.Inference

# OpenAI-compatible REST API
dotnet add package DotNetFoundryLLM.Api

# Tokenization
dotnet add package DotNetFoundryLLM.Tokenization
```

## Supported Platforms

DotNetFoundryLLM runs on CPU-only environments across multiple platforms:

- **Architectures**: x64, arm64
- **Operating Systems**: Windows, Linux, macOS
- **Note**: GPU support is not available in v1.0.0

## Known Limitations (v1.0.0)

- **SafeTensors format**: Not yet supported (planned for Phase 2)
- **Embeddings API**: Placeholder only, returns 501 Not Implemented
- **CLI tool**: Placeholder, planned for Phase 2
- **GPU / hardware acceleration**: Not supported in this release

## Performance

DotNetFoundryLLM includes comprehensive benchmarks using BenchmarkDotNet. To run benchmarks:

```bash
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.ForwardPass
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Tokenization
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Quantization
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Tensors
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started, coding standards, and the pull request process.

## Security

For security concerns and vulnerability reports, please see [SECURITY.md](SECURITY.md). Do not open public issues for security vulnerabilities.

## License

MIT — see [LICENSE](LICENSE).

