# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-26

### Added
- Dependency-free GGUF model loading with full metadata and tensor type support
- LLaMA 3, Mistral, Gemma 2, and Qwen 2 architecture implementations
- Comprehensive dequantization support: F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K
- BPE tokenizer with full vocabulary and merge support
- Chat templates for all four supported architectures (LLaMA 3, Mistral, Gemma 2, Qwen 2)
- Top-K and Top-P (nucleus) sampling with temperature and seed control
- OpenAI-compatible REST API with `/v1/completions` and `/v1/chat/completions` endpoints
- Streaming and non-streaming response support for all API endpoints
- ASP.NET Core hosting integration (DotNetFoundryLLM.Hosting)
- Telemetry and performance metrics layer (DotNetFoundryLLM.Telemetry)
- Comprehensive benchmark suite using BenchmarkDotNet for tensor operations, dequantization, tokenization, and forward pass
- Sample projects: ConsoleChat, AspNetMinimalHost, BatchCompletion, EmbeddingsDemo
- Full XML documentation on all public APIs
- .NET 10 support with AOT and trimming compatibility

### Not Included (Planned for Phase 2)
- SafeTensors model format support
- Embeddings generation API (currently returns 501 Not Implemented)
- CLI tool (placeholder only)
- GPU and hardware acceleration support

[1.0.0]: https://github.com/justinamiller/dotnetfoundryllm/releases/tag/v1.0.0
