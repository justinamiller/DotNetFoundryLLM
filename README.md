# DotNetFoundryLLM

A dependency-free LLM inference engine built from first principles in .NET 10.
It loads GGUF model files directly, runs inference on CPU without any native
bindings or third-party ML libraries, and exposes an OpenAI-compatible REST API.

## Prerequisites

- .NET 10 SDK (no other runtime dependencies)

## Build and test

```sh
dotnet build DotNetFoundryLLM.slnx -c Release
dotnet test DotNetFoundryLLM.slnx -c Release
```

## Quick start — ConsoleChat

```sh
dotnet run --project samples/ConsoleChat -- /path/to/model.gguf
```

Or set the environment variable instead of passing the path on the command line:

```sh
FOUNDRY_MODEL_PATH=/path/to/model.gguf dotnet run --project samples/ConsoleChat
```

## REST API — AspNetMinimalHost

Start the server:

```sh
FOUNDRY_MODEL_PATH=/path/to/model.gguf dotnet run --project samples/AspNetMinimalHost
```

### Text completion (non-streaming)

```sh
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 128}'
```

### Text completion (streaming)

```sh
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 128, "stream": true}'
```

### Chat completion (non-streaming)

```sh
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

### Chat completion (streaming)

```sh
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": true}'
```

## Supported model architectures

- LLaMA 3
- Mistral
- Gemma 2
- Qwen 2

## Supported GGUF quantizations

| Format | Description     |
|--------|-----------------|
| F32    | 32-bit float    |
| F16    | 16-bit float    |
| BF16   | bfloat16        |
| Q4_0   | 4-bit, scheme 0 |
| Q8_0   | 8-bit, scheme 0 |
| Q4_K   | 4-bit, K-quant  |
| Q5_K   | 5-bit, K-quant  |
| Q6_K   | 6-bit, K-quant  |
| Q8_K   | 8-bit, K-quant  |

## Environment variables

| Variable             | Description                                                           |
|----------------------|-----------------------------------------------------------------------|
| `FOUNDRY_MODEL_PATH` | Absolute path to the `.gguf` model file (required for the REST API)  |

## Project layout

```
src/      Core libraries: inference engine, quantization, tokenization,
          chat templates, GGUF loader, abstractions, telemetry, sampling
tests/    Unit and integration tests for each library
samples/  Runnable samples: ConsoleChat, AspNetMinimalHost, BatchCompletion
bench/    BenchmarkDotNet projects for inference, tensors, quantization, tokenization
tools/    Developer utilities: GgufInspector, TokenizerProbe
```

## Not yet implemented (Phase 2)

- SafeTensors format support
- Embedding generation
- CLI (`dotnet-foundry` tool)
