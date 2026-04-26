# Contributing to DotNetFoundryLLM

Thank you for your interest in contributing to DotNetFoundryLLM! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites
- .NET 10 SDK or later
- Git
- A code editor (Visual Studio, Visual Studio Code, or Rider recommended)

### Clone and Build
```bash
git clone https://github.com/justinamiller/dotnetfoundryllm.git
cd dotnetfoundryllm
dotnet build
dotnet test
```

All tests should pass before you begin making changes.

## Branch Naming

Use descriptive branch names with appropriate prefixes:

- `feature/` - New features (e.g., `feature/add-rope-scaling`)
- `fix/` - Bug fixes (e.g., `fix/tokenizer-edge-case`)
- `perf/` - Performance improvements (e.g., `perf/optimize-matmul`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)
- `test/` - Test additions or improvements (e.g., `test/add-sampling-tests`)
- `chore/` - Maintenance tasks (e.g., `chore/update-dependencies`)

## Commit Style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `perf:` - Performance improvements
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks
- `refactor:` - Code refactoring without functional changes

Example:
```
feat: add RoPE scaling support for extended context lengths
fix: handle edge case in BPE tokenizer for emoji sequences
perf: optimize matrix multiplication using vectorization
```

## Pull Request Checklist

Before submitting a pull request, ensure:

- ✅ All tests pass (`dotnet test`)
- ✅ No new warnings introduced (`dotnet build` with TreatWarningsAsErrors=true)
- ✅ XML documentation comments added to all public APIs
- ✅ No `TODO`, `FIXME`, or `HACK` markers in committed code
- ✅ Code follows existing patterns and conventions
- ✅ Nullable reference types handled correctly (nullable is enabled project-wide)
- ✅ Changes are focused and atomic (one logical change per PR)

## Coding Standards

### General Principles
- Follow the existing code style and patterns in the codebase
- Refer to `.editorconfig` for formatting rules
- All warnings are treated as errors (`TreatWarningsAsErrors` is enabled)
- Maintain nullable reference type safety throughout

### Documentation
- All public APIs must have XML documentation comments (`///`)
- Include `<summary>`, `<param>`, `<returns>`, and `<exception>` tags where applicable
- Write clear, concise descriptions
- Document non-obvious behavior and edge cases

### Testing
- Add unit tests for new functionality
- Maintain or improve code coverage
- Ensure tests are deterministic and isolated
- Use descriptive test names that explain what is being tested

## Running Benchmarks

To validate performance changes, run the relevant benchmarks:

```bash
# Tensor operations
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Tensors

# Quantization/dequantization
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Quantization

# Tokenization
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Tokenization

# Full inference forward pass
dotnet run -c Release --project bench/DotNetFoundryLLM.Bench.Inference
```

Include before/after benchmark results in your PR description for performance-related changes.

## Reporting Bugs

To report a bug:

1. Check existing [GitHub Issues](https://github.com/justinamiller/dotnetfoundryllm/issues) to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Environment details (.NET version, OS, architecture)
   - Stack trace or error messages (if applicable)

For security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Questions?

If you have questions about contributing:
- Open a [GitHub Discussion](https://github.com/justinamiller/dotnetfoundryllm/discussions)
- Review existing issues and pull requests
- Reach out to maintainers

We appreciate your contributions to making DotNetFoundryLLM better!
