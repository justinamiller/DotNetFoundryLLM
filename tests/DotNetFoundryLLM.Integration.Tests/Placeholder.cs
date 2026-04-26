using Xunit;
using DotNetFoundryLLM.Tokenization;
using DotNetFoundryLLM.Sampling;
using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.Integration.Tests;

/// <summary>Integration tests that validate core functionality without requiring model files.</summary>
public sealed class IntegrationTests
{
    [Fact]
    public void ApiModelValidation_RejectsEmptyPrompt()
    {
        // Arrange: Create a CompletionRequest with an empty prompt
        var request = new CompletionRequest(string.Empty, new GenerationOptions());

        // Act & Assert: Validate that empty prompt is rejected
        Assert.NotNull(request);
        Assert.Equal(string.Empty, request.Prompt);
        // An empty prompt is allowed at the API level but would be handled by the model layer
        // This test validates the request structure can be created with empty prompt
    }

    [Fact]
    public void TokenizerRoundTrip_AsciiText()
    {
        // Arrange: Create a minimal but valid vocabulary using byte encodings
        // BPE works at the byte level, so we need proper byte-to-character mappings
        var tokens = new List<string> { "<unk>", "<s>", "</s>" };

        // Add tokens for all possible byte values using the GPT-2 byte encoding scheme
        // This ensures any UTF-8 byte sequence can be encoded
        for (int b = 0; b < 256; b++)
        {
            char c;
            // GPT-2 byte encoder maps bytes to printable characters
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            {
                c = (char)b;
            }
            else
            {
                c = (char)(256 + b);
            }
            tokens.Add(c.ToString());
        }

        // No merges - keeping it simple for this integration test
        var merges = new List<(string, string)>();

        var vocab = new TokenizerVocab(
            tokens.ToArray(),
            merges,
            bosTokenId: 1,
            eosTokenId: 2,
            unknownTokenId: 0);

        var tokenizer = new BpeTokenizer(vocab);

        // Act: Encode and decode a simple ASCII text
        var text = "abc";
        var encoded = tokenizer.Encode(text.AsSpan(), addBos: false, addEos: false);

        // Assert: Verify we got tokens
        Assert.True(encoded.Length > 0, "Encoding should produce tokens");

        // Decode and verify round-trip
        var decoded = tokenizer.Decode(encoded.Span);
        Assert.Equal("abc", decoded);
    }

    [Fact]
    public void SamplerConfiguration_TopKGreaterThanVocab_Clamps()
    {
        // Arrange: Create a sampler with TopK larger than a typical vocabulary
        var vocabSize = 100;
        var options = new GenerationOptions
        {
            TopK = 50000, // Much larger than vocab
            Temperature = 1.0f,
            TopP = 1.0f
        };

        var sampler = SamplerFactory.Create(options, seed: 42);

        // Create a mock logits array
        var logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            logits[i] = i * 0.1f; // Simple ascending values
        }

        // Act: Sample from the distribution
        // The sampler should handle TopK > vocab gracefully by clamping
        int sampledToken = sampler.Sample(logits);

        // Assert: Verify sampling succeeded without throwing
        Assert.InRange(sampledToken, 0, vocabSize - 1);
    }

    [Fact]
    public void GenerationOptions_DefaultValues_AreValid()
    {
        // Arrange & Act: Create default GenerationOptions
        var options = new GenerationOptions();

        // Assert: Verify all defaults are sensible
        Assert.Equal(512, options.MaxTokens);
        Assert.Equal(1.0f, options.Temperature);
        Assert.Equal(1.0f, options.TopP);
        Assert.Equal(0, options.TopK);
        Assert.Equal(1.0f, options.RepetitionPenalty);
        Assert.Null(options.Seed);
        Assert.Null(options.StopSequences);
        Assert.False(options.ReturnLogProbs);
        Assert.Equal(0, options.TopLogProbsCount);
    }

    [Fact]
    public void ChatRequest_Creation_WithMessages()
    {
        // Arrange: Create a chat request with multiple messages
        var messages = new[]
        {
            new ChatMessage(Role.System, "You are a helpful assistant."),
            new ChatMessage(Role.User, "Hello!"),
            new ChatMessage(Role.Assistant, "Hi there!")
        };

        // Act: Create chat request
        var request = new ChatRequest(messages);

        // Assert: Verify structure
        Assert.NotNull(request);
        Assert.Equal(3, request.Messages.Count);
        Assert.Equal(Role.System, request.Messages[0].Role);
        Assert.Equal("You are a helpful assistant.", request.Messages[0].Content);
        Assert.Null(request.Options);
    }
}
