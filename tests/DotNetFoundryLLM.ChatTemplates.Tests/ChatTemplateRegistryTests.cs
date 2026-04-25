using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="ChatTemplateRegistry"/>.</summary>
public sealed class ChatTemplateRegistryTests
{
    [Fact]
    public void DefaultRegistry_ContainsAllBuiltInTemplates()
    {
        var registry = new ChatTemplateRegistry();
        Assert.True(registry.TryGet("llama3", out _));
        Assert.True(registry.TryGet("mistral", out _));
        Assert.True(registry.TryGet("phi3", out _));
        Assert.True(registry.TryGet("qwen2", out _));
        Assert.True(registry.TryGet("gemma2", out _));
    }

    [Fact]
    public void TryGet_CaseInsensitive()
    {
        var registry = new ChatTemplateRegistry();
        Assert.True(registry.TryGet("LLAMA3", out _));
        Assert.True(registry.TryGet("Mistral", out _));
        Assert.True(registry.TryGet("PHI3", out _));
    }

    [Fact]
    public void TryGet_UnknownFamily_ReturnsFalse()
    {
        var registry = new ChatTemplateRegistry();
        Assert.False(registry.TryGet("nonexistent", out var template));
        Assert.Null(template);
    }

    [Fact]
    public void Get_UnknownFamily_ThrowsKeyNotFoundException()
    {
        var registry = new ChatTemplateRegistry();
        Assert.Throws<KeyNotFoundException>(() => registry.Get("nonexistent"));
    }

    [Fact]
    public void Register_AddsCustomTemplate()
    {
        var registry = new ChatTemplateRegistry();
        var custom = new FakeChatTemplate("custom-family");
        registry.Register(custom);

        Assert.True(registry.TryGet("custom-family", out var retrieved));
        Assert.Same(custom, retrieved);
    }

    [Fact]
    public void Register_OverridesExistingTemplate()
    {
        var registry = new ChatTemplateRegistry();
        var custom = new FakeChatTemplate("llama3"); // override llama3
        registry.Register(custom);

        var retrieved = registry.Get("llama3");
        Assert.IsType<FakeChatTemplate>(retrieved);
    }

    [Fact]
    public void Register_NullTemplate_Throws()
    {
        var registry = new ChatTemplateRegistry();
        Assert.Throws<ArgumentNullException>(() => registry.Register(null!));
    }

    [Fact]
    public void RegisteredFamilies_ContainsAllBuiltIns()
    {
        var registry = new ChatTemplateRegistry();
        Assert.Contains("llama3", registry.RegisteredFamilies);
        Assert.Contains("mistral", registry.RegisteredFamilies);
        Assert.Contains("phi3", registry.RegisteredFamilies);
        Assert.Contains("qwen2", registry.RegisteredFamilies);
        Assert.Contains("gemma2", registry.RegisteredFamilies);
    }

    // Minimal test double.
    private sealed class FakeChatTemplate(string family) : IChatTemplate
    {
        public string ModelFamily => family;
        public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true) => string.Empty;
    }
}
