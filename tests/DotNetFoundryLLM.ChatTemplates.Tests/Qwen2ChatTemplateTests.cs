using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="Qwen2ChatTemplate"/>.</summary>
public sealed class Qwen2ChatTemplateTests
{
    private static readonly Qwen2ChatTemplate Template = new();

    [Fact]
    public void ModelFamily_IsQwen2()
    {
        Assert.Equal("qwen2", Template.ModelFamily);
    }

    [Fact]
    public void Render_UserMessage_UsesChatMlTags()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hello") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|im_start|>user", result);
        Assert.Contains("Hello", result);
        Assert.Contains("<|im_end|>", result);
    }

    [Fact]
    public void Render_SystemMessage_UsesChatMlTags()
    {
        var messages = new[] { new ChatMessage(Role.System, "Be concise.") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|im_start|>system", result);
        Assert.Contains("Be concise.", result);
    }

    [Fact]
    public void Render_AssistantMessage_UsesChatMlTags()
    {
        var messages = new[] { new ChatMessage(Role.Assistant, "OK!") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|im_start|>assistant", result);
        Assert.Contains("OK!", result);
    }

    [Fact]
    public void Render_AddGenerationPromptTrue_EndsWithAssistantStart()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: true);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void Render_AddGenerationPromptFalse_DoesNotEndWithAssistantStart()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.False(result.TrimEnd().EndsWith("<|im_start|>assistant", StringComparison.Ordinal));
    }
}
