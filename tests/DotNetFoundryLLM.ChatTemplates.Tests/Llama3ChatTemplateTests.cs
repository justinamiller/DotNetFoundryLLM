using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="Llama3ChatTemplate"/>.</summary>
public sealed class Llama3ChatTemplateTests
{
    private static readonly Llama3ChatTemplate Template = new();

    [Fact]
    public void ModelFamily_IsLlama3()
    {
        Assert.Equal("llama3", Template.ModelFamily);
    }

    [Fact]
    public void Render_StartsWithBeginOfText()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages);
        Assert.StartsWith("<|begin_of_text|>", result);
    }

    [Fact]
    public void Render_UserMessage_ContainsHeaderAndContent()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hello") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>", result);
        Assert.Contains("Hello", result);
    }

    [Fact]
    public void Render_SystemMessage_IncludedBeforeUser()
    {
        var messages = new ChatMessage[]
        {
            new(Role.System, "You are helpful."),
            new(Role.User, "Hello"),
        };
        var result = Template.Render(messages, addGenerationPrompt: false);
        int systemPos = result.IndexOf("system", StringComparison.Ordinal);
        int userPos = result.IndexOf("user", StringComparison.Ordinal);
        Assert.True(systemPos < userPos, "System header should appear before user header.");
        Assert.Contains("You are helpful.", result);
    }

    [Fact]
    public void Render_AssistantMessage_HasEotId()
    {
        var messages = new ChatMessage[]
        {
            new(Role.User, "Hi"),
            new(Role.Assistant, "Hello there!"),
        };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("Hello there!", result);
        Assert.Contains("<|eot_id|>", result);
    }

    [Fact]
    public void Render_AddGenerationPromptTrue_EndsWithAssistantHeader()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: true);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", result);
    }

    [Fact]
    public void Render_AddGenerationPromptFalse_DoesNotEndWithAssistantHeader()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.False(result.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", StringComparison.Ordinal));
    }

    [Fact]
    public void Render_NullMessages_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Template.Render(null!));
    }

    [Fact]
    public void Render_EmptyMessages_ReturnsBosOnly()
    {
        var result = Template.Render([], addGenerationPrompt: false);
        Assert.Equal("<|begin_of_text|>", result);
    }
}
