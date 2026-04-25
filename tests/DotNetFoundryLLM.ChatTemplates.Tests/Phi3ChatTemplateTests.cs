using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="Phi3ChatTemplate"/>.</summary>
public sealed class Phi3ChatTemplateTests
{
    private static readonly Phi3ChatTemplate Template = new();

    [Fact]
    public void ModelFamily_IsPhi3()
    {
        Assert.Equal("phi3", Template.ModelFamily);
    }

    [Fact]
    public void Render_UserMessage_UsesUserTag()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hello") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|user|>", result);
        Assert.Contains("Hello", result);
        Assert.Contains("<|end|>", result);
    }

    [Fact]
    public void Render_SystemMessage_UsesSystemTag()
    {
        var messages = new[] { new ChatMessage(Role.System, "Be helpful.") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|system|>", result);
        Assert.Contains("Be helpful.", result);
    }

    [Fact]
    public void Render_AssistantMessage_UsesAssistantTag()
    {
        var messages = new[] { new ChatMessage(Role.Assistant, "Sure!") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|assistant|>", result);
        Assert.Contains("Sure!", result);
    }

    [Fact]
    public void Render_AddGenerationPromptTrue_EndsWithAssistantTag()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: true);
        Assert.EndsWith("<|assistant|>\n", result);
    }

    [Fact]
    public void Render_MultiTurn_CorrectStructure()
    {
        var messages = new ChatMessage[]
        {
            new(Role.System, "You are Phi."),
            new(Role.User, "Question 1"),
            new(Role.Assistant, "Answer 1"),
            new(Role.User, "Question 2"),
        };
        var result = Template.Render(messages, addGenerationPrompt: true);
        Assert.Contains("<|system|>", result);
        Assert.Contains("<|user|>", result);
        Assert.Contains("<|assistant|>", result);
        Assert.Contains("You are Phi.", result);
        Assert.Contains("Question 1", result);
        Assert.Contains("Answer 1", result);
        Assert.Contains("Question 2", result);
    }
}
