using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="Gemma2ChatTemplate"/>.</summary>
public sealed class Gemma2ChatTemplateTests
{
    private static readonly Gemma2ChatTemplate Template = new();

    [Fact]
    public void ModelFamily_IsGemma2()
    {
        Assert.Equal("gemma2", Template.ModelFamily);
    }

    [Fact]
    public void Render_StartsWithBosToken()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages);
        Assert.StartsWith("<bos>", result);
    }

    [Fact]
    public void Render_UserMessage_UsesUserTurn()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hello") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<start_of_turn>user", result);
        Assert.Contains("Hello", result);
        Assert.Contains("<end_of_turn>", result);
    }

    [Fact]
    public void Render_AssistantMessage_UsesModelTurn()
    {
        var messages = new[] { new ChatMessage(Role.Assistant, "Answer") };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<start_of_turn>model", result);
        Assert.Contains("Answer", result);
    }

    [Fact]
    public void Render_SystemMessage_PrependedToFirstUser()
    {
        var messages = new ChatMessage[]
        {
            new(Role.System, "System prompt."),
            new(Role.User, "Hello"),
        };
        var result = Template.Render(messages, addGenerationPrompt: false);
        int systemPos = result.IndexOf("System prompt.", StringComparison.Ordinal);
        int userPos = result.IndexOf("Hello", StringComparison.Ordinal);
        Assert.True(systemPos < userPos, "System content should appear before user content.");
    }

    [Fact]
    public void Render_AddGenerationPromptTrue_EndsWithModelTurnStart()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages, addGenerationPrompt: true);
        Assert.EndsWith("<start_of_turn>model\n", result);
    }
}
