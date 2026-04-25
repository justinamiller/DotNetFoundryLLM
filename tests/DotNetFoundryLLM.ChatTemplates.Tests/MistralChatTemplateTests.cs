using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.ChatTemplates;
using Xunit;

namespace DotNetFoundryLLM.ChatTemplates.Tests;

/// <summary>Tests for <see cref="MistralChatTemplate"/>.</summary>
public sealed class MistralChatTemplateTests
{
    private static readonly MistralChatTemplate Template = new();

    [Fact]
    public void ModelFamily_IsMistral()
    {
        Assert.Equal("mistral", Template.ModelFamily);
    }

    [Fact]
    public void Render_StartsWithBosToken()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hi") };
        var result = Template.Render(messages);
        Assert.StartsWith("<s>", result);
    }

    [Fact]
    public void Render_UserMessage_WrappedInInst()
    {
        var messages = new[] { new ChatMessage(Role.User, "Hello") };
        var result = Template.Render(messages);
        Assert.Contains("[INST]", result);
        Assert.Contains("[/INST]", result);
        Assert.Contains("Hello", result);
    }

    [Fact]
    public void Render_SystemMessage_PrependedToFirstUser()
    {
        var messages = new ChatMessage[]
        {
            new(Role.System, "Be helpful."),
            new(Role.User, "Hello"),
        };
        var result = Template.Render(messages);
        int systemPos = result.IndexOf("Be helpful.", StringComparison.Ordinal);
        int userPos = result.IndexOf("Hello", StringComparison.Ordinal);
        Assert.True(systemPos < userPos, "System content should appear before user content.");
    }

    [Fact]
    public void Render_AssistantReply_WrappedWithCloseTag()
    {
        var messages = new ChatMessage[]
        {
            new(Role.User, "Hi"),
            new(Role.Assistant, "Hello!"),
        };
        var result = Template.Render(messages, addGenerationPrompt: false);
        Assert.Contains("Hello!", result);
        Assert.Contains("</s>", result);
    }

    [Fact]
    public void Render_MultiTurn_ContainsMultipleInstPairs()
    {
        var messages = new ChatMessage[]
        {
            new(Role.User, "Turn 1"),
            new(Role.Assistant, "Answer 1"),
            new(Role.User, "Turn 2"),
        };
        var result = Template.Render(messages);
        Assert.Equal(2, CountOccurrences(result, "[INST]"));
    }

    private static int CountOccurrences(string text, string pattern)
    {
        int count = 0, idx = 0;
        while ((idx = text.IndexOf(pattern, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += pattern.Length;
        }

        return count;
    }
}
