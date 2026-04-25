using DotNetFoundryLLM.Abstractions;
using System.Text;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// Chat template for Mistral and Mixtral models.
/// <para>
/// Format (system message is prepended to the first user turn):
/// <code>
/// &lt;s&gt;[INST] {system_content}
///
/// {user_content} [/INST] {assistant_content}&lt;/s&gt;[INST] {user_content} [/INST]
/// </code>
/// </para>
/// </summary>
public sealed class MistralChatTemplate : IChatTemplate
{
    /// <inheritdoc />
    public string ModelFamily => "mistral";

    /// <inheritdoc />
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();
        sb.Append("<s>");

        string? systemContent = null;
        bool firstUserSeen = false;

        foreach (var message in messages)
        {
            switch (message.Role)
            {
                case Role.System:
                    systemContent = message.Content;
                    break;

                case Role.User:
                    sb.Append("[INST] ");
                    if (!firstUserSeen && systemContent is not null)
                    {
                        sb.Append(systemContent).Append("\n\n");
                        firstUserSeen = true;
                    }

                    sb.Append(message.Content).Append(" [/INST]");
                    break;

                case Role.Assistant:
                    sb.Append(' ').Append(message.Content).Append("</s>");
                    break;

                case Role.Tool:
                    sb.Append("[TOOL_RESULTS] ").Append(message.Content).Append(" [/TOOL_RESULTS]");
                    break;
            }
        }

        // addGenerationPrompt has no explicit token in Mistral; the [/INST] at the end of
        // the last user turn already opens the assistant response slot. Nothing extra needed.
        _ = addGenerationPrompt;

        return sb.ToString();
    }
}
