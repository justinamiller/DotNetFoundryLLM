using DotNetFoundryLLM.Abstractions;
using System.Text;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// Chat template for Google Gemma 2 and Gemma 3 models.
/// <para>
/// Format:
/// <code>
/// &lt;bos&gt;&lt;start_of_turn&gt;user
/// {user_content}&lt;end_of_turn&gt;
/// &lt;start_of_turn&gt;model
/// {assistant_content}&lt;end_of_turn&gt;
/// &lt;start_of_turn&gt;model
/// </code>
/// System messages are prepended to the first user turn (Gemma has no dedicated system role token).
/// </para>
/// </summary>
public sealed class Gemma2ChatTemplate : IChatTemplate
{
    /// <inheritdoc />
    public string ModelFamily => "gemma2";

    /// <inheritdoc />
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();
        sb.Append("<bos>");

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
                    sb.Append("<start_of_turn>user\n");
                    if (!firstUserSeen && systemContent is not null)
                    {
                        sb.Append(systemContent).Append('\n');
                        firstUserSeen = true;
                    }

                    sb.Append(message.Content).Append("<end_of_turn>\n");
                    break;

                case Role.Assistant:
                    sb.Append("<start_of_turn>model\n")
                      .Append(message.Content)
                      .Append("<end_of_turn>\n");
                    break;

                case Role.Tool:
                    sb.Append("<start_of_turn>tool\n")
                      .Append(message.Content)
                      .Append("<end_of_turn>\n");
                    break;
            }
        }

        if (addGenerationPrompt)
        {
            sb.Append("<start_of_turn>model\n");
        }

        return sb.ToString();
    }
}
