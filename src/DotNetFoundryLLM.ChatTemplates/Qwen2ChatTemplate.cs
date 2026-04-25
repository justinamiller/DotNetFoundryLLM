using DotNetFoundryLLM.Abstractions;
using System.Text;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// Chat template for Qwen2 and Qwen3 models using the ChatML format.
/// <para>
/// Format:
/// <code>
/// &lt;|im_start|&gt;system
/// {system_content}&lt;|im_end|&gt;
/// &lt;|im_start|&gt;user
/// {user_content}&lt;|im_end|&gt;
/// &lt;|im_start|&gt;assistant
/// {assistant_content}&lt;|im_end|&gt;
/// ...
/// &lt;|im_start|&gt;assistant
/// </code>
/// </para>
/// </summary>
public sealed class Qwen2ChatTemplate : IChatTemplate
{
    /// <inheritdoc />
    public string ModelFamily => "qwen2";

    /// <inheritdoc />
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                Role.System => "system",
                Role.User => "user",
                Role.Assistant => "assistant",
                Role.Tool => "tool",
                _ => "user"
            };

            sb.Append("<|im_start|>")
              .Append(role)
              .Append('\n')
              .Append(message.Content)
              .Append("<|im_end|>\n");
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
