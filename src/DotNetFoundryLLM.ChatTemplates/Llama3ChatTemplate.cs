using DotNetFoundryLLM.Abstractions;
using System.Text;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// Chat template for LLaMA 3.x models.
/// <para>
/// Format:
/// <code>
/// &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;
///
/// {system_content}&lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;
///
/// {user_content}&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;
///
/// {assistant_content}&lt;|eot_id|&gt;...&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;
/// </code>
/// </para>
/// </summary>
public sealed class Llama3ChatTemplate : IChatTemplate
{
    /// <inheritdoc />
    public string ModelFamily => "llama3";

    /// <inheritdoc />
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();
        sb.Append("<|begin_of_text|>");

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

            sb.Append("<|start_header_id|>")
              .Append(role)
              .Append("<|end_header_id|>")
              .Append("\n\n")
              .Append(message.Content)
              .Append("<|eot_id|>");
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return sb.ToString();
    }
}
