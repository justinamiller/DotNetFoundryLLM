using DotNetFoundryLLM.Abstractions;
using System.Text;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// Chat template for Microsoft Phi-3 and Phi-4 models.
/// <para>
/// Format:
/// <code>
/// &lt;|system|&gt;
/// {system_content}&lt;|end|&gt;
/// &lt;|user|&gt;
/// {user_content}&lt;|end|&gt;
/// &lt;|assistant|&gt;
/// {assistant_content}&lt;|end|&gt;
/// &lt;|user|&gt;
/// ...
/// &lt;|assistant|&gt;
/// </code>
/// </para>
/// </summary>
public sealed class Phi3ChatTemplate : IChatTemplate
{
    /// <inheritdoc />
    public string ModelFamily => "phi3";

    /// <inheritdoc />
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);

        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            switch (message.Role)
            {
                case Role.System:
                    sb.Append("<|system|>\n")
                      .Append(message.Content)
                      .Append("<|end|>\n");
                    break;

                case Role.User:
                    sb.Append("<|user|>\n")
                      .Append(message.Content)
                      .Append("<|end|>\n");
                    break;

                case Role.Assistant:
                    sb.Append("<|assistant|>\n")
                      .Append(message.Content)
                      .Append("<|end|>\n");
                    break;

                case Role.Tool:
                    sb.Append("<|tool|>\n")
                      .Append(message.Content)
                      .Append("<|end|>\n");
                    break;
            }
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|assistant|>\n");
        }

        return sb.ToString();
    }
}
