using DotNetFoundryLLM.Abstractions;

namespace DotNetFoundryLLM.ChatTemplates;

/// <summary>
/// A registry of <see cref="IChatTemplate"/> implementations, looked up by model family name.
/// <para>
/// Built-in families (case-insensitive):
/// <list type="table">
///   <item><term>llama3</term><description><see cref="Llama3ChatTemplate"/></description></item>
///   <item><term>mistral</term><description><see cref="MistralChatTemplate"/></description></item>
///   <item><term>phi3</term><description><see cref="Phi3ChatTemplate"/></description></item>
///   <item><term>qwen2</term><description><see cref="Qwen2ChatTemplate"/></description></item>
///   <item><term>gemma2</term><description><see cref="Gemma2ChatTemplate"/></description></item>
/// </list>
/// Additional templates can be registered at runtime via <see cref="Register"/>.
/// </para>
/// </summary>
public sealed class ChatTemplateRegistry
{
    private readonly Dictionary<string, IChatTemplate> _templates =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Initializes a new registry pre-populated with all built-in templates.
    /// </summary>
    public ChatTemplateRegistry()
    {
        Register(new Llama3ChatTemplate());
        Register(new MistralChatTemplate());
        Register(new Phi3ChatTemplate());
        Register(new Qwen2ChatTemplate());
        Register(new Gemma2ChatTemplate());
    }

    /// <summary>
    /// Adds or replaces a template in the registry.
    /// The template is indexed by <see cref="IChatTemplate.ModelFamily"/> (case-insensitive).
    /// </summary>
    public void Register(IChatTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        _templates[template.ModelFamily] = template;
    }

    /// <summary>
    /// Tries to find a template for the given model family name.
    /// </summary>
    /// <param name="modelFamily">Model family name, e.g. <c>"llama3"</c>, <c>"mistral"</c>.</param>
    /// <param name="template">The matching template, or <see langword="null"/> if not found.</param>
    /// <returns><see langword="true"/> if a matching template was found.</returns>
    public bool TryGet(string modelFamily, out IChatTemplate? template)
    {
        ArgumentNullException.ThrowIfNull(modelFamily);
        return _templates.TryGetValue(modelFamily, out template);
    }

    /// <summary>
    /// Returns the template for the given model family, or throws if it is not registered.
    /// </summary>
    /// <exception cref="KeyNotFoundException">Thrown when no template matches <paramref name="modelFamily"/>.</exception>
    public IChatTemplate Get(string modelFamily)
    {
        if (!TryGet(modelFamily, out var template))
        {
            throw new KeyNotFoundException($"No chat template registered for model family '{modelFamily}'.");
        }

        return template!;
    }

    /// <summary>
    /// Returns the names of all registered model families.
    /// </summary>
    public IReadOnlyCollection<string> RegisteredFamilies => _templates.Keys;
}
