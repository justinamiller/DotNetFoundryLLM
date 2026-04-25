namespace DotNetFoundryLLM.Tokenization;

/// <summary>
/// Holds the vocabulary and BPE merge rules for a <see cref="BpeTokenizer"/>.
/// Maps token IDs to their byte-encoded string representations and vice versa,
/// and stores the ordered list of BPE merge pairs.
/// </summary>
public sealed class TokenizerVocab
{
    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly IReadOnlyDictionary<(string Left, string Right), int> _mergeRanks;

    /// <summary>
    /// Initializes a new vocabulary from the given token list and merge rules.
    /// </summary>
    /// <param name="tokens">
    /// Token strings indexed by token ID. Each string uses the GPT-2 byte encoding
    /// (see <see cref="ByteEncoder"/>), so every character represents one byte.
    /// </param>
    /// <param name="merges">
    /// Ordered list of BPE merge pairs. The index within the list is the merge rank —
    /// pairs listed earlier (rank 0, 1, 2 …) are applied before later pairs.
    /// </param>
    /// <param name="bosTokenId">ID of the beginning-of-sequence special token.</param>
    /// <param name="eosTokenId">ID of the end-of-sequence special token.</param>
    /// <param name="unknownTokenId">
    /// ID used when a symbol produced by BPE has no entry in the vocabulary.
    /// </param>
    /// <param name="addedTokens">
    /// Optional additional special tokens (e.g. chat templates, tool tokens).
    /// These are registered on top of the main vocabulary and override any
    /// entry that shares the same string.
    /// </param>
    public TokenizerVocab(
        string[] tokens,
        IReadOnlyList<(string Left, string Right)> merges,
        int bosTokenId = 1,
        int eosTokenId = 2,
        int unknownTokenId = 0,
        IReadOnlyDictionary<string, int>? addedTokens = null)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(merges);

        _idToToken = tokens;
        _tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
        {
            _tokenToId[tokens[i]] = i;
        }

        if (addedTokens is not null)
        {
            foreach (var (token, id) in addedTokens)
            {
                _tokenToId[token] = id;
            }
        }

        var mergeDict = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; i++)
        {
            mergeDict[merges[i]] = i;
        }

        _mergeRanks = mergeDict;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        UnknownTokenId = unknownTokenId;
    }

    /// <summary>Total number of tokens in the vocabulary.</summary>
    public int VocabSize => _idToToken.Length;

    /// <summary>Token ID for the beginning-of-sequence token.</summary>
    public int BosTokenId { get; }

    /// <summary>Token ID for the end-of-sequence token.</summary>
    public int EosTokenId { get; }

    /// <summary>Token ID used when a symbol cannot be found in the vocabulary.</summary>
    public int UnknownTokenId { get; }

    /// <summary>
    /// Merge rules, keyed by the pair of adjacent token strings and valued by merge rank.
    /// Lower rank means the pair is merged first.
    /// </summary>
    public IReadOnlyDictionary<(string Left, string Right), int> MergeRanks => _mergeRanks;

    /// <summary>Attempts to look up the token ID for the given token string.</summary>
    public bool TryGetId(string token, out int id) =>
        _tokenToId.TryGetValue(token, out id);

    /// <summary>Returns the byte-encoded token string for the given token ID.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown if <paramref name="id"/> is outside the vocabulary range.
    /// </exception>
    public string GetToken(int id)
    {
        if ((uint)id >= (uint)_idToToken.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(id), id,
                "Token ID is outside the vocabulary range.");
        }

        return _idToToken[id];
    }
}
