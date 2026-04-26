using System.Text;
using System.Text.RegularExpressions;
using DotNetFoundryLLM.Abstractions;
using DotNetFoundryLLM.Core;

namespace DotNetFoundryLLM.Tokenization;

/// <summary>
/// A Byte Pair Encoding (BPE) tokenizer compatible with GPT-2 and LLaMA-family models.
/// <para>
/// Encoding proceeds in three stages:
/// <list type="number">
///   <item><description>
///     <b>Pre-tokenization</b> — the input string is split into coarse chunks using a
///     configurable regular expression (defaults to the GPT-2 / LLaMA pattern).
///   </description></item>
///   <item><description>
///     <b>Byte encoding</b> — each chunk is converted to its UTF-8 byte sequence and
///     every byte is mapped to a unique Unicode character via <see cref="ByteEncoder"/>,
///     producing an initial list of single-character symbols.
///   </description></item>
///   <item><description>
///     <b>BPE merges</b> — adjacent symbol pairs are repeatedly merged according to the
///     <see cref="TokenizerVocab.MergeRanks"/> table (lowest rank = highest priority) until
///     no more applicable merge rules remain.
///   </description></item>
/// </list>
/// The resulting symbols are then looked up in the vocabulary to yield token IDs.
/// </para>
/// </summary>
public sealed partial class BpeTokenizer : ITokenizer
{
    private readonly TokenizerVocab _vocab;
    private readonly Regex _splitPattern;

    // Default GPT-2 / LLaMA-style pre-tokenization pattern.
    // Handles English contractions, Unicode words, numbers, punctuation, and whitespace.
    [GeneratedRegex(
        @"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.CultureInvariant)]
    private static partial Regex GetDefaultSplitPattern();

    private static readonly Regex DefaultSplitPattern = GetDefaultSplitPattern();

    /// <summary>
    /// Initializes a new BPE tokenizer with the given vocabulary and an optional split pattern.
    /// </summary>
    /// <param name="vocab">The vocabulary and merge rules to use.</param>
    /// <param name="splitPattern">
    /// A regex used to pre-tokenize text into chunks before BPE merges are applied.
    /// Pass <see langword="null"/> to use the built-in GPT-2 / LLaMA pattern.
    /// </param>
    public BpeTokenizer(TokenizerVocab vocab, Regex? splitPattern = null)
    {
        _vocab = Ensure.NotNull(vocab);
        _splitPattern = splitPattern ?? DefaultSplitPattern;
    }

    /// <inheritdoc />
    public int VocabSize => _vocab.VocabSize;

    /// <inheritdoc />
    public ReadOnlyMemory<int> Encode(ReadOnlySpan<char> text, bool addBos = true, bool addEos = false)
    {
        var result = new List<int>();

        if (addBos)
        {
            result.Add(_vocab.BosTokenId);
        }

        // Pre-tokenize and BPE-encode each chunk.
        var textStr = text.ToString();
        foreach (Match match in _splitPattern.Matches(textStr))
        {
            EncodeChunk(match.Value, result);
        }

        if (addEos)
        {
            result.Add(_vocab.EosTokenId);
        }

        return result.ToArray();
    }

    /// <inheritdoc />
    /// <remarks>
    /// BOS and EOS tokens are silently skipped so that special control tokens do not
    /// appear as literal text in the decoded output.
    /// </remarks>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder();

        foreach (var id in tokenIds)
        {
            if (id == _vocab.BosTokenId || id == _vocab.EosTokenId)
            {
                continue;
            }

            sb.Append(DecodeToken(id));
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public string DecodeToken(int tokenId)
    {
        var tokenStr = _vocab.GetToken(tokenId);

        if (tokenStr.Contains('\u2581'))
        {
            return tokenStr.Replace('\u2581', ' ');
        }

        var bytes = new byte[tokenStr.Length];
        for (int i = 0; i < tokenStr.Length; i++)
        {
            bytes[i] = ByteEncoder.CharToByte(tokenStr[i]);
        }

        return Encoding.UTF8.GetString(bytes);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Encodes a single pre-tokenized chunk by converting it to byte-encoded symbols
    /// and then applying BPE merges before appending the resulting token IDs to
    /// <paramref name="output"/>.
    /// </summary>
    private void EncodeChunk(string piece, List<int> output)
    {
        // Encode each UTF-8 byte of the chunk as a single BPE character symbol.
        var bytes = Encoding.UTF8.GetBytes(piece);
        var symbols = new List<string>(bytes.Length);
        foreach (var b in bytes)
        {
            symbols.Add(ByteEncoder.ByteToChar(b).ToString());
        }

        ApplyMerges(symbols);

        foreach (var symbol in symbols)
        {
            output.Add(_vocab.TryGetId(symbol, out var id) ? id : _vocab.UnknownTokenId);
        }
    }

    /// <summary>
    /// Iteratively merges adjacent symbol pairs in <paramref name="symbols"/> according to
    /// the vocabulary's merge rank table.  In each iteration the pair with the lowest rank
    /// (highest priority) is merged; the process repeats until no applicable pair remains.
    /// </summary>
    private void ApplyMerges(List<string> symbols)
    {
        while (symbols.Count >= 2)
        {
            int bestRank = int.MaxValue;
            int bestPos = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_vocab.MergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out var rank)
                    && rank < bestRank)
                {
                    bestRank = rank;
                    bestPos = i;
                }
            }

            if (bestPos < 0)
            {
                break;
            }

            symbols[bestPos] = symbols[bestPos] + symbols[bestPos + 1];
            symbols.RemoveAt(bestPos + 1);
        }
    }
}
