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
        var byteBuffer = new List<byte>();

        foreach (var id in tokenIds)
        {
            if (id == _vocab.BosTokenId || id == _vocab.EosTokenId)
            {
                continue;
            }

            try
            {
                var tokenStr = _vocab.GetToken(id);
                foreach (var c in tokenStr)
                {
                    byteBuffer.Add(ByteEncoder.CharToByte(c));
                }
            }
            catch (ArgumentOutOfRangeException)
            {
                // Skip invalid token IDs
                continue;
            }
        }

        return Encoding.UTF8.GetString(
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(byteBuffer));
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
    /// the vocabulary's merge rank table. In each iteration the pair with the lowest rank
    /// (highest priority) is merged; the process repeats until no applicable pair remains.
    /// </summary>
    private void ApplyMerges(List<string> symbols)
    {
        if (symbols.Count < 2)
        {
            return;
        }

        int n = symbols.Count;
        var values = new string[n];
        var prev = new int[n];
        var next = new int[n];
        var alive = new bool[n];

        for (int i = 0; i < n; i++)
        {
            values[i] = symbols[i];
            prev[i] = i - 1;
            next[i] = i + 1;
            alive[i] = true;
        }

        next[n - 1] = -1;

        var pq = new PriorityQueue<MergeCandidate, int>();

        void EnqueueIfMergeable(int left)
        {
            if (left < 0 || !alive[left])
            {
                return;
            }

            int right = next[left];
            if (right < 0 || !alive[right])
            {
                return;
            }

            if (_vocab.MergeRanks.TryGetValue((values[left], values[right]), out int rank))
            {
                pq.Enqueue(new MergeCandidate(left, right, rank), rank);
            }
        }

        for (int i = 0; i < n - 1; i++)
        {
            EnqueueIfMergeable(i);
        }

        while (pq.Count > 0)
        {
            var candidate = pq.Dequeue();
            int left = candidate.Left;
            int right = candidate.Right;

            if (left < 0 || right < 0 || !alive[left] || !alive[right])
            {
                continue;
            }

            if (next[left] != right || prev[right] != left)
            {
                continue;
            }

            if (!_vocab.MergeRanks.TryGetValue((values[left], values[right]), out int currentRank) ||
                currentRank != candidate.Rank)
            {
                continue;
            }

            values[left] = values[left] + values[right];
            alive[right] = false;

            int rightNext = next[right];
            next[left] = rightNext;
            if (rightNext >= 0)
            {
                prev[rightNext] = left;
            }

            EnqueueIfMergeable(prev[left]);
            EnqueueIfMergeable(left);
        }

        symbols.Clear();
        int head = 0;
        while (head >= 0 && !alive[head])
        {
            head = next[head];
        }

        for (int i = head; i >= 0; i = next[i])
        {
            if (alive[i])
            {
                symbols.Add(values[i]);
            }
        }
    }

    private readonly record struct MergeCandidate(int Left, int Right, int Rank);
}
