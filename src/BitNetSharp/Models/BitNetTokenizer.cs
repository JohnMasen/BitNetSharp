using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace BitNetSharp.Models
{
    public sealed class BitNetTokenizer
    {
        private static readonly ReadOnlyDictionary<byte, char> ByteEncoder = new(BuildByteEncoder());
        private static readonly ReadOnlyDictionary<char, byte> ByteDecoder = new(BuildByteDecoder(ByteEncoder));

        private static readonly Regex EncodePattern = new(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private readonly BitNetTokenizerConfig tokenizerConfig;
        private readonly Dictionary<string, int> tokenToId;
        private readonly Dictionary<(string Left, string Right), int> mergeRanks;

        public BitNetTokenizer(BitNetTokenizerConfig tokenizerConfig)
        {
            ArgumentNullException.ThrowIfNull(tokenizerConfig);

            if (!string.Equals(tokenizerConfig.TokenizerModelType, "gpt2", StringComparison.OrdinalIgnoreCase))
            {
                throw new NotSupportedException($"Tokenizer model '{tokenizerConfig.TokenizerModelType}' is not supported.");
            }

            this.tokenizerConfig = tokenizerConfig;
            tokenToId = new Dictionary<string, int>(tokenizerConfig.Tokens.Count, StringComparer.Ordinal);
            for (int index = 0; index < tokenizerConfig.Tokens.Count; index++)
            {
                tokenToId[tokenizerConfig.Tokens[index]] = index;
            }

            mergeRanks = new Dictionary<(string Left, string Right), int>(tokenizerConfig.Merges.Count);
            for (int index = 0; index < tokenizerConfig.Merges.Count; index++)
            {
                string merge = tokenizerConfig.Merges[index];
                int separatorIndex = merge.IndexOf(' ');
                if (separatorIndex <= 0 || separatorIndex >= merge.Length - 1)
                {
                    continue;
                }

                string left = merge[..separatorIndex];
                string right = merge[(separatorIndex + 1)..];
                mergeRanks[(left, right)] = index;
            }
        }

        /// <summary>
        /// Encodes text into token ids using the GGUF tokenizer vocabulary and merges.
        /// </summary>
        public IReadOnlyList<int> EncodeToIds(string text, bool addBos = false, bool addEos = false)
        {
            ArgumentNullException.ThrowIfNull(text);

            List<int> tokenIds = new();

            if (addBos)
            {
                tokenIds.Add(GetRequiredSpecialTokenId(tokenizerConfig.BosTokenId, "BOS"));
            }

            foreach (Match match in EncodePattern.Matches(text))
            {
                if (!match.Success || match.Length == 0)
                {
                    continue;
                }

                EncodeMatch(match.Value, tokenIds);
            }

            if (addEos)
            {
                tokenIds.Add(GetRequiredSpecialTokenId(tokenizerConfig.EosTokenId, "EOS"));
            }

            return tokenIds;
        }

        /// <summary>
        /// Decodes a sequence of token ids back into text.
        /// </summary>
        public string Decode(IEnumerable<int> tokenIds)
        {
            ArgumentNullException.ThrowIfNull(tokenIds);

            StringBuilder builder = new();
            List<byte> byteBuffer = new();

            foreach (int tokenId in tokenIds)
            {
                if ((uint)tokenId >= (uint)tokenizerConfig.Tokens.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {tokenId} is outside the vocabulary range 0..{tokenizerConfig.Tokens.Count - 1}.");
                }

                if (IsSpecialTokenId(tokenId))
                {
                    continue;
                }

                AppendMappedTokenBytes(tokenizerConfig.Tokens[tokenId], byteBuffer);
            }

            FlushByteBuffer(byteBuffer, builder);
            return builder.ToString();
        }

        private void EncodeMatch(string matchText, List<int> tokenIds)
        {
            string mappedText = MapTextToByteEncoding(matchText);
            foreach (string piece in ApplyBytePairEncoding(mappedText))
            {
                if (!tokenToId.TryGetValue(piece, out int tokenId))
                {
                    throw new InvalidOperationException($"Tokenizer vocabulary is missing encoded piece '{piece}'.");
                }

                tokenIds.Add(tokenId);
            }
        }

        private IEnumerable<string> ApplyBytePairEncoding(string mappedText)
        {
            if (string.IsNullOrEmpty(mappedText))
            {
                yield break;
            }

            List<string> symbols = mappedText.Select(character => character.ToString(CultureInfo.InvariantCulture)).ToList();
            if (symbols.Count == 1)
            {
                yield return symbols[0];
                yield break;
            }

            while (symbols.Count > 1)
            {
                int bestPairIndex = -1;
                int bestPairRank = int.MaxValue;

                for (int index = 0; index < symbols.Count - 1; index++)
                {
                    if (!mergeRanks.TryGetValue((symbols[index], symbols[index + 1]), out int rank) || rank >= bestPairRank)
                    {
                        continue;
                    }

                    bestPairRank = rank;
                    bestPairIndex = index;
                }

                if (bestPairIndex < 0)
                {
                    break;
                }

                symbols[bestPairIndex] += symbols[bestPairIndex + 1];
                symbols.RemoveAt(bestPairIndex + 1);
            }

            foreach (string symbol in symbols)
            {
                yield return symbol;
            }
        }

        private static string MapTextToByteEncoding(string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            char[] chars = new char[bytes.Length];
            for (int index = 0; index < bytes.Length; index++)
            {
                chars[index] = ByteEncoder[bytes[index]];
            }

            return new string(chars);
        }

        private static void AppendMappedTokenBytes(string tokenText, List<byte> bytes)
        {
            foreach (char character in tokenText)
            {
                if (!ByteDecoder.TryGetValue(character, out byte value))
                {
                    throw new InvalidOperationException($"Tokenizer byte decoder is missing mapping for character U+{(int)character:X4}.");
                }

                bytes.Add(value);
            }
        }

        private static void FlushByteBuffer(List<byte> bytes, StringBuilder builder)
        {
            if (bytes.Count == 0)
            {
                return;
            }

            builder.Append(Encoding.UTF8.GetString(bytes.ToArray()));
            bytes.Clear();
        }

        private static Dictionary<byte, char> BuildByteEncoder()
        {
            List<int> orderedBytes = new();
            orderedBytes.AddRange(Enumerable.Range('!', 126 - '!' + 1));
            orderedBytes.AddRange(Enumerable.Range(161, 172 - 161 + 1));
            orderedBytes.AddRange(Enumerable.Range(174, 255 - 174 + 1));

            int extraCodePoint = 256;
            for (int value = 0; value <= 255; value++)
            {
                if (orderedBytes.Contains(value))
                {
                    continue;
                }

                orderedBytes.Add(value);
            }

            Dictionary<byte, char> map = new(256);
            for (int index = 0; index < orderedBytes.Count; index++)
            {
                byte byteValue = (byte)orderedBytes[index];
                int codePoint = index < 188 ? orderedBytes[index] : extraCodePoint++;
                map[byteValue] = (char)codePoint;
            }

            return map;
        }

        private static Dictionary<char, byte> BuildByteDecoder(IReadOnlyDictionary<byte, char> encoder)
        {
            Dictionary<char, byte> map = new(encoder.Count);
            foreach ((byte key, char value) in encoder)
            {
                map[value] = key;
            }

            return map;
        }

        private static int GetRequiredSpecialTokenId(uint tokenId, string tokenKind)
        {
            return checked((int)tokenId);
        }

        private bool IsSpecialTokenId(int tokenId)
        {
            return tokenId == checked((int)tokenizerConfig.BosTokenId)
                || tokenId == checked((int)tokenizerConfig.EosTokenId)
                || tokenId == checked((int)tokenizerConfig.PaddingTokenId);
        }
    }
}
