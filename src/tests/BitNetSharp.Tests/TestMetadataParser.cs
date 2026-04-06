using GGUFSharp;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Tests
{
    internal static class TestMetadataParser
    {
        public static Models.BitNetMetadataParseResult ParseDefaultBitNetMetadata(GGUFFile file)
        {
            ArgumentNullException.ThrowIfNull(file);

            return new Models.BitNetMetadataParseResult
            {
                ModelConfig = new Models.BitNetModelConfig(
                    GetRequiredString(file, "general.architecture"),
                    GetRequiredString(file, "general.name"),
                    GetRequiredUInt32(file, "bitnet-b1.58.vocab_size"),
                    GetRequiredUInt32(file, "bitnet-b1.58.context_length"),
                    GetRequiredUInt32(file, "bitnet-b1.58.embedding_length"),
                    GetRequiredUInt32(file, "bitnet-b1.58.block_count"),
                    GetRequiredUInt32(file, "bitnet-b1.58.feed_forward_length"),
                    GetRequiredUInt32(file, "bitnet-b1.58.attention.head_count"),
                    GetRequiredUInt32(file, "bitnet-b1.58.attention.head_count_kv"),
                    GetRequiredUInt32(file, "bitnet-b1.58.rope.dimension_count"),
                    GetRequiredSingle(file, "bitnet-b1.58.rope.freq_base"),
                    GetRequiredSingle(file, "bitnet-b1.58.attention.layer_norm_rms_epsilon"),
                    GetRequiredUInt32(file, "general.file_type"),
                    GetRequiredUInt32(file, "general.quantization_version")),
                TokenizerConfig = new Models.BitNetTokenizerConfig(
                    GetRequiredString(file, "tokenizer.ggml.model"),
                    GetRequiredBoolean(file, "tokenizer.ggml.add_bos_token"),
                    GetRequiredUInt32(file, "tokenizer.ggml.bos_token_id"),
                    GetRequiredUInt32(file, "tokenizer.ggml.eos_token_id"),
                    GetRequiredUInt32(file, "tokenizer.ggml.padding_token_id"),
                    GetRequiredString(file, "tokenizer.chat_template"),
                    GetRequiredStringArray(file, "tokenizer.ggml.tokens"),
                    GetRequiredStringArray(file, "tokenizer.ggml.merges"),
                    GetRequiredSingleArray(file, "tokenizer.ggml.scores"),
                    GetRequiredInt32Array(file, "tokenizer.ggml.token_type")),
            };
        }

        private static GGUFMetaItem GetRequiredMetaItem(GGUFFile file, string metaItemName)
        {
            return file.MetaItems.Single(x => x.Name == metaItemName);
        }

        private static string GetRequiredString(GGUFFile file, string metaItemName)
        {
            return Encoding.UTF8.GetString(GetRequiredMetaItem(file, metaItemName).RawData);
        }

        private static uint GetRequiredUInt32(GGUFFile file, string metaItemName)
        {
            return BitConverter.ToUInt32(GetRequiredMetaItem(file, metaItemName).RawData, 0);
        }

        private static float GetRequiredSingle(GGUFFile file, string metaItemName)
        {
            return BitConverter.ToSingle(GetRequiredMetaItem(file, metaItemName).RawData, 0);
        }

        private static bool GetRequiredBoolean(GGUFFile file, string metaItemName)
        {
            return GetRequiredMetaItem(file, metaItemName).RawData[0] != 0;
        }

        private static string[] GetRequiredStringArray(GGUFFile file, string metaItemName)
        {
            return GetRequiredMetaItem(file, metaItemName).ArrayStrings;
        }

        private static int[] GetRequiredInt32Array(GGUFFile file, string metaItemName)
        {
            return MemoryMarshal.Cast<byte, int>(GetRequiredMetaItem(file, metaItemName).RawData).ToArray();
        }

        private static float[] GetRequiredSingleArray(GGUFFile file, string metaItemName)
        {
            return MemoryMarshal.Cast<byte, float>(GetRequiredMetaItem(file, metaItemName).RawData).ToArray();
        }
    }
}
