using GGUFSharp;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Models
{
    /// <summary>
    /// Loads BitNet model metadata, tokenizer data, tensor descriptors, and layer definitions from GGUF files.
    /// </summary>
    public sealed class BitNetModelLoader
    {
        /// <summary>
        /// Loads a BitNet model from a GGUF file.
        /// </summary>
        public BitNetModel Load(string ggufPath)
        {
            return Load(ggufPath, options: null);
        }

        /// <summary>
        /// Loads a BitNet model from a GGUF file using the provided load options.
        /// </summary>
        public BitNetModel Load(string ggufPath, BitNetModelLoadOptions? options)
        {
            if (string.IsNullOrWhiteSpace(ggufPath))
            {
                throw new ArgumentException("GGUF path must not be empty.", nameof(ggufPath));
            }

            GGUFReader reader = new();
            GGUFFile file = reader.Read(ggufPath);
            BitNetMetadataParser metadataParser = options?.MetadataParser ?? ParseDefaultMetadata;
            BitNetMetadataParseResult metadata = metadataParser(file) ?? throw new InvalidOperationException("Metadata parser returned no result.");

            BitNetModelConfig config = metadata.ModelConfig ?? throw new InvalidOperationException("Metadata parser returned no model configuration.");
            BitNetTokenizerConfig tokenizerConfig = metadata.TokenizerConfig ?? throw new InvalidOperationException("Metadata parser returned no tokenizer configuration.");
            BitNetTokenizer tokenizer = new(tokenizerConfig);
            IReadOnlyDictionary<string, BitNetTensorInfo> tensorIndex = BitNetTensorIndexBuilder.Create(file.TensorInfos);
            IReadOnlyDictionary<string, GGUFTensorInfo> rawTensorIndex = file.TensorInfos.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
            BitNetGlobalTensors globalTensors = BitNetTensorIndexBuilder.CreateGlobalTensors(tensorIndex);
            IReadOnlyList<BitNetLayerDefinition> layers = BitNetLayerBuilder.Create(tensorIndex, checked((int)config.BlockCount));

            return new BitNetModel(
                reader,
                file,
                config,
                tokenizerConfig,
                tokenizer,
                globalTensors,
                layers,
                tensorIndex,
                rawTensorIndex);
        }

        private static BitNetMetadataParseResult ParseDefaultMetadata(GGUFFile file)
        {
            ArgumentNullException.ThrowIfNull(file);

            GgufMetadataReader metadataReader = new(file);
            string architectureName = metadataReader.GetArchitectureName();

            return architectureName switch
            {
                "bitnet-b1.58" => new BitNetMetadataParseResult
                {
                    ModelConfig = metadataReader.CreateBitNetB158ModelConfig(),
                    TokenizerConfig = metadataReader.CreateTokenizerConfig(),
                },
                _ => throw new NotSupportedException($"Unsupported BitNet architecture '{architectureName}'."),
            };
        }

        private sealed class GgufMetadataReader(GGUFFile file)
        {
            public string GetArchitectureName()
            {
                return GetRequiredString("general.architecture");
            }

            public BitNetModelConfig CreateBitNetB158ModelConfig()
            {
                return new BitNetModelConfig(
                    GetRequiredString("general.architecture"),
                    GetRequiredString("general.name"),
                    GetRequiredUInt32("bitnet-b1.58.vocab_size"),
                    GetRequiredUInt32("bitnet-b1.58.context_length"),
                    GetRequiredUInt32("bitnet-b1.58.embedding_length"),
                    GetRequiredUInt32("bitnet-b1.58.block_count"),
                    GetRequiredUInt32("bitnet-b1.58.feed_forward_length"),
                    GetRequiredUInt32("bitnet-b1.58.attention.head_count"),
                    GetRequiredUInt32("bitnet-b1.58.attention.head_count_kv"),
                    GetRequiredUInt32("bitnet-b1.58.rope.dimension_count"),
                    GetRequiredSingle("bitnet-b1.58.rope.freq_base"),
                    GetRequiredSingle("bitnet-b1.58.attention.layer_norm_rms_epsilon"),
                    GetRequiredUInt32("general.file_type"),
                    GetRequiredUInt32("general.quantization_version"));
            }

            public BitNetTokenizerConfig CreateTokenizerConfig()
            {
                return new BitNetTokenizerConfig(
                    GetRequiredString("tokenizer.ggml.model"),
                    GetRequiredBoolean("tokenizer.ggml.add_bos_token"),
                    GetRequiredUInt32("tokenizer.ggml.bos_token_id"),
                    GetRequiredUInt32("tokenizer.ggml.eos_token_id"),
                    GetRequiredUInt32("tokenizer.ggml.padding_token_id"),
                    GetRequiredString("tokenizer.chat_template"),
                    GetRequiredStringArray("tokenizer.ggml.tokens"),
                    GetRequiredStringArray("tokenizer.ggml.merges"),
                    GetRequiredSingleArray("tokenizer.ggml.scores"),
                    GetRequiredInt32Array("tokenizer.ggml.token_type"));
            }

            private GGUFMetaItem GetRequiredMetaItem(string metaItemName)
            {
                return file.MetaItems.Single(x => x.Name == metaItemName);
            }

            private string GetRequiredString(string metaItemName)
            {
                return Encoding.UTF8.GetString(GetRequiredMetaItem(metaItemName).RawData);
            }

            private uint GetRequiredUInt32(string metaItemName)
            {
                return BitConverter.ToUInt32(GetRequiredMetaItem(metaItemName).RawData, 0);
            }

            private float GetRequiredSingle(string metaItemName)
            {
                return BitConverter.ToSingle(GetRequiredMetaItem(metaItemName).RawData, 0);
            }

            private bool GetRequiredBoolean(string metaItemName)
            {
                return GetRequiredMetaItem(metaItemName).RawData[0] != 0;
            }

            private string[] GetRequiredStringArray(string metaItemName)
            {
                return GetRequiredMetaItem(metaItemName).ArrayStrings;
            }

            private int[] GetRequiredInt32Array(string metaItemName)
            {
                var rawData = GetRequiredMetaItem(metaItemName).RawData;
                return MemoryMarshal.Cast<byte, int>(rawData).ToArray();
            }

            private float[] GetRequiredSingleArray(string metaItemName)
            {
                var rawData = GetRequiredMetaItem(metaItemName).RawData;
                return MemoryMarshal.Cast<byte, float>(rawData).ToArray();
            }
        }

        private static class BitNetTensorIndexBuilder
        {
            private static readonly BitNetTensorRole[] QuantizedTensorRoles =
            [
                BitNetTensorRole.AttentionQueryWeight,
                BitNetTensorRole.AttentionKeyWeight,
                BitNetTensorRole.AttentionValueWeight,
                BitNetTensorRole.AttentionOutputWeight,
                BitNetTensorRole.FeedForwardDownWeight,
                BitNetTensorRole.FeedForwardGateWeight,
                BitNetTensorRole.FeedForwardUpWeight,
            ];

            public static IReadOnlyDictionary<string, BitNetTensorInfo> Create(IReadOnlyList<GGUFTensorInfo> tensorInfos)
            {
                ArgumentNullException.ThrowIfNull(tensorInfos);

                Dictionary<string, BitNetTensorInfo> tensors = new(StringComparer.Ordinal);
                foreach (var tensorInfo in tensorInfos)
                {
                    var tensor = CreateTensorInfo(tensorInfo);
                    tensors.Add(tensor.Name, tensor);
                }

                return tensors;
            }

            public static BitNetGlobalTensors CreateGlobalTensors(IReadOnlyDictionary<string, BitNetTensorInfo> tensors)
            {
                ArgumentNullException.ThrowIfNull(tensors);

                return new BitNetGlobalTensors(
                    GetRequiredTensor(tensors, "token_embd.weight"),
                    GetRequiredTensor(tensors, "output_norm.weight"));
            }

            private static BitNetTensorInfo CreateTensorInfo(GGUFTensorInfo tensorInfo)
            {
                ArgumentNullException.ThrowIfNull(tensorInfo);

                var tensorName = BitNetTensorNameParser.Parse(tensorInfo.Name);
                return new BitNetTensorInfo(
                    tensorInfo.Name,
                    tensorName.LayerIndex,
                    tensorName.Role,
                    tensorInfo.TensorType,
                    tensorInfo.Dimensions,
                    tensorInfo.Offset,
                    tensorInfo.Size,
                    QuantizedTensorRoles.Contains(tensorName.Role),
                    tensorName.LayerIndex is null);
            }

            private static BitNetTensorInfo GetRequiredTensor(IReadOnlyDictionary<string, BitNetTensorInfo> tensors, string tensorName)
            {
                if (!tensors.TryGetValue(tensorName, out var tensor))
                {
                    throw new InvalidOperationException($"Required tensor '{tensorName}' was not found.");
                }

                return tensor;
            }
        }

        private static class BitNetLayerBuilder
        {
            public static IReadOnlyList<BitNetLayerDefinition> Create(IReadOnlyDictionary<string, BitNetTensorInfo> tensors, int blockCount)
            {
                ArgumentNullException.ThrowIfNull(tensors);

                List<BitNetLayerDefinition> layers = new(blockCount);
                for (int layerIndex = 0; layerIndex < blockCount; layerIndex++)
                {
                    layers.Add(CreateLayer(tensors, layerIndex));
                }

                return layers;
            }

            private static BitNetLayerDefinition CreateLayer(IReadOnlyDictionary<string, BitNetTensorInfo> tensors, int layerIndex)
            {
                return new BitNetLayerDefinition(
                    layerIndex,
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_norm.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_sub_norm.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_q.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_k.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_v.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.attn_output.weight"),
                    GetOptionalTensor(tensors, $"blk.{layerIndex}.attn_output.scale"),
                    GetOptionalTensor(tensors, $"blk.{layerIndex}.attn_output.bias"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.ffn_norm.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.ffn_sub_norm.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.ffn_gate.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.ffn_up.weight"),
                    GetRequiredTensor(tensors, $"blk.{layerIndex}.ffn_down.weight"));
            }

            private static BitNetTensorInfo? GetOptionalTensor(IReadOnlyDictionary<string, BitNetTensorInfo> tensors, string tensorName)
            {
                tensors.TryGetValue(tensorName, out BitNetTensorInfo? tensor);
                return tensor;
            }

            private static BitNetTensorInfo GetRequiredTensor(IReadOnlyDictionary<string, BitNetTensorInfo> tensors, string tensorName)
            {
                if (!tensors.TryGetValue(tensorName, out var tensor))
                {
                    throw new InvalidOperationException($"Required tensor '{tensorName}' was not found.");
                }

                return tensor;
            }
        }

        private static class BitNetTensorNameParser
        {
            public static (int? LayerIndex, BitNetTensorRole Role) Parse(string tensorName)
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(tensorName);

                if (tensorName == "token_embd.weight")
                {
                    return (null, BitNetTensorRole.TokenEmbedding);
                }

                if (tensorName == "output_norm.weight")
                {
                    return (null, BitNetTensorRole.OutputNorm);
                }

                if (!tensorName.StartsWith("blk.", StringComparison.Ordinal))
                {
                    return (null, BitNetTensorRole.Unknown);
                }

                string[] segments = tensorName.Split('.', StringSplitOptions.RemoveEmptyEntries);
                if (segments.Length != 4 || !int.TryParse(segments[1], out int layerIndex))
                {
                    return (null, BitNetTensorRole.Unknown);
                }

                return (layerIndex, ResolveRole(tensorName));
            }

            private static BitNetTensorRole ResolveRole(string tensorName)
            {
                return tensorName switch
                {
                    var name when name.EndsWith("attn_norm.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionNorm,
                    var name when name.EndsWith("attn_sub_norm.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionSubNorm,
                    var name when name.EndsWith("attn_q.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionQueryWeight,
                    var name when name.EndsWith("attn_k.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionKeyWeight,
                    var name when name.EndsWith("attn_v.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionValueWeight,
                    var name when name.EndsWith("attn_output.scale", StringComparison.Ordinal) => BitNetTensorRole.AttentionOutputScale,
                    var name when name.EndsWith("attn_output.bias", StringComparison.Ordinal) => BitNetTensorRole.AttentionOutputBias,
                    var name when name.EndsWith("attn_output.weight", StringComparison.Ordinal) => BitNetTensorRole.AttentionOutputWeight,
                    var name when name.EndsWith("ffn_norm.weight", StringComparison.Ordinal) => BitNetTensorRole.FeedForwardNorm,
                    var name when name.EndsWith("ffn_sub_norm.weight", StringComparison.Ordinal) => BitNetTensorRole.FeedForwardSubNorm,
                    var name when name.EndsWith("ffn_gate.weight", StringComparison.Ordinal) => BitNetTensorRole.FeedForwardGateWeight,
                    var name when name.EndsWith("ffn_up.weight", StringComparison.Ordinal) => BitNetTensorRole.FeedForwardUpWeight,
                    var name when name.EndsWith("ffn_down.weight", StringComparison.Ordinal) => BitNetTensorRole.FeedForwardDownWeight,
                    _ => BitNetTensorRole.Unknown,
                };
            }
        }
    }
}
