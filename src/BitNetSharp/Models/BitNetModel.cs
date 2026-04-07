using GGUFSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Models
{
    public class BitNetModel : IDisposable
    {
        private bool disposed;
        private GGUFFile? loadedFile;
        private GGUFReader? loadedReader;
        private IReadOnlyDictionary<string, GGUFTensorInfo> rawTensorIndex = new Dictionary<string, GGUFTensorInfo>(StringComparer.Ordinal);

        public BitNetModelConfig? Config { get; private set; }

        public BitNetTokenizerConfig? TokenizerConfig { get; private set; }

        public BitNetGlobalTensors? GlobalTensors { get; private set; }

        public IReadOnlyList<BitNetLayerDefinition> Layers { get; private set; } = [];

        public IReadOnlyDictionary<string, BitNetTensorInfo> TensorIndex { get; private set; } =
            new Dictionary<string, BitNetTensorInfo>(StringComparer.Ordinal);

        public BitNetTokenizer? Tokenizer { get; private set; }

        public bool UsesTiedEmbeddings => GlobalTensors is not null;

        /// <summary>
        /// Loads model metadata, tokenizer data, tensor descriptors, and layer definitions from a GGUF file.
        /// </summary>
        public void Load(string ggufPath)
        {
            Load(ggufPath, options: null);
        }

        /// <summary>
        /// Loads model metadata, tokenizer data, tensor descriptors, and layer definitions from a GGUF file using the provided load options.
        /// </summary>
        public void Load(string ggufPath, BitNetModelLoadOptions? options)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(ggufPath))
            {
                throw new ArgumentException("GGUF path must not be empty.", nameof(ggufPath));
            }

            GGUFReader reader = new GGUFReader();
            GGUFFile file = reader.Read(ggufPath);
            BitNetMetadataParser metadataParser = options?.MetadataParser ?? ParseDefaultMetadata;
            BitNetMetadataParseResult metadata = metadataParser(file) ?? throw new InvalidOperationException("Metadata parser returned no result.");

            loadedReader = reader;
            loadedFile = file;

            Config = metadata.ModelConfig ?? throw new InvalidOperationException("Metadata parser returned no model configuration.");
            TokenizerConfig = metadata.TokenizerConfig ?? throw new InvalidOperationException("Metadata parser returned no tokenizer configuration.");
            Tokenizer = BitNetTokenizerFactory.Create(TokenizerConfig);

            var tensorIndex = BitNetTensorIndexBuilder.Create(file.TensorInfos);
            TensorIndex = tensorIndex;
            rawTensorIndex = file.TensorInfos.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
            GlobalTensors = BitNetTensorIndexBuilder.CreateGlobalTensors(tensorIndex);
            Layers = BitNetLayerBuilder.Create(tensorIndex, checked((int)Config.BlockCount));
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file.
        /// </summary>
        public IMemoryOwner<byte> ReadTensorData(string tensorName)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
            }

            if (loadedFile is null || loadedReader is null)
            {
                throw new InvalidOperationException("The model must be loaded before tensor data can be read.");
            }

            if (!rawTensorIndex.TryGetValue(tensorName, out GGUFTensorInfo? tensorInfo))
            {
                throw new InvalidOperationException($"Required tensor '{tensorName}' was not found.");
            }

            return loadedReader.ReadTensorData(loadedFile, tensorInfo);
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file.
        /// </summary>
        public IMemoryOwner<byte> ReadTensorData(BitNetTensorInfo tensorInfo)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentNullException.ThrowIfNull(tensorInfo);

            return ReadTensorData(tensorInfo.Name);
        }

        /// <summary>
        /// Releases the loaded model state so test fixtures and callers can free the model after use.
        /// </summary>
        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            Tokenizer = null;
            TokenizerConfig = null;
            GlobalTensors = null;
            Config = null;
            Layers = [];
            TensorIndex = new Dictionary<string, BitNetTensorInfo>(StringComparer.Ordinal);
            rawTensorIndex = new Dictionary<string, GGUFTensorInfo>(StringComparer.Ordinal);
            loadedFile = null;
            loadedReader = null;
            disposed = true;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Returns the requested layer definition from the loaded model.
        /// </summary>
        public BitNetLayerDefinition GetLayer(int layerIndex)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before layers can be accessed.");
            }

            if ((uint)layerIndex >= Config.BlockCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            return Layers[layerIndex];
        }

        /// <summary>
        /// Tries to find a tensor by its GGUF name.
        /// </summary>
        public bool TryGetTensor(string tensorName, out BitNetTensorInfo? tensor)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
            }

            return TensorIndex.TryGetValue(tensorName, out tensor);
        }

        private static BitNetMetadataParseResult ParseDefaultMetadata(GGUFFile file)
        {
            ArgumentNullException.ThrowIfNull(file);

            GgufMetadataReader metadataReader = new(file);
            return new BitNetMetadataParseResult
            {
                ModelConfig = metadataReader.CreateModelConfig(),
                TokenizerConfig = metadataReader.CreateTokenizerConfig(),
            };
        }

        private sealed class GgufMetadataReader(GGUFFile file)
        {
            public BitNetModelConfig CreateModelConfig()
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

        private static class BitNetTokenizerFactory
        {
            public static BitNetTokenizer Create(BitNetTokenizerConfig tokenizerConfig)
            {
                return new BitNetTokenizer(tokenizerConfig);
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
