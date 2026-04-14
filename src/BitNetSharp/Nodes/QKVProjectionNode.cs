using BitNetSharp.Core;
using BitNetSharp.Models;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Projects normalized hidden states into query, key, and value tensors on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class QKVProjectionNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo queryTensor;
        private readonly BitNetTensorInfo keyTensor;
        private readonly BitNetTensorInfo valueTensor;
        private readonly IOPProvider opProvider;
        private PackedProjectionWeights? cachedQueryWeights;
        private PackedProjectionWeights? cachedKeyWeights;
        private PackedProjectionWeights? cachedValueWeights;
        private bool isInitialized;

        public QKVProjectionNode(
            BitNetModel model,
            BitNetTensorInfo queryTensor,
            BitNetTensorInfo keyTensor,
            BitNetTensorInfo valueTensor,
            bool enableCache = false,
            Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(queryTensor);
            ArgumentNullException.ThrowIfNull(keyTensor);
            ArgumentNullException.ThrowIfNull(valueTensor);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the QKV projection node can be created.");
            }

            this.model = model;
            this.queryTensor = queryTensor;
            this.keyTensor = keyTensor;
            this.valueTensor = valueTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig;
            opProvider = InferenceConfig.OPProvider;

            int embeddingLength = checked((int)model.Config.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            ValidateTensor(queryTensor, embeddingLength, embeddingLength, "query");
            ValidateTensor(keyTensor, embeddingLength, keyValueLength, "key");
            ValidateTensor(valueTensor, embeddingLength, keyValueLength, "value");
        }

        public bool EnableCache { get; }

        public Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedQueryWeights();
                _ = EnsureCachedKeyWeights();
                _ = EnsureCachedValueWeights();
            }

            isInitialized = true;
        }

        /// <summary>
        /// Projects the normalized attention input stored on the session into query, key, and value vectors.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.RmsNormKey))
            {
                throw new InvalidOperationException("Session does not contain RMSNorm output.");
            }

            RuntimeTensor input = session.RmsNormTensor;
            RuntimeTensor query = session.QKVQueryTensor;
            RuntimeTensor key = session.QKVKeyTensor;
            RuntimeTensor value = session.QKVValueTensor;
            if (!input.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> inputMemory) || inputMemory.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            int expectedInputLength = checked((int)model.Config!.EmbeddingLength);
            if (inputMemory.Length != expectedInputLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            int queryOutputLength = checked((int)model.Config.EmbeddingLength);
            int keyValueOutputLength = checked((int)model.Config.KeyValueProjectionSize);

            if (EnableCache)
            {
                PackedProjectionWeights cachedQueryWeights = EnsureCachedQueryWeights();
                PackedProjectionWeights cachedKeyWeights = EnsureCachedKeyWeights();
                PackedProjectionWeights cachedValueWeights = EnsureCachedValueWeights();
                using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
                Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..inputMemory.Length];
                RuntimeTensor quantizedTensor = RuntimeTensor.CreateWritable("QKVQuantizedValues", quantizedValues, [inputMemory.Length]);
                (float activationScale, _) = opProvider.QuantizeBitNetActivations(input, quantizedTensor);
                opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(cachedQueryWeights.PackedWeights, "CachedQKVQueryWeights"), queryOutputLength, cachedQueryWeights.Scale, query);
                opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(cachedKeyWeights.PackedWeights, "CachedQKVKeyWeights"), keyValueOutputLength, cachedKeyWeights.Scale, key);
                opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(cachedValueWeights.PackedWeights, "CachedQKVValueWeights"), keyValueOutputLength, cachedValueWeights.Scale, value);
                return;
            }

            using IMemoryOwner<byte> queryTensorData = model.ReadTensorData(queryTensor);
            using IMemoryOwner<byte> keyTensorData = model.ReadTensorData(keyTensor);
            using IMemoryOwner<byte> valueTensorData = model.ReadTensorData(valueTensor);
            PackedProjectionWeights queryWeights = ParsePackedWeights(queryTensorData.Memory, queryTensor, "QKV query");
            PackedProjectionWeights keyWeights = ParsePackedWeights(keyTensorData.Memory, keyTensor, "QKV key");
            PackedProjectionWeights valueWeights = ParsePackedWeights(valueTensorData.Memory, valueTensor, "QKV value");

            using IMemoryOwner<sbyte> uncachedQuantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
            Memory<sbyte> uncachedQuantizedValues = uncachedQuantizedValuesOwner.Memory[..inputMemory.Length];
            RuntimeTensor uncachedQuantizedTensor = RuntimeTensor.CreateWritable("QKVQuantizedValues", uncachedQuantizedValues, [inputMemory.Length]);
            (float uncachedActivationScale, _) = opProvider.QuantizeBitNetActivations(input, uncachedQuantizedTensor);
            opProvider.ProjectBitNetI2(uncachedQuantizedTensor, uncachedActivationScale, CreatePackedWeightTensor(queryWeights.PackedWeights, "QKVQueryWeights"), queryOutputLength, queryWeights.Scale, query);
            opProvider.ProjectBitNetI2(uncachedQuantizedTensor, uncachedActivationScale, CreatePackedWeightTensor(keyWeights.PackedWeights, "QKVKeyWeights"), keyValueOutputLength, keyWeights.Scale, key);
            opProvider.ProjectBitNetI2(uncachedQuantizedTensor, uncachedActivationScale, CreatePackedWeightTensor(valueWeights.PackedWeights, "QKVValueWeights"), keyValueOutputLength, valueWeights.Scale, value);
        }

        private static RuntimeTensor CreatePackedWeightTensor(ReadOnlyMemory<byte> packedWeights, string name)
        {
            return RuntimeTensor.CreateReadOnly<byte>(name, packedWeights, [packedWeights.Length]);
        }

        private PackedProjectionWeights ReadPackedWeights(BitNetTensorInfo tensor)
        {
            using var tensorData = model.ReadTensorData(tensor);
            PackedProjectionWeights weights = ParsePackedWeights(tensorData.Memory, tensor, "Packed QKV");
            return new PackedProjectionWeights(weights.PackedWeights.ToArray(), weights.Scale);
        }

        private static PackedProjectionWeights ParsePackedWeights(ReadOnlyMemory<byte> tensorBytes, BitNetTensorInfo tensor, string tensorLabel)
        {
            //each byte stores four 2-bit ternary weights, and the packed payload is followed by one float scale
            int packedWeightByteCount = checked(((int)tensor.Dimensions[0] * (int)tensor.Dimensions[1]) / 4);
            if (tensorBytes.Length < packedWeightByteCount + sizeof(float))
            {
                throw new InvalidOperationException($"{tensorLabel} tensor '{tensor.Name}' is incomplete.");
            }

            return new PackedProjectionWeights(
                tensorBytes[..packedWeightByteCount],
                MemoryMarshal.Read<float>(tensorBytes.Span.Slice(packedWeightByteCount, sizeof(float))));
        }

        private PackedProjectionWeights EnsureCachedQueryWeights()
        {
            return cachedQueryWeights ??= ReadPackedWeights(queryTensor);
        }

        private PackedProjectionWeights EnsureCachedKeyWeights()
        {
            return cachedKeyWeights ??= ReadPackedWeights(keyTensor);
        }

        private PackedProjectionWeights EnsureCachedValueWeights()
        {
            return cachedValueWeights ??= ReadPackedWeights(valueTensor);
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The node must be initialized by calling Init before Forward.");
            }
        }

        private static void ValidateTensor(BitNetTensorInfo tensor, int expectedInputLength, int expectedOutputLength, string tensorLabel)
        {
            if (!tensor.IsQuantized)
            {
                throw new NotSupportedException($"QKV {tensorLabel} tensor type '{tensor.TensorType}' is not supported.");
            }

            if (tensor.Dimensions.Count < 2)
            {
                throw new InvalidOperationException($"QKV {tensorLabel} tensor dimensions are incomplete.");
            }

            int actualInputLength = checked((int)tensor.Dimensions[0]);
            int actualOutputLength = checked((int)tensor.Dimensions[1]);
            if (actualInputLength != expectedInputLength || actualOutputLength != expectedOutputLength)
            {
                throw new InvalidOperationException($"QKV {tensorLabel} tensor dimensions do not match the loaded model configuration.");
            }
        }

        private sealed record PackedProjectionWeights(ReadOnlyMemory<byte> PackedWeights, float Scale);
    }
}
