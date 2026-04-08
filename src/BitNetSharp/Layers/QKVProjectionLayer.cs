using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Projects normalized hidden states into query, key, and value tensors on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class QKVProjectionLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo queryTensor;
        private readonly BitNetTensorInfo keyTensor;
        private readonly BitNetTensorInfo valueTensor;
        private PackedProjectionWeights? cachedQueryWeights;
        private PackedProjectionWeights? cachedKeyWeights;
        private PackedProjectionWeights? cachedValueWeights;
        private bool isInitialized;

        public QKVProjectionLayer(
            BitNetModel model,
            BitNetTensorInfo queryTensor,
            BitNetTensorInfo keyTensor,
            BitNetTensorInfo valueTensor,
            bool enableCache = false,
            InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(queryTensor);
            ArgumentNullException.ThrowIfNull(keyTensor);
            ArgumentNullException.ThrowIfNull(valueTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the QKV projection layer can be created.");
            }

            this.model = model;
            this.queryTensor = queryTensor;
            this.keyTensor = keyTensor;
            this.valueTensor = valueTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();

            int embeddingLength = checked((int)model.Config.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            ValidateTensor(queryTensor, embeddingLength, embeddingLength, "query");
            ValidateTensor(keyTensor, embeddingLength, keyValueLength, "key");
            ValidateTensor(valueTensor, embeddingLength, keyValueLength, "value");
        }

        public bool EnableCache { get; }

        public InferenceConfig InferenceConfig { get; }

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

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
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

            ForwardCore(session.RmsNorm, session.QKVQuery, session.QKVKey, session.QKVValue);
        }

        private void ForwardCore(ReadOnlyMemory<float> input, Memory<float> query, Memory<float> key, Memory<float> value)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            int expectedInputLength = checked((int)model.Config!.EmbeddingLength);
            if (input.Length != expectedInputLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            //load packed q / k / v weights from the model or reuse the cached copy
            //quantize the normalized attention input once, then reuse it across the three projection branches
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = MathHelper.QuantizeBitNetActivations(input.Span, quantizedValues.Span);
            int queryOutputLength = checked((int)model.Config.EmbeddingLength);
            int keyValueOutputLength = checked((int)model.Config.KeyValueProjectionSize);
            int threads = InferenceConfig.ThreadCount;

            if (EnableCache)
            {
                Project(quantizedValues, activationScale, EnsureCachedQueryWeights(), queryOutputLength, query, threads);
                Project(quantizedValues, activationScale, EnsureCachedKeyWeights(), keyValueOutputLength, key, threads);
                Project(quantizedValues, activationScale, EnsureCachedValueWeights(), keyValueOutputLength, value, threads);
                return;
            }

            using (IMemoryOwner<byte> queryTensorData = model.ReadTensorData(queryTensor))
            {
                Project(quantizedValues, activationScale, ParsePackedWeights(queryTensorData.Memory, queryTensor, "QKV query"), queryOutputLength, query, threads);
            }

            using (IMemoryOwner<byte> keyTensorData = model.ReadTensorData(keyTensor))
            {
                Project(quantizedValues, activationScale, ParsePackedWeights(keyTensorData.Memory, keyTensor, "QKV key"), keyValueOutputLength, key, threads);
            }

            using (IMemoryOwner<byte> valueTensorData = model.ReadTensorData(valueTensor))
            {
                Project(quantizedValues, activationScale, ParsePackedWeights(valueTensorData.Memory, valueTensor, "QKV value"), keyValueOutputLength, value, threads);
            }
        }

        private void Project(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, PackedProjectionWeights weights, int outputLength, Memory<float> output, int threads)
        {
            switch (InferenceConfig.Backend)
            {
                case InferenceBackend.CPU:
                    MathHelper.ProjectBitNetI2Cpu(quantizedValues, activationScale, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                case InferenceBackend.Tensor:
                    MathHelper.ProjectBitNetI2Tensor(quantizedValues, activationScale, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                case InferenceBackend.SIMD:
                    MathHelper.ProjectBitNetI2Simd(quantizedValues, activationScale, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                default:
                    throw new NotSupportedException($"QKV backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
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
                throw new InvalidOperationException("The layer must be initialized by calling Init before Forward.");
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
