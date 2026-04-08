using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Applies the feed-forward block projection, squared-ReLU gating, sub-norm, and down projection.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class FeedForwardLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo subNormTensor;
        private readonly BitNetTensorInfo gateTensor;
        private readonly BitNetTensorInfo upTensor;
        private readonly BitNetTensorInfo downTensor;
        private float[]? cachedSubNormWeights;
        private PackedProjectionWeights? cachedGateWeights;
        private PackedProjectionWeights? cachedUpWeights;
        private PackedProjectionWeights? cachedDownWeights;
        private bool isInitialized;

        /// <summary>
        /// Creates a feed-forward layer for the loaded model.
        /// </summary>
        public FeedForwardLayer(
            BitNetModel model,
            BitNetTensorInfo subNormTensor,
            BitNetTensorInfo gateTensor,
            BitNetTensorInfo upTensor,
            BitNetTensorInfo downTensor,
            bool enableCache = false,
            InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(subNormTensor);
            ArgumentNullException.ThrowIfNull(gateTensor);
            ArgumentNullException.ThrowIfNull(upTensor);
            ArgumentNullException.ThrowIfNull(downTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the feed-forward layer can be created.");
            }

            this.model = model;
            this.subNormTensor = subNormTensor;
            this.gateTensor = gateTensor;
            this.upTensor = upTensor;
            this.downTensor = downTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();

            ValidateSubNormTensor();
            ValidateProjectionTensor(gateTensor, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.FeedForwardLength), "gate");
            ValidateProjectionTensor(upTensor, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.FeedForwardLength), "up");
            ValidateProjectionTensor(downTensor, checked((int)model.Config.FeedForwardLength), checked((int)model.Config.EmbeddingLength), "down");
        }

        public bool EnableCache { get; }

        public InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedSubNormWeights();
                _ = EnsureCachedGateWeights();
                _ = EnsureCachedUpWeights();
                _ = EnsureCachedDownWeights();
            }

            isInitialized = true;
        }

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
        }

        /// <summary>
        /// Applies the feed-forward block to the session feed-forward norm buffer.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.FeedForwardNormKey))
            {
                throw new InvalidOperationException("Session does not contain feed-forward norm output.");
            }

            ForwardCore(session.FeedForwardNorm, session.FeedForwardSubNorm, session.FeedForwardOutput);
        }

        private void ForwardCore(ReadOnlyMemory<float> input, Memory<float> subNorm, Memory<float> output)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            if (input.Length != embeddingLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            int feedForwardLength = checked((int)model.Config.FeedForwardLength);
            if (subNorm.Length < feedForwardLength)
            {
                throw new ArgumentException("Feed-forward sub-norm output length does not match the model feed-forward length.", nameof(subNorm));
            }

            if (output.Length < embeddingLength)
            {
                throw new ArgumentException("Feed-forward output length does not match the model embedding length.", nameof(output));
            }

            using IMemoryOwner<float> upOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<float> gateOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            Memory<float> up = upOwner.Memory[..feedForwardLength];
            Memory<float> gate = gateOwner.Memory[..feedForwardLength];
            int threads = InferenceConfig.ThreadCount;

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = MathHelper.QuantizeBitNetActivations(input.Span, quantizedValues.Span);

            if (EnableCache)
            {
                ExecuteProjection(quantizedValues, activationScale, EnsureCachedUpWeights(), feedForwardLength, up, threads);
                ExecuteProjection(quantizedValues, activationScale, EnsureCachedGateWeights(), feedForwardLength, gate, threads);
                ApplySquaredReluGate(gate, up, threads);
                ExecuteSubNorm(up, EnsureCachedSubNormWeights().AsMemory(0, feedForwardLength), subNorm, threads);
                ExecuteProjection(subNorm, EnsureCachedDownWeights(), embeddingLength, output, threads);
                return;
            }

            using (IMemoryOwner<byte> upTensorData = model.ReadTensorData(upTensor))
            using (IMemoryOwner<byte> gateTensorData = model.ReadTensorData(gateTensor))
            {
                ExecuteProjection(quantizedValues, activationScale, ParsePackedWeights(upTensorData.Memory, upTensor, "Feed-forward up"), feedForwardLength, up, threads);
                ExecuteProjection(quantizedValues, activationScale, ParsePackedWeights(gateTensorData.Memory, gateTensor, "Feed-forward gate"), feedForwardLength, gate, threads);
            }

            ApplySquaredReluGate(gate, up, threads);

            using (IMemoryOwner<byte> subNormTensorData = model.ReadTensorData(subNormTensor))
            using (IMemoryOwner<float> subNormWeightsOwner = MemoryPool<float>.Shared.Rent(feedForwardLength))
            {
                Memory<float> subNormWeights = subNormWeightsOwner.Memory[..feedForwardLength];
                FillFloatValues(subNormTensorData.Memory.Span, subNormTensor.TensorType, subNormWeights.Span, "Feed-forward sub-norm");
                ExecuteSubNorm(up, subNormWeights, subNorm, threads);
            }

            using (IMemoryOwner<byte> downTensorData = model.ReadTensorData(downTensor))
            {
                ExecuteProjection(subNorm, ParsePackedWeights(downTensorData.Memory, downTensor, "Feed-forward down"), embeddingLength, output, threads);
            }
        }

        private void ExecuteProjection(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, PackedProjectionWeights weights, int outputLength, Memory<float> output, int threads)
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
                    throw new NotSupportedException($"Feed-forward backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
        }

        private void ExecuteProjection(ReadOnlyMemory<float> input, PackedProjectionWeights weights, int outputLength, Memory<float> output, int threads)
        {
            switch (InferenceConfig.Backend)
            {
                case InferenceBackend.CPU:
                    MathHelper.ProjectBitNetI2Cpu(input, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                case InferenceBackend.Tensor:
                    MathHelper.ProjectBitNetI2Tensor(input, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                case InferenceBackend.SIMD:
                    MathHelper.ProjectBitNetI2Simd(input, weights.PackedWeights, outputLength, weights.Scale, output, threads);
                    return;
                default:
                    throw new NotSupportedException($"Feed-forward backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
        }

        private void ExecuteSubNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> subNormWeights, Memory<float> output, int threads)
        {
            switch (InferenceConfig.Backend)
            {
                case InferenceBackend.CPU:
                    MathHelper.ForwardRmsNormCpuStandard(input, subNormWeights, model.Config!.AttentionLayerNormRmsEpsilon, output, threads);
                    return;
                case InferenceBackend.Tensor:
                    MathHelper.ForwardRmsNormTensor(input, subNormWeights, model.Config!.AttentionLayerNormRmsEpsilon, output, threads);
                    return;
                case InferenceBackend.SIMD:
                    MathHelper.ForwardRmsNormSimd(input, subNormWeights, model.Config!.AttentionLayerNormRmsEpsilon, output, threads);
                    return;
                default:
                    throw new NotSupportedException($"Feed-forward backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
        }

        private static void ApplySquaredReluGate(ReadOnlyMemory<float> gate, Memory<float> up, int threads)
        {
            if (gate.Length != up.Length)
            {
                throw new ArgumentException("Feed-forward gate length must match the up projection length.", nameof(gate));
            }

            if (threads == 1 || up.Length <= 1)
            {
                ApplySquaredReluGateRange(gate.Span, up.Span, 0, up.Length);
                return;
            }

            ThreadHelper.ForEachRange(up.Length, (startIndex, endIndex) =>
                ApplySquaredReluGateRange(gate.Span, up.Span, startIndex, endIndex), threads, sizeof(float));
        }

        private static void ApplySquaredReluGateRange(ReadOnlySpan<float> gate, Span<float> up, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                float relu = MathF.Max(gate[index], 0f);
                up[index] *= relu * relu;
            }
        }

        private void ValidateSubNormTensor()
        {
            int expectedLength = checked((int)model.Config!.FeedForwardLength);
            int actualLength = GetElementCount(subNormTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("Feed-forward sub-norm tensor dimensions do not match the loaded model configuration.");
            }

            if (subNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && subNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Feed-forward sub-norm tensor type '{subNormTensor.TensorType}' is not supported.");
            }
        }

        private static void ValidateProjectionTensor(BitNetTensorInfo tensor, int expectedInputLength, int expectedOutputLength, string tensorLabel)
        {
            if (!tensor.IsQuantized)
            {
                throw new NotSupportedException($"Feed-forward {tensorLabel} tensor type '{tensor.TensorType}' is not supported.");
            }

            if (tensor.Dimensions.Count < 2)
            {
                throw new InvalidOperationException($"Feed-forward {tensorLabel} tensor dimensions are incomplete.");
            }

            int actualInputLength = checked((int)tensor.Dimensions[0]);
            int actualOutputLength = checked((int)tensor.Dimensions[1]);
            if (actualInputLength != expectedInputLength || actualOutputLength != expectedOutputLength)
            {
                throw new InvalidOperationException($"Feed-forward {tensorLabel} tensor dimensions do not match the loaded model configuration.");
            }
        }

        private float[] ReadSubNormWeights()
        {
            using var tensorData = model.ReadTensorData(subNormTensor);
            return subNormTensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"Feed-forward sub-norm tensor type '{subNormTensor.TensorType}' is not supported."),
            };
        }

        private float[] EnsureCachedSubNormWeights()
        {
            return cachedSubNormWeights ??= ReadSubNormWeights();
        }

        private PackedProjectionWeights ReadPackedWeights(BitNetTensorInfo tensor, string tensorLabel)
        {
            using var tensorData = model.ReadTensorData(tensor);
            PackedProjectionWeights weights = ParsePackedWeights(tensorData.Memory, tensor, tensorLabel);
            return new PackedProjectionWeights(weights.PackedWeights.ToArray(), weights.Scale);
        }

        private static PackedProjectionWeights ParsePackedWeights(ReadOnlyMemory<byte> tensorBytes, BitNetTensorInfo tensor, string tensorLabel)
        {
            int packedWeightByteCount = checked(((int)tensor.Dimensions[0] * (int)tensor.Dimensions[1]) / 4);
            if (tensorBytes.Length < packedWeightByteCount + sizeof(float))
            {
                throw new InvalidOperationException($"{tensorLabel} tensor '{tensor.Name}' is incomplete.");
            }

            return new PackedProjectionWeights(
                tensorBytes[..packedWeightByteCount],
                MemoryMarshal.Read<float>(tensorBytes.Span.Slice(packedWeightByteCount, sizeof(float))));
        }

        private PackedProjectionWeights EnsureCachedGateWeights()
        {
            return cachedGateWeights ??= ReadPackedWeights(gateTensor, "Feed-forward gate");
        }

        private PackedProjectionWeights EnsureCachedUpWeights()
        {
            return cachedUpWeights ??= ReadPackedWeights(upTensor, "Feed-forward up");
        }

        private PackedProjectionWeights EnsureCachedDownWeights()
        {
            return cachedDownWeights ??= ReadPackedWeights(downTensor, "Feed-forward down");
        }

        private static void FillFloatValues(ReadOnlySpan<byte> source, GGUFTensorType tensorType, Span<float> destination, string tensorLabel)
        {
            switch (tensorType)
            {
                case GGUFTensorType.GGML_TYPE_F32:
                    MemoryMarshal.Cast<byte, float>(source[..checked(destination.Length * sizeof(float))]).CopyTo(destination);
                    return;
                case GGUFTensorType.GGML_TYPE_F16:
                    ConvertHalfBytesToSingle(source[..checked(destination.Length * sizeof(ushort))], destination);
                    return;
                default:
                    throw new NotSupportedException($"{tensorLabel} tensor type '{tensorType}' is not supported.");
            }
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The layer must be initialized by calling Init before Forward.");
            }
        }

        private static float[] ConvertHalfToSingle(ReadOnlySpan<Half> source)
        {
            float[] values = new float[source.Length];
            ConvertHalfToSingle(source, values);
            return values;
        }

        private static void ConvertHalfToSingle(ReadOnlySpan<Half> source, Span<float> destination)
        {
            for (int index = 0; index < source.Length; index++)
            {
                destination[index] = (float)source[index];
            }
        }

        private static void ConvertHalfBytesToSingle(ReadOnlySpan<byte> source, Span<float> destination)
        {
            ReadOnlySpan<ushort> halfBits = MemoryMarshal.Cast<byte, ushort>(source);
            for (int index = 0; index < halfBits.Length; index++)
            {
                destination[index] = (float)BitConverter.UInt16BitsToHalf(halfBits[index]);
            }
        }

        private static int GetElementCount(IReadOnlyList<ulong> dimensions)
        {
            if (dimensions.Count == 0)
            {
                throw new InvalidOperationException("Feed-forward tensor dimensions are incomplete.");
            }

            ulong total = 1;
            foreach (ulong dimension in dimensions)
            {
                total = checked(total * dimension);
            }

            return checked((int)total);
        }

        private sealed record PackedProjectionWeights(ReadOnlyMemory<byte> PackedWeights, float Scale);
    }
}
