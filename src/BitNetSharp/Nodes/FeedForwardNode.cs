using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Applies the feed-forward block projection, squared-ReLU gating, sub-norm, and down projection.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class FeedForwardNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo subNormTensor;
        private readonly BitNetTensorInfo gateTensor;
        private readonly BitNetTensorInfo upTensor;
        private readonly BitNetTensorInfo downTensor;
        private readonly IOPProvider2 opProvider;
        private float[]? cachedSubNormWeights;
        private PackedProjectionWeights? cachedGateWeights;
        private PackedProjectionWeights? cachedUpWeights;
        private PackedProjectionWeights? cachedDownWeights;
        private bool isInitialized;

        /// <summary>
        /// Creates a feed-forward node for the loaded model.
        /// </summary>
        public FeedForwardNode(
            BitNetModel model,
            BitNetTensorInfo subNormTensor,
            BitNetTensorInfo gateTensor,
            BitNetTensorInfo upTensor,
            BitNetTensorInfo downTensor,
            bool enableCache = false,
            global::BitNetSharp.Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(subNormTensor);
            ArgumentNullException.ThrowIfNull(gateTensor);
            ArgumentNullException.ThrowIfNull(upTensor);
            ArgumentNullException.ThrowIfNull(downTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the feed-forward node can be created.");
            }

            this.model = model;
            this.subNormTensor = subNormTensor;
            this.gateTensor = gateTensor;
            this.upTensor = upTensor;
            this.downTensor = downTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
            opProvider = InferenceConfig.Backend switch
            {
                global::BitNetSharp.Nodes.InferenceBackend.CPU => new CPUDefaultOPProvider(InferenceConfig.ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.Tensor => new CPUTensorOPProvider(InferenceConfig.ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.SIMD => new CPUSimdOPProvider(InferenceConfig.ThreadCount),
                _ => throw new NotSupportedException($"Backend '{InferenceConfig.Backend}' is not implemented yet."),
            };

            ValidateSubNormTensor();
            ValidateProjectionTensor(gateTensor, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.FeedForwardLength), "gate");
            ValidateProjectionTensor(upTensor, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.FeedForwardLength), "up");
            ValidateProjectionTensor(downTensor, checked((int)model.Config.FeedForwardLength), checked((int)model.Config.EmbeddingLength), "down");
        }

        public bool EnableCache { get; }

        public global::BitNetSharp.Nodes.InferenceConfig InferenceConfig { get; }

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

        private static global::BitNetSharp.Nodes.InferenceConfig CreateDefaultInferenceConfig()
        {
            return new global::BitNetSharp.Nodes.InferenceConfig(global::BitNetSharp.Nodes.InferenceBackend.SIMD, global::BitNetSharp.Nodes.InferenceConfig.AutoThreadCount);
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

            if (EnableCache)
            {
                PackedProjectionWeights cachedGateWeights = EnsureCachedGateWeights();
                PackedProjectionWeights cachedUpWeights = EnsureCachedUpWeights();
                PackedProjectionWeights cachedDownWeights = EnsureCachedDownWeights();
                opProvider.ForwardFeedForward(input, EnsureCachedSubNormWeights().AsMemory(0, feedForwardLength), model.Config!.AttentionLayerNormRmsEpsilon, cachedGateWeights.PackedWeights, cachedGateWeights.Scale, cachedUpWeights.PackedWeights, cachedUpWeights.Scale, cachedDownWeights.PackedWeights, cachedDownWeights.Scale, embeddingLength, feedForwardLength, subNorm, output);
                return;
            }

            PackedProjectionWeights upWeights;
            using (IMemoryOwner<byte> upTensorData = model.ReadTensorData(upTensor))
            {
                PackedProjectionWeights weights = ParsePackedWeights(upTensorData.Memory, upTensor, "Feed-forward up");
                upWeights = new PackedProjectionWeights(weights.PackedWeights.ToArray(), weights.Scale);
            }

            PackedProjectionWeights gateWeights;
            using (IMemoryOwner<byte> gateTensorData = model.ReadTensorData(gateTensor))
            {
                PackedProjectionWeights weights = ParsePackedWeights(gateTensorData.Memory, gateTensor, "Feed-forward gate");
                gateWeights = new PackedProjectionWeights(weights.PackedWeights.ToArray(), weights.Scale);
            }

            float[] subNormWeights;
            using (IMemoryOwner<byte> subNormTensorData = model.ReadTensorData(subNormTensor))
            {
                subNormWeights = new float[feedForwardLength];
                FillFloatValues(subNormTensorData.Memory.Span, subNormTensor.TensorType, subNormWeights, "Feed-forward sub-norm");
            }

            PackedProjectionWeights downWeights;
            using (IMemoryOwner<byte> downTensorData = model.ReadTensorData(downTensor))
            {
                PackedProjectionWeights weights = ParsePackedWeights(downTensorData.Memory, downTensor, "Feed-forward down");
                downWeights = new PackedProjectionWeights(weights.PackedWeights.ToArray(), weights.Scale);
            }

            opProvider.ForwardFeedForward(input, subNormWeights, model.Config!.AttentionLayerNormRmsEpsilon, gateWeights.PackedWeights, gateWeights.Scale, upWeights.PackedWeights, upWeights.Scale, downWeights.PackedWeights, downWeights.Scale, embeddingLength, feedForwardLength, subNorm, output);
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
                throw new InvalidOperationException("The node must be initialized by calling Init before Forward.");
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
