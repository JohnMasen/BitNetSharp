using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Builds the current-token attention output from the session QKV tensors, applies the attention sub-norm,
    /// and projects the result with the attention output weights.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class AttentionNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo subNormTensor;
        private readonly BitNetTensorInfo outputTensor;
        private readonly BitNetTensorInfo? outputScaleTensor;
        private readonly BitNetTensorInfo? outputBiasTensor;
        private readonly IOPProvider2 opProvider;
        private float[]? cachedSubNormWeights;
        private PackedProjectionWeights? cachedOutputWeights;
        private float[]? cachedOutputScaleValues;
        private float[]? cachedOutputBiasValues;
        private bool isInitialized;

        /// <summary>
        /// Creates an attention node that consumes QKV projection output, applies attention sub-norm,
        /// and then performs the attention output projection.
        /// </summary>
        /// <param name="model">The loaded model that owns the attention tensors.</param>
        /// <param name="subNormTensor">The attention post-reduction RMSNorm weight tensor.</param>
        /// <param name="outputTensor">The packed attention output projection weight tensor.</param>
        /// <param name="outputScaleTensor">
        /// An optional scalar tensor applied after the attention output projection when the model provides
        /// an extra <c>attn_output.scale</c> tensor.
        /// </param>
        /// <param name="outputBiasTensor">
        /// An optional bias tensor applied after the attention output projection when the model provides
        /// an extra <c>attn_output.bias</c> tensor.
        /// </param>
        /// <param name="enableCache">Whether tensor data should be loaded eagerly and cached during <see cref="Init"/>.</param>
        /// <param name="inferenceConfig">The backend and threading configuration for the node.</param>
        /// <remarks>
        /// Some models, including the current baseline test GGUF, do not contain <c>attn_output.scale</c>
        /// or <c>attn_output.bias</c>. Callers may pass <see langword="null"/> for those optional tensors in
        /// that case, but models that do provide them should forward both tensors so the runtime order remains
        /// <c>wo -&gt; optional scale -&gt; optional bias</c>.
        /// </remarks>
        public AttentionNode(
            BitNetModel model,
            BitNetTensorInfo subNormTensor,
            BitNetTensorInfo outputTensor,
            BitNetTensorInfo? outputScaleTensor = null,
            BitNetTensorInfo? outputBiasTensor = null,
            bool enableCache = false,
            global::BitNetSharp.Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(subNormTensor);
            ArgumentNullException.ThrowIfNull(outputTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the attention node can be created.");
            }

            this.model = model;
            this.subNormTensor = subNormTensor;
            this.outputTensor = outputTensor;
            this.outputScaleTensor = outputScaleTensor;
            this.outputBiasTensor = outputBiasTensor;
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
            ValidateOutputTensor();
            ValidateOutputScaleTensor();
            ValidateOutputBiasTensor();
        }

        public bool EnableCache { get; }

        public global::BitNetSharp.Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedSubNormWeights();
                _ = EnsureCachedOutputWeights();
                _ = TryEnsureCachedOutputScaleValues();
                _ = TryEnsureCachedOutputBiasValues();
            }

            isInitialized = true;
        }

        private static global::BitNetSharp.Nodes.InferenceConfig CreateDefaultInferenceConfig()
        {
            return new global::BitNetSharp.Nodes.InferenceConfig(global::BitNetSharp.Nodes.InferenceBackend.SIMD, global::BitNetSharp.Nodes.InferenceConfig.AutoThreadCount);
        }

        /// <summary>
        /// Computes the current-token attention output using the QKV projection stored on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            bool hasQuery = session.HasMemory<float>(BitNetSession.QKVQueryKey);
            bool hasKey = session.HasMemory<float>(BitNetSession.QKVKeyKey);
            bool hasValue = session.HasMemory<float>(BitNetSession.QKVValueKey);
            if (!hasQuery && !hasKey && !hasValue)
            {
                throw new InvalidOperationException("Session does not contain QKV projection output.");
            }

            if (!hasQuery || !hasKey || !hasValue)
            {
                throw new InvalidOperationException("Session does not contain complete QKV projection output.");
            }

            ForwardCore(session.QKVQuery, session.QKVKey, session.QKVValue, session.AttentionSubNorm, session.AttentionOutput);
        }

        private void ForwardCore(ReadOnlyMemory<float> query, ReadOnlyMemory<float> key, ReadOnlyMemory<float> value, Memory<float> subNorm, Memory<float> output)
        {
            if (EnableCache)
            {
                PackedProjectionWeights cachedOutputWeights = EnsureCachedOutputWeights();
                float[] cachedOutputScaleValues = TryEnsureCachedOutputScaleValues().ToArray();
                float[] cachedOutputBiasValues = TryEnsureCachedOutputBiasValues().ToArray();
                opProvider.ForwardAttention(query, key, value, EnsureCachedSubNormWeights().AsMemory(), model.Config!.AttentionLayerNormRmsEpsilon, cachedOutputWeights.PackedWeights, cachedOutputWeights.Scale, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.KeyValueProjectionSize), checked((int)model.Config.AttentionHeadCount), checked((int)model.Config.AttentionKeyValueHeadCount), checked((int)model.Config.AttentionHeadDimension), subNorm, output, cachedOutputScaleValues, cachedOutputBiasValues);
                return;
            }

            float[] subNormWeights;
            using (IMemoryOwner<byte> subNormTensorData = model.ReadTensorData(subNormTensor))
            {
                subNormWeights = new float[checked((int)model.Config!.EmbeddingLength)];
                FillFloatValues(subNormTensorData.Memory.Span, subNormTensor.TensorType, subNormWeights, "Attention sub-norm");
            }

            PackedProjectionWeights outputWeights;
            using (IMemoryOwner<byte> outputTensorData = model.ReadTensorData(outputTensor))
            {
                outputWeights = ParsePackedWeights(outputTensorData.Memory, outputTensor, "Packed attention");
            }

            float[] outputScaleValues = [];
            if (outputScaleTensor is not null)
            {
                using IMemoryOwner<byte> outputScaleTensorData = model.ReadTensorData(outputScaleTensor);
                outputScaleValues = new float[1];
                FillFloatValues(outputScaleTensorData.Memory.Span, outputScaleTensor.TensorType, outputScaleValues, "Attention output scale");
            }

            float[] outputBiasValues = [];
            if (outputBiasTensor is not null)
            {
                using IMemoryOwner<byte> outputBiasTensorData = model.ReadTensorData(outputBiasTensor);
                outputBiasValues = new float[output.Length];
                FillFloatValues(outputBiasTensorData.Memory.Span, outputBiasTensor.TensorType, outputBiasValues, "Attention output bias");
            }

            opProvider.ForwardAttention(query, key, value, subNormWeights, model.Config!.AttentionLayerNormRmsEpsilon, outputWeights.PackedWeights, outputWeights.Scale, checked((int)model.Config.EmbeddingLength), checked((int)model.Config.KeyValueProjectionSize), checked((int)model.Config.AttentionHeadCount), checked((int)model.Config.AttentionKeyValueHeadCount), checked((int)model.Config.AttentionHeadDimension), subNorm, output, outputScaleValues, outputBiasValues);
        }

        private void ValidateOutputScaleTensor()
        {
            if (outputScaleTensor is null)
            {
                return;
            }

            if (outputScaleTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && outputScaleTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Attention output scale tensor type '{outputScaleTensor.TensorType}' is not supported.");
            }

            int actualLength = GetElementCount(outputScaleTensor.Dimensions);
            if (actualLength != 1)
            {
                throw new InvalidOperationException("Attention output scale tensor dimensions do not match the expected scalar shape.");
            }
        }

        private void ValidateOutputBiasTensor()
        {
            if (outputBiasTensor is null)
            {
                return;
            }

            if (outputBiasTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && outputBiasTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Attention output bias tensor type '{outputBiasTensor.TensorType}' is not supported.");
            }

            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            int actualLength = GetElementCount(outputBiasTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("Attention output bias tensor dimensions do not match the loaded model configuration.");
            }
        }

        private static float RoundTripThroughHalf(float value)
        {
            return (float)(Half)value;
        }

        private void ValidateProjectionShape(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            if (query.Length != embeddingLength)
            {
                throw new InvalidOperationException("Attention query length does not match the loaded model configuration.");
            }

            if (key.Length != keyValueLength)
            {
                throw new InvalidOperationException("Attention key length does not match the loaded model configuration.");
            }

            if (value.Length != keyValueLength)
            {
                throw new InvalidOperationException("Attention value length does not match the loaded model configuration.");
            }
        }

        private void ValidateSubNormTensor()
        {
            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            int actualLength = GetElementCount(subNormTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("Attention sub-norm tensor dimensions do not match the loaded model configuration.");
            }

            if (subNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && subNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Attention sub-norm tensor type '{subNormTensor.TensorType}' is not supported.");
            }
        }

        private void ValidateOutputTensor()
        {
            if (!outputTensor.IsQuantized)
            {
                throw new NotSupportedException($"Attention output tensor type '{outputTensor.TensorType}' is not supported.");
            }

            if (outputTensor.Dimensions.Count < 2)
            {
                throw new InvalidOperationException("Attention output tensor dimensions are incomplete.");
            }

            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int actualInputLength = checked((int)outputTensor.Dimensions[0]);
            int actualOutputLength = checked((int)outputTensor.Dimensions[1]);
            if (actualInputLength != embeddingLength || actualOutputLength != embeddingLength)
            {
                throw new InvalidOperationException("Attention output tensor dimensions do not match the loaded model configuration.");
            }
        }

        private float[] ReadSubNormWeights()
        {
            using var tensorData = model.ReadTensorData(subNormTensor);
            return subNormTensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"Attention sub-norm tensor type '{subNormTensor.TensorType}' is not supported."),
            };
        }

        private float[] EnsureCachedSubNormWeights()
        {
            return cachedSubNormWeights ??= ReadSubNormWeights();
        }

        private ReadOnlySpan<float> TryEnsureCachedOutputScaleValues()
        {
            return outputScaleTensor is null ? [] : cachedOutputScaleValues ??= ReadFloatTensor(outputScaleTensor);
        }

        private ReadOnlySpan<float> TryEnsureCachedOutputBiasValues()
        {
            return outputBiasTensor is null ? [] : cachedOutputBiasValues ??= ReadFloatTensor(outputBiasTensor);
        }

        private PackedProjectionWeights ReadPackedWeights(BitNetTensorInfo tensor)
        {
            using var tensorData = model.ReadTensorData(tensor);
            PackedProjectionWeights weights = ParsePackedWeights(tensorData.Memory, tensor, "Packed attention");
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

        private PackedProjectionWeights EnsureCachedOutputWeights()
        {
            return cachedOutputWeights ??= ReadPackedWeights(outputTensor);
        }

        private float[] ReadFloatTensor(BitNetTensorInfo tensor)
        {
            using var tensorData = model.ReadTensorData(tensor);
            return tensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"Attention tensor type '{tensor.TensorType}' is not supported."),
            };
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
                throw new InvalidOperationException("Attention sub-norm tensor dimensions are incomplete.");
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
