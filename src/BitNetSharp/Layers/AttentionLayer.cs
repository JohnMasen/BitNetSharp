using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Builds the current-token attention output from the session QKV tensors, applies the attention sub-norm,
    /// and projects the result with the attention output weights.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class AttentionLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo subNormTensor;
        private readonly BitNetTensorInfo outputTensor;
        private readonly BitNetTensorInfo? outputScaleTensor;
        private readonly BitNetTensorInfo? outputBiasTensor;
        private float[]? cachedSubNormWeights;
        private PackedProjectionWeights? cachedOutputWeights;
        private float[]? cachedOutputScaleValues;
        private float[]? cachedOutputBiasValues;
        private bool isInitialized;

        /// <summary>
        /// Creates an attention layer that consumes QKV projection output, applies attention sub-norm,
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
        /// <param name="inferenceConfig">The backend and threading configuration for the layer.</param>
        /// <remarks>
        /// Some models, including the current baseline test GGUF, do not contain <c>attn_output.scale</c>
        /// or <c>attn_output.bias</c>. Callers may pass <see langword="null"/> for those optional tensors in
        /// that case, but models that do provide them should forward both tensors so the runtime order remains
        /// <c>wo -&gt; optional scale -&gt; optional bias</c>.
        /// </remarks>
        public AttentionLayer(
            BitNetModel model,
            BitNetTensorInfo subNormTensor,
            BitNetTensorInfo outputTensor,
            BitNetTensorInfo? outputScaleTensor = null,
            BitNetTensorInfo? outputBiasTensor = null,
            bool enableCache = false,
            InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(subNormTensor);
            ArgumentNullException.ThrowIfNull(outputTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the attention layer can be created.");
            }

            this.model = model;
            this.subNormTensor = subNormTensor;
            this.outputTensor = outputTensor;
            this.outputScaleTensor = outputScaleTensor;
            this.outputBiasTensor = outputBiasTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();

            ValidateSubNormTensor();
            ValidateOutputTensor();
            ValidateOutputScaleTensor();
            ValidateOutputBiasTensor();
        }

        public bool EnableCache { get; }

        public InferenceConfig InferenceConfig { get; }

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

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
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

            QKVProjectionOutput projection = session.QKVProjection ?? throw new InvalidOperationException("Session does not contain QKV projection output.");
            (float[] subNorm, float[] output) = ForwardCore(projection);
            session.AttentionSubNorm = subNorm;
            session.AttentionOutput = output;
        }

        private (float[] SubNorm, float[] Output) ForwardCore(QKVProjectionOutput projection)
        {
            ArgumentNullException.ThrowIfNull(projection);

            ValidateProjectionShape(projection);
            float[] attentionContext = BuildSingleTokenAttentionContext(projection.Value);
            ReadOnlySpan<float> subNormWeights = EnableCache
                ? EnsureCachedSubNormWeights()
                : ReadSubNormWeights();

            if (subNormWeights.Length < attentionContext.Length)
            {
                throw new InvalidOperationException("Attention sub-norm weight length does not match the attention context length.");
            }

            int threads = InferenceConfig.ThreadCount;
            float[] subNorm = InferenceConfig.Backend switch
            {
                InferenceBackend.CPU => MathHelper.ForwardRmsNormCpuStandard(attentionContext, subNormWeights[..attentionContext.Length], model.Config!.AttentionLayerNormRmsEpsilon, threads),
                _ => throw new NotSupportedException($"Attention backend '{InferenceConfig.Backend}' is not implemented yet."),
            };

            PackedProjectionWeights outputWeights = EnableCache ? EnsureCachedOutputWeights() : ReadPackedWeights(outputTensor);
            float[] output = InferenceConfig.Backend switch
            {
                InferenceBackend.CPU => MathHelper.ProjectBitNetI2Cpu(subNorm, outputWeights.PackedWeights, checked((int)model.Config!.EmbeddingLength), outputWeights.Scale, threads),
                _ => throw new NotSupportedException($"Attention backend '{InferenceConfig.Backend}' is not implemented yet."),
            };

            ReadOnlySpan<float> outputScaleValues = EnableCache
                ? TryEnsureCachedOutputScaleValues()
                : TryReadOptionalFloatTensor(outputScaleTensor);
            ApplyOptionalOutputScale(output, outputScaleValues);

            ReadOnlySpan<float> outputBiasValues = EnableCache
                ? TryEnsureCachedOutputBiasValues()
                : TryReadOptionalFloatTensor(outputBiasTensor);
            ApplyOptionalOutputBias(output, outputBiasValues);

            return (subNorm, output);
        }

        private float[] BuildSingleTokenAttentionContext(ReadOnlySpan<float> value)
        {
            int headCount = checked((int)model.Config!.AttentionHeadCount);
            int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
            int headDimension = checked((int)model.Config.AttentionHeadDimension);
            if (headCount % keyValueHeadCount != 0)
            {
                throw new InvalidOperationException("Attention head count must be divisible by the key/value head count.");
            }

            int expectedValueLength = checked(keyValueHeadCount * headDimension);
            if (value.Length != expectedValueLength)
            {
                throw new InvalidOperationException("Attention value length does not match the loaded model configuration.");
            }

            int groupSize = headCount / keyValueHeadCount;
            float[] context = new float[checked(headCount * headDimension)];
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                int sourceHeadIndex = headIndex / groupSize;
                int sourceOffset = sourceHeadIndex * headDimension;
                int outputOffset = headIndex * headDimension;
                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    context[outputOffset + dimensionIndex] = RoundTripThroughHalf(value[sourceOffset + dimensionIndex]);
                }
            }

            return context;
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

        private void ValidateProjectionShape(QKVProjectionOutput projection)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            if (projection.Query.Length != embeddingLength)
            {
                throw new InvalidOperationException("Attention query length does not match the loaded model configuration.");
            }

            if (projection.Key.Length != keyValueLength)
            {
                throw new InvalidOperationException("Attention key length does not match the loaded model configuration.");
            }

            if (projection.Value.Length != keyValueLength)
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
            ReadOnlySpan<byte> tensorBytes = tensorData.Memory.Span;

            int packedWeightByteCount = checked(((int)tensor.Dimensions[0] * (int)tensor.Dimensions[1]) / 4);
            if (tensorBytes.Length < packedWeightByteCount + sizeof(float))
            {
                throw new InvalidOperationException($"Packed attention tensor '{tensor.Name}' is incomplete.");
            }

            byte[] packedWeights = tensorBytes[..packedWeightByteCount].ToArray();
            float scale = MemoryMarshal.Read<float>(tensorBytes.Slice(packedWeightByteCount, sizeof(float)));
            return new PackedProjectionWeights(packedWeights, scale);
        }

        private PackedProjectionWeights EnsureCachedOutputWeights()
        {
            return cachedOutputWeights ??= ReadPackedWeights(outputTensor);
        }

        private ReadOnlySpan<float> TryReadOptionalFloatTensor(BitNetTensorInfo? tensor)
        {
            return tensor is null ? [] : ReadFloatTensor(tensor);
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

        private static void ApplyOptionalOutputScale(float[] output, ReadOnlySpan<float> outputScaleValues)
        {
            if (outputScaleValues.IsEmpty)
            {
                return;
            }

            float outputScale = outputScaleValues[0];
            for (int index = 0; index < output.Length; index++)
            {
                output[index] *= outputScale;
            }
        }

        private static void ApplyOptionalOutputBias(float[] output, ReadOnlySpan<float> outputBiasValues)
        {
            if (outputBiasValues.IsEmpty)
            {
                return;
            }

            for (int index = 0; index < output.Length; index++)
            {
                output[index] += outputBiasValues[index];
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
            for (int index = 0; index < source.Length; index++)
            {
                values[index] = (float)source[index];
            }

            return values;
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

        private sealed record PackedProjectionWeights(byte[] PackedWeights, float Scale);
    }
}
