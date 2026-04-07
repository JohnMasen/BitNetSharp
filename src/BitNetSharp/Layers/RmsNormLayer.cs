using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Applies RMS normalization to the embedding data stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class RmsNormLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo normTensor;
        private float[]? cachedNormWeights;
        private bool isInitialized;

        public RmsNormLayer(BitNetModel model, BitNetTensorInfo normTensor, bool enableCache = false, InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(normTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the RMSNorm layer can be created.");
            }

            this.model = model;
            this.normTensor = normTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();

            ValidateTensorShape();
            ValidateTensorType();
        }

        public bool EnableCache { get; }

        public InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedNormWeights();
            }

            isInitialized = true;
        }

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.SIMD, 1);
        }

        /// <summary>
        /// Applies RMSNorm to the embedding stored on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            float[] input = session.Embedding ?? throw new InvalidOperationException("Session does not contain embedding output.");
            session.RmsNorm = ForwardCore(input);
        }

        private float[] ForwardCore(ReadOnlySpan<float> input)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            if (input.Length != expectedLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            ReadOnlySpan<float> normWeights = EnableCache
                ? EnsureCachedNormWeights()
                : ReadNormWeights();

            if (normWeights.Length < input.Length)
            {
                throw new InvalidOperationException("RMSNorm weight length does not match the input length.");
            }

            normWeights = normWeights[..input.Length];
            int threads = InferenceConfig.ThreadCount;

            return InferenceConfig.Backend switch
            {
                InferenceBackend.CPU => MathHelper.ForwardRmsNormCpuStandard(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon, threads),
                InferenceBackend.SIMD => MathHelper.ForwardRmsNormSimd(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon, threads),
                InferenceBackend.Tensor => MathHelper.ForwardRmsNormTensor(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon, threads),
                _ => throw new NotSupportedException($"RMSNorm backend '{InferenceConfig.Backend}' is not implemented yet."),
            };
        }

        private void ValidateTensorShape()
        {
            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            int actualLength = GetElementCount(normTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("RMSNorm tensor dimensions do not match the loaded model configuration.");
            }
        }

        private void ValidateTensorType()
        {
            if (normTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && normTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"RMSNorm tensor type '{normTensor.TensorType}' is not supported.");
            }
        }

        private float[] ReadNormWeights()
        {
            using var tensorData = model.ReadTensorData(normTensor);
            return normTensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"RMSNorm tensor type '{normTensor.TensorType}' is not supported."),
            };
        }

        private float[] EnsureCachedNormWeights()
        {
            return cachedNormWeights ??= ReadNormWeights();
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
                throw new InvalidOperationException("RMSNorm tensor dimensions are incomplete.");
            }

            ulong total = 1;
            foreach (ulong dimension in dimensions)
            {
                total = checked(total * dimension);
            }

            return checked((int)total);
        }
    }
}
