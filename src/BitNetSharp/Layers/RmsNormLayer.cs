using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    public sealed class RmsNormLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo normTensor;
        private float[]? cachedNormWeights;

        public RmsNormLayer(BitNetModel model, BitNetTensorInfo normTensor, RmsNormBackend backend = RmsNormBackend.CPUStandard, bool enableCache = false)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(normTensor);

            if (backend != RmsNormBackend.CPUStandard && backend != RmsNormBackend.Tensor && backend != RmsNormBackend.SIMD)
            {
                throw new NotSupportedException($"RMSNorm backend '{backend}' is not implemented yet.");
            }

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the RMSNorm layer can be created.");
            }

            this.model = model;
            this.normTensor = normTensor;
            Backend = backend;
            EnableCache = enableCache;

            ValidateTensorShape();
            ValidateTensorType();
        }

        public RmsNormBackend Backend { get; }

        public bool EnableCache { get; }

        /// <summary>
        /// Applies RMSNorm to the provided hidden state using the configured norm weights.
        /// </summary>
        public float[] Forward(ReadOnlySpan<float> input)
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

            return Backend switch
            {
                RmsNormBackend.CPUStandard => MathHelper.ForwardRmsNormCpuStandard(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon),
                RmsNormBackend.SIMD => MathHelper.ForwardRmsNormSimd(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon),
                RmsNormBackend.Tensor => MathHelper.ForwardRmsNormTensor(input, normWeights, model.Config.AttentionLayerNormRmsEpsilon),
                _ => throw new NotSupportedException($"RMSNorm backend '{Backend}' is not implemented yet."),
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
