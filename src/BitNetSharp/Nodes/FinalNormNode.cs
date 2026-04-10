using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Applies RMS normalization to the final hidden state stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class FinalNormNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo outputNormTensor;
        private readonly IOPProvider opProvider;
        private float[]? cachedNormWeights;
        private bool isInitialized;

        public FinalNormNode(BitNetModel model, bool enableCache = false, Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the final norm node can be created.");
            }

            this.model = model;
            outputNormTensor = model.GlobalTensors?.OutputNorm ?? throw new InvalidOperationException("The model must be loaded before the final norm node can be created.");
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig;
            opProvider = InferenceConfig.OPProvider;

            ValidateTensorShape();
            ValidateTensorType();
        }

        public bool EnableCache { get; }

        public Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedNormWeights();
            }

            isInitialized = true;
        }

        /// <summary>
        /// Applies RMSNorm to the final hidden state stored on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.EmbeddingKey))
            {
                throw new InvalidOperationException("Session does not contain final hidden state.");
            }

            ReadOnlyMemory<float> input = session.Embedding;
            Memory<float> output = session.FinalNormOutput;
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            if (input.Length != expectedLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            int requiredLength = input.Length;
            if (EnableCache)
            {
                opProvider.ForwardRmsNorm(input, EnsureCachedNormWeights().AsMemory(0, requiredLength), model.Config!.AttentionLayerNormRmsEpsilon, output);
                return;
            }

            using var tensorData = model.ReadTensorData(outputNormTensor);
            using IMemoryOwner<float> normWeightsOwner = MemoryPool<float>.Shared.Rent(requiredLength);
            Memory<float> normWeights = normWeightsOwner.Memory[..requiredLength];
            FillFloatValues(tensorData.Memory.Span, outputNormTensor.TensorType, normWeights.Span);
            opProvider.ForwardRmsNorm(input, normWeights, model.Config!.AttentionLayerNormRmsEpsilon, output);
        }

        private void ValidateTensorShape()
        {
            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            int actualLength = GetElementCount(outputNormTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("Final norm tensor dimensions do not match the loaded model configuration.");
            }
        }

        private void ValidateTensorType()
        {
            if (outputNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && outputNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Final norm tensor type '{outputNormTensor.TensorType}' is not supported.");
            }
        }

        private float[] ReadNormWeights()
        {
            using var tensorData = model.ReadTensorData(outputNormTensor);
            return outputNormTensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"Final norm tensor type '{outputNormTensor.TensorType}' is not supported."),
            };
        }

        private float[] EnsureCachedNormWeights()
        {
            return cachedNormWeights ??= ReadNormWeights();
        }

        private static void FillFloatValues(ReadOnlySpan<byte> source, GGUFTensorType tensorType, Span<float> destination)
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
                    throw new NotSupportedException($"Final norm tensor type '{tensorType}' is not supported.");
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
                throw new InvalidOperationException("Final norm tensor dimensions are incomplete.");
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
