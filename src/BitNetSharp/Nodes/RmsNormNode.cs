using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Applies RMS normalization to the embedding data stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class RmsNormNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo normTensor;
        private readonly IOPProvider opProvider;
        private float[]? cachedNormWeights;
        private bool isInitialized;

        public RmsNormNode(BitNetModel model, BitNetTensorInfo normTensor, bool enableCache = false, Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(normTensor);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the RMSNorm node can be created.");
            }

            this.model = model;
            this.normTensor = normTensor;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
            opProvider = InferenceConfig.Backend switch
            {
                Nodes.InferenceBackend.CPU => new CPUDefaultOPProvider(InferenceConfig.ThreadCount),
                Nodes.InferenceBackend.Tensor => new CPUTensorOPProvider(InferenceConfig.ThreadCount),
                Nodes.InferenceBackend.SIMD => new CPUSimdOPProvider(InferenceConfig.ThreadCount),
                _ => throw new NotSupportedException($"Backend '{InferenceConfig.Backend}' is not implemented yet."),
            };

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

        private static Nodes.InferenceConfig CreateDefaultInferenceConfig()
        {
            return new Nodes.InferenceConfig(Nodes.InferenceBackend.SIMD, 1);
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

            if (!session.HasMemory<float>(BitNetSession.EmbeddingKey))
            {
                throw new InvalidOperationException("Session does not contain embedding output.");
            }

            ReadOnlyMemory<float> input = session.Embedding;
            Memory<float> output = session.RmsNorm;
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

            using var tensorData = model.ReadTensorData(normTensor);
            using IMemoryOwner<float> normWeightsOwner = MemoryPool<float>.Shared.Rent(requiredLength);
            Memory<float> normWeights = normWeightsOwner.Memory[..requiredLength];
            FillFloatValues(tensorData.Memory.Span, normTensor.TensorType, normWeights.Span);
            opProvider.ForwardRmsNorm(input, normWeights, model.Config!.AttentionLayerNormRmsEpsilon, output);
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
                    throw new NotSupportedException($"RMSNorm tensor type '{tensorType}' is not supported.");
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
