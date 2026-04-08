using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Projects the final normalized hidden state stored on a <see cref="BitNetSession"/> into vocabulary logits using the tied token embedding weights.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class LmHeadLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo tokenEmbedding;
        private Half[]? cachedEmbeddingWeights;
        private bool isInitialized;

        public LmHeadLayer(BitNetModel model, bool enableCache = false, InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the LM head layer can be created.");
            }

            this.model = model;
            tokenEmbedding = model.GlobalTensors?.TokenEmbedding ?? throw new InvalidOperationException("The model must be loaded before the LM head layer can be created.");
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
                _ = EnsureCachedEmbeddingWeights();
            }

            isInitialized = true;
        }

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
        }

        /// <summary>
        /// Projects the final normalized hidden state on the session into vocabulary logits.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.FinalNormOutputKey))
            {
                throw new InvalidOperationException("Session does not contain final norm output.");
            }

            ForwardCore(session.FinalNormOutput, session.Logits);
        }

        private void ForwardCore(ReadOnlyMemory<float> input, Memory<float> output)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int vocabularySize = checked((int)model.Config.VocabularySize);
            if (input.Length != embeddingLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            if (output.Length < vocabularySize)
            {
                throw new ArgumentException("Output length does not match the model vocabulary size.", nameof(output));
            }

            if (EnableCache)
            {
                ExecuteForward(input.Span, EnsureCachedEmbeddingWeights(), output.Span[..vocabularySize]);
                return;
            }

            using var tensorData = model.ReadTensorData(tokenEmbedding);
            ReadOnlySpan<Half> embeddingWeights = MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span);
            ExecuteForward(input.Span, embeddingWeights, output.Span[..vocabularySize]);
        }

        private void ExecuteForward(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, Span<float> output)
        {
            int rowLength = checked((int)model.Config!.EmbeddingLength);
            switch (InferenceConfig.Backend)
            {
                case InferenceBackend.CPU:
                    ProjectCpu(input, embeddingWeights, rowLength, output, InferenceConfig.ThreadCount);
                    return;
                case InferenceBackend.Tensor:
                    ProjectTensor(input, embeddingWeights, rowLength, output, InferenceConfig.ThreadCount);
                    return;
                case InferenceBackend.SIMD:
                    ProjectCpu(input, embeddingWeights, rowLength, output, InferenceConfig.ThreadCount);
                    return;
                default:
                    throw new NotSupportedException($"LM head backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
        }

        private void ValidateTensorShape()
        {
            if (tokenEmbedding.Dimensions.Count < 2)
            {
                throw new InvalidOperationException("LM head tensor dimensions are incomplete.");
            }

            int expectedRowLength = checked((int)model.Config!.EmbeddingLength);
            int expectedRowCount = checked((int)model.Config.VocabularySize);
            int actualRowLength = checked((int)tokenEmbedding.Dimensions[0]);
            int actualRowCount = checked((int)tokenEmbedding.Dimensions[1]);
            if (actualRowLength != expectedRowLength || actualRowCount != expectedRowCount)
            {
                throw new InvalidOperationException("LM head tensor dimensions do not match the loaded model configuration.");
            }
        }

        private void ValidateTensorType()
        {
            if (tokenEmbedding.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"LM head tensor type '{tokenEmbedding.TensorType}' is not supported.");
            }
        }

        private Half[] ReadEmbeddingWeights()
        {
            using var tensorData = model.ReadTensorData(tokenEmbedding);
            return MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span).ToArray();
        }

        private Half[] EnsureCachedEmbeddingWeights()
        {
            return cachedEmbeddingWeights ??= ReadEmbeddingWeights();
        }

        private static void ProjectCpu(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int threads)
        {
            if (threads == 1 || output.Length <= 1)
            {
                ProjectCpuRange(input, embeddingWeights, rowLength, output, 0, output.Length);
                return;
            }

            float[] inputBuffer = input.ToArray();
            Half[] embeddingWeightsBuffer = embeddingWeights.ToArray();
            float[] outputBuffer = new float[output.Length];
            ThreadHelper.ForEachRange(outputBuffer.AsSpan(), (startIndex, endIndex) =>
                ProjectCpuRange(inputBuffer, embeddingWeightsBuffer, rowLength, outputBuffer, startIndex, endIndex), threads);
            outputBuffer.AsSpan().CopyTo(output);
        }

        private static void ProjectCpuRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                int rowOffset = outputIndex * rowLength;
                float sum = 0f;
                for (int inputIndex = 0; inputIndex < rowLength; inputIndex++)
                {
                    sum += input[inputIndex] * (float)embeddingWeights[rowOffset + inputIndex];
                }

                output[outputIndex] = sum;
            }
        }

        private static void ProjectTensor(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int threads)
        {
            if (threads == 1 || output.Length <= 1)
            {
                using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                ProjectTensorRange(input, embeddingWeights, rowLength, output, 0, output.Length, rowOwner.Memory.Span[..rowLength]);
                return;
            }

            float[] inputBuffer = input.ToArray();
            Half[] embeddingWeightsBuffer = embeddingWeights.ToArray();
            float[] outputBuffer = new float[output.Length];
            ThreadHelper.ForEachRange(outputBuffer.AsSpan(), (startIndex, endIndex) =>
            {
                using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                ProjectTensorRange(inputBuffer, embeddingWeightsBuffer, rowLength, outputBuffer, startIndex, endIndex, rowOwner.Memory.Span[..rowLength]);
            }, threads);
            outputBuffer.AsSpan().CopyTo(output);
        }

        private static void ProjectTensorRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int startIndex, int endIndex, Span<float> rowBuffer)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                int rowOffset = outputIndex * rowLength;
                ConvertHalfToSingle(embeddingWeights.Slice(rowOffset, rowLength), rowBuffer);
                output[outputIndex] = TensorPrimitives.Dot(input, rowBuffer);
            }
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The layer must be initialized by calling Init before Forward.");
            }
        }

        private static void ConvertHalfToSingle(ReadOnlySpan<Half> source, Span<float> destination)
        {
            for (int index = 0; index < source.Length; index++)
            {
                destination[index] = (float)source[index];
            }
        }
    }
}
