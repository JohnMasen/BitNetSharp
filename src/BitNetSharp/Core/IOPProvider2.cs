using System.Buffers;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides backend-specific higher-level inference operations.
    /// </summary>
    /// <remarks>
    /// This currently extends <see cref="IOPProvider1"/> so one provider instance can serve both node-level
    /// orchestration and lower-level kernels. If the two layers need to diverge later, they can still be split.
    /// </remarks>
    public interface IOPProvider2 : IOPProvider1
    {
        void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output);

        void ForwardQKVProjection(
            ReadOnlyMemory<float> input,
            ReadOnlyMemory<byte> queryPackedWeights,
            float queryWeightScale,
            ReadOnlyMemory<byte> keyPackedWeights,
            float keyWeightScale,
            ReadOnlyMemory<byte> valuePackedWeights,
            float valueWeightScale,
            int queryOutputLength,
            int keyValueOutputLength,
            Memory<float> query,
            Memory<float> key,
            Memory<float> value)
        {
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedValues);

            ProjectBitNetI2(quantizedValues, activationScale, queryPackedWeights, queryOutputLength, queryWeightScale, query);
            ProjectBitNetI2(quantizedValues, activationScale, keyPackedWeights, keyValueOutputLength, keyWeightScale, key);
            ProjectBitNetI2(quantizedValues, activationScale, valuePackedWeights, keyValueOutputLength, valueWeightScale, value);
        }

        void ForwardAttention(
            ReadOnlyMemory<float> query,
            ReadOnlyMemory<float> key,
            ReadOnlyMemory<float> value,
            ReadOnlyMemory<float> subNormWeights,
            float epsilon,
            ReadOnlyMemory<byte> outputPackedWeights,
            float outputWeightScale,
            int embeddingLength,
            int keyValueLength,
            int headCount,
            int keyValueHeadCount,
            int headDimension,
            Memory<float> subNorm,
            Memory<float> output,
            ReadOnlyMemory<float> outputScaleValues = default,
            ReadOnlyMemory<float> outputBiasValues = default)
        {
            ValidateAttentionProjection(query.Span, key.Span, value.Span, embeddingLength, keyValueLength);

            using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
            Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
            BuildSingleTokenAttentionContext(query, key, value, attentionContext, headCount, keyValueHeadCount, headDimension);

            ForwardRmsNorm(attentionContext, subNormWeights[..attentionContext.Length], epsilon, subNorm);
            ProjectBitNetI2(subNorm, outputPackedWeights, embeddingLength, outputWeightScale, output);
            ApplyScale(output[..embeddingLength], outputScaleValues);
            ApplyBias(output[..embeddingLength], outputBiasValues);
        }

        void ForwardFeedForward(
            ReadOnlyMemory<float> input,
            ReadOnlyMemory<float> subNormWeights,
            float epsilon,
            ReadOnlyMemory<byte> gatePackedWeights,
            float gateWeightScale,
            ReadOnlyMemory<byte> upPackedWeights,
            float upWeightScale,
            ReadOnlyMemory<byte> downPackedWeights,
            float downWeightScale,
            int embeddingLength,
            int feedForwardLength,
            Memory<float> subNormOutput,
            Memory<float> output)
        {
            if (input.Length != embeddingLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            if (subNormOutput.Length < feedForwardLength)
            {
                throw new ArgumentException("Feed-forward sub-norm output length does not match the model feed-forward length.", nameof(subNormOutput));
            }

            if (output.Length < embeddingLength)
            {
                throw new ArgumentException("Feed-forward output length does not match the model embedding length.", nameof(output));
            }

            using IMemoryOwner<float> upOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<float> gateOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<float> up = upOwner.Memory[..feedForwardLength];
            Memory<float> gate = gateOwner.Memory[..feedForwardLength];
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedValues);

            ProjectBitNetI2(quantizedValues, activationScale, upPackedWeights, feedForwardLength, upWeightScale, up);
            ProjectBitNetI2(quantizedValues, activationScale, gatePackedWeights, feedForwardLength, gateWeightScale, gate);
            ApplySquaredReluGate(gate, up);
            ForwardRmsNorm(up, subNormWeights[..feedForwardLength], epsilon, subNormOutput);
            ProjectBitNetI2(subNormOutput, downPackedWeights, embeddingLength, downWeightScale, output);
        }

        void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output);

        private void BuildSingleTokenAttentionContext(
            ReadOnlyMemory<float> query,
            ReadOnlyMemory<float> key,
            ReadOnlyMemory<float> value,
            Memory<float> context,
            int headCount,
            int keyValueHeadCount,
            int headDimension)
        {
            if (headCount % keyValueHeadCount != 0)
            {
                throw new InvalidOperationException("Attention head count must be divisible by the key/value head count.");
            }

            int groupSize = headCount / keyValueHeadCount;
            float scoreScale = 1f / MathF.Sqrt(headDimension);
            if (ThreadCount == 1 || headCount <= 1)
            {
                BuildSingleTokenAttentionContextRange(query.Span, key.Span, value.Span, context.Span, groupSize, headDimension, scoreScale, 0, headCount);
                return;
            }

            ThreadHelper.ForEachRange(
                headCount,
                (startIndex, endIndex) => BuildSingleTokenAttentionContextRange(query.Span, key.Span, value.Span, context.Span, groupSize, headDimension, scoreScale, startIndex, endIndex),
                ThreadCount,
                sizeof(float));
        }

        private void BuildSingleTokenAttentionContextRange(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> key,
            ReadOnlySpan<float> value,
            Span<float> context,
            int groupSize,
            int headDimension,
            float scoreScale,
            int startHeadIndex,
            int endHeadIndex)
        {
            Span<float> attentionScore = stackalloc float[1];
            Span<float> attentionWeight = stackalloc float[1];
            for (int headIndex = startHeadIndex; headIndex < endHeadIndex; headIndex++)
            {
                int sourceHeadIndex = headIndex / groupSize;
                int queryOffset = headIndex * headDimension;
                int keyOffset = sourceHeadIndex * headDimension;
                int sourceOffset = sourceHeadIndex * headDimension;
                int outputOffset = headIndex * headDimension;

                attentionScore[0] = ComputeScaledAttentionScore(query, queryOffset, key, keyOffset, headDimension, scoreScale);
                ForwardSoftmax(attentionScore, attentionWeight);

                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    context[outputOffset + dimensionIndex] = RoundTripThroughHalf(value[sourceOffset + dimensionIndex] * attentionWeight[0]);
                }
            }
        }

        private void ApplySquaredReluGate(ReadOnlyMemory<float> gate, Memory<float> up)
        {
            if (gate.Length != up.Length)
            {
                throw new ArgumentException("Feed-forward gate length must match the up projection length.", nameof(gate));
            }

            if (ThreadCount == 1 || up.Length <= 1)
            {
                ApplySquaredReluGateRange(gate.Span, up.Span, 0, up.Length);
                return;
            }

            ThreadHelper.ForEachRange(
                up.Length,
                (startIndex, endIndex) => ApplySquaredReluGateRange(gate.Span, up.Span, startIndex, endIndex),
                ThreadCount,
                sizeof(float));
        }

        private static void ValidateAttentionProjection(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value, int embeddingLength, int keyValueLength)
        {
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

        private static void ApplySquaredReluGateRange(ReadOnlySpan<float> gate, Span<float> up, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                float relu = MathF.Max(gate[index], 0f);
                up[index] *= relu * relu;
            }
        }

        private void ApplyScale(Memory<float> output, ReadOnlyMemory<float> scaleValues)
        {
            if (scaleValues.IsEmpty)
            {
                return;
            }

            float outputScale = scaleValues.Span[0];
            if (ThreadCount == 1 || output.Length <= 1)
            {
                ApplyScaleRange(output.Span, outputScale, 0, output.Length);
                return;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                ApplyScaleRange(output.Span, outputScale, startIndex, endIndex), ThreadCount, sizeof(float));
        }

        private void ApplyBias(Memory<float> output, ReadOnlyMemory<float> biasValues)
        {
            if (biasValues.IsEmpty)
            {
                return;
            }

            ValidationHelper.ValidateAddDestination(output.Span, biasValues.Span, output.Span);
            if (ThreadCount == 1 || output.Length <= 1)
            {
                ApplyBiasRange(output.Span, biasValues.Span, 0, output.Length);
                return;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                ApplyBiasRange(output.Span, biasValues.Span, startIndex, endIndex), ThreadCount, sizeof(float));
        }

        private static void ApplyScaleRange(Span<float> output, float outputScale, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] *= outputScale;
            }
        }

        private static void ApplyBiasRange(Span<float> output, ReadOnlySpan<float> biasValues, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] += biasValues[index];
            }
        }

        private static float ComputeScaledAttentionScore(ReadOnlySpan<float> query, int queryOffset, ReadOnlySpan<float> key, int keyOffset, int headDimension, float scoreScale)
        {
            float dotProduct = 0f;
            for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
            {
                dotProduct += query[queryOffset + dimensionIndex] * key[keyOffset + dimensionIndex];
            }

            return dotProduct * scoreScale;
        }

        private static float RoundTripThroughHalf(float value)
        {
            return (float)(Half)value;
        }

        private (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlyMemory<float> input, Memory<sbyte> quantizedValues)
        {
            return Backend switch
            {
                global::BitNetSharp.Nodes.InferenceBackend.CPU => CPUDefaultOPProvider.QuantizeBitNetActivations(input, quantizedValues, ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.Tensor => CPUTensorOPProvider.QuantizeBitNetActivations(input, quantizedValues, ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.SIMD => CPUSimdOPProvider.QuantizeBitNetActivations(input, quantizedValues, ThreadCount),
                _ => throw new NotSupportedException($"Backend '{Backend}' is not implemented yet."),
            };
        }
    }
}
