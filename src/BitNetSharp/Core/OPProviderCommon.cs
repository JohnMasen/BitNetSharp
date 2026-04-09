using System;
using System.Buffers;
using System.Numerics.Tensors;

namespace BitNetSharp.Core
{
    internal static class OPProviderCommon
    {
        private const float MinimumBitNetQuantizationMax = 0.00001f;

        internal static void ForwardQKVProjection(
            IOPProvider1 provider,
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
            (float activationScale, _) = QuantizeBitNetActivations(input.Span, quantizedValues.Span);

            provider.ProjectBitNetI2(quantizedValues, activationScale, queryPackedWeights, queryOutputLength, queryWeightScale, query, "QKV");
            provider.ProjectBitNetI2(quantizedValues, activationScale, keyPackedWeights, keyValueOutputLength, keyWeightScale, key, "QKV");
            provider.ProjectBitNetI2(quantizedValues, activationScale, valuePackedWeights, keyValueOutputLength, valueWeightScale, value, "QKV");
        }

        internal static void ForwardAttention(
            IOPProvider2 provider,
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
            BuildSingleTokenAttentionContext(provider, query, key, value, attentionContext, headCount, keyValueHeadCount, headDimension);

            provider.ForwardRmsNorm(attentionContext, subNormWeights[..attentionContext.Length], epsilon, subNorm);
            provider.ProjectBitNetI2(subNorm, outputPackedWeights, embeddingLength, outputWeightScale, output, "Attention");
            ApplyScale(output.Span[..embeddingLength], outputScaleValues.Span);
            ApplyBias(output.Span[..embeddingLength], outputBiasValues.Span);
        }

        internal static void ForwardFeedForward(
            IOPProvider2 provider,
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
            (float activationScale, _) = QuantizeBitNetActivations(input.Span, quantizedValues.Span);

            provider.ProjectBitNetI2(quantizedValues, activationScale, upPackedWeights, feedForwardLength, upWeightScale, up, "Feed-forward");
            provider.ProjectBitNetI2(quantizedValues, activationScale, gatePackedWeights, feedForwardLength, gateWeightScale, gate, "Feed-forward");
            ApplySquaredReluGate(gate, up, provider.ThreadCount);
            provider.ForwardRmsNorm(up, subNormWeights[..feedForwardLength], epsilon, subNormOutput);
            provider.ProjectBitNetI2(subNormOutput, downPackedWeights, embeddingLength, downWeightScale, output, "Feed-forward");
        }

        internal static void ForwardLmHeadCpu(ReadOnlyMemory<float> input, ReadOnlyMemory<Half> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output, int threadCount)
        {
            ValidateLmHeadArguments(input, rowLength, vocabularySize, output);
            ProjectLmHeadCpu(input.Span, embeddingWeights.Span, rowLength, output.Span[..vocabularySize], threadCount);
        }

        internal static void ForwardLmHeadTensor(ReadOnlyMemory<float> input, ReadOnlyMemory<Half> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output, int threadCount)
        {
            ValidateLmHeadArguments(input, rowLength, vocabularySize, output);
            ProjectLmHeadTensor(input.Span, embeddingWeights.Span, rowLength, output.Span[..vocabularySize], threadCount);
        }

        private static void ValidateLmHeadArguments(ReadOnlyMemory<float> input, int rowLength, int vocabularySize, Memory<float> output)
        {
            if (input.Length != rowLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            if (output.Length < vocabularySize)
            {
                throw new ArgumentException("Output length does not match the model vocabulary size.", nameof(output));
            }
        }

        private static void BuildSingleTokenAttentionContext(
            IOPProvider1 provider,
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
            if (provider.ThreadCount == 1 || headCount <= 1)
            {
                BuildSingleTokenAttentionContextRange(provider, query.Span, key.Span, value.Span, context.Span, groupSize, headDimension, scoreScale, 0, headCount);
                return;
            }

            ThreadHelper.ForEachRange(
                headCount,
                (startIndex, endIndex) => BuildSingleTokenAttentionContextRange(provider, query.Span, key.Span, value.Span, context.Span, groupSize, headDimension, scoreScale, startIndex, endIndex),
                provider.ThreadCount,
                sizeof(float));
        }

        private static void BuildSingleTokenAttentionContextRange(
            IOPProvider1 provider,
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
                provider.ForwardSoftmax(attentionScore, attentionWeight, "Attention");

                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    context[outputOffset + dimensionIndex] = RoundTripThroughHalf(value[sourceOffset + dimensionIndex] * attentionWeight[0]);
                }
            }
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

            ThreadHelper.ForEachRange(
                up.Length,
                (startIndex, endIndex) => ApplySquaredReluGateRange(gate.Span, up.Span, startIndex, endIndex),
                threads,
                sizeof(float));
        }

        private static void ApplySquaredReluGateRange(ReadOnlySpan<float> gate, Span<float> up, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                float relu = MathF.Max(gate[index], 0f);
                up[index] *= relu * relu;
            }
        }

        private static void ApplyScale(Span<float> output, ReadOnlySpan<float> scaleValues)
        {
            if (scaleValues.IsEmpty)
            {
                return;
            }

            float outputScale = scaleValues[0];
            for (int index = 0; index < output.Length; index++)
            {
                output[index] *= outputScale;
            }
        }

        private static void ApplyBias(Span<float> output, ReadOnlySpan<float> biasValues)
        {
            if (biasValues.IsEmpty)
            {
                return;
            }

            ValidateAddDestination(output, biasValues, output);
            for (int index = 0; index < output.Length; index++)
            {
                output[index] += biasValues[index];
            }
        }

        private static void ProjectLmHeadCpu(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int threads)
        {
            if (threads == 1 || output.Length <= 1)
            {
                ProjectLmHeadCpuRange(input, embeddingWeights, rowLength, output, 0, output.Length);
                return;
            }

            float[] inputBuffer = input.ToArray();
            Half[] embeddingWeightsBuffer = embeddingWeights.ToArray();
            float[] outputBuffer = new float[output.Length];
            ThreadHelper.ForEachRange(
                outputBuffer.AsSpan(),
                (startIndex, endIndex) => ProjectLmHeadCpuRange(inputBuffer, embeddingWeightsBuffer, rowLength, outputBuffer, startIndex, endIndex),
                threads);
            outputBuffer.AsSpan().CopyTo(output);
        }

        private static void ProjectLmHeadCpuRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int startIndex, int endIndex)
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

        private static void ProjectLmHeadTensor(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int threads)
        {
            if (threads == 1 || output.Length <= 1)
            {
                using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                ProjectLmHeadTensorRange(input, embeddingWeights, rowLength, output, 0, output.Length, rowOwner.Memory.Span[..rowLength]);
                return;
            }

            float[] inputBuffer = input.ToArray();
            Half[] embeddingWeightsBuffer = embeddingWeights.ToArray();
            float[] outputBuffer = new float[output.Length];
            ThreadHelper.ForEachRange(
                outputBuffer.AsSpan(),
                (startIndex, endIndex) =>
                {
                    using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                    ProjectLmHeadTensorRange(inputBuffer, embeddingWeightsBuffer, rowLength, outputBuffer, startIndex, endIndex, rowOwner.Memory.Span[..rowLength]);
                },
                threads);
            outputBuffer.AsSpan().CopyTo(output);
        }

        private static void ProjectLmHeadTensorRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int startIndex, int endIndex, Span<float> rowBuffer)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                int rowOffset = outputIndex * rowLength;
                ConvertHalfToSingle(embeddingWeights.Slice(rowOffset, rowLength), rowBuffer);
                output[outputIndex] = TensorPrimitives.Dot(input, rowBuffer);
            }
        }

        private static void ConvertHalfToSingle(ReadOnlySpan<Half> source, Span<float> destination)
        {
            for (int index = 0; index < source.Length; index++)
            {
                destination[index] = (float)source[index];
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

        internal static void ValidateAddDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (addend.Length < input.Length)
            {
                throw new ArgumentException("Addend length must be at least the input length.", nameof(addend));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        internal static void ValidateRmsNormDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (normWeights.Length < input.Length)
            {
                throw new ArgumentException("RMSNorm weight length must be at least the input length.", nameof(normWeights));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        internal static void ValidateProjectionDestination(int outputLength, Span<float> output)
        {
            if (output.Length < outputLength)
            {
                throw new ArgumentException("Output length must be at least the projection output length.", nameof(output));
            }
        }

        internal static void ValidateSoftmaxDestination(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        internal static void ValidateBitNetProjectionArguments(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (outputLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputLength));
            }

            if (input.Length % 4 != 0)
            {
                throw new ArgumentException("BitNet projection input length must be divisible by 4.", nameof(input));
            }

            int expectedPackedWeightByteCount = GetBitNetPackedWeightByteCount(input.Length, outputLength);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }

        internal static void ValidateBitNetProjectionArguments(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int outputLength)
        {
            if (quantizedValues.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(quantizedValues));
            }

            if (outputLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputLength));
            }

            if (quantizedValues.Length % 4 != 0)
            {
                throw new ArgumentException("BitNet projection input length must be divisible by 4.", nameof(quantizedValues));
            }

            int expectedPackedWeightByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, outputLength);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }

        internal static int GetBitNetPackedWeightByteCount(int inputLength, int outputLength)
        {
            return checked((inputLength * outputLength) / 4);
        }

        internal static float FinalizeBitNetMappedProjection(int mappedDot, float activationScale, float weightScale)
        {
            return (mappedDot / activationScale) * weightScale;
        }

        internal static float FinalizeBitNetMappedProjection(float mappedDot, float activationScale, float weightScale)
        {
            return (mappedDot / activationScale) * weightScale;
        }

        internal static int DecodeBitNetWeight(byte packedWeight, int groupIndex)
        {
            int weightCode = (packedWeight >> (6 - (groupIndex * 2))) & 0b_0000_0011;
            return weightCode == 0b_0000_0011 ? 0 : weightCode - 1;
        }

        internal static void ExpandBitNetRowWeights(ReadOnlySpan<byte> packedRow, Span<float> rowWeights)
        {
            const int PackedGroupWidth = 32;
            const int ActivationBlockWidth = 128;

            int packedOffset = 0;
            for (int activationBlockStart = 0; activationBlockStart < rowWeights.Length; activationBlockStart += ActivationBlockWidth)
            {
                int packedBlockByteCount = Math.Min(PackedGroupWidth, packedRow.Length - packedOffset);
                ReadOnlySpan<byte> packedBlock = packedRow.Slice(packedOffset, packedBlockByteCount);
                int remainingActivations = rowWeights.Length - activationBlockStart;
                int groupCount = Math.Min(4, (remainingActivations + PackedGroupWidth - 1) / PackedGroupWidth);

                for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
                {
                    int activationOffset = activationBlockStart + (groupIndex * PackedGroupWidth);
                    int groupActivationCount = Math.Min(PackedGroupWidth, rowWeights.Length - activationOffset);
                    Span<float> rowWeightGroup = rowWeights.Slice(activationOffset, groupActivationCount);
                    for (int packedIndex = 0; packedIndex < packedBlockByteCount && packedIndex < groupActivationCount; packedIndex++)
                    {
                        rowWeightGroup[packedIndex] = DecodeBitNetWeight(packedBlock[packedIndex], groupIndex);
                    }
                }

                packedOffset += packedBlockByteCount;
            }
        }

        internal static (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlySpan<float> input, Span<sbyte> quantizedValues)
        {
            if (quantizedValues.Length < input.Length)
            {
                throw new ArgumentException("Quantized output length must be at least the input length.", nameof(quantizedValues));
            }

            float maxAbs = MinimumBitNetQuantizationMax;
            for (int index = 0; index < input.Length; index++)
            {
                float absValue = MathF.Abs(input[index]);
                if (absValue > maxAbs)
                {
                    maxAbs = absValue;
                }
            }

            float activationScale = 127f / maxAbs;
            int activationSum = 0;
            for (int index = 0; index < input.Length; index++)
            {
                int quantizedValue = RoundBitNetQuantizedValue(input[index] * activationScale);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return (activationScale, activationSum);
        }

        private static int RoundBitNetQuantizedValue(float value)
        {
            int roundedValue = (int)MathF.Round(value, MidpointRounding.ToEven);
            return Math.Clamp(roundedValue, sbyte.MinValue, sbyte.MaxValue);
        }
    }
}
