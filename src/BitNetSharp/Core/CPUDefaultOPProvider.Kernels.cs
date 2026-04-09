using System;
using System.Buffers;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    public sealed partial class CPUDefaultOPProvider
    {
        private void ExecuteForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            double inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquare(inputSpan, epsilon);
            if (ThreadCount != 1)
            {
                ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                    FillRmsNormRange(input.Span, normWeights.Span, inverseRootMeanSquare, output.Span, startIndex, endIndex), ThreadCount);

                return;
            }

            FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
        }

        private void ExecuteAdd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> addendSpan = addend.Span;
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateAddDestination(inputSpan, addendSpan, outputSpan);

            if (ThreadCount == 1 || input.Length <= 1)
            {
                FillAddRange(inputSpan, addendSpan, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillAddRange(input.Span, addend.Span, output.Span, startIndex, endIndex), ThreadCount, sizeof(float));
        }

        private void ExecuteForwardSoftmax(ReadOnlySpan<float> input, Span<float> output)
        {
            OPProviderCommon.ValidateSoftmaxDestination(input, output);

            if (ThreadCount == 1 || input.Length <= 1)
            {
                ForwardSoftmaxCore(input, output);
                return;
            }

            using IMemoryOwner<float> inputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> outputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Memory<float> inputMemory = inputOwner.Memory[..input.Length];
            Memory<float> outputMemory = outputOwner.Memory[..input.Length];
            input.CopyTo(inputMemory.Span);
            ExecuteForwardSoftmaxMemory(inputMemory, outputMemory);
            outputMemory.Span.CopyTo(output);
        }

        private void ExecuteProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            OPProviderCommon.ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            OPProviderCommon.ValidateProjectionDestination(outputLength, output.Span);

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = OPProviderCommon.QuantizeBitNetActivations(input.Span, quantizedValues.Span);
            ExecuteProjectBitNetI2(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output);
        }

        private void ExecuteProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            OPProviderCommon.ValidateBitNetProjectionArguments(quantizedValues.Span, packedWeights.Span, outputLength);
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateProjectionDestination(outputLength, outputSpan);

            int packedRowByteCount = OPProviderCommon.GetBitNetPackedWeightByteCount(quantizedValues.Length, 1);
            if (ThreadCount == 1 || outputLength <= 1)
            {
                ProjectBitNetI2Range(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, outputSpan, 0, outputLength);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2Range(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, output.Span, startIndex, endIndex), ThreadCount);
        }

        private static void ForwardSoftmaxCore(ReadOnlySpan<float> input, Span<float> output)
        {
            float maxValue = input[0];
            for (int index = 1; index < input.Length; index++)
            {
                if (input[index] > maxValue)
                {
                    maxValue = input[index];
                }
            }

            float sum = 0f;
            for (int index = 0; index < input.Length; index++)
            {
                float exponent = MathF.Exp(input[index] - maxValue);
                output[index] = exponent;
                sum += exponent;
            }

            for (int index = 0; index < input.Length; index++)
            {
                output[index] /= sum;
            }
        }

        private static float ComputeSoftmaxMaxRange(ReadOnlySpan<float> input, int startIndex, int endIndex)
        {
            float maxValue = input[startIndex];
            for (int index = startIndex + 1; index < endIndex; index++)
            {
                if (input[index] > maxValue)
                {
                    maxValue = input[index];
                }
            }

            return maxValue;
        }

        private static double FillSoftmaxExponentRange(ReadOnlySpan<float> input, Span<float> output, float maxValue, int startIndex, int endIndex)
        {
            double sum = 0d;
            for (int index = startIndex; index < endIndex; index++)
            {
                float exponent = MathF.Exp(input[index] - maxValue);
                output[index] = exponent;
                sum += exponent;
            }

            return sum;
        }

        private static void NormalizeSoftmaxRange(Span<float> output, float inverseSum, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] *= inverseSum;
            }
        }

        private void ExecuteForwardSoftmaxMemory(ReadOnlyMemory<float> input, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateSoftmaxDestination(inputSpan, outputSpan);

            if (ThreadCount == 1 || input.Length <= 1)
            {
                ForwardSoftmaxCore(inputSpan, outputSpan);
                return;
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, ThreadCount, sizeof(float));
            if (ranges.Length <= 1)
            {
                ForwardSoftmaxCore(inputSpan, outputSpan);
                return;
            }

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = ComputeSoftmaxMaxRange(input.Span, range.StartIndex, range.EndIndex);
            });

            float maxValue = partialMaxima[0];
            for (int index = 1; index < partialMaxima.Length; index++)
            {
                if (partialMaxima[index] > maxValue)
                {
                    maxValue = partialMaxima[index];
                }
            }

            double[] partialSums = new double[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialSums[rangeIndex] = FillSoftmaxExponentRange(input.Span, output.Span, maxValue, range.StartIndex, range.EndIndex);
            });

            double sum = 0d;
            for (int index = 0; index < partialSums.Length; index++)
            {
                sum += partialSums[index];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                NormalizeSoftmaxRange(output.Span, inverseSum, startIndex, endIndex), ThreadCount, sizeof(float));
        }

        private static double ComputeRmsNormInverseRootMeanSquare(ReadOnlySpan<float> input, float epsilon)
        {
            double sumSquares = 0d;
            for (int index = 0; index < input.Length; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            double meanSquare = sumSquares / input.Length;
            return 1d / Math.Sqrt(meanSquare + epsilon);
        }

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, double inverseRootMeanSquare, Span<float> output, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * normWeights[index]);
            }
        }

        private static void FillAddRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] = input[index] + addend[index];
            }
        }

        private static void ProjectBitNetI2Range(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                int mappedDot = ComputeBitNetMappedDot(quantizedValues, packedRow);
                output[outputIndex] = OPProviderCommon.FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }

        private static int ComputeBitNetMappedDot(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedRow)
        {
            const int PackedGroupWidth = 32;
            const int ActivationBlockWidth = 128;

            int mappedDot = 0;
            int packedOffset = 0;
            for (int activationBlockStart = 0; activationBlockStart < quantizedValues.Length; activationBlockStart += ActivationBlockWidth)
            {
                int packedBlockByteCount = Math.Min(PackedGroupWidth, packedRow.Length - packedOffset);
                ReadOnlySpan<byte> packedBlock = packedRow.Slice(packedOffset, packedBlockByteCount);
                int remainingActivations = quantizedValues.Length - activationBlockStart;
                int groupCount = Math.Min(4, (remainingActivations + PackedGroupWidth - 1) / PackedGroupWidth);

                for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
                {
                    int activationOffset = activationBlockStart + (groupIndex * PackedGroupWidth);
                    int groupActivationCount = Math.Min(PackedGroupWidth, quantizedValues.Length - activationOffset);
                    ReadOnlySpan<sbyte> activationGroup = quantizedValues.Slice(activationOffset, groupActivationCount);
                    for (int packedIndex = 0; packedIndex < packedBlockByteCount && packedIndex < groupActivationCount; packedIndex++)
                    {
                        mappedDot += activationGroup[packedIndex] * OPProviderCommon.DecodeBitNetWeight(packedBlock[packedIndex], groupIndex);
                    }
                }

                packedOffset += packedBlockByteCount;
            }

            return mappedDot;
        }
    }
}
