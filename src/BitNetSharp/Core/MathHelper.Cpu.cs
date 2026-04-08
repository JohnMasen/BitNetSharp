using System;
using System.Buffers;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static void ForwardRmsNormCpuStandard(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            double inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareCpuStandard(inputSpan, epsilon);
            if (threads != 1)
            {
                ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                    FillRmsNormCpuStandardRange(input.Span, normWeights.Span, inverseRootMeanSquare, output.Span, startIndex, endIndex), threads);

                return;
            }

            FillRmsNormCpuStandardRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
        }

        public static float[] ForwardRmsNormCpuStandard(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, int threads = 0)
        {
            float[] output = new float[input.Length];
            ForwardRmsNormCpuStandard(input, normWeights, epsilon, output, threads);
            return output;
        }

        internal static void ForwardSoftmaxCpuStandard(ReadOnlySpan<float> input, Span<float> output, int threads = 0)
        {
            ValidateSoftmaxDestination(input, output);

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxCpuStandardCore(input, output);
                return;
            }

            using IMemoryOwner<float> inputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> outputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Memory<float> inputMemory = inputOwner.Memory[..input.Length];
            Memory<float> outputMemory = outputOwner.Memory[..input.Length];
            input.CopyTo(inputMemory.Span);
            ForwardSoftmaxCpuStandard(inputMemory, outputMemory, threads);
            outputMemory.Span.CopyTo(output);
        }

        internal static void ForwardSoftmaxCpuStandard(ReadOnlyMemory<float> input, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            Span<float> outputSpan = output.Span;
            ValidateSoftmaxDestination(inputSpan, outputSpan);

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxCpuStandardCore(inputSpan, outputSpan);
                return;
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threads, sizeof(float));
            if (ranges.Length <= 1)
            {
                ForwardSoftmaxCpuStandardCore(inputSpan, outputSpan);
                return;
            }

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = ComputeSoftmaxMaxCpuRange(input.Span, range.StartIndex, range.EndIndex);
            });

            float maxValue = partialMaxima[0];
            for (int rangeIndex = 1; rangeIndex < partialMaxima.Length; rangeIndex++)
            {
                if (partialMaxima[rangeIndex] > maxValue)
                {
                    maxValue = partialMaxima[rangeIndex];
                }
            }

            double[] partialSums = new double[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialSums[rangeIndex] = FillSoftmaxExponentCpuRange(input.Span, output.Span, maxValue, range.StartIndex, range.EndIndex);
            });

            double sum = 0d;
            for (int rangeIndex = 0; rangeIndex < partialSums.Length; rangeIndex++)
            {
                sum += partialSums[rangeIndex];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                NormalizeSoftmaxRange(output.Span, inverseSum, startIndex, endIndex), threads, sizeof(float));
        }

        private static void ForwardSoftmaxCpuStandardCore(ReadOnlySpan<float> input, Span<float> output)
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

        private static float ComputeSoftmaxMaxCpuRange(ReadOnlySpan<float> input, int startIndex, int endIndex)
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

        private static double FillSoftmaxExponentCpuRange(ReadOnlySpan<float> input, Span<float> output, float maxValue, int startIndex, int endIndex)
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

        private static double ComputeRmsNormInverseRootMeanSquareCpuStandard(ReadOnlySpan<float> input, float epsilon)
        {
            double sumSquares = 0;
            for (int index = 0; index < input.Length; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            double meanSquare = sumSquares / input.Length;
            return 1d / Math.Sqrt(meanSquare + epsilon);
        }

        private static void FillRmsNormCpuStandardRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, double inverseRootMeanSquare, Span<float> output, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * normWeights[index]);
            }
        }

        public static void ProjectBitNetI2Cpu(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            ValidateProjectionDestination(outputLength, output.Span);

            //quantize the float activations once so each output row can reuse the same ternary-friendly input block
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input.Span, quantizedValues.Span);
            ProjectBitNetI2Cpu(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output, threads);
        }

        public static float[] ProjectBitNetI2Cpu(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Cpu(input, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        internal static void ProjectBitNetI2Cpu(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(quantizedValues.Span, packedWeights.Span, outputLength);
            Span<float> outputSpan = output.Span;
            ValidateProjectionDestination(outputLength, outputSpan);

            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, 1);
            if (threads == 1 || outputLength <= 1)
            {
                ProjectBitNetI2CpuRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, outputSpan, 0, outputLength);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2CpuRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, output.Span, startIndex, endIndex), threads);
        }

        internal static float[] ProjectBitNetI2Cpu(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Cpu(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        private static void ProjectBitNetI2CpuRange(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //load one packed weight row, decode its mapped dot product, then restore the final scaled float result
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                int mappedDot = ComputeBitNetMappedDotCpu(quantizedValues, packedRow);
                output[outputIndex] = FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }

        private static int ComputeBitNetMappedDotCpu(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedRow)
        {
            const int PackedGroupWidth = 32;
            const int ActivationBlockWidth = 128;

            int mappedDot = 0;
            int packedOffset = 0;
            for (int activationBlockStart = 0; activationBlockStart < quantizedValues.Length; activationBlockStart += ActivationBlockWidth)
            {
                //each packed block stores up to 128 activations as four 32-value groups
                int packedBlockByteCount = Math.Min(PackedGroupWidth, packedRow.Length - packedOffset);
                ReadOnlySpan<byte> packedBlock = packedRow.Slice(packedOffset, packedBlockByteCount);
                int remainingActivations = quantizedValues.Length - activationBlockStart;
                int groupCount = Math.Min(4, (remainingActivations + PackedGroupWidth - 1) / PackedGroupWidth);

                for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
                {
                    int activationOffset = activationBlockStart + (groupIndex * PackedGroupWidth);
                    int groupActivationCount = Math.Min(PackedGroupWidth, quantizedValues.Length - activationOffset);
                    ReadOnlySpan<sbyte> activationGroup = quantizedValues.Slice(activationOffset, groupActivationCount);

                    //decode the selected 2-bit lane from each packed byte and accumulate the mapped integer dot product
                    for (int packedIndex = 0; packedIndex < packedBlockByteCount && packedIndex < groupActivationCount; packedIndex++)
                    {
                        mappedDot += activationGroup[packedIndex] * DecodeBitNetWeight(packedBlock[packedIndex], groupIndex);
                    }
                }

                packedOffset += packedBlockByteCount;
            }

            return mappedDot;
        }
    }
}
