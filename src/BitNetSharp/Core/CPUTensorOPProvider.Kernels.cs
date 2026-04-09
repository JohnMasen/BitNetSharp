using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    public sealed partial class CPUTensorOPProvider
    {
        private void ExecuteForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float sumSquares = TensorPrimitives.Dot(inputSpan, inputSpan);
            float meanSquare = sumSquares / input.Length;
            float inverseRootMeanSquare = 1f / MathF.Sqrt(meanSquare + epsilon);
            if (ThreadCount == 1 || input.Length <= 1)
            {
                FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillRmsNormRange(input.Span, normWeights.Span, output.Span, inverseRootMeanSquare, startIndex, endIndex), ThreadCount);
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
                FillAddRange(input.Span, addend.Span, output.Span, startIndex, endIndex), ThreadCount);
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
                partialMaxima[rangeIndex] = TensorPrimitives.Max(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
                Span<float> outputRange = output.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                ReadOnlySpan<float> inputRange = input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                for (int index = 0; index < inputRange.Length; index++)
                {
                    outputRange[index] = inputRange[index] - maxValue;
                }

                TensorPrimitives.Exp(outputRange, outputRange);
                partialSums[rangeIndex] = TensorPrimitives.Sum(outputRange);
            });

            double sum = 0d;
            for (int index = 0; index < partialSums.Length; index++)
            {
                sum += partialSums[index];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
            {
                Span<float> outputRange = output.Span.Slice(startIndex, endIndex - startIndex);
                TensorPrimitives.Multiply(outputRange, inverseSum, outputRange);
            }, ThreadCount);
        }

        private void ExecuteProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            OPProviderCommon.ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            OPProviderCommon.ValidateProjectionDestination(outputLength, output.Span);

            using IMemoryOwner<sbyte> quantizedValuesRawOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValuesRaw = quantizedValuesRawOwner.Memory[..input.Length];
            (float activationScale, _) = OPProviderCommon.QuantizeBitNetActivations(input.Span, quantizedValuesRaw.Span);
            ExecuteProjectBitNetI2(quantizedValuesRaw, activationScale, packedWeights, outputLength, weightScale, output);
        }

        private void ExecuteProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValuesRaw, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            OPProviderCommon.ValidateBitNetProjectionArguments(quantizedValuesRaw.Span, packedWeights.Span, outputLength);
            Span<float> outputSpan = output.Span;
            OPProviderCommon.ValidateProjectionDestination(outputLength, outputSpan);

            using IMemoryOwner<float> quantizedValuesOwner = MemoryPool<float>.Shared.Rent(quantizedValuesRaw.Length);
            Memory<float> quantizedValues = quantizedValuesOwner.Memory[..quantizedValuesRaw.Length];
            ReadOnlySpan<sbyte> quantizedValuesRawSpan = quantizedValuesRaw.Span;
            for (int index = 0; index < quantizedValuesRawSpan.Length; index++)
            {
                quantizedValues.Span[index] = quantizedValuesRawSpan[index];
            }

            int packedRowByteCount = OPProviderCommon.GetBitNetPackedWeightByteCount(quantizedValuesRaw.Length, 1);
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
            float maxValue = TensorPrimitives.Max(input);
            for (int index = 0; index < input.Length; index++)
            {
                output[index] = input[index] - maxValue;
            }

            TensorPrimitives.Exp(output[..input.Length], output);
            float sum = TensorPrimitives.Sum(output[..input.Length]);
            for (int index = 0; index < input.Length; index++)
            {
                output[index] /= sum;
            }
        }

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float inverseRootMeanSquare, Span<float> output, int startIndex, int endIndex)
        {
            Span<float> outputRange = output.Slice(startIndex, endIndex - startIndex);
            TensorPrimitives.Multiply(input.Slice(startIndex, endIndex - startIndex), inverseRootMeanSquare, outputRange);
            TensorPrimitives.Multiply(outputRange, normWeights.Slice(startIndex, endIndex - startIndex), outputRange);
        }

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, Span<float> output, float inverseRootMeanSquare, int startIndex, int endIndex)
        {
            FillRmsNormRange(input, normWeights, inverseRootMeanSquare, output, startIndex, endIndex);
        }

        private static void FillAddRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output, int startIndex, int endIndex)
        {
            TensorPrimitives.Add(input.Slice(startIndex, endIndex - startIndex), addend.Slice(startIndex, endIndex - startIndex), output.Slice(startIndex, endIndex - startIndex));
        }

        private static void ProjectBitNetI2Range(ReadOnlySpan<float> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output, int startIndex, int endIndex)
        {
            using IMemoryOwner<float> rowWeightsOwner = MemoryPool<float>.Shared.Rent(quantizedValues.Length);
            Span<float> rowWeights = rowWeightsOwner.Memory.Span[..quantizedValues.Length];
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                rowWeights.Clear();
                OPProviderCommon.ExpandBitNetRowWeights(packedRow, rowWeights);
                float mappedDot = TensorPrimitives.Dot(rowWeights, quantizedValues);
                output[outputIndex] = OPProviderCommon.FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }
    }
}
