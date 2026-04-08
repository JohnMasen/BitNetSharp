using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static void ForwardRmsNormTensor(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float sumSquares = TensorPrimitives.Dot(inputSpan, inputSpan);
            float meanSquare = sumSquares / input.Length;
            float inverseRootMeanSquare = 1f / MathF.Sqrt(meanSquare + epsilon);
            if (threads == 1 || input.Length <= 1)
            {
                FillRmsNormTensorRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillRmsNormTensorRange(input.Span, normWeights.Span, inverseRootMeanSquare, output.Span, startIndex, endIndex), threads);
        }

        public static float[] ForwardRmsNormTensor(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, int threads = 0)
        {
            float[] output = new float[input.Length];
            ForwardRmsNormTensor(input, normWeights, epsilon, output, threads);
            return output;
        }

        public static void AddTensor(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> addendSpan = addend.Span;
            Span<float> outputSpan = output.Span;
            ValidateAddDestination(inputSpan, addendSpan, outputSpan);

            if (threads == 1 || input.Length <= 1)
            {
                FillAddTensorRange(inputSpan, addendSpan, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillAddTensorRange(input.Span, addend.Span, output.Span, startIndex, endIndex), threads);
        }

        public static float[] AddTensor(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, int threads = 0)
        {
            float[] output = new float[input.Length];
            AddTensor(input, addend, output, threads);
            return output;
        }

        internal static void ForwardSoftmaxTensor(ReadOnlySpan<float> input, Span<float> output, int threads = 0)
        {
            ValidateSoftmaxDestination(input, output);

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxTensorCore(input, output);
                return;
            }

            using IMemoryOwner<float> inputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> outputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Memory<float> inputMemory = inputOwner.Memory[..input.Length];
            Memory<float> outputMemory = outputOwner.Memory[..input.Length];
            input.CopyTo(inputMemory.Span);
            ForwardSoftmaxTensor(inputMemory, outputMemory, threads);
            outputMemory.Span.CopyTo(output);
        }

        internal static void ForwardSoftmaxTensor(ReadOnlyMemory<float> input, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            Span<float> outputSpan = output.Span;
            ValidateSoftmaxDestination(inputSpan, outputSpan);

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxTensorCore(inputSpan, outputSpan);
                return;
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threads, sizeof(float));
            if (ranges.Length <= 1)
            {
                ForwardSoftmaxTensorCore(inputSpan, outputSpan);
                return;
            }

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = TensorPrimitives.Max(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
            for (int rangeIndex = 0; rangeIndex < partialSums.Length; rangeIndex++)
            {
                sum += partialSums[rangeIndex];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
            {
                Span<float> outputRange = output.Span.Slice(startIndex, endIndex - startIndex);
                TensorPrimitives.Multiply(outputRange, inverseSum, outputRange);
            }, threads);
        }

        private static void ForwardSoftmaxTensorCore(ReadOnlySpan<float> input, Span<float> output)
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

        private static void FillRmsNormTensorRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float inverseRootMeanSquare, Span<float> output, int startIndex, int endIndex)
        {
            Span<float> outputRange = output.Slice(startIndex, endIndex - startIndex);
            TensorPrimitives.Multiply(input.Slice(startIndex, endIndex - startIndex), inverseRootMeanSquare, outputRange);
            TensorPrimitives.Multiply(outputRange, normWeights.Slice(startIndex, endIndex - startIndex), outputRange);
        }

        private static void FillAddTensorRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output, int startIndex, int endIndex)
        {
            TensorPrimitives.Add(input.Slice(startIndex, endIndex - startIndex), addend.Slice(startIndex, endIndex - startIndex), output.Slice(startIndex, endIndex - startIndex));
        }

        public static void ProjectBitNetI2Tensor(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            ValidateProjectionDestination(outputLength, output.Span);

            //quantize the float activations first, then convert them to float so TensorPrimitives can consume them
            using IMemoryOwner<sbyte> quantizedValuesRawOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValuesRaw = quantizedValuesRawOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input.Span, quantizedValuesRaw.Span);
            ProjectBitNetI2Tensor(quantizedValuesRaw, activationScale, packedWeights, outputLength, weightScale, output, threads);
        }

        public static float[] ProjectBitNetI2Tensor(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Tensor(input, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        internal static void ProjectBitNetI2Tensor(ReadOnlyMemory<sbyte> quantizedValuesRaw, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(quantizedValuesRaw.Span, packedWeights.Span, outputLength);
            Span<float> outputSpan = output.Span;
            ValidateProjectionDestination(outputLength, outputSpan);

            using IMemoryOwner<float> quantizedValuesOwner = MemoryPool<float>.Shared.Rent(quantizedValuesRaw.Length);
            Memory<float> quantizedValues = quantizedValuesOwner.Memory[..quantizedValuesRaw.Length];
            ReadOnlySpan<sbyte> quantizedValuesRawSpan = quantizedValuesRaw.Span;
            for (int index = 0; index < quantizedValuesRawSpan.Length; index++)
            {
                quantizedValues.Span[index] = quantizedValuesRawSpan[index];
            }

            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValuesRaw.Length, 1);
            if (threads == 1 || outputLength <= 1)
            {
                ProjectBitNetI2TensorRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, outputSpan, 0, outputLength);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2TensorRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, output.Span, startIndex, endIndex), threads);
        }

        internal static float[] ProjectBitNetI2Tensor(ReadOnlyMemory<sbyte> quantizedValuesRaw, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Tensor(quantizedValuesRaw, activationScale, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        private static void ProjectBitNetI2TensorRange(ReadOnlySpan<float> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output, int startIndex, int endIndex)
        {
            using IMemoryOwner<float> rowWeightsOwner = MemoryPool<float>.Shared.Rent(quantizedValues.Length);
            Span<float> rowWeights = rowWeightsOwner.Memory.Span[..quantizedValues.Length];
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //expand one packed row into dense float weights, compute the mapped dot, then rescale it back to the projection result
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                rowWeights.Clear();
                ExpandBitNetRowWeights(packedRow, rowWeights);

                float mappedDot = TensorPrimitives.Dot(rowWeights, quantizedValues);
                output[outputIndex] = FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }
    }
}
