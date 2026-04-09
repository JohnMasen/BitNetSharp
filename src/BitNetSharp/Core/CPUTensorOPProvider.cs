using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides the tensor-accelerated CPU implementation of math operations.
    /// </summary>
    public sealed class CPUTensorOPProvider : IOPProvider2
    {
        public CPUTensorOPProvider(int threadCount = global::BitNetSharp.Nodes.InferenceConfig.AutoThreadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            ThreadCount = threadCount;
        }

        public global::BitNetSharp.Nodes.InferenceBackend Backend => global::BitNetSharp.Nodes.InferenceBackend.Tensor;

        public int ThreadCount { get; }

        public void Add(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output, string operationName = "Add")
        {
            ExecuteAdd(input, addend, output);
        }

        public void ProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, string operationName = "BitNet projection")
        {
            ExecuteProjectBitNetI2(input, packedWeights, outputLength, weightScale, output);
        }

        public void ProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, string operationName = "BitNet projection")
        {
            ExecuteProjectBitNetI2(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output);
        }

        public void ForwardSoftmax(ReadOnlySpan<float> input, Span<float> output, string operationName = "Softmax")
        {
            ExecuteForwardSoftmax(input, output);
        }

        public void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ExecuteForwardRmsNorm(input, normWeights, epsilon, output);
        }

        public void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<Half> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output)
        {
            ValidationHelper.ValidateLmHeadArguments(input, rowLength, vocabularySize, output);

            Span<float> outputSpan = output.Span[..vocabularySize];
            if (ThreadCount == 1 || vocabularySize <= 1)
            {
                using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                ProjectLmHeadRange(input.Span, embeddingWeights.Span, rowLength, outputSpan, 0, vocabularySize, rowOwner.Memory.Span[..rowLength]);
                return;
            }

            float[] inputBuffer = input.ToArray();
            Half[] embeddingWeightsBuffer = embeddingWeights.ToArray();
            float[] outputBuffer = new float[vocabularySize];
            ThreadHelper.ForEachRange(
                outputBuffer.AsSpan(),
                (startIndex, endIndex) =>
                {
                    using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                    ProjectLmHeadRange(inputBuffer, embeddingWeightsBuffer, rowLength, outputBuffer, startIndex, endIndex, rowOwner.Memory.Span[..rowLength]);
                },
                ThreadCount);
            outputBuffer.AsSpan().CopyTo(outputSpan);
        }

        private void ExecuteForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquare(input, epsilon, ThreadCount);
            if (ThreadCount == 1 || input.Length <= 1)
            {
                FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillRmsNormRange(input.Span, normWeights.Span, inverseRootMeanSquare, output.Span, startIndex, endIndex), ThreadCount);
        }

        private void ExecuteAdd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> addendSpan = addend.Span;
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateAddDestination(inputSpan, addendSpan, outputSpan);

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
            ValidationHelper.ValidateSoftmaxDestination(input, output);

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
            ValidationHelper.ValidateSoftmaxDestination(inputSpan, outputSpan);

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
            ValidationHelper.ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            ValidationHelper.ValidateProjectionDestination(outputLength, output.Span);

            using IMemoryOwner<sbyte> quantizedValuesRawOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValuesRaw = quantizedValuesRawOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedValuesRaw, ThreadCount);
            ExecuteProjectBitNetI2(quantizedValuesRaw, activationScale, packedWeights, outputLength, weightScale, output);
        }

        private void ExecuteProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValuesRaw, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            ValidationHelper.ValidateBitNetProjectionArguments(quantizedValuesRaw.Span, packedWeights.Span, outputLength);
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateProjectionDestination(outputLength, outputSpan);

            using IMemoryOwner<float> quantizedValuesOwner = MemoryPool<float>.Shared.Rent(quantizedValuesRaw.Length);
            Memory<float> quantizedValues = quantizedValuesOwner.Memory[..quantizedValuesRaw.Length];
            if (ThreadCount == 1 || quantizedValuesRaw.Length <= 1)
            {
                ConvertQuantizedValuesToSingle(quantizedValuesRaw.Span, quantizedValues.Span, 0, quantizedValuesRaw.Length);
            }
            else
            {
                ThreadHelper.ForEachRange(quantizedValuesRaw.Length, (startIndex, endIndex) =>
                    ConvertQuantizedValuesToSingle(quantizedValuesRaw.Span, quantizedValues.Span, startIndex, endIndex), ThreadCount, sizeof(sbyte));
            }

            // Each packed byte stores four 2-bit weights, so one output row uses inputLength / 4 bytes.
            int packedRowByteCount = checked(quantizedValuesRaw.Length / 4);
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

        private static float ComputeRmsNormInverseRootMeanSquare(ReadOnlyMemory<float> input, float epsilon, int threadCount)
        {
            if (threadCount == 1 || input.Length <= 1)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threadCount, sizeof(float));
            if (ranges.Length <= 1)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            float[] partialSums = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                ReadOnlySpan<float> inputRange = input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                partialSums[rangeIndex] = TensorPrimitives.Dot(inputRange, inputRange);
            });

            double sumSquares = 0d;
            for (int index = 0; index < partialSums.Length; index++)
            {
                sumSquares += partialSums[index];
            }

            double meanSquare = sumSquares / input.Length;
            return (float)(1d / Math.Sqrt(meanSquare + epsilon));
        }

        private static float ComputeRmsNormInverseRootMeanSquareSingleThread(ReadOnlySpan<float> input, float epsilon)
        {
            float sumSquares = TensorPrimitives.Dot(input, input);
            float meanSquare = sumSquares / input.Length;
            return 1f / MathF.Sqrt(meanSquare + epsilon);
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
                CPUDefaultOPProvider.ExpandBitNetRowWeights(packedRow, rowWeights);
                float mappedDot = TensorPrimitives.Dot(rowWeights, quantizedValues);
                // Restore the final float projection value from the mapped dot product and the activation/weight scales.
                output[outputIndex] = (mappedDot / activationScale) * weightScale;
            }
        }

        private static void ProjectLmHeadRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, int startIndex, int endIndex, Span<float> rowBuffer)
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

        private static void ConvertQuantizedValuesToSingle(ReadOnlySpan<sbyte> source, Span<float> destination, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                destination[index] = source[index];
            }
        }

        internal static (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlyMemory<float> input, Memory<sbyte> quantizedValues, int threadCount)
        {
            const float MinimumBitNetQuantizationMax = 0.00001f;

            if (quantizedValues.Length < input.Length)
            {
                throw new ArgumentException("Quantized output length must be at least the input length.", nameof(quantizedValues));
            }

            if (threadCount == 1 || input.Length <= 1)
            {
                return QuantizeBitNetActivationsSingleThread(input.Span, quantizedValues.Span);
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threadCount, sizeof(float));
            if (ranges.Length <= 1)
            {
                return QuantizeBitNetActivationsSingleThread(input.Span, quantizedValues.Span);
            }

            using IMemoryOwner<float> absoluteValuesOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> scaledValuesOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Memory<float> absoluteValuesMemory = absoluteValuesOwner.Memory[..input.Length];
            Memory<float> scaledValuesMemory = scaledValuesOwner.Memory[..input.Length];

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                ReadOnlySpan<float> inputRange = input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                Span<float> absoluteRange = absoluteValuesMemory.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                TensorPrimitives.Abs(inputRange, absoluteRange);
                partialMaxima[rangeIndex] = TensorPrimitives.Max(absoluteRange);
            });

            float maxAbs = MinimumBitNetQuantizationMax;
            for (int index = 0; index < partialMaxima.Length; index++)
            {
                if (partialMaxima[index] > maxAbs)
                {
                    maxAbs = partialMaxima[index];
                }
            }

            float activationScale = 127f / maxAbs;
            int[] partialSums = new int[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                ReadOnlySpan<float> inputRange = input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                Span<float> scaledRange = scaledValuesMemory.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex);
                TensorPrimitives.Multiply(inputRange, activationScale, scaledRange);
                partialSums[rangeIndex] = QuantizeScaledValuesRange(scaledRange, quantizedValues.Span, range.StartIndex);
            });

            int activationSum = 0;
            for (int index = 0; index < partialSums.Length; index++)
            {
                activationSum += partialSums[index];
            }

            return (activationScale, activationSum);
        }

        private static (float ActivationScale, int ActivationSum) QuantizeBitNetActivationsSingleThread(ReadOnlySpan<float> input, Span<sbyte> quantizedValues)
        {
            const float MinimumBitNetQuantizationMax = 0.00001f;

            using IMemoryOwner<float> absoluteValuesOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> scaledValuesOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Span<float> absoluteValues = absoluteValuesOwner.Memory.Span[..input.Length];
            Span<float> scaledValues = scaledValuesOwner.Memory.Span[..input.Length];

            TensorPrimitives.Abs(input, absoluteValues);
            float maxAbs = MathF.Max(TensorPrimitives.Max(absoluteValues), MinimumBitNetQuantizationMax);
            float activationScale = 127f / maxAbs;

            TensorPrimitives.Multiply(input, activationScale, scaledValues);
            int activationSum = 0;
            for (int index = 0; index < scaledValues.Length; index++)
            {
                int quantizedValue = (int)MathF.Round(scaledValues[index], MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return (activationScale, activationSum);
        }

        private static int QuantizeScaledValuesRange(ReadOnlySpan<float> scaledValues, Span<sbyte> quantizedValues, int startIndex)
        {
            int activationSum = 0;
            for (int index = 0; index < scaledValues.Length; index++)
            {
                int quantizedValue = (int)MathF.Round(scaledValues[index], MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[startIndex + index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return activationSum;
        }
    }
}
