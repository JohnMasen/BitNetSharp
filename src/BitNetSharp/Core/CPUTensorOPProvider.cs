using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides the tensor-accelerated CPU implementation of math operations.
    /// </summary>
    public sealed class CPUTensorOPProvider : IOPProvider
    {
        public CPUTensorOPProvider(int threadCount = Nodes.InferenceConfig.AutoThreadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            ThreadCount = threadCount;
        }

        public string Backend => "Tensor";

        public int ThreadCount { get; }

        public (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(RuntimeTensor input, RuntimeTensor quantizedValues)
        {
            return QuantizeBitNetActivations(
                RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input)),
                RuntimeTensorBufferHelper.GetMemory<sbyte>(quantizedValues, nameof(quantizedValues)),
                ThreadCount);
        }

        public void Add(RuntimeTensor input, RuntimeTensor addend, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            ReadOnlyMemory<float> addendMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(addend, nameof(addend));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ReadOnlySpan<float> inputSpan = inputMemory.Span;
            ReadOnlySpan<float> addendSpan = addendMemory.Span;
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateAddDestination(inputSpan, addendSpan, outputSpan);

            if (ThreadCount == 1 || inputMemory.Length <= 1)
            {
                FillAddRange(inputSpan, addendSpan, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(inputMemory.Length, (startIndex, endIndex) =>
                FillAddRange(
                    inputMemory.Span.Slice(startIndex, endIndex - startIndex),
                    addendMemory.Span.Slice(startIndex, endIndex - startIndex),
                    outputMemory.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(float));
        }

        public void ProjectBitNetI2(RuntimeTensor input, RuntimeTensor packedWeights, int outputLength, float weightScale, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            ReadOnlyMemory<byte> packedWeightsMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<byte>(packedWeights, nameof(packedWeights));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ValidationHelper.ValidateBitNetProjectionArguments(inputMemory.Span, packedWeightsMemory.Span, outputLength);
            ValidationHelper.ValidateProjectionDestination(outputLength, outputMemory.Span);

            using IMemoryOwner<sbyte> quantizedValuesRawOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
            Memory<sbyte> quantizedValuesRaw = quantizedValuesRawOwner.Memory[..inputMemory.Length];
            RuntimeTensor quantizedTensor = RuntimeTensor.CreateWritable("QuantizedBitNetActivations", quantizedValuesRaw, [inputMemory.Length]);
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedTensor);
            ProjectBitNetI2(quantizedTensor, activationScale, packedWeights, outputLength, weightScale, output);
        }

        public void ProjectBitNetI2(RuntimeTensor quantizedValues, float activationScale, RuntimeTensor packedWeights, int outputLength, float weightScale, RuntimeTensor output)
        {
            ReadOnlyMemory<sbyte> quantizedValuesMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<sbyte>(quantizedValues, nameof(quantizedValues));
            ReadOnlyMemory<byte> packedWeightsMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<byte>(packedWeights, nameof(packedWeights));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ValidationHelper.ValidateBitNetProjectionArguments(quantizedValuesMemory.Span, packedWeightsMemory.Span, outputLength);
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateProjectionDestination(outputLength, outputSpan);

            using IMemoryOwner<float> quantizedValuesOwner = MemoryPool<float>.Shared.Rent(quantizedValuesMemory.Length);
            Memory<float> quantizedValuesSingle = quantizedValuesOwner.Memory[..quantizedValuesMemory.Length];
            if (ThreadCount == 1 || quantizedValuesMemory.Length <= 1)
            {
                ConvertQuantizedValuesToSingle(quantizedValuesMemory.Span, quantizedValuesSingle.Span);
            }
            else
            {
                ThreadHelper.ForEachRange(quantizedValuesMemory.Length, (startIndex, endIndex) =>
                    ConvertQuantizedValuesToSingle(
                        quantizedValuesMemory.Span.Slice(startIndex, endIndex - startIndex),
                        quantizedValuesSingle.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(sbyte));
            }

            // Each packed byte stores four 2-bit weights, so one output row uses inputLength / 4 bytes.
            int packedRowByteCount = checked(quantizedValuesMemory.Length / 4);
            if (ThreadCount == 1 || outputLength <= 1)
            {
                ProjectBitNetI2Range(quantizedValuesSingle.Span, packedWeightsMemory.Span, packedRowByteCount, activationScale, weightScale, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2Range(
                    quantizedValuesSingle.Span,
                    packedWeightsMemory.Span.Slice(startIndex * packedRowByteCount, (endIndex - startIndex) * packedRowByteCount),
                    packedRowByteCount,
                    activationScale,
                    weightScale,
                    outputMemory.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, packedRowByteCount);
        }

        public void ForwardSoftmax(RuntimeTensor input, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ReadOnlySpan<float> inputSpan = inputMemory.Span;
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateSoftmaxDestination(inputSpan, outputSpan);

            if (ThreadCount == 1 || inputMemory.Length <= 1)
            {
                ForwardSoftmaxCore(inputSpan, outputSpan);
                return;
            }

            ExecuteForwardSoftmaxMemory(inputMemory, outputMemory);
        }

        public void ForwardRmsNorm(RuntimeTensor input, RuntimeTensor normWeights, float epsilon, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            ReadOnlyMemory<float> normWeightsMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(normWeights, nameof(normWeights));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ReadOnlySpan<float> inputSpan = inputMemory.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeightsMemory.Span;
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquare(inputMemory, epsilon, ThreadCount);
            if (ThreadCount == 1 || inputMemory.Length <= 1)
            {
                FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(inputMemory.Length, (startIndex, endIndex) =>
                FillRmsNormRange(
                    inputMemory.Span.Slice(startIndex, endIndex - startIndex),
                    normWeightsMemory.Span.Slice(startIndex, endIndex - startIndex),
                    inverseRootMeanSquare,
                    outputMemory.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(float));
        }

        public void ForwardLmHead(RuntimeTensor input, RuntimeTensor embeddingWeights, int rowLength, int vocabularySize, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            ReadOnlyMemory<byte> embeddingWeightsMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<byte>(embeddingWeights, nameof(embeddingWeights));
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ValidationHelper.ValidateLmHeadArguments(inputMemory, embeddingWeightsMemory, rowLength, vocabularySize, outputMemory);
            ReadOnlySpan<Half> embeddingWeightsSpan = MemoryMarshal.Cast<byte, Half>(embeddingWeightsMemory.Span);

            if (ThreadCount == 1 || vocabularySize <= 1)
            {
                using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                ProjectLmHeadRange(inputMemory.Span, embeddingWeightsSpan, rowLength, outputMemory.Span[..vocabularySize], rowOwner.Memory.Span[..rowLength]);
                return;
            }

            ThreadHelper.ForEachRange(
                vocabularySize,
                (startIndex, endIndex) =>
                {
                    using IMemoryOwner<float> rowOwner = MemoryPool<float>.Shared.Rent(rowLength);
                    ProjectLmHeadRange(
                        inputMemory.Span,
                        MemoryMarshal.Cast<byte, Half>(embeddingWeightsMemory.Span.Slice(startIndex * rowLength * sizeof(ushort), (endIndex - startIndex) * rowLength * sizeof(ushort))),
                        rowLength,
                        outputMemory.Span.Slice(startIndex, endIndex - startIndex),
                        rowOwner.Memory.Span[..rowLength]);
                },
                ThreadCount,
                checked(rowLength * sizeof(ushort)));
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
            }, ThreadCount, sizeof(float));
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

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float inverseRootMeanSquare, Span<float> output)
        {
            TensorPrimitives.Multiply(input, inverseRootMeanSquare, output);
            TensorPrimitives.Multiply(output, normWeights, output);
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

        private static void FillAddRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
        {
            TensorPrimitives.Add(input, addend, output);
        }

        private static void ProjectBitNetI2Range(ReadOnlySpan<float> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output)
        {
            using IMemoryOwner<float> rowWeightsOwner = MemoryPool<float>.Shared.Rent(quantizedValues.Length);
            Span<float> rowWeights = rowWeightsOwner.Memory.Span[..quantizedValues.Length];
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
            {
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                rowWeights.Clear();
                CPUDefaultOPProvider.ExpandBitNetRowWeights(packedRow, rowWeights);
                float mappedDot = TensorPrimitives.Dot(rowWeights, quantizedValues);
                // Restore the final float projection value from the mapped dot product and the activation/weight scales.
                output[outputIndex] = (mappedDot / activationScale) * weightScale;
            }
        }

        private static void ProjectLmHeadRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output, Span<float> rowBuffer)
        {
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
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

        private static void ConvertQuantizedValuesToSingle(ReadOnlySpan<sbyte> source, Span<float> destination)
        {
            for (int index = 0; index < source.Length; index++)
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
                partialSums[rangeIndex] = QuantizeScaledValuesRange(
                    scaledRange,
                    quantizedValues.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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

        private static int QuantizeScaledValuesRange(ReadOnlySpan<float> scaledValues, Span<sbyte> quantizedValues)
        {
            int activationSum = 0;
            for (int index = 0; index < scaledValues.Length; index++)
            {
                int quantizedValue = (int)MathF.Round(scaledValues[index], MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return activationSum;
        }
    }
}
