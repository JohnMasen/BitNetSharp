using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides the standard CPU implementation of math operations.
    /// </summary>
    public sealed class CPUDefaultOPProvider : IOPProvider
    {
        public CPUDefaultOPProvider(int threadCount = Nodes.InferenceConfig.AutoThreadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            ThreadCount = threadCount;
        }

        public string Backend => "CPU";

        public int ThreadCount { get; }

        public (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(RuntimeTensor input, RuntimeTensor quantizedValues)
        {
            return QuantizeBitNetActivations(
                input.GetReadOnlyMemory<float>(),
                quantizedValues.GetMemory<sbyte>(),
                ThreadCount);
        }

        public void Add(RuntimeTensor input, RuntimeTensor addend, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            ReadOnlyMemory<float> addendMemory = addend.GetReadOnlyMemory<float>();
            Memory<float> outputMemory = output.GetMemory<float>();

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
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            ReadOnlyMemory<byte> packedWeightsMemory = packedWeights.GetReadOnlyMemory<byte>();
            Memory<float> outputMemory = output.GetMemory<float>();

            ValidationHelper.ValidateBitNetProjectionArguments(inputMemory.Span, packedWeightsMemory.Span, outputLength);
            ValidationHelper.ValidateProjectionDestination(outputLength, outputMemory.Span);

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..inputMemory.Length];
            RuntimeTensor quantizedTensor = RuntimeTensor.CreateWritable("QuantizedBitNetActivations", quantizedValues, [inputMemory.Length]);
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedTensor);
            ProjectBitNetI2(quantizedTensor, activationScale, packedWeights, outputLength, weightScale, output);
        }

        public void ProjectBitNetI2(RuntimeTensor quantizedValues, float activationScale, RuntimeTensor packedWeights, int outputLength, float weightScale, RuntimeTensor output)
        {
            ReadOnlyMemory<sbyte> quantizedValuesMemory = quantizedValues.GetReadOnlyMemory<sbyte>();
            ReadOnlyMemory<byte> packedWeightsMemory = packedWeights.GetReadOnlyMemory<byte>();
            Memory<float> outputMemory = output.GetMemory<float>();
            ValidationHelper.ValidateBitNetProjectionArguments(quantizedValuesMemory.Span, packedWeightsMemory.Span, outputLength);
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateProjectionDestination(outputLength, outputSpan);

            // Each packed byte stores four 2-bit weights, so one output row uses inputLength / 4 bytes.
            int packedRowByteCount = checked(quantizedValuesMemory.Length / 4);
            if (ThreadCount == 1 || outputLength <= 1)
            {
                ProjectBitNetI2Range(quantizedValuesMemory.Span, packedWeightsMemory.Span, packedRowByteCount, activationScale, weightScale, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2Range(
                    quantizedValuesMemory.Span,
                    packedWeightsMemory.Span.Slice(startIndex * packedRowByteCount, (endIndex - startIndex) * packedRowByteCount),
                    packedRowByteCount,
                    activationScale,
                    weightScale,
                    outputMemory.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, packedRowByteCount);
        }

        public void ForwardSoftmax(RuntimeTensor input, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            Memory<float> outputMemory = output.GetMemory<float>();
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
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            ReadOnlyMemory<float> normWeightsMemory = normWeights.GetReadOnlyMemory<float>();
            Memory<float> outputMemory = output.GetMemory<float>();
            ReadOnlySpan<float> inputSpan = inputMemory.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeightsMemory.Span;
            Span<float> outputSpan = outputMemory.Span;
            ValidationHelper.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            double inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquare(inputMemory, epsilon, ThreadCount);
            if (ThreadCount != 1)
            {
                ThreadHelper.ForEachRange(inputMemory.Length, (startIndex, endIndex) =>
                    FillRmsNormRange(
                        inputMemory.Span.Slice(startIndex, endIndex - startIndex),
                        normWeightsMemory.Span.Slice(startIndex, endIndex - startIndex),
                        inverseRootMeanSquare,
                        outputMemory.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(float));
                return;
            }

            FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan);
        }

        public void ForwardLmHead(RuntimeTensor input, RuntimeTensor embeddingWeights, int rowLength, int vocabularySize, RuntimeTensor output)
        {
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            ReadOnlyMemory<byte> embeddingWeightsMemory = embeddingWeights.GetReadOnlyMemory<byte>();
            Memory<float> outputMemory = output.GetMemory<float>();
            ValidationHelper.ValidateLmHeadArguments(inputMemory, embeddingWeightsMemory, rowLength, vocabularySize, outputMemory);
            ReadOnlySpan<Half> embeddingWeightsSpan = MemoryMarshal.Cast<byte, Half>(embeddingWeightsMemory.Span);

            if (ThreadCount == 1 || vocabularySize <= 1)
            {
                ProjectLmHeadRange(inputMemory.Span, embeddingWeightsSpan, rowLength, outputMemory.Span[..vocabularySize]);
                return;
            }

            ThreadHelper.ForEachRange(
                vocabularySize,
                (startIndex, endIndex) => ProjectLmHeadRange(
                    inputMemory.Span,
                    MemoryMarshal.Cast<byte, Half>(embeddingWeightsMemory.Span.Slice(startIndex * rowLength * sizeof(ushort), (endIndex - startIndex) * rowLength * sizeof(ushort))),
                    rowLength,
                    outputMemory.Span.Slice(startIndex, endIndex - startIndex)),
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
                partialMaxima[rangeIndex] = ComputeSoftmaxMaxRange(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
                partialSums[rangeIndex] = FillSoftmaxExponentRange(
                    input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex),
                    output.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex),
                    maxValue);
            });

            double sum = 0d;
            for (int index = 0; index < partialSums.Length; index++)
            {
                sum += partialSums[index];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                NormalizeSoftmaxRange(output.Span.Slice(startIndex, endIndex - startIndex), inverseSum), ThreadCount, sizeof(float));
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

        private static float ComputeSoftmaxMaxRange(ReadOnlySpan<float> input)
        {
            float maxValue = input[0];
            for (int index = 1; index < input.Length; index++)
            {
                if (input[index] > maxValue)
                {
                    maxValue = input[index];
                }
            }

            return maxValue;
        }

        private static double FillSoftmaxExponentRange(ReadOnlySpan<float> input, Span<float> output, float maxValue)
        {
            double sum = 0d;
            for (int index = 0; index < input.Length; index++)
            {
                float exponent = MathF.Exp(input[index] - maxValue);
                output[index] = exponent;
                sum += exponent;
            }

            return sum;
        }

        private static void NormalizeSoftmaxRange(Span<float> output, float inverseSum)
        {
            for (int index = 0; index < output.Length; index++)
            {
                output[index] *= inverseSum;
            }
        }

        private static double ComputeRmsNormInverseRootMeanSquare(ReadOnlyMemory<float> input, float epsilon, int threadCount)
        {
            if (threadCount == 1 || input.Length <= 1)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            ThreadHelper.WorkRange[] workRanges = ThreadHelper.CreateRanges(input.Length, threadCount, sizeof(float));
            (int StartIndex, int EndIndex)[] ranges = new (int StartIndex, int EndIndex)[workRanges.Length];
            for (int rangeIndex = 0; rangeIndex < workRanges.Length; rangeIndex++)
            {
                ThreadHelper.WorkRange range = workRanges[rangeIndex];
                ranges[rangeIndex] = (range.StartIndex, range.EndIndex);
            }

            if (ranges.Length <= 1)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            double[] partialSums = new double[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                (int startIndex, int endIndex) = ranges[rangeIndex];
                partialSums[rangeIndex] = AccumulateRmsNormSumSquares(input.Span.Slice(startIndex, endIndex - startIndex));
            });

            double sumSquares = 0d;
            for (int index = 0; index < partialSums.Length; index++)
            {
                sumSquares += partialSums[index];
            }

            double meanSquare = sumSquares / input.Length;
            return 1d / Math.Sqrt(meanSquare + epsilon);
        }

        private static double ComputeRmsNormInverseRootMeanSquareSingleThread(ReadOnlySpan<float> input, float epsilon)
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

        private static double AccumulateRmsNormSumSquares(ReadOnlySpan<float> input)
        {
            double sumSquares = 0d;
            for (int index = 0; index < input.Length; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            return sumSquares;
        }

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, double inverseRootMeanSquare, Span<float> output)
        {
            for (int index = 0; index < output.Length; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * normWeights[index]);
            }
        }

        private static void FillAddRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
        {
            for (int index = 0; index < output.Length; index++)
            {
                output[index] = input[index] + addend[index];
            }
        }

        private static void ProjectBitNetI2Range(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output)
        {
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
            {
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                int mappedDot = ComputeBitNetMappedDot(quantizedValues, packedRow);
                // Restore the final float projection value from the mapped dot product and the activation/weight scales.
                output[outputIndex] = (mappedDot / activationScale) * weightScale;
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
                        mappedDot += activationGroup[packedIndex] * DecodeBitNetWeight(packedBlock[packedIndex], groupIndex);
                    }
                }

                packedOffset += packedBlockByteCount;
            }

            return mappedDot;
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

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = ComputeQuantizationMaxRange(
                    input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex),
                    MinimumBitNetQuantizationMax);
            });

            float maxAbs = partialMaxima[0];
            for (int index = 1; index < partialMaxima.Length; index++)
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
                partialSums[rangeIndex] = QuantizeBitNetActivationsRange(
                    input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex),
                    quantizedValues.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex),
                    activationScale);
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
                int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return (activationScale, activationSum);
        }

        private static float ComputeQuantizationMaxRange(ReadOnlySpan<float> input, float minimum)
        {
            float maxAbs = minimum;
            for (int index = 0; index < input.Length; index++)
            {
                float absValue = MathF.Abs(input[index]);
                if (absValue > maxAbs)
                {
                    maxAbs = absValue;
                }
            }

            return maxAbs;
        }

        private static int QuantizeBitNetActivationsRange(ReadOnlySpan<float> input, Span<sbyte> quantizedValues, float activationScale)
        {
            int activationSum = 0;
            for (int index = 0; index < input.Length; index++)
            {
                int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return activationSum;
        }

        private static void ProjectLmHeadRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output)
        {
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
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
    }
}
