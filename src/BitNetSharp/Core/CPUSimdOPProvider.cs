using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides the SIMD-accelerated CPU implementation of math operations.
    /// </summary>
    public sealed class CPUSimdOPProvider : IOPProvider
    {
        private static readonly Vector256<byte> v256_3 = Vector256.Create((byte)0b_0000_0011);
        private static readonly Vector256<byte> v256_2 = Vector256.Create((byte)0b_0000_0010);
        public CPUSimdOPProvider(int threadCount = Nodes.InferenceConfig.AutoThreadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            ThreadCount = threadCount;
        }

        public string Backend => "SIMD";

        public int ThreadCount { get; }

        public (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlyMemory<float> input, Memory<sbyte> quantizedValues)
        {
            return QuantizeBitNetActivations(input, quantizedValues, ThreadCount);
        }

        public void Add(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> addendSpan = addend.Span;
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateAddDestination(inputSpan, addendSpan, outputSpan);
            EnsureAddSupported();

            if (ThreadCount == 1 || input.Length <= Vector256<float>.Count)
            {
                FillAddRange(inputSpan, addendSpan, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                FillAddRange(
                    input.Span.Slice(startIndex, endIndex - startIndex),
                    addend.Span.Slice(startIndex, endIndex - startIndex),
                    output.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
        }

        public void ProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            ValidationHelper.ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            EnsureBitNetProjectionSupported();
            ValidationHelper.ValidateProjectionDestination(outputLength, output.Span);

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input, quantizedValues, ThreadCount);
            ProjectBitNetI2(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output);
        }

        public void ProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output)
        {
            ValidationHelper.ValidateBitNetProjectionArguments(quantizedValues.Span, packedWeights.Span, outputLength);
            EnsureBitNetProjectionSupported();
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateProjectionDestination(outputLength, outputSpan);

            const int PackedVectorWidth = 32;
            // Each packed byte stores four 2-bit weights, so one output row uses inputLength / 4 bytes.
            int packedRowByteCount = checked(quantizedValues.Length / 4);
            if (ThreadCount == 1 || outputLength <= 1)
            {
                ProjectBitNetI2Range(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(outputLength, (startIndex, endIndex) =>
                ProjectBitNetI2Range(
                    quantizedValues.Span,
                    packedWeights.Span.Slice(startIndex * packedRowByteCount, (endIndex - startIndex) * packedRowByteCount),
                    packedRowByteCount,
                    activationScale,
                    weightScale,
                    output.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, packedRowByteCount, PackedVectorWidth);
        }

        public void ForwardSoftmax(ReadOnlySpan<float> input, Span<float> output)
        {
            ValidationHelper.ValidateSoftmaxDestination(input, output);
            EnsureSoftmaxSupported();

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

        public void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquare(input, epsilon, ThreadCount);
            if (ThreadCount == 1 || input.Length <= Vector256<float>.Count)
            {
                FillRmsNormRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan);
                return;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                FillRmsNormRange(
                    input.Span.Slice(startIndex, endIndex - startIndex),
                    normWeights.Span.Slice(startIndex, endIndex - startIndex),
                    inverseRootMeanSquare,
                    output.Span.Slice(startIndex, endIndex - startIndex)), ThreadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
        }

        public void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output)
        {
            ValidationHelper.ValidateLmHeadArguments(input, embeddingWeights, rowLength, vocabularySize, output);
            ReadOnlySpan<Half> embeddingWeightsSpan = MemoryMarshal.Cast<byte, Half>(embeddingWeights.Span);

            if (ThreadCount == 1 || vocabularySize <= 1)
            {
                ProjectLmHeadSimdRange(input.Span, embeddingWeightsSpan, rowLength, output.Span[..vocabularySize]);
                return;
            }

            ThreadHelper.ForEachRange(
                vocabularySize,
                (startIndex, endIndex) => ProjectLmHeadSimdRange(
                    input.Span,
                    MemoryMarshal.Cast<byte, Half>(embeddingWeights.Span.Slice(startIndex * rowLength * sizeof(ushort), (endIndex - startIndex) * rowLength * sizeof(ushort))),
                    rowLength,
                    output.Span.Slice(startIndex, endIndex - startIndex)),
                ThreadCount,
                checked(rowLength * sizeof(ushort)),
                Vector256<float>.Count * sizeof(float));
        }

        private void ExecuteForwardSoftmaxMemory(ReadOnlyMemory<float> input, Memory<float> output)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            Span<float> outputSpan = output.Span;
            ValidationHelper.ValidateSoftmaxDestination(inputSpan, outputSpan);
            EnsureSoftmaxSupported();

            if (ThreadCount == 1 || input.Length <= 1)
            {
                ForwardSoftmaxCore(inputSpan, outputSpan);
                return;
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, ThreadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
            if (ranges.Length <= 1)
            {
                ForwardSoftmaxCore(inputSpan, outputSpan);
                return;
            }

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = ComputeSoftmaxMax(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                NormalizeSoftmaxOutputRange(output.Span.Slice(startIndex, endIndex - startIndex), inverseSum), ThreadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
        }

        private static void ForwardSoftmaxCore(ReadOnlySpan<float> input, Span<float> output)
        {
            float maxValue = ComputeSoftmaxMax(input);
            float sum = 0f;
            for (int index = 0; index < input.Length; index++)
            {
                float exponent = MathF.Exp(input[index] - maxValue);
                output[index] = exponent;
                sum += exponent;
            }

            NormalizeSoftmaxOutput(output[..input.Length], sum);
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

        private static void FillRmsNormRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float inverseRootMeanSquare, Span<float> output)
        {
            const int SimdWidth = 8;
            const int UnrolledWidth = SimdWidth * 4;

            Vector256<float> inverseRootMeanSquareVector = Vector256.Create(inverseRootMeanSquare);
            ref float inputRef = ref MemoryMarshal.GetReference(input);
            ref float normWeightsRef = ref MemoryMarshal.GetReference(normWeights);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            int unrolledLength = output.Length - (output.Length % UnrolledWidth);
            for (int index = 0; index < unrolledLength; index += UnrolledWidth)
            {
                Vector256<float> inputVector0 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> normWeightVector0 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 0)));
                Avx.Multiply(Avx.Multiply(inputVector0, inverseRootMeanSquareVector), normWeightVector0).StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 0)));

                Vector256<float> inputVector1 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> normWeightVector1 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 1)));
                Avx.Multiply(Avx.Multiply(inputVector1, inverseRootMeanSquareVector), normWeightVector1).StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 1)));

                Vector256<float> inputVector2 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> normWeightVector2 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 2)));
                Avx.Multiply(Avx.Multiply(inputVector2, inverseRootMeanSquareVector), normWeightVector2).StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 2)));

                Vector256<float> inputVector3 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> normWeightVector3 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 3)));
                Avx.Multiply(Avx.Multiply(inputVector3, inverseRootMeanSquareVector), normWeightVector3).StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 3)));
            }

            for (int index = unrolledLength; index < output.Length; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> normWeightVector = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)index);
                Avx.Multiply(Avx.Multiply(inputVector, inverseRootMeanSquareVector), normWeightVector).StoreUnsafe(ref outputRef, (nuint)index);
            }
        }

        private static void FillAddRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
        {
            const int SimdWidth = 8;
            ref float inputRef = ref MemoryMarshal.GetReference(input);
            ref float addendRef = ref MemoryMarshal.GetReference(addend);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            int vectorizedLength = output.Length - (output.Length % SimdWidth);
            for (int index = 0; index < vectorizedLength; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> addendVector = Vector256.LoadUnsafe(ref addendRef, (nuint)index);
                Avx.Add(inputVector, addendVector).StoreUnsafe(ref outputRef, (nuint)index);
            }

            for (int index = vectorizedLength; index < output.Length; index++)
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

        private static void ProjectLmHeadSimdRange(ReadOnlySpan<float> input, ReadOnlySpan<Half> embeddingWeights, int rowLength, Span<float> output)
        {
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
            {
                int rowOffset = outputIndex * rowLength;
                output[outputIndex] = ComputeLmHeadDotSimd(input, embeddingWeights.Slice(rowOffset, rowLength));
            }
        }

        private static float ComputeLmHeadDotSimd(ReadOnlySpan<float> input, ReadOnlySpan<Half> weights)
        {
            //use tensor conver to enable hardware-acclerated convert, the only way to use FP16C
            Span<float> weights_float = stackalloc float[weights.Length];
            TensorPrimitives.ConvertToSingle(weights, weights_float);
            float sum = TensorPrimitives.Dot(input, weights_float);
            return sum;
            
        }

        internal static (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlyMemory<float> input, Memory<sbyte> quantizedValues, int threadCount)
        {
            const float MinimumBitNetQuantizationMax = 0.00001f;

            if (quantizedValues.Length < input.Length)
            {
                throw new ArgumentException("Quantized output length must be at least the input length.", nameof(quantizedValues));
            }

            if (threadCount == 1 || input.Length < Vector256<float>.Count)
            {
                return QuantizeBitNetActivationsSingleThread(input.Span, quantizedValues.Span);
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
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
            if (Avx.IsSupported && input.Length >= Vector256<float>.Count)
            {
                const int SimdWidth = 8;
                int vectorizedLength = input.Length - (input.Length % SimdWidth);
                ref float inputRef = ref MemoryMarshal.GetReference(input);
                Vector256<float> signMask = Vector256.Create(-0.0f);
                Vector256<float> maxVector = Vector256.Create(maxAbs);

                for (int index = 0; index < vectorizedLength; index += SimdWidth)
                {
                    Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                    Vector256<float> absoluteVector = Avx.AndNot(signMask, inputVector);
                    maxVector = Avx.Max(maxVector, absoluteVector);
                }

                maxAbs = MathF.Max(maxAbs, HorizontalMax(maxVector));
                for (int index = vectorizedLength; index < input.Length; index++)
                {
                    float absValue = MathF.Abs(input[index]);
                    if (absValue > maxAbs)
                    {
                        maxAbs = absValue;
                    }
                }
            }
            else
            {
                for (int index = 0; index < input.Length; index++)
                {
                    float absValue = MathF.Abs(input[index]);
                    if (absValue > maxAbs)
                    {
                        maxAbs = absValue;
                    }
                }
            }

            float activationScale = 127f / maxAbs;
            int activationSum = 0;
            if (Avx.IsSupported && input.Length >= Vector256<float>.Count)
            {
                const int SimdWidth = 8;
                int vectorizedLength = input.Length - (input.Length % SimdWidth);
                ref float inputRef = ref MemoryMarshal.GetReference(input);
                Vector256<float> scaleVector = Vector256.Create(activationScale);
                Span<float> scaledValues = stackalloc float[SimdWidth];

                for (int index = 0; index < vectorizedLength; index += SimdWidth)
                {
                    Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                    Avx.Multiply(inputVector, scaleVector).CopyTo(scaledValues);
                    for (int offset = 0; offset < SimdWidth; offset++)
                    {
                        int quantizedValue = (int)MathF.Round(scaledValues[offset], MidpointRounding.ToEven);
                        quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                        quantizedValues[index + offset] = (sbyte)quantizedValue;
                        activationSum += quantizedValue;
                    }
                }

                for (int index = vectorizedLength; index < input.Length; index++)
                {
                    int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                    quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                    quantizedValues[index] = (sbyte)quantizedValue;
                    activationSum += quantizedValue;
                }
            }
            else
            {
                for (int index = 0; index < input.Length; index++)
                {
                    int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                    quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                    quantizedValues[index] = (sbyte)quantizedValue;
                    activationSum += quantizedValue;
                }
            }

            return (activationScale, activationSum);
        }

        private static float ComputeQuantizationMaxRange(ReadOnlySpan<float> input, float minimum)
        {
            float maxAbs = minimum;
            if (Avx.IsSupported && input.Length >= Vector256<float>.Count)
            {
                const int SimdWidth = 8;
                int vectorizedLength = input.Length - (input.Length % SimdWidth);
                ref float inputRef = ref MemoryMarshal.GetReference(input);
                Vector256<float> signMask = Vector256.Create(-0.0f);
                Vector256<float> maxVector = Vector256.Create(maxAbs);
                for (int index = 0; index < vectorizedLength; index += SimdWidth)
                {
                    Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                    Vector256<float> absoluteVector = Avx.AndNot(signMask, inputVector);
                    maxVector = Avx.Max(maxVector, absoluteVector);
                }

                maxAbs = MathF.Max(maxAbs, HorizontalMax(maxVector));
                for (int index = vectorizedLength; index < input.Length; index++)
                {
                    float absValue = MathF.Abs(input[index]);
                    if (absValue > maxAbs)
                    {
                        maxAbs = absValue;
                    }
                }

                return maxAbs;
            }

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
            if (Avx.IsSupported && input.Length >= Vector256<float>.Count)
            {
                const int SimdWidth = 8;
                int vectorizedLength = input.Length - (input.Length % SimdWidth);
                ref float inputRef = ref MemoryMarshal.GetReference(input);
                Vector256<float> scaleVector = Vector256.Create(activationScale);
                Span<float> scaledValues = stackalloc float[SimdWidth];

                for (int index = 0; index < vectorizedLength; index += SimdWidth)
                {
                    Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                    Avx.Multiply(inputVector, scaleVector).CopyTo(scaledValues);
                    for (int offset = 0; offset < SimdWidth; offset++)
                    {
                        int quantizedValue = (int)MathF.Round(scaledValues[offset], MidpointRounding.ToEven);
                        quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                        quantizedValues[index + offset] = (sbyte)quantizedValue;
                        activationSum += quantizedValue;
                    }
                }

                for (int index = vectorizedLength; index < input.Length; index++)
                {
                    int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                    quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                    quantizedValues[index] = (sbyte)quantizedValue;
                    activationSum += quantizedValue;
                }

                return activationSum;
            }

            for (int index = 0; index < input.Length; index++)
            {
                int quantizedValue = (int)MathF.Round(input[index] * activationScale, MidpointRounding.ToEven);
                quantizedValue = Math.Clamp(quantizedValue, sbyte.MinValue, sbyte.MaxValue);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return activationSum;
        }

        private static void EnsureBitNetProjectionSupported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                throw new NotSupportedException("BitNet SIMD projection requires AVX2 support.");
            }
        }

        private static void EnsureAddSupported()
        {
            if (!Avx.IsSupported)
            {
                throw new NotSupportedException("Residual SIMD implementation requires AVX support.");
            }
        }

        private static void EnsureSoftmaxSupported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                throw new NotSupportedException("Softmax SIMD implementation requires AVX2 support.");
            }
        }

        private static void EnsureRmsNormSupported(int inputLength)
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                throw new NotSupportedException("RMSNorm SIMD implementation requires AVX2 support.");
            }

            if (inputLength % Vector256<float>.Count != 0)
            {
                throw new NotSupportedException("RMSNorm SIMD implementation requires an input length divisible by 8.");
            }
        }

        private static int ComputeBitNetMappedDot(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedRow)
        {
            const int PackedVectorWidth = 32;
            const int ActivationVectorWidth = 128;

            int mappedDot = 0;
            int vectorizedPackedLength = packedRow.Length - (packedRow.Length % PackedVectorWidth);
            for (int packedIndex = 0; packedIndex < vectorizedPackedLength; packedIndex += PackedVectorWidth)
            {
                int activationIndex = packedIndex * 4;
                mappedDot += VectorProcessOne(quantizedValues.Slice(activationIndex, ActivationVectorWidth), packedRow.Slice(packedIndex, PackedVectorWidth));
            }

            int scalarActivationIndex = vectorizedPackedLength * 4;
            for (int packedIndex = vectorizedPackedLength; packedIndex < packedRow.Length; packedIndex++)
            {
                byte packedWeight = packedRow[packedIndex];
                mappedDot += quantizedValues[scalarActivationIndex++] * CPUDefaultOPProvider.DecodeBitNetWeight(packedWeight, 0);
                mappedDot += quantizedValues[scalarActivationIndex++] * CPUDefaultOPProvider.DecodeBitNetWeight(packedWeight, 1);
                mappedDot += quantizedValues[scalarActivationIndex++] * CPUDefaultOPProvider.DecodeBitNetWeight(packedWeight, 2);
                mappedDot += quantizedValues[scalarActivationIndex++] * CPUDefaultOPProvider.DecodeBitNetWeight(packedWeight, 3);
            }

            return mappedDot;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ComputeSoftmaxMax(ReadOnlySpan<float> input)
        {
            const int SimdWidth = 8;
            int vectorizedLength = input.Length - (input.Length % SimdWidth);
            if (vectorizedLength == 0)
            {
                return input[0];
            }

            ref float inputRef = ref MemoryMarshal.GetReference(input);
            Vector256<float> maxVector = Vector256.LoadUnsafe(ref inputRef);
            for (int index = SimdWidth; index < vectorizedLength; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                maxVector = Avx.Max(maxVector, inputVector);
            }

            float maxValue = HorizontalMax(maxVector);
            for (int index = vectorizedLength; index < input.Length; index++)
            {
                if (input[index] > maxValue)
                {
                    maxValue = input[index];
                }
            }

            return maxValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NormalizeSoftmaxOutput(Span<float> output, float sum)
        {
            const int SimdWidth = 8;
            int vectorizedLength = output.Length - (output.Length % SimdWidth);
            Vector256<float> sumVector = Vector256.Create(sum);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            for (int index = 0; index < vectorizedLength; index += SimdWidth)
            {
                Vector256<float> outputVector = Vector256.LoadUnsafe(ref outputRef, (nuint)index);
                Avx.Divide(outputVector, sumVector).StoreUnsafe(ref outputRef, (nuint)index);
            }

            for (int index = vectorizedLength; index < output.Length; index++)
            {
                output[index] /= sum;
            }
        }

        private static void NormalizeSoftmaxOutputRange(Span<float> output, float inverseSum)
        {
            const int SimdWidth = 8;
            int vectorizedLength = output.Length - (output.Length % SimdWidth);
            Vector256<float> inverseSumVector = Vector256.Create(inverseSum);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            for (int index = 0; index < vectorizedLength; index += SimdWidth)
            {
                Vector256<float> outputVector = Vector256.LoadUnsafe(ref outputRef, (nuint)index);
                Avx.Multiply(outputVector, inverseSumVector).StoreUnsafe(ref outputRef, (nuint)index);
            }

            for (int index = vectorizedLength; index < output.Length; index++)
            {
                output[index] *= inverseSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ComputeRmsNormInverseRootMeanSquare(ReadOnlyMemory<float> input, float epsilon, int threadCount)
        {
            if (threadCount == 1 || input.Length <= Vector256<float>.Count)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threadCount, sizeof(float), Vector256<float>.Count * sizeof(float));
            if (ranges.Length <= 1)
            {
                return ComputeRmsNormInverseRootMeanSquareSingleThread(input.Span, epsilon);
            }

            double[] partialSums = new double[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialSums[rangeIndex] = AccumulateRmsNormSumSquares(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
            double meanSquare = AccumulateRmsNormSumSquares(input) / input.Length;
            return (float)(1d / Math.Sqrt(meanSquare + epsilon));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalMax(Vector256<float> vector)
        {
            Span<float> values = stackalloc float[Vector256<float>.Count];
            vector.CopyTo(values);

            float maxValue = values[0];
            for (int index = 1; index < values.Length; index++)
            {
                if (values[index] > maxValue)
                {
                    maxValue = values[index];
                }
            }

            return maxValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double AccumulateRmsNormSumSquares(ReadOnlySpan<float> source)
        {
            const int SimdWidth = 8;
            const int UnrolledWidth = SimdWidth * 4;

            Vector256<float> sum0 = Vector256<float>.Zero;
            Vector256<float> sum1 = Vector256<float>.Zero;
            Vector256<float> sum2 = Vector256<float>.Zero;
            Vector256<float> sum3 = Vector256<float>.Zero;
            ref float sourceRef = ref MemoryMarshal.GetReference(source);
            int unrolledLength = source.Length - (source.Length % UnrolledWidth);
            for (int index = 0; index < unrolledLength; index += UnrolledWidth)
            {
                Vector256<float> inputVector0 = Vector256.LoadUnsafe(ref sourceRef, (nuint)(index + (SimdWidth * 0)));
                sum0 = Avx.Add(sum0, Avx.Multiply(inputVector0, inputVector0));
                Vector256<float> inputVector1 = Vector256.LoadUnsafe(ref sourceRef, (nuint)(index + (SimdWidth * 1)));
                sum1 = Avx.Add(sum1, Avx.Multiply(inputVector1, inputVector1));
                Vector256<float> inputVector2 = Vector256.LoadUnsafe(ref sourceRef, (nuint)(index + (SimdWidth * 2)));
                sum2 = Avx.Add(sum2, Avx.Multiply(inputVector2, inputVector2));
                Vector256<float> inputVector3 = Vector256.LoadUnsafe(ref sourceRef, (nuint)(index + (SimdWidth * 3)));
                sum3 = Avx.Add(sum3, Avx.Multiply(inputVector3, inputVector3));
            }

            Vector256<float> sumVector = Avx.Add(Avx.Add(sum0, sum1), Avx.Add(sum2, sum3));
            for (int index = unrolledLength; index < source.Length; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref sourceRef, (nuint)index);
                sumVector = Avx.Add(sumVector, Avx.Multiply(inputVector, inputVector));
            }

            return HorizontalSum(sumVector);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> vector)
        {
            Vector256<float> sum = Avx.HorizontalAdd(vector, vector);
            sum = Avx.HorizontalAdd(sum, sum);
            Vector256<float> upper = Avx.Permute2x128(sum, sum, 0b_0000_0001);
            return Avx.Add(sum, upper).ToScalar();
        }

        private static int VectorProcessOne(ReadOnlySpan<sbyte> dataBlock, ReadOnlySpan<byte> weightBlock)
        {
            if (dataBlock.Length != 128)
            {
                throw new ArgumentException("dataBlock length must be 128", nameof(dataBlock));
            }

            if (weightBlock.Length != 32)
            {
                throw new ArgumentException("weightBlock length must be 32", nameof(weightBlock));
            }

            Vector256<sbyte> data0 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock));
            Vector256<sbyte> data1 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(32)));
            Vector256<sbyte> data2 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(64)));
            Vector256<sbyte> data3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(96)));
            Vector256<byte> weight3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weightBlock));

            Vector256<byte> weight0 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 6), v256_3);
            Vector256<byte> weight1 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 4), v256_3);
            Vector256<byte> weight2 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 2), v256_3);
            weight3 = NormalizeBitNetWeightCodes(weight3, v256_3);

            return Sum4(ProcessBlock(data0, weight0), ProcessBlock(data1, weight1), ProcessBlock(data2, weight2), ProcessBlock(data3, weight3));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> ProcessBlock(Vector256<sbyte> data, Vector256<byte> weight)
        {
            Vector256<short> result0 = Avx2.MultiplyAddAdjacent(weight, data);
            Vector256<short> expandedData = Avx2.MultiplyAddAdjacent(Vector256<byte>.One, data);
            Vector256<short> mapped = Avx2.Subtract(result0, expandedData);
            return Avx2.MultiplyAddAdjacent(mapped, Vector256<short>.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Sum4(Vector256<int> v1, Vector256<int> v2, Vector256<int> v3, Vector256<int> v4)
        {
            Vector256<int> sum = Avx2.Add(Avx2.Add(v1, v2), Avx2.Add(v3, v4));
            sum = Avx2.HorizontalAdd(sum, sum);
            sum = Avx2.HorizontalAdd(sum, sum);
            Vector256<int> upper = Avx2.Permute2x128(sum, sum, 0b_0000_0001);
            return Avx2.Add(sum, upper).ToScalar();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> NormalizeBitNetWeightCodes(Vector256<byte> packedWeightCodes, Vector256<byte> bitMask)
        {
            Vector256<byte> weightCodes = Vector256.BitwiseAnd(packedWeightCodes, bitMask);
            Vector256<byte> threeMask = Avx2.CompareEqual(weightCodes, v256_3);
            Vector256<byte> correction = Vector256.BitwiseAnd(threeMask, v256_2);
            return Avx2.Subtract(weightCodes, correction);
        }
    }
}
