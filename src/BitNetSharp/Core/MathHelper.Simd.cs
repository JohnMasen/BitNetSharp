using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static void ForwardRmsNormSimd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output, int threads = 0)
        {
            EnsureRmsNormSimdSupported(input.Length);
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> normWeightsSpan = normWeights.Span;
            Span<float> outputSpan = output.Span;
            ValidateRmsNormDestination(inputSpan, normWeightsSpan, outputSpan);

            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareSimd(inputSpan, epsilon);
            if (threads == 1 || input.Length <= Vector256<float>.Count)
            {
                FillRmsNormSimdRange(inputSpan, normWeightsSpan, inverseRootMeanSquare, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(
                output.Span,
                (startIndex, endIndex) => FillRmsNormSimdRange(input.Span, normWeights.Span, inverseRootMeanSquare, output.Span, startIndex, endIndex),
                threads,
                Vector256<float>.Count * sizeof(float));
        }

        public static float[] ForwardRmsNormSimd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, int threads = 0)
        {
            float[] output = new float[input.Length];
            ForwardRmsNormSimd(input, normWeights, epsilon, output, threads);
            return output;
        }

        public static void AddSimd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            ReadOnlySpan<float> addendSpan = addend.Span;
            Span<float> outputSpan = output.Span;
            ValidateAddDestination(inputSpan, addendSpan, outputSpan);
            EnsureAddSimdSupported();

            if (threads == 1 || input.Length <= Vector256<float>.Count)
            {
                FillAddSimdRange(inputSpan, addendSpan, outputSpan, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(
                output.Span,
                (startIndex, endIndex) => FillAddSimdRange(input.Span, addend.Span, output.Span, startIndex, endIndex),
                threads,
                Vector256<float>.Count * sizeof(float));
        }

        public static float[] AddSimd(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, int threads = 0)
        {
            float[] output = new float[input.Length];
            AddSimd(input, addend, output, threads);
            return output;
        }

        internal static void ForwardSoftmaxSimd(ReadOnlySpan<float> input, Span<float> output, int threads = 0)
        {
            ValidateSoftmaxDestination(input, output);
            EnsureSoftmaxSimdSupported();

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxSimdCore(input, output);
                return;
            }

            using IMemoryOwner<float> inputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            using IMemoryOwner<float> outputOwner = MemoryPool<float>.Shared.Rent(input.Length);
            Memory<float> inputMemory = inputOwner.Memory[..input.Length];
            Memory<float> outputMemory = outputOwner.Memory[..input.Length];
            input.CopyTo(inputMemory.Span);
            ForwardSoftmaxSimd(inputMemory, outputMemory, threads);
            outputMemory.Span.CopyTo(output);
        }

        internal static void ForwardSoftmaxSimd(ReadOnlyMemory<float> input, Memory<float> output, int threads = 0)
        {
            ReadOnlySpan<float> inputSpan = input.Span;
            Span<float> outputSpan = output.Span;
            ValidateSoftmaxDestination(inputSpan, outputSpan);
            EnsureSoftmaxSimdSupported();

            if (threads == 1 || input.Length <= 1)
            {
                ForwardSoftmaxSimdCore(inputSpan, outputSpan);
                return;
            }

            ThreadHelper.WorkRange[] ranges = ThreadHelper.CreateRanges(input.Length, threads, sizeof(float), Vector256<float>.Count * sizeof(float));
            if (ranges.Length <= 1)
            {
                ForwardSoftmaxSimdCore(inputSpan, outputSpan);
                return;
            }

            float[] partialMaxima = new float[ranges.Length];
            Parallel.For(0, ranges.Length, new ParallelOptions { MaxDegreeOfParallelism = ranges.Length }, rangeIndex =>
            {
                ThreadHelper.WorkRange range = ranges[rangeIndex];
                partialMaxima[rangeIndex] = ComputeSoftmaxMaxSimd(input.Span.Slice(range.StartIndex, range.EndIndex - range.StartIndex));
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
                partialSums[rangeIndex] = FillSoftmaxExponentSimdRange(input.Span, output.Span, maxValue, range.StartIndex, range.EndIndex);
            });

            double sum = 0d;
            for (int rangeIndex = 0; rangeIndex < partialSums.Length; rangeIndex++)
            {
                sum += partialSums[rangeIndex];
            }

            float inverseSum = (float)(1d / sum);
            ThreadHelper.ForEachRange(output.Span, (startIndex, endIndex) =>
                NormalizeSoftmaxOutputSimdRange(output.Span, inverseSum, startIndex, endIndex), threads, Vector256<float>.Count * sizeof(float));
        }

        private static void ForwardSoftmaxSimdCore(ReadOnlySpan<float> input, Span<float> output)
        {
            float maxValue = ComputeSoftmaxMaxSimd(input);

            float sum = 0f;
            for (int index = 0; index < input.Length; index++)
            {
                float exponent = MathF.Exp(input[index] - maxValue);
                output[index] = exponent;
                sum += exponent;
            }

            NormalizeSoftmaxOutputSimd(output[..input.Length], sum);
        }

        private static double FillSoftmaxExponentSimdRange(ReadOnlySpan<float> input, Span<float> output, float maxValue, int startIndex, int endIndex)
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

        private static void FillRmsNormSimdRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float inverseRootMeanSquare, Span<float> output, int startIndex, int endIndex)
        {
            const int SimdWidth = 8;
            const int UnrolledWidth = SimdWidth * 4;

            Vector256<float> inverseRootMeanSquareVector = Vector256.Create(inverseRootMeanSquare);
            ref float inputRef = ref MemoryMarshal.GetReference(input);
            ref float normWeightsRef = ref MemoryMarshal.GetReference(normWeights);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            int unrolledEnd = endIndex - ((endIndex - startIndex) % UnrolledWidth);
            for (int index = startIndex; index < unrolledEnd; index += UnrolledWidth)
            {
                Vector256<float> inputVector0 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> normWeightVector0 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> outputVector0 = Avx.Multiply(Avx.Multiply(inputVector0, inverseRootMeanSquareVector), normWeightVector0);
                outputVector0.StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 0)));

                Vector256<float> inputVector1 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> normWeightVector1 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> outputVector1 = Avx.Multiply(Avx.Multiply(inputVector1, inverseRootMeanSquareVector), normWeightVector1);
                outputVector1.StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 1)));

                Vector256<float> inputVector2 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> normWeightVector2 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> outputVector2 = Avx.Multiply(Avx.Multiply(inputVector2, inverseRootMeanSquareVector), normWeightVector2);
                outputVector2.StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 2)));

                Vector256<float> inputVector3 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> normWeightVector3 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> outputVector3 = Avx.Multiply(Avx.Multiply(inputVector3, inverseRootMeanSquareVector), normWeightVector3);
                outputVector3.StoreUnsafe(ref outputRef, (nuint)(index + (SimdWidth * 3)));
            }

            for (int index = unrolledEnd; index < endIndex; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> normWeightVector = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)index);
                Vector256<float> outputVector = Avx.Multiply(Avx.Multiply(inputVector, inverseRootMeanSquareVector), normWeightVector);
                outputVector.StoreUnsafe(ref outputRef, (nuint)index);
            }
        }

        private static void FillAddSimdRange(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output, int startIndex, int endIndex)
        {
            const int SimdWidth = 8;
            ref float inputRef = ref MemoryMarshal.GetReference(input);
            ref float addendRef = ref MemoryMarshal.GetReference(addend);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            int vectorizedEnd = endIndex - ((endIndex - startIndex) % SimdWidth);
            for (int index = startIndex; index < vectorizedEnd; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> addendVector = Vector256.LoadUnsafe(ref addendRef, (nuint)index);
                Avx.Add(inputVector, addendVector).StoreUnsafe(ref outputRef, (nuint)index);
            }

            for (int index = vectorizedEnd; index < endIndex; index++)
            {
                output[index] = input[index] + addend[index];
            }
        }

        public static void ProjectBitNetI2Simd(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input.Span, packedWeights.Span, outputLength);
            EnsureBitNetProjectionSimdSupported();
            ValidateProjectionDestination(outputLength, output.Span);

            //quantize the float activations once so the SIMD kernel can reuse the same 128-value blocks for every output row
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = QuantizeBitNetActivations(input.Span, quantizedValues.Span);
            ProjectBitNetI2Simd(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output, threads);
        }

        public static float[] ProjectBitNetI2Simd(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Simd(input, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        internal static void ProjectBitNetI2Simd(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, int threads = 0)
        {
            ValidateBitNetProjectionArguments(quantizedValues.Span, packedWeights.Span, outputLength);
            EnsureBitNetProjectionSimdSupported();
            Span<float> outputSpan = output.Span;
            ValidateProjectionDestination(outputLength, outputSpan);

            const int PackedVectorWidth = 32;
            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, 1);
            if (threads == 1 || outputLength <= 1)
            {
                ProjectBitNetI2SimdRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, outputSpan, 0, outputLength);
                return;
            }

            ThreadHelper.ForEachRange(
                outputLength,
                (startIndex, endIndex) => ProjectBitNetI2SimdRange(quantizedValues.Span, packedWeights.Span, packedRowByteCount, activationScale, weightScale, output.Span, startIndex, endIndex),
                threads,
                packedRowByteCount,
                PackedVectorWidth);
        }

        internal static float[] ProjectBitNetI2Simd(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            float[] output = new float[outputLength];
            ProjectBitNetI2Simd(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output, threads);
            return output;
        }

        private static void ProjectBitNetI2SimdRange(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int packedRowByteCount, float activationScale, float weightScale, Span<float> output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //run the packed row through the SIMD mapped-dot kernel, then restore the final float projection value
                ReadOnlySpan<byte> packedRow = packedWeights.Slice(outputIndex * packedRowByteCount, packedRowByteCount);
                int mappedDot = ComputeBitNetMappedDotSimd(quantizedValues, packedRow);
                output[outputIndex] = FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }

        private static void EnsureBitNetProjectionSimdSupported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                throw new NotSupportedException("BitNet SIMD projection requires AVX2 support.");
            }
        }

        private static void EnsureAddSimdSupported()
        {
            if (!Avx.IsSupported)
            {
                throw new NotSupportedException("Residual SIMD implementation requires AVX support.");
            }
        }

        private static void EnsureSoftmaxSimdSupported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                throw new NotSupportedException("Softmax SIMD implementation requires AVX2 support.");
            }
        }

        private static int ComputeBitNetMappedDotSimd(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedRow)
        {
            const int PackedVectorWidth = 32;
            const int ActivationVectorWidth = 128;

            int mappedDot = 0;
            int vectorizedPackedLength = packedRow.Length - (packedRow.Length % PackedVectorWidth);
            for (int packedIndex = 0; packedIndex < vectorizedPackedLength; packedIndex += PackedVectorWidth)
            {
                //each 32-byte packed vector expands to one 128-activation block handled by VectorProcessOne
                int activationIndex = packedIndex * 4;
                mappedDot += VectorProcessOne(
                    quantizedValues.Slice(activationIndex, ActivationVectorWidth),
                    packedRow.Slice(packedIndex, PackedVectorWidth));
            }

            int scalarActivationIndex = vectorizedPackedLength * 4;
            for (int packedIndex = vectorizedPackedLength; packedIndex < packedRow.Length; packedIndex++)
            {
                //finish the tail with scalar decoding when the packed row is not an exact SIMD-width multiple
                byte packedWeight = packedRow[packedIndex];
                mappedDot += quantizedValues[scalarActivationIndex++] * DecodeBitNetWeight(packedWeight, 0);
                mappedDot += quantizedValues[scalarActivationIndex++] * DecodeBitNetWeight(packedWeight, 1);
                mappedDot += quantizedValues[scalarActivationIndex++] * DecodeBitNetWeight(packedWeight, 2);
                mappedDot += quantizedValues[scalarActivationIndex++] * DecodeBitNetWeight(packedWeight, 3);
            }

            return mappedDot;
        }

        private static void EnsureRmsNormSimdSupported(int inputLength)
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ComputeSoftmaxMaxSimd(ReadOnlySpan<float> input)
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
        private static void NormalizeSoftmaxOutputSimd(Span<float> output, float sum)
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

        private static void NormalizeSoftmaxOutputSimdRange(Span<float> output, float inverseSum, int startIndex, int endIndex)
        {
            const int SimdWidth = 8;
            int vectorizedEnd = endIndex - ((endIndex - startIndex) % SimdWidth);
            Vector256<float> inverseSumVector = Vector256.Create(inverseSum);
            ref float outputRef = ref MemoryMarshal.GetReference(output);
            for (int index = startIndex; index < vectorizedEnd; index += SimdWidth)
            {
                Vector256<float> outputVector = Vector256.LoadUnsafe(ref outputRef, (nuint)index);
                Avx.Multiply(outputVector, inverseSumVector).StoreUnsafe(ref outputRef, (nuint)index);
            }

            for (int index = vectorizedEnd; index < endIndex; index++)
            {
                output[index] *= inverseSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ComputeRmsNormInverseRootMeanSquareSimd(ReadOnlySpan<float> input, float epsilon)
        {
            double meanSquare = AccumulateRmsNormSumSquaresSimd(input) / input.Length;
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
        private static double AccumulateRmsNormSumSquaresSimd(ReadOnlySpan<float> source)
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
            int remainderStart = unrolledLength;
            for (int index = remainderStart; index < source.Length; index += SimdWidth)
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
    }
}
