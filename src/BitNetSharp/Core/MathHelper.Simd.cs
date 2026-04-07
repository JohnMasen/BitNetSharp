using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormSimd(float[] input, float[] normWeights, float epsilon, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(normWeights);

            EnsureRmsNormSimdSupported(input.Length);
            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareSimd(input, epsilon);

            float[] output = new float[input.Length];
            if (threads == 1 || output.Length <= Vector256<float>.Count)
            {
                FillRmsNormSimdRange(input, normWeights, inverseRootMeanSquare, output, 0, output.Length);
                return output;
            }

            ThreadHelper.ForEachRange(
                output.AsSpan(),
                (startIndex, endIndex) => FillRmsNormSimdRange(input, normWeights, inverseRootMeanSquare, output, startIndex, endIndex),
                threads,
                Vector256<float>.Count * sizeof(float));

            return output;
        }

        public static float[] ForwardRmsNormSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon, int threads = 0)
        {
            if (threads != 1)
            {
                return ForwardRmsNormSimd(input.ToArray(), normWeights.ToArray(), epsilon, threads);
            }

            EnsureRmsNormSimdSupported(input.Length);
            float inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareSimd(input, epsilon);

            float[] output = new float[input.Length];
            const int SimdWidth = 8;
            const int UnrolledWidth = SimdWidth * 4;
            Vector256<float> inverseRootMeanSquareVector = Vector256.Create(inverseRootMeanSquare);
            ref float inputRef = ref MemoryMarshal.GetReference(input);
            ref float normWeightsRef = ref MemoryMarshal.GetReference(normWeights);
            int unrolledLength = input.Length - (input.Length % UnrolledWidth);
            for (int index = 0; index < unrolledLength; index += UnrolledWidth)
            {
                Vector256<float> inputVector0 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> normWeightVector0 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> outputVector0 = Avx.Multiply(Avx.Multiply(inputVector0, inverseRootMeanSquareVector), normWeightVector0);
                outputVector0.CopyTo(output.AsSpan(index + (SimdWidth * 0), SimdWidth));

                Vector256<float> inputVector1 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> normWeightVector1 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> outputVector1 = Avx.Multiply(Avx.Multiply(inputVector1, inverseRootMeanSquareVector), normWeightVector1);
                outputVector1.CopyTo(output.AsSpan(index + (SimdWidth * 1), SimdWidth));

                Vector256<float> inputVector2 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> normWeightVector2 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> outputVector2 = Avx.Multiply(Avx.Multiply(inputVector2, inverseRootMeanSquareVector), normWeightVector2);
                outputVector2.CopyTo(output.AsSpan(index + (SimdWidth * 2), SimdWidth));

                Vector256<float> inputVector3 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> normWeightVector3 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> outputVector3 = Avx.Multiply(Avx.Multiply(inputVector3, inverseRootMeanSquareVector), normWeightVector3);
                outputVector3.CopyTo(output.AsSpan(index + (SimdWidth * 3), SimdWidth));
            }

            int remainderStart = unrolledLength;
            for (int index = remainderStart; index < input.Length; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> normWeightVector = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)index);
                Vector256<float> outputVector = Avx.Multiply(Avx.Multiply(inputVector, inverseRootMeanSquareVector), normWeightVector);
                outputVector.CopyTo(output.AsSpan(index, SimdWidth));
            }

            return output;
        }

        private static void FillRmsNormSimdRange(float[] input, float[] normWeights, float inverseRootMeanSquare, float[] output, int startIndex, int endIndex)
        {
            const int SimdWidth = 8;
            const int UnrolledWidth = SimdWidth * 4;

            Vector256<float> inverseRootMeanSquareVector = Vector256.Create(inverseRootMeanSquare);
            ref float inputRef = ref MemoryMarshal.GetArrayDataReference(input);
            ref float normWeightsRef = ref MemoryMarshal.GetArrayDataReference(normWeights);
            int unrolledEnd = endIndex - ((endIndex - startIndex) % UnrolledWidth);
            for (int index = startIndex; index < unrolledEnd; index += UnrolledWidth)
            {
                Vector256<float> inputVector0 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> normWeightVector0 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 0)));
                Vector256<float> outputVector0 = Avx.Multiply(Avx.Multiply(inputVector0, inverseRootMeanSquareVector), normWeightVector0);
                outputVector0.CopyTo(output.AsSpan(index + (SimdWidth * 0), SimdWidth));

                Vector256<float> inputVector1 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> normWeightVector1 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 1)));
                Vector256<float> outputVector1 = Avx.Multiply(Avx.Multiply(inputVector1, inverseRootMeanSquareVector), normWeightVector1);
                outputVector1.CopyTo(output.AsSpan(index + (SimdWidth * 1), SimdWidth));

                Vector256<float> inputVector2 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> normWeightVector2 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 2)));
                Vector256<float> outputVector2 = Avx.Multiply(Avx.Multiply(inputVector2, inverseRootMeanSquareVector), normWeightVector2);
                outputVector2.CopyTo(output.AsSpan(index + (SimdWidth * 2), SimdWidth));

                Vector256<float> inputVector3 = Vector256.LoadUnsafe(ref inputRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> normWeightVector3 = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)(index + (SimdWidth * 3)));
                Vector256<float> outputVector3 = Avx.Multiply(Avx.Multiply(inputVector3, inverseRootMeanSquareVector), normWeightVector3);
                outputVector3.CopyTo(output.AsSpan(index + (SimdWidth * 3), SimdWidth));
            }

            for (int index = unrolledEnd; index < endIndex; index += SimdWidth)
            {
                Vector256<float> inputVector = Vector256.LoadUnsafe(ref inputRef, (nuint)index);
                Vector256<float> normWeightVector = Vector256.LoadUnsafe(ref normWeightsRef, (nuint)index);
                Vector256<float> outputVector = Avx.Multiply(Avx.Multiply(inputVector, inverseRootMeanSquareVector), normWeightVector);
                outputVector.CopyTo(output.AsSpan(index, SimdWidth));
            }
        }

        public static float[] ProjectBitNetI2Simd(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);
            EnsureBitNetProjectionSimdSupported();

            //quantize the float activations once so the SIMD kernel can reuse the same 128-value blocks for every output row
            (sbyte[] quantizedValues, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Simd(quantizedValues, activationScale, packedWeights.ToArray(), outputLength, weightScale, threads);
        }

        public static float[] ProjectBitNetI2Simd(ReadOnlySpan<float> input, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);
            EnsureBitNetProjectionSimdSupported();

            //quantize the float activations once so the SIMD kernel can reuse the same 128-value blocks for every output row
            (sbyte[] quantizedValues, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Simd(quantizedValues, activationScale, packedWeights, outputLength, weightScale, threads);
        }

        internal static float[] ProjectBitNetI2Simd(sbyte[] quantizedValues, float activationScale, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(quantizedValues);
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(quantizedValues, packedWeights, outputLength);
            EnsureBitNetProjectionSimdSupported();

            const int PackedVectorWidth = 32;
            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, 1);
            float[] output = new float[outputLength];
            if (threads == 1 || output.Length <= 1)
            {
                ProjectBitNetI2SimdRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, 0, output.Length);
                return output;
            }

            ThreadHelper.ForEachRange(
                output.Length,
                (startIndex, endIndex) => ProjectBitNetI2SimdRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, startIndex, endIndex),
                threads,
                packedRowByteCount,
                PackedVectorWidth);

            return output;
        }

        private static void ProjectBitNetI2SimdRange(sbyte[] quantizedValues, byte[] packedWeights, int packedRowByteCount, float activationScale, float weightScale, float[] output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //run the packed row through the SIMD mapped-dot kernel, then restore the final float projection value
                ReadOnlySpan<byte> packedRow = packedWeights.AsSpan(outputIndex * packedRowByteCount, packedRowByteCount);
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
        private static float ComputeRmsNormInverseRootMeanSquareSimd(ReadOnlySpan<float> input, float epsilon)
        {
            double meanSquare = AccumulateRmsNormSumSquaresSimd(input) / input.Length;
            return (float)(1d / Math.Sqrt(meanSquare + epsilon));
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
