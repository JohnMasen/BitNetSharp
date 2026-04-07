using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon)
        {
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
