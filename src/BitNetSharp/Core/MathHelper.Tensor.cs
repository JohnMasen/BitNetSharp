using System;
using System.Numerics.Tensors;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormTensor(float[] input, float[] normWeights, float epsilon, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(normWeights);

            float sumSquares = TensorPrimitives.Dot(input, input);
            float meanSquare = sumSquares / input.Length;
            float inverseRootMeanSquare = 1f / MathF.Sqrt(meanSquare + epsilon);

            float[] output = new float[input.Length];
            if (threads == 1 || input.Length <= 1)
            {
                FillRmsNormTensorRange(input, normWeights, inverseRootMeanSquare, output, 0, input.Length);
                return output;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillRmsNormTensorRange(input, normWeights, inverseRootMeanSquare, output, startIndex, endIndex), threads);

            return output;
        }

        public static float[] ForwardRmsNormTensor(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon, int threads = 0)
        {
            if (threads != 1)
            {
                return ForwardRmsNormTensor(input.ToArray(), normWeights.ToArray(), epsilon, threads);
            }

            float sumSquares = TensorPrimitives.Dot(input, input);
            float meanSquare = sumSquares / input.Length;
            float inverseRootMeanSquare = 1f / MathF.Sqrt(meanSquare + epsilon);

            float[] normalized = new float[input.Length];
            TensorPrimitives.Multiply(input, inverseRootMeanSquare, normalized);

            float[] output = new float[input.Length];
            TensorPrimitives.Multiply(normalized, normWeights, output);
            return output;
        }

        private static void FillRmsNormTensorRange(float[] input, float[] normWeights, float inverseRootMeanSquare, float[] output, int startIndex, int endIndex)
        {
            Span<float> outputRange = output.AsSpan(startIndex, endIndex - startIndex);
            TensorPrimitives.Multiply(input.AsSpan(startIndex, endIndex - startIndex), inverseRootMeanSquare, outputRange);
            TensorPrimitives.Multiply(outputRange, normWeights.AsSpan(startIndex, endIndex - startIndex), outputRange);
        }

        public static float[] ProjectBitNetI2Tensor(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);

            //quantize the float activations first, then convert them to float so TensorPrimitives can consume them
            (sbyte[] quantizedValuesRaw, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Tensor(quantizedValuesRaw, activationScale, packedWeights.ToArray(), outputLength, weightScale, threads);
        }

        public static float[] ProjectBitNetI2Tensor(ReadOnlySpan<float> input, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);

            //quantize the float activations first, then convert them to float so TensorPrimitives can consume them
            (sbyte[] quantizedValuesRaw, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Tensor(quantizedValuesRaw, activationScale, packedWeights, outputLength, weightScale, threads);
        }

        internal static float[] ProjectBitNetI2Tensor(sbyte[] quantizedValuesRaw, float activationScale, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(quantizedValuesRaw);
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(quantizedValuesRaw, packedWeights, outputLength);

            float[] quantizedValues = new float[quantizedValuesRaw.Length];
            for (int index = 0; index < quantizedValuesRaw.Length; index++)
            {
                quantizedValues[index] = quantizedValuesRaw[index];
            }

            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValuesRaw.Length, 1);
            float[] output = new float[outputLength];
            if (threads == 1 || output.Length <= 1)
            {
                ProjectBitNetI2TensorRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, 0, output.Length);
                return output;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                ProjectBitNetI2TensorRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, startIndex, endIndex), threads);

            return output;
        }

        private static void ProjectBitNetI2TensorRange(float[] quantizedValues, byte[] packedWeights, int packedRowByteCount, float activationScale, float weightScale, float[] output, int startIndex, int endIndex)
        {
            float[] rowWeights = new float[quantizedValues.Length];
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //expand one packed row into dense float weights, compute the mapped dot, then rescale it back to the projection result
                ReadOnlySpan<byte> packedRow = packedWeights.AsSpan(outputIndex * packedRowByteCount, packedRowByteCount);
                Array.Clear(rowWeights);
                ExpandBitNetRowWeights(packedRow, rowWeights);

                float mappedDot = TensorPrimitives.Dot(rowWeights, quantizedValues);
                output[outputIndex] = FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }
    }
}
