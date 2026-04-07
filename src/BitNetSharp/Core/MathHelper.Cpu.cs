using System;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormCpuStandard(float[] input, float[] normWeights, float epsilon, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(normWeights);

            double inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareCpuStandard(input, epsilon);
            float[] output = new float[input.Length];
            if (threads == 1 || input.Length <= 1)
            {
                FillRmsNormCpuStandardRange(input, normWeights, inverseRootMeanSquare, output, 0, input.Length);
                return output;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                FillRmsNormCpuStandardRange(input, normWeights, inverseRootMeanSquare, output, startIndex, endIndex), threads);

            return output;
        }

        public static float[] ForwardRmsNormCpuStandard(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon, int threads = 0)
        {
            if (threads != 1)
            {
                return ForwardRmsNormCpuStandard(input.ToArray(), normWeights.ToArray(), epsilon, threads);
            }

            double inverseRootMeanSquare = ComputeRmsNormInverseRootMeanSquareCpuStandard(input, epsilon);
            float[] output = new float[input.Length];
            FillRmsNormCpuStandardRange(input, normWeights, inverseRootMeanSquare, output, 0, input.Length);

            return output;
        }

        private static double ComputeRmsNormInverseRootMeanSquareCpuStandard(ReadOnlySpan<float> input, float epsilon)
        {
            double sumSquares = 0;
            for (int index = 0; index < input.Length; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            double meanSquare = sumSquares / input.Length;
            return 1d / Math.Sqrt(meanSquare + epsilon);
        }

        private static void FillRmsNormCpuStandardRange(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, double inverseRootMeanSquare, float[] output, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * normWeights[index]);
            }
        }

        public static float[] ProjectBitNetI2Cpu(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);

            //quantize the float activations once so each output row can reuse the same ternary-friendly input block
            (sbyte[] quantizedValues, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Cpu(quantizedValues, activationScale, packedWeights.ToArray(), outputLength, weightScale, threads);
        }

        public static float[] ProjectBitNetI2Cpu(ReadOnlySpan<float> input, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(input, packedWeights, outputLength);

            //quantize the float activations once so each output row can reuse the same ternary-friendly input block
            (sbyte[] quantizedValues, float activationScale, _) = QuantizeBitNetActivations(input);
            return ProjectBitNetI2Cpu(quantizedValues, activationScale, packedWeights, outputLength, weightScale, threads);
        }

        internal static float[] ProjectBitNetI2Cpu(sbyte[] quantizedValues, float activationScale, byte[] packedWeights, int outputLength, float weightScale, int threads = 0)
        {
            ArgumentNullException.ThrowIfNull(quantizedValues);
            ArgumentNullException.ThrowIfNull(packedWeights);

            ValidateBitNetProjectionArguments(quantizedValues, packedWeights, outputLength);

            int packedRowByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, 1);
            float[] output = new float[outputLength];
            if (threads == 1 || output.Length <= 1)
            {
                ProjectBitNetI2CpuRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, 0, output.Length);
                return output;
            }

            ThreadHelper.ForEachRange(output.Length, (startIndex, endIndex) =>
                ProjectBitNetI2CpuRange(quantizedValues, packedWeights, packedRowByteCount, activationScale, weightScale, output, startIndex, endIndex), threads);

            return output;
        }

        private static void ProjectBitNetI2CpuRange(sbyte[] quantizedValues, byte[] packedWeights, int packedRowByteCount, float activationScale, float weightScale, float[] output, int startIndex, int endIndex)
        {
            for (int outputIndex = startIndex; outputIndex < endIndex; outputIndex++)
            {
                //load one packed weight row, decode its mapped dot product, then restore the final scaled float result
                ReadOnlySpan<byte> packedRow = packedWeights.AsSpan(outputIndex * packedRowByteCount, packedRowByteCount);
                int mappedDot = ComputeBitNetMappedDotCpu(quantizedValues, packedRow);
                output[outputIndex] = FinalizeBitNetMappedProjection(mappedDot, activationScale, weightScale);
            }
        }

        private static int ComputeBitNetMappedDotCpu(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedRow)
        {
            const int PackedGroupWidth = 32;
            const int ActivationBlockWidth = 128;

            int mappedDot = 0;
            int packedOffset = 0;
            for (int activationBlockStart = 0; activationBlockStart < quantizedValues.Length; activationBlockStart += ActivationBlockWidth)
            {
                //each packed block stores up to 128 activations as four 32-value groups
                int packedBlockByteCount = Math.Min(PackedGroupWidth, packedRow.Length - packedOffset);
                ReadOnlySpan<byte> packedBlock = packedRow.Slice(packedOffset, packedBlockByteCount);
                int remainingActivations = quantizedValues.Length - activationBlockStart;
                int groupCount = Math.Min(4, (remainingActivations + PackedGroupWidth - 1) / PackedGroupWidth);

                for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
                {
                    int activationOffset = activationBlockStart + (groupIndex * PackedGroupWidth);
                    int groupActivationCount = Math.Min(PackedGroupWidth, quantizedValues.Length - activationOffset);
                    ReadOnlySpan<sbyte> activationGroup = quantizedValues.Slice(activationOffset, groupActivationCount);

                    //decode the selected 2-bit lane from each packed byte and accumulate the mapped integer dot product
                    for (int packedIndex = 0; packedIndex < packedBlockByteCount && packedIndex < groupActivationCount; packedIndex++)
                    {
                        mappedDot += activationGroup[packedIndex] * DecodeBitNetWeight(packedBlock[packedIndex], groupIndex);
                    }
                }

                packedOffset += packedBlockByteCount;
            }

            return mappedDot;
        }
    }
}
