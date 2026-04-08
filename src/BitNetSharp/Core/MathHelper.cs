using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        private const float MinimumBitNetQuantizationMax = 0.00001f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateRmsNormDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (normWeights.Length < input.Length)
            {
                throw new ArgumentException("RMSNorm weight length must be at least the input length.", nameof(normWeights));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateAddDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (addend.Length < input.Length)
            {
                throw new ArgumentException("Addend length must be at least the input length.", nameof(addend));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateProjectionDestination(int outputLength, Span<float> output)
        {
            if (output.Length < outputLength)
            {
                throw new ArgumentException("Output length must be at least the projection output length.", nameof(output));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateSoftmaxDestination(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output length must be at least the input length.", nameof(output));
            }
        }

        public static int VectorProcessOne(ReadOnlyMemory<sbyte> dataBlock, ReadOnlyMemory<byte> weightBlock)
        {
            return VectorProcessOne(dataBlock.Span, weightBlock.Span);
        }

        private static int VectorProcessOne(ReadOnlySpan<sbyte> dataBlock, ReadOnlySpan<byte> weightBlock)
        {
            if (dataBlock.Length != 128)
            {
                throw new ArgumentException("dataBlock length must be 128");
            }
            if (weightBlock.Length != 32)
            {
                throw new ArgumentException("weightBlock length must be 32");
            }
            //load data
            var data0 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock));
            var data1 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(32)));
            var data2 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(64)));
            var data3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(96)));
            var bitMask = Vector256.Create<byte>(0b_0000_0011);
            //load weight
            var weight3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weightBlock));

            var weight0 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 6), bitMask);

            var weight1 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 4), bitMask);

            var weight2 = NormalizeBitNetWeightCodes(Vector256.ShiftRightLogical(weight3, 2), bitMask);

            weight3 = NormalizeBitNetWeightCodes(weight3, bitMask);

            //fast compress test



            return sum4(
                processBlock(data0, weight0),
                 processBlock(data1, weight1),
                 processBlock(data2, weight2),
                 processBlock(data3, weight3));
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static Vector256<int> processBlock(Vector256<sbyte> data, Vector256<byte> weight)
            {
                //result=data * block - data 
                //map logic
                // x*(-1|0|1)=x*[(0,1,2)-1]
                var result0 = Avx2.MultiplyAddAdjacent(weight, data);
                var sData = Avx2.MultiplyAddAdjacent(Vector256<byte>.One, data);//expand sbyte to short
                var w = Avx2.Subtract(result0, sData);

                //expand to int32 with adjacent add, prepare for sum
                return Avx2.MultiplyAddAdjacent(w, Vector256<short>.One);
            }

            //sums values in each Vector 256
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static int sum4(Vector256<int> v1, Vector256<int> v2, Vector256<int> v3, Vector256<int> v4)
            {

                var r = Avx2.Add(Avx2.Add(v1, v2), Avx2.Add(v3, v4));
                r = Avx2.HorizontalAdd(r, r);
                r = Avx2.HorizontalAdd(r, r);
                Vector256<int> r1 = Avx2.Permute2x128(r, r, 0b_0000_0001);//switch lanes
                return Avx2.Add(r, r1).ToScalar();
            }



        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateBitNetProjectionArguments(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength)
        {
            if (input.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            if (outputLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputLength));
            }

            if (input.Length % 4 != 0)
            {
                throw new ArgumentException("BitNet projection input length must be divisible by 4.", nameof(input));
            }

            int expectedPackedWeightByteCount = GetBitNetPackedWeightByteCount(input.Length, outputLength);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateBitNetProjectionArguments(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int outputLength)
        {
            if (quantizedValues.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(quantizedValues));
            }

            if (outputLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputLength));
            }

            if (quantizedValues.Length % 4 != 0)
            {
                throw new ArgumentException("BitNet projection input length must be divisible by 4.", nameof(quantizedValues));
            }

            int expectedPackedWeightByteCount = GetBitNetPackedWeightByteCount(quantizedValues.Length, outputLength);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetBitNetPackedWeightByteCount(int inputLength, int outputLength)
        {
            return checked((inputLength * outputLength) / 4);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float FinalizeBitNetMappedProjection(int mappedDot, float activationScale, float weightScale)
        {
            return (mappedDot / activationScale) * weightScale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float FinalizeBitNetMappedProjection(float mappedDot, float activationScale, float weightScale)
        {
            return (mappedDot / activationScale) * weightScale;
        }

        internal static (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlySpan<float> input, Span<sbyte> quantizedValues)
        {
            if (quantizedValues.Length < input.Length)
            {
                throw new ArgumentException("Quantized output length must be at least the input length.", nameof(quantizedValues));
            }

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
                int quantizedValue = RoundBitNetQuantizedValue(input[index] * activationScale);
                quantizedValues[index] = (sbyte)quantizedValue;
                activationSum += quantizedValue;
            }

            return (activationScale, activationSum);
        }

        internal static (sbyte[] QuantizedValues, float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlySpan<float> input)
        {
            sbyte[] quantizedValues = new sbyte[input.Length];
            (float activationScale, int activationSum) = QuantizeBitNetActivations(input, quantizedValues);
            return (quantizedValues, activationScale, activationSum);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RoundBitNetQuantizedValue(float value)
        {
            int roundedValue = (int)MathF.Round(value, MidpointRounding.ToEven);
            return Math.Clamp(roundedValue, sbyte.MinValue, sbyte.MaxValue);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int DecodeBitNetWeight(byte packedWeight, int groupIndex)
        {
            int weightCode = (packedWeight >> (6 - (groupIndex * 2))) & 0b_0000_0011;
            return weightCode == 0b_0000_0011 ? 0 : weightCode - 1;
        }

        private static void ExpandBitNetRowWeights(ReadOnlySpan<byte> packedRow, Span<float> rowWeights)
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> NormalizeBitNetWeightCodes(Vector256<byte> packedWeightCodes, Vector256<byte> bitMask)
        {
            Vector256<byte> weightCodes = Vector256.BitwiseAnd(packedWeightCodes, bitMask);
            Vector256<byte> threeMask = Avx2.CompareEqual(weightCodes, Vector256.Create((byte)0b_0000_0011));
            Vector256<byte> correction = Vector256.BitwiseAnd(threeMask, Vector256.Create((byte)0b_0000_0010));
            return Avx2.Subtract(weightCodes, correction);
        }
    }
}
