using System.Runtime.CompilerServices;

namespace BitNetSharp.Core
{
    internal static class ValidationHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateLmHeadArguments(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output)
        {
            if (input.Length != rowLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            int requiredWeightByteCount = checked(rowLength * vocabularySize * sizeof(ushort));
            if (embeddingWeights.Length < requiredWeightByteCount)
            {
                throw new ArgumentException("Embedding weights length does not match the LM head tensor size.", nameof(embeddingWeights));
            }

            if (output.Length < vocabularySize)
            {
                throw new ArgumentException("Output length does not match the model vocabulary size.", nameof(output));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateAddDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> addend, Span<float> output)
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
        internal static void ValidateRmsNormDestination(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, Span<float> output)
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
        internal static void ValidateProjectionDestination(int outputLength, Span<float> output)
        {
            if (output.Length < outputLength)
            {
                throw new ArgumentException("Output length must be at least the projection output length.", nameof(output));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateSoftmaxDestination(ReadOnlySpan<float> input, Span<float> output)
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateBitNetProjectionArguments(ReadOnlySpan<float> input, ReadOnlySpan<byte> packedWeights, int outputLength)
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

            int expectedPackedWeightByteCount = checked((input.Length * outputLength) / 4);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateBitNetProjectionArguments(ReadOnlySpan<sbyte> quantizedValues, ReadOnlySpan<byte> packedWeights, int outputLength)
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

            int expectedPackedWeightByteCount = checked((quantizedValues.Length * outputLength) / 4);
            if (packedWeights.Length != expectedPackedWeightByteCount)
            {
                throw new ArgumentException("Packed BitNet weight length does not match the expected tensor shape.", nameof(packedWeights));
            }
        }
    }
}
