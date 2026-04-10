namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides backend-specific low-level math operations.
    /// </summary>
    public interface IOPProvider
    {
        string Backend { get; }

        void Add(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output);

        (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(ReadOnlyMemory<float> input, Memory<sbyte> quantizedValues);

        void ProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output);

        void ProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output);

        void ForwardSoftmax(ReadOnlySpan<float> input, Span<float> output);

        void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output);

        void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output);
    }
}
