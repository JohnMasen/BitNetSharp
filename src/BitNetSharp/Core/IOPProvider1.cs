namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides backend-specific low-level math operations.
    /// </summary>
    public interface IOPProvider1
    {
        global::BitNetSharp.Nodes.InferenceBackend Backend { get; }

        int ThreadCount { get; }

        void Add(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output);

        void ProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output);

        void ProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output);

        void ForwardSoftmax(ReadOnlySpan<float> input, Span<float> output);
    }
}
