using System;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides backend-specific higher-level inference operations.
    /// </summary>
    /// <remarks>
    /// This currently extends <see cref="IOPProvider1"/> so one provider instance can serve both node-level
    /// orchestration and lower-level kernels. If the two layers need to diverge later, they can still be split.
    /// </remarks>
    public interface IOPProvider2 : IOPProvider1
    {
        void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output);

        void ForwardQKVProjection(
            ReadOnlyMemory<float> input,
            ReadOnlyMemory<byte> queryPackedWeights,
            float queryWeightScale,
            ReadOnlyMemory<byte> keyPackedWeights,
            float keyWeightScale,
            ReadOnlyMemory<byte> valuePackedWeights,
            float valueWeightScale,
            int queryOutputLength,
            int keyValueOutputLength,
            Memory<float> query,
            Memory<float> key,
            Memory<float> value);

        void ForwardAttention(
            ReadOnlyMemory<float> query,
            ReadOnlyMemory<float> key,
            ReadOnlyMemory<float> value,
            ReadOnlyMemory<float> subNormWeights,
            float epsilon,
            ReadOnlyMemory<byte> outputPackedWeights,
            float outputWeightScale,
            int embeddingLength,
            int keyValueLength,
            int headCount,
            int keyValueHeadCount,
            int headDimension,
            Memory<float> subNorm,
            Memory<float> output,
            ReadOnlyMemory<float> outputScaleValues = default,
            ReadOnlyMemory<float> outputBiasValues = default);

        void ForwardFeedForward(
            ReadOnlyMemory<float> input,
            ReadOnlyMemory<float> subNormWeights,
            float epsilon,
            ReadOnlyMemory<byte> gatePackedWeights,
            float gateWeightScale,
            ReadOnlyMemory<byte> upPackedWeights,
            float upWeightScale,
            ReadOnlyMemory<byte> downPackedWeights,
            float downWeightScale,
            int embeddingLength,
            int feedForwardLength,
            Memory<float> subNormOutput,
            Memory<float> output);

        void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<Half> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output);
    }
}
