using System;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides the standard CPU implementation of math operations.
    /// </summary>
    public sealed partial class CPUDefaultOPProvider : IOPProvider2
    {
        public CPUDefaultOPProvider(int threadCount = global::BitNetSharp.Nodes.InferenceConfig.AutoThreadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            ThreadCount = threadCount;
        }

        public global::BitNetSharp.Nodes.InferenceBackend Backend => global::BitNetSharp.Nodes.InferenceBackend.CPU;

        public int ThreadCount { get; }

        public void Add(ReadOnlyMemory<float> input, ReadOnlyMemory<float> addend, Memory<float> output, string operationName = "Add")
        {
            ExecuteAdd(input, addend, output);
        }

        public void ProjectBitNetI2(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, string operationName = "BitNet projection")
        {
            ExecuteProjectBitNetI2(input, packedWeights, outputLength, weightScale, output);
        }

        public void ProjectBitNetI2(ReadOnlyMemory<sbyte> quantizedValues, float activationScale, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale, Memory<float> output, string operationName = "BitNet projection")
        {
            ExecuteProjectBitNetI2(quantizedValues, activationScale, packedWeights, outputLength, weightScale, output);
        }

        public void ForwardSoftmax(ReadOnlySpan<float> input, Span<float> output, string operationName = "Softmax")
        {
            ExecuteForwardSoftmax(input, output);
        }

        public void ForwardRmsNorm(ReadOnlyMemory<float> input, ReadOnlyMemory<float> normWeights, float epsilon, Memory<float> output)
        {
            ExecuteForwardRmsNorm(input, normWeights, epsilon, output);
        }

        public void ForwardQKVProjection(ReadOnlyMemory<float> input, ReadOnlyMemory<byte> queryPackedWeights, float queryWeightScale, ReadOnlyMemory<byte> keyPackedWeights, float keyWeightScale, ReadOnlyMemory<byte> valuePackedWeights, float valueWeightScale, int queryOutputLength, int keyValueOutputLength, Memory<float> query, Memory<float> key, Memory<float> value)
        {
            OPProviderCommon.ForwardQKVProjection(this, input, queryPackedWeights, queryWeightScale, keyPackedWeights, keyWeightScale, valuePackedWeights, valueWeightScale, queryOutputLength, keyValueOutputLength, query, key, value);
        }

        public void ForwardAttention(ReadOnlyMemory<float> query, ReadOnlyMemory<float> key, ReadOnlyMemory<float> value, ReadOnlyMemory<float> subNormWeights, float epsilon, ReadOnlyMemory<byte> outputPackedWeights, float outputWeightScale, int embeddingLength, int keyValueLength, int headCount, int keyValueHeadCount, int headDimension, Memory<float> subNorm, Memory<float> output, ReadOnlyMemory<float> outputScaleValues = default, ReadOnlyMemory<float> outputBiasValues = default)
        {
            OPProviderCommon.ForwardAttention(this, query, key, value, subNormWeights, epsilon, outputPackedWeights, outputWeightScale, embeddingLength, keyValueLength, headCount, keyValueHeadCount, headDimension, subNorm, output, outputScaleValues, outputBiasValues);
        }

        public void ForwardFeedForward(ReadOnlyMemory<float> input, ReadOnlyMemory<float> subNormWeights, float epsilon, ReadOnlyMemory<byte> gatePackedWeights, float gateWeightScale, ReadOnlyMemory<byte> upPackedWeights, float upWeightScale, ReadOnlyMemory<byte> downPackedWeights, float downWeightScale, int embeddingLength, int feedForwardLength, Memory<float> subNormOutput, Memory<float> output)
        {
            OPProviderCommon.ForwardFeedForward(this, input, subNormWeights, epsilon, gatePackedWeights, gateWeightScale, upPackedWeights, upWeightScale, downPackedWeights, downWeightScale, embeddingLength, feedForwardLength, subNormOutput, output);
        }

        public void ForwardLmHead(ReadOnlyMemory<float> input, ReadOnlyMemory<Half> embeddingWeights, int rowLength, int vocabularySize, Memory<float> output)
        {
            OPProviderCommon.ForwardLmHeadCpu(input, embeddingWeights, rowLength, vocabularySize, output, ThreadCount);
        }
    }
}
