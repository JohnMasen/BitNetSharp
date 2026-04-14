namespace BitNetSharp.Core
{
    /// <summary>
    /// Provides backend-specific low-level math operations.
    /// </summary>
    public interface IOPProvider
    {
        /// <summary>
        /// Gets the backend name exposed by this provider.
        /// </summary>
        string Backend { get; }

        /// <summary>
        /// Gets the preferred CPU thread count used by this provider.
        /// </summary>
        int ThreadCount { get; }

        /// <summary>
        /// Adds two input tensors element-wise and writes the result to the output tensor.
        /// </summary>
        void Add(RuntimeTensor input, RuntimeTensor addend, RuntimeTensor output);

        /// <summary>
        /// Quantizes floating-point activations into signed 8-bit values and returns the quantization metadata.
        /// </summary>
        (float ActivationScale, int ActivationSum) QuantizeBitNetActivations(RuntimeTensor input, RuntimeTensor quantizedValues);

        /// <summary>
        /// Projects floating-point activations through packed BitNet weights and writes the result to the output tensor.
        /// </summary>
        void ProjectBitNetI2(RuntimeTensor input, RuntimeTensor packedWeights, int outputLength, float weightScale, RuntimeTensor output);

        /// <summary>
        /// Projects pre-quantized activations through packed BitNet weights and writes the result to the output tensor.
        /// </summary>
        void ProjectBitNetI2(RuntimeTensor quantizedValues, float activationScale, RuntimeTensor packedWeights, int outputLength, float weightScale, RuntimeTensor output);

        /// <summary>
        /// Applies softmax to the input tensor and writes the normalized probabilities to the output tensor.
        /// </summary>
        void ForwardSoftmax(RuntimeTensor input, RuntimeTensor output);

        /// <summary>
        /// Applies RMS normalization using the provided norm weights and writes the normalized values to the output tensor.
        /// </summary>
        void ForwardRmsNorm(RuntimeTensor input, RuntimeTensor normWeights, float epsilon, RuntimeTensor output);

        /// <summary>
        /// Projects the hidden state tensor into vocabulary logits using the embedding weights.
        /// </summary>
        void ForwardLmHead(RuntimeTensor input, RuntimeTensor embeddingWeights, int rowLength, int vocabularySize, RuntimeTensor output);
    }
}
