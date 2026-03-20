using System;
using BitNetSharp.Core;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// BitLinear – a drop-in replacement for a standard linear (dense) layer that
    /// uses ternary {-1, 0, +1} weights and 8-bit quantized activations.
    ///
    /// Forward pass:
    ///   1. Normalize input with RMSNorm (SubLN in the paper).
    ///   2. Quantize activations to int8 via absmax.
    ///   3. Quantize weights to ternary via absmean.
    ///   4. Compute integer matrix multiply.
    ///   5. Dequantize output.
    ///
    /// No bias term is used (consistent with the BitNet paper).
    /// </summary>
    public sealed class BitLinear
    {
        private readonly RMSNorm _norm;

        /// <summary>Weight matrix of shape (outputSize, inputSize).</summary>
        public Tensor Weights { get; }

        public int InputSize { get; }
        public int OutputSize { get; }

        public BitLinear(int inputSize, int outputSize)
        {
            InputSize = inputSize;
            OutputSize = outputSize;

            Weights = new Tensor(new[] { outputSize, inputSize });
            _norm = new RMSNorm(inputSize);

            InitializeWeights();
        }

        /// <summary>
        /// Forward pass for a single token vector (1-D tensor of length InputSize).
        /// Returns a 1-D tensor of length OutputSize.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input.Size != InputSize)
                throw new ArgumentException(
                    $"Input size {input.Size} does not match BitLinear input size {InputSize}.");

            // 1. SubLN: normalize input before quantization
            Tensor normedInput = _norm.Forward(input);

            // 2. Quantize activations (absmax, 8-bit)
            var (quantizedInput, activationScale) = Quantization.QuantizeActivations(normedInput);

            // 3. Quantize weights (absmean, ternary)
            var (quantizedWeights, weightScale) = Quantization.QuantizeWeights(Weights);

            // 4. Integer matrix multiply: result = W_q * x_q
            //    input  shape: (InputSize,)  → treat as (InputSize, 1)
            //    weight shape: (OutputSize, InputSize)
            //    output shape: (OutputSize, 1) → flatten to (OutputSize,)
            var inputCol = quantizedInput.Reshape(InputSize, 1);
            var rawOutput = Tensor.MatMul(quantizedWeights, inputCol);  // (OutputSize, 1)

            // 5. Dequantize
            Tensor dequantized = Quantization.Dequantize(rawOutput, activationScale, weightScale);

            return dequantized.Reshape(OutputSize);
        }

        /// <summary>
        /// Forward pass for a batch of tokens (2-D tensor of shape batchSize × InputSize).
        /// Returns a 2-D tensor of shape batchSize × OutputSize.
        /// </summary>
        public Tensor ForwardBatch(Tensor input)
        {
            if (input.Cols != InputSize)
                throw new ArgumentException(
                    $"Input cols {input.Cols} does not match BitLinear input size {InputSize}.");

            int batchSize = input.Rows;

            // 1. SubLN: normalize each row
            Tensor normedInput = _norm.ForwardBatch(input);

            // 2. Quantize activations (absmax, per-tensor)
            var (quantizedInput, activationScale) = Quantization.QuantizeActivations(normedInput);

            // 3. Quantize weights (absmean, ternary)
            var (quantizedWeights, weightScale) = Quantization.QuantizeWeights(Weights);

            // 4. Matrix multiply: (batchSize, InputSize) x (InputSize, OutputSize) = (batchSize, OutputSize)
            Tensor wT = quantizedWeights.Transpose();   // (InputSize, OutputSize)
            Tensor rawOutput = Tensor.MatMul(quantizedInput, wT);   // (batchSize, OutputSize)

            // 5. Dequantize
            return Quantization.Dequantize(rawOutput, activationScale, weightScale);
        }

        // ------------------------------------------------------------------ //
        //  Helpers                                                            //
        // ------------------------------------------------------------------ //

        /// <summary>
        /// Initialises weights with Xavier/Glorot uniform initialisation.
        /// This gives a good starting distribution prior to training.
        /// </summary>
        private void InitializeWeights()
        {
            float limit = MathF.Sqrt(6f / (InputSize + OutputSize));
            var rng = new Random(42);

            for (int i = 0; i < Weights.Size; i++)
                Weights.Data[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        }
    }
}
