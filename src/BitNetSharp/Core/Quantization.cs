using System;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Quantization utilities used by BitNet layers.
    ///
    /// Weight quantization (absmean):
    ///   scale  = mean(|W|) + eps
    ///   W_q    = clip(round(W / scale), -1, 1)   → ternary {-1, 0, +1}
    ///
    /// Activation quantization (absmax, 8-bit signed):
    ///   scale  = 127 / (max(|x|) + eps)
    ///   x_q    = clip(round(x * scale), -128, 127)
    /// </summary>
    public static class Quantization
    {
        private const float Eps = 1e-8f;

        // ------------------------------------------------------------------ //
        //  Weight quantization                                                //
        // ------------------------------------------------------------------ //

        /// <summary>
        /// Quantizes a weight tensor to ternary values {-1, 0, +1} using absmean
        /// quantization as described in the BitNet paper.
        /// Returns the quantized tensor and the scale factor used.
        /// </summary>
        public static (Tensor quantized, float scale) QuantizeWeights(Tensor weights)
        {
            // Compute absmean scale
            float sumAbs = 0f;
            for (int i = 0; i < weights.Size; i++)
                sumAbs += MathF.Abs(weights.Data[i]);
            float scale = sumAbs / weights.Size + Eps;

            var quantized = new Tensor(weights.Shape, new float[weights.Size]);
            for (int i = 0; i < weights.Size; i++)
            {
                float val = MathF.Round(weights.Data[i] / scale);
                quantized.Data[i] = Math.Clamp(val, -1f, 1f);
            }

            return (quantized, scale);
        }

        /// <summary>
        /// Quantizes an activation tensor to 8-bit signed integers using absmax
        /// quantization (per-tensor).  Returns the quantized tensor and scale.
        /// </summary>
        public static (Tensor quantized, float scale) QuantizeActivations(Tensor activations)
        {
            float maxAbs = 0f;
            for (int i = 0; i < activations.Size; i++)
            {
                float abs = MathF.Abs(activations.Data[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = 127f / (maxAbs + Eps);

            var quantized = new Tensor(activations.Shape, new float[activations.Size]);
            for (int i = 0; i < activations.Size; i++)
            {
                float val = MathF.Round(activations.Data[i] * scale);
                quantized.Data[i] = Math.Clamp(val, -128f, 127f);
            }

            return (quantized, scale);
        }

        /// <summary>
        /// Dequantizes an output tensor that was produced from quantized inputs.
        /// output = quantized_output / (activation_scale * weight_scale)
        /// </summary>
        public static Tensor Dequantize(Tensor quantizedOutput, float activationScale, float weightScale)
        {
            float denom = activationScale * weightScale;
            var result = new Tensor(quantizedOutput.Shape, new float[quantizedOutput.Size]);
            for (int i = 0; i < quantizedOutput.Size; i++)
                result.Data[i] = quantizedOutput.Data[i] / denom;
            return result;
        }
    }
}
