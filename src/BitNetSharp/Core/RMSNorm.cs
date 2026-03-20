using System;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Root Mean Square Layer Normalization (RMSNorm).
    ///
    /// Formula: y = x / RMS(x) * gamma
    /// where   RMS(x) = sqrt(mean(x^2) + eps)
    ///
    /// No learnable bias is used, consistent with the BitNet paper.
    /// </summary>
    public sealed class RMSNorm
    {
        private const float DefaultEps = 1e-6f;

        public float[] Gamma { get; }
        public float Eps { get; }

        public int Size => Gamma.Length;

        public RMSNorm(int size, float eps = DefaultEps)
        {
            Eps = eps;
            Gamma = new float[size];
            // Initialise gamma to 1 (identity transform)
            Array.Fill(Gamma, 1f);
        }

        /// <summary>
        /// Applies RMSNorm to a 1-D input vector (length must equal Size).
        /// Returns a new 1-D tensor.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input.Size != Size)
                throw new ArgumentException(
                    $"Input size {input.Size} does not match RMSNorm size {Size}.");

            float meanSq = 0f;
            for (int i = 0; i < input.Size; i++)
                meanSq += input.Data[i] * input.Data[i];
            meanSq /= input.Size;

            float rms = MathF.Sqrt(meanSq + Eps);

            var output = new Tensor(input.Shape, new float[input.Size]);
            for (int i = 0; i < input.Size; i++)
                output.Data[i] = input.Data[i] / rms * Gamma[i];

            return output;
        }

        /// <summary>
        /// Applies RMSNorm row-wise to a 2-D input tensor (batchSize x Size).
        /// Returns a new 2-D tensor of the same shape.
        /// </summary>
        public Tensor ForwardBatch(Tensor input)
        {
            if (input.Cols != Size)
                throw new ArgumentException(
                    $"Input cols {input.Cols} does not match RMSNorm size {Size}.");

            int rows = input.Rows;
            var output = new Tensor(input.Shape, new float[input.Size]);

            for (int r = 0; r < rows; r++)
            {
                float meanSq = 0f;
                for (int c = 0; c < Size; c++)
                {
                    float v = input[r, c];
                    meanSq += v * v;
                }
                meanSq /= Size;
                float rms = MathF.Sqrt(meanSq + Eps);

                for (int c = 0; c < Size; c++)
                    output[r, c] = input[r, c] / rms * Gamma[c];
            }

            return output;
        }
    }
}
