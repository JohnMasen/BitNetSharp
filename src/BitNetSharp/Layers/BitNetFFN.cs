using System;
using BitNetSharp.Core;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// BitNet Feed-Forward Network (FFN) using two BitLinear layers and
    /// Squared ReLU (ReLU²) as the activation function, as specified in the
    /// BitNet paper.
    ///
    /// Architecture:
    ///   x → BitLinear(embedDim → ffnDim) → ReLU²  → BitLinear(ffnDim → embedDim)
    ///
    /// No bias terms are used.
    /// </summary>
    public sealed class BitNetFFN
    {
        private readonly BitLinear _expand;
        private readonly BitLinear _project;

        public int EmbedDim { get; }
        public int FFNDim { get; }

        public BitNetFFN(int embedDim, int ffnDim)
        {
            EmbedDim = embedDim;
            FFNDim = ffnDim;

            _expand  = new BitLinear(embedDim, ffnDim);
            _project = new BitLinear(ffnDim, embedDim);
        }

        /// <summary>
        /// Forward pass for a batch of tokens.
        /// input: (seqLen, embedDim) → output: (seqLen, embedDim)
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            // Expand
            Tensor hidden = _expand.ForwardBatch(input);   // (seqLen, ffnDim)

            // Squared ReLU activation
            for (int i = 0; i < hidden.Size; i++)
            {
                float v = hidden.Data[i];
                v = v > 0f ? v * v : 0f;
                hidden.Data[i] = v;
            }

            // Project back
            return _project.ForwardBatch(hidden);           // (seqLen, embedDim)
        }
    }
}
