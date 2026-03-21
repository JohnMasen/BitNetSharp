using BitNetSharp.Core;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// A single BitNet Transformer block.
    ///
    /// Structure (Pre-LN variant):
    ///   x = x + Attention(RMSNorm(x))
    ///   x = x + FFN(RMSNorm(x))
    ///
    /// Note: SubLN normalisation is applied inside BitLinear for the Q/K/V/Out
    /// projections.  The additional RMSNorm before each sub-layer provides
    /// the outer pre-norm residual stabilisation.
    /// </summary>
    public sealed class BitNetBlock
    {
        private readonly RMSNorm _attNorm;
        private readonly RMSNorm _ffnNorm;
        private readonly BitNetAttention _attention;
        private readonly BitNetFFN _ffn;

        public int EmbedDim { get; }

        public BitNetBlock(int embedDim, int numHeads, int ffnDim, int maxSeqLen = 2048)
        {
            EmbedDim = embedDim;

            _attNorm   = new RMSNorm(embedDim);
            _ffnNorm   = new RMSNorm(embedDim);
            _attention = new BitNetAttention(embedDim, numHeads, maxSeqLen);
            _ffn       = new BitNetFFN(embedDim, ffnDim);
        }

        /// <summary>
        /// Forward pass.
        /// input: (seqLen, embedDim) → output: (seqLen, embedDim)
        /// </summary>
        public Tensor Forward(Tensor input, bool causalMask = true, int startPosition = 0)
        {
            // Self-attention sub-layer with residual
            Tensor normedAtt = _attNorm.ForwardBatch(input);
            Tensor attOut    = _attention.Forward(normedAtt, causalMask, startPosition);
            Tensor afterAtt  = Tensor.Add(input, attOut);

            // FFN sub-layer with residual
            Tensor normedFFN = _ffnNorm.ForwardBatch(afterAtt);
            Tensor ffnOut    = _ffn.Forward(normedFFN);
            return Tensor.Add(afterAtt, ffnOut);
        }
    }
}
