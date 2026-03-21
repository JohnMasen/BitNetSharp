using System;

namespace BitNetSharp.Core
{
    /// <summary>
    /// Rotary Position Embeddings (RoPE).
    ///
    /// Applies position-dependent rotation to query and key tensors in
    /// the attention mechanism.  Each pair of dimensions (2i, 2i+1) is
    /// rotated by angle θ_i * position, where θ_i = 1 / (10000 ^ (2i / d_head)).
    ///
    /// Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    /// (Su et al., 2021) as adopted by BitNet.
    /// </summary>
    public sealed class RotaryEmbedding
    {
        private readonly float[] _cosCache;
        private readonly float[] _sinCache;

        public int HeadDim { get; }
        public int MaxSequenceLength { get; }

        public RotaryEmbedding(int headDim, int maxSeqLen = 2048, float baseFreq = 10000f)
        {
            if (headDim % 2 != 0)
                throw new ArgumentException("headDim must be even for RoPE.");

            HeadDim = headDim;
            MaxSequenceLength = maxSeqLen;

            int halfDim = headDim / 2;
            _cosCache = new float[maxSeqLen * halfDim];
            _sinCache = new float[maxSeqLen * halfDim];

            for (int pos = 0; pos < maxSeqLen; pos++)
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float theta = pos / MathF.Pow(baseFreq, 2f * i / headDim);
                    _cosCache[pos * halfDim + i] = MathF.Cos(theta);
                    _sinCache[pos * halfDim + i] = MathF.Sin(theta);
                }
            }
        }

        /// <summary>
        /// Applies RoPE in-place to a 2-D query or key tensor of shape (seqLen, headDim).
        /// The tensor is modified in place and also returned for convenience.
        /// </summary>
        public Tensor Apply(Tensor qOrK, int startPosition = 0)
        {
            int seqLen = qOrK.Rows;
            int dim = qOrK.Cols;

            if (dim != HeadDim)
                throw new ArgumentException(
                    $"Tensor column dimension {dim} does not match HeadDim {HeadDim}.");

            int halfDim = HeadDim / 2;

            for (int pos = 0; pos < seqLen; pos++)
            {
                int absPos = startPosition + pos;
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = qOrK[pos, i];
                    float x1 = qOrK[pos, i + halfDim];

                    float cos = _cosCache[absPos * halfDim + i];
                    float sin = _sinCache[absPos * halfDim + i];

                    qOrK[pos, i]           = x0 * cos - x1 * sin;
                    qOrK[pos, i + halfDim] = x0 * sin + x1 * cos;
                }
            }

            return qOrK;
        }
    }
}
