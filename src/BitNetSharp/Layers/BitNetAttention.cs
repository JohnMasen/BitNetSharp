using System;
using BitNetSharp.Core;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// BitNet multi-head self-attention layer.
    ///
    /// Uses BitLinear projections for Q, K, V and the output projection.
    /// Applies Rotary Position Embeddings (RoPE) to Q and K.
    /// Supports an optional causal (autoregressive) mask.
    /// </summary>
    public sealed class BitNetAttention
    {
        private readonly BitLinear _queryProj;
        private readonly BitLinear _keyProj;
        private readonly BitLinear _valueProj;
        private readonly BitLinear _outputProj;
        private readonly RotaryEmbedding _rope;

        public int EmbedDim { get; }
        public int NumHeads { get; }
        public int HeadDim { get; }

        public BitNetAttention(int embedDim, int numHeads, int maxSeqLen = 2048)
        {
            if (embedDim % numHeads != 0)
                throw new ArgumentException("embedDim must be divisible by numHeads.");

            EmbedDim = embedDim;
            NumHeads = numHeads;
            HeadDim = embedDim / numHeads;

            _queryProj  = new BitLinear(embedDim, embedDim);
            _keyProj    = new BitLinear(embedDim, embedDim);
            _valueProj  = new BitLinear(embedDim, embedDim);
            _outputProj = new BitLinear(embedDim, embedDim);
            _rope = new RotaryEmbedding(HeadDim, maxSeqLen);
        }

        /// <summary>
        /// Forward pass.
        /// input: (seqLen, embedDim)
        /// Returns: (seqLen, embedDim)
        /// </summary>
        public Tensor Forward(Tensor input, bool causalMask = true, int startPosition = 0)
        {
            int seqLen = input.Rows;
            int d = EmbedDim;

            // Project Q, K, V using BitLinear
            Tensor q = _queryProj.ForwardBatch(input);   // (seqLen, d)
            Tensor k = _keyProj.ForwardBatch(input);     // (seqLen, d)
            Tensor v = _valueProj.ForwardBatch(input);   // (seqLen, d)

            // Apply RoPE head by head
            ApplyRopeMultiHead(q, seqLen, startPosition);
            ApplyRopeMultiHead(k, seqLen, startPosition);

            // Compute attention head outputs
            Tensor context = MultiHeadAttention(q, k, v, seqLen, causalMask);

            // Final output projection
            return _outputProj.ForwardBatch(context);
        }

        // ------------------------------------------------------------------ //
        //  Private helpers                                                    //
        // ------------------------------------------------------------------ //

        private void ApplyRopeMultiHead(Tensor x, int seqLen, int startPosition)
        {
            // x shape: (seqLen, embedDim) → process each head slice separately
            for (int h = 0; h < NumHeads; h++)
            {
                int offset = h * HeadDim;
                // Extract head slice into a (seqLen, headDim) tensor
                var headSlice = new Tensor(new[] { seqLen, HeadDim });
                for (int t = 0; t < seqLen; t++)
                    for (int i = 0; i < HeadDim; i++)
                        headSlice[t, i] = x[t, offset + i];

                _rope.Apply(headSlice, startPosition);

                // Write back
                for (int t = 0; t < seqLen; t++)
                    for (int i = 0; i < HeadDim; i++)
                        x[t, offset + i] = headSlice[t, i];
            }
        }

        private Tensor MultiHeadAttention(Tensor q, Tensor k, Tensor v, int seqLen, bool causalMask)
        {
            float scale = 1f / MathF.Sqrt(HeadDim);
            var output = new Tensor(new[] { seqLen, EmbedDim });

            for (int h = 0; h < NumHeads; h++)
            {
                int offset = h * HeadDim;

                // Compute attention scores: (seqLen, seqLen)
                var scores = new float[seqLen * seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < HeadDim; d++)
                            dot += q[i, offset + d] * k[j, offset + d];
                        scores[i * seqLen + j] = dot * scale;
                    }
                }

                // Apply causal mask (set future positions to -inf)
                if (causalMask)
                {
                    for (int i = 0; i < seqLen; i++)
                        for (int j = i + 1; j < seqLen; j++)
                            scores[i * seqLen + j] = float.NegativeInfinity;
                }

                // Softmax row-wise
                Softmax(scores, seqLen);

                // Weighted sum of values
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < HeadDim; d++)
                    {
                        float sum = 0f;
                        for (int j = 0; j < seqLen; j++)
                            sum += scores[i * seqLen + j] * v[j, offset + d];
                        output[i, offset + d] = sum;
                    }
                }
            }

            return output;
        }

        private static void Softmax(float[] scores, int seqLen)
        {
            for (int i = 0; i < seqLen; i++)
            {
                int rowStart = i * seqLen;

                // Numerically stable: subtract row max
                float max = float.NegativeInfinity;
                for (int j = 0; j < seqLen; j++)
                    if (scores[rowStart + j] > max)
                        max = scores[rowStart + j];

                float sum = 0f;
                for (int j = 0; j < seqLen; j++)
                {
                    scores[rowStart + j] = MathF.Exp(scores[rowStart + j] - max);
                    sum += scores[rowStart + j];
                }

                for (int j = 0; j < seqLen; j++)
                    scores[rowStart + j] /= sum;
            }
        }
    }
}
