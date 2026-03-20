using System;
using BitNetSharp.Core;
using BitNetSharp.Layers;

namespace BitNetSharp.Models
{
    /// <summary>
    /// Full BitNet transformer language model.
    ///
    /// Architecture:
    ///   token_ids → Embedding → [BitNetBlock × NumLayers] → RMSNorm → LM Head → logits
    ///
    /// The LM head is a standard (unquantized) linear projection from embedDim
    /// to vocabSize.  All intermediate projections use BitLinear (ternary weights).
    /// </summary>
    public sealed class BitNetModel
    {
        private readonly float[] _embedding;      // (vocabSize, embedDim)
        private readonly BitNetBlock[] _blocks;
        private readonly RMSNorm _finalNorm;
        private readonly float[] _lmHead;         // (vocabSize, embedDim)

        public BitNetConfig Config { get; }

        public BitNetModel(BitNetConfig config)
        {
            Config = config;

            // Token embedding table
            _embedding = new float[config.VocabSize * config.EmbedDim];
            InitEmbedding(_embedding, config.EmbedDim);

            // Transformer blocks
            _blocks = new BitNetBlock[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
                _blocks[i] = new BitNetBlock(
                    config.EmbedDim, config.NumHeads, config.FFNDim, config.MaxSequenceLength);

            // Final norm
            _finalNorm = new RMSNorm(config.EmbedDim);

            // LM head (shared weights with embedding, untied copy here for clarity)
            _lmHead = new float[config.VocabSize * config.EmbedDim];
            Array.Copy(_embedding, _lmHead, _embedding.Length);
        }

        /// <summary>
        /// Forward pass.
        ///
        /// tokenIds: integer token indices of shape (seqLen,)
        /// Returns logits tensor of shape (seqLen, vocabSize).
        /// </summary>
        public Tensor Forward(int[] tokenIds, int startPosition = 0)
        {
            if (tokenIds == null || tokenIds.Length == 0)
                throw new ArgumentException("tokenIds must not be null or empty.");

            int seqLen = tokenIds.Length;
            int embedDim = Config.EmbedDim;

            // Look up embeddings → (seqLen, embedDim)
            var hidden = new Tensor(new[] { seqLen, embedDim });
            for (int t = 0; t < seqLen; t++)
            {
                int tokenId = tokenIds[t];
                if (tokenId < 0 || tokenId >= Config.VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds),
                        $"Token id {tokenId} is out of range [0, {Config.VocabSize}).");

                int embStart = tokenId * embedDim;
                for (int d = 0; d < embedDim; d++)
                    hidden[t, d] = _embedding[embStart + d];
            }

            // Pass through transformer blocks
            foreach (BitNetBlock block in _blocks)
                hidden = block.Forward(hidden, causalMask: true, startPosition: startPosition);

            // Final norm
            hidden = _finalNorm.ForwardBatch(hidden);

            // LM head: (seqLen, embedDim) × (embedDim, vocabSize) → (seqLen, vocabSize)
            var logits = new Tensor(new[] { seqLen, Config.VocabSize });
            for (int t = 0; t < seqLen; t++)
            {
                for (int v = 0; v < Config.VocabSize; v++)
                {
                    float dot = 0f;
                    int lmStart = v * embedDim;
                    for (int d = 0; d < embedDim; d++)
                        dot += hidden[t, d] * _lmHead[lmStart + d];
                    logits[t, v] = dot;
                }
            }

            return logits;
        }

        /// <summary>
        /// Greedy next-token prediction: returns the argmax of the last position's logits.
        /// </summary>
        public int PredictNextToken(int[] tokenIds)
        {
            Tensor logits = Forward(tokenIds);
            int seqLen = tokenIds.Length;
            int bestToken = 0;
            float bestScore = float.NegativeInfinity;
            for (int v = 0; v < Config.VocabSize; v++)
            {
                float score = logits[seqLen - 1, v];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestToken = v;
                }
            }
            return bestToken;
        }

        // ------------------------------------------------------------------ //
        //  Helpers                                                            //
        // ------------------------------------------------------------------ //

        private static void InitEmbedding(float[] table, int embedDim)
        {
            // Small normal-distributed initialisation
            var rng = new Random(0);
            float std = 0.02f;
            for (int i = 0; i < table.Length; i++)
            {
                // Box-Muller
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                table[i] = (float)(normal * std);
            }
        }
    }
}
