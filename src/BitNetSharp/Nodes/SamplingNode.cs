namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Selects the next token from the logits stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class SamplingNode
    {
        internal const string GreedyArgmaxStrategy = "greedy_argmax";
        private bool isInitialized;

        public SamplingNode(int topK = 10)
        {
            if (topK <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(topK));
            }

            TopK = topK;
        }

        public int TopK { get; }

        public void Init()
        {
            isInitialized = true;
        }

        /// <summary>
        /// Selects the greedy next token and stores sampling metadata on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!session.HasMemory<float>(BitNetSession.LogitsKey))
            {
                throw new InvalidOperationException("Session does not contain logits.");
            }

            ReadOnlySpan<float> logits = session.Logits.Span;
            if (logits.IsEmpty)
            {
                throw new InvalidOperationException("Session logits must not be empty.");
            }

            int topKCount = Math.Min(TopK, logits.Length);
            int[] topKTokenIds = new int[topKCount];
            float[] topKLogits = new float[topKCount];
            Array.Fill(topKTokenIds, -1);
            Array.Fill(topKLogits, float.NegativeInfinity);

            int argmaxTokenId = 0;
            float argmaxLogit = logits[0];
            for (int tokenId = 0; tokenId < logits.Length; tokenId++)
            {
                float logit = logits[tokenId];
                if (logit > argmaxLogit)
                {
                    argmaxLogit = logit;
                    argmaxTokenId = tokenId;
                }

                int insertIndex = -1;
                for (int rank = 0; rank < topKCount; rank++)
                {
                    if (logit > topKLogits[rank])
                    {
                        insertIndex = rank;
                        break;
                    }
                }

                if (insertIndex < 0)
                {
                    continue;
                }

                for (int rank = topKCount - 1; rank > insertIndex; rank--)
                {
                    topKLogits[rank] = topKLogits[rank - 1];
                    topKTokenIds[rank] = topKTokenIds[rank - 1];
                }

                topKLogits[insertIndex] = logit;
                topKTokenIds[insertIndex] = tokenId;
            }

            session.ArgmaxTokenId = argmaxTokenId;
            session.ArgmaxLogit = argmaxLogit;
            session.NextTokenId = argmaxTokenId;
            session.NextTokenLogit = argmaxLogit;
            session.NextTokenStrategy = GreedyArgmaxStrategy;
            session.TopKTokenIds = topKTokenIds;
            session.TopKLogits = topKLogits;
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The step must be initialized by calling Init before Forward.");
            }
        }
    }
}
