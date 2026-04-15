using System.Diagnostics;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Selects the next token from the logits stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class SamplingNode
    {
        internal const string GreedyArgmaxStrategy = "greedy_argmax";
        internal const string TopKSamplingStrategy = "top_k_sampling";
        private const float DefaultTemperature = 0.80f;
        private const float DefaultTopP = 0.95f;
        private const float DefaultMinP = 0.05f;
        private const int DefaultRepeatLastN = 64;
        private const float DefaultRepeatPenalty = 1.00f;
        private readonly bool enableSampling;
        private readonly Random? random;
        private readonly float temperature;
        private readonly float topP;
        private readonly float minP;
        private readonly int repeatLastN;
        private readonly float repeatPenalty;
        private bool isInitialized;

        public SamplingNode(int topK = 40, bool enableSampling = false, int? randomSeed = null, float temperature = DefaultTemperature, float topP = DefaultTopP, float minP = DefaultMinP, int repeatLastN = DefaultRepeatLastN, float repeatPenalty = DefaultRepeatPenalty)
        {
            if (topK <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(topK));
            }

            if (temperature < 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(temperature));
            }

            if (topP <= 0f || topP > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(topP));
            }

            if (minP < 0f || minP > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(minP));
            }

            if (repeatLastN < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(repeatLastN));
            }

            if (repeatPenalty <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(repeatPenalty));
            }

            TopK = topK;
            this.enableSampling = enableSampling;
            this.temperature = temperature;
            this.topP = topP;
            this.minP = minP;
            this.repeatLastN = repeatLastN;
            this.repeatPenalty = repeatPenalty;
            random = enableSampling ? (randomSeed.HasValue ? new Random(randomSeed.Value) : Random.Shared) : null;
        }

        public int TopK { get; }

        public bool EnableSampling => enableSampling;

        public float Temperature => temperature;

        public float TopP => topP;

        public float MinP => minP;

        public int RepeatLastN => repeatLastN;

        public float RepeatPenalty => repeatPenalty;

        public void Init()
        {
            isInitialized = true;
        }

        /// <summary>
        /// Selects the next token and stores sampling metadata on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();
            long samplingStartTimestamp = Stopwatch.GetTimestamp();

            if (!session.HasMemory<float>(BitNetSession.LogitsKey))
            {
                throw new InvalidOperationException("Session does not contain logits.");
            }

            ReadOnlySpan<float> logits = session.Logits.Span;
            if (logits.IsEmpty)
            {
                throw new InvalidOperationException("Session logits must not be empty.");
            }

            float[] adjustedLogits = logits.ToArray();
            ApplyRepeatPenalty(session, adjustedLogits);

            int topKCount = Math.Min(TopK, adjustedLogits.Length);
            int[] topKTokenIds = new int[topKCount];
            float[] topKLogits = new float[topKCount];
            Array.Fill(topKTokenIds, -1);
            Array.Fill(topKLogits, float.NegativeInfinity);

            int argmaxTokenId = 0;
            float argmaxLogit = adjustedLogits[0];
            for (int tokenId = 0; tokenId < adjustedLogits.Length; tokenId++)
            {
                float logit = adjustedLogits[tokenId];
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
            if (enableSampling && temperature > 0f && topKTokenIds.Length > 1)
            {
                (int nextTokenId, float nextTokenLogit) = SampleTopK(topKTokenIds, topKLogits);
                session.NextTokenId = nextTokenId;
                session.NextTokenLogit = nextTokenLogit;
                session.NextTokenStrategy = TopKSamplingStrategy;
            }
            else
            {
                session.NextTokenId = argmaxTokenId;
                session.NextTokenLogit = argmaxLogit;
                session.NextTokenStrategy = GreedyArgmaxStrategy;
            }

            session.TopKTokenIds = topKTokenIds;
            session.TopKLogits = topKLogits;
            session.LastSamplingElapsedMilliseconds = Stopwatch.GetElapsedTime(samplingStartTimestamp).Milliseconds;
        }

        private (int NextTokenId, float NextTokenLogit) SampleTopK(ReadOnlySpan<int> topKTokenIds, ReadOnlySpan<float> topKLogits)
        {
            float maxLogit = topKLogits[0];
            for (int index = 1; index < topKLogits.Length; index++)
            {
                if (topKLogits[index] > maxLogit)
                {
                    maxLogit = topKLogits[index];
                }
            }

            float[] probabilities = new float[topKLogits.Length];
            double sum = 0d;
            float minLogitThreshold = float.NegativeInfinity;
            if (minP > 0f)
            {
                minLogitThreshold = maxLogit + MathF.Log(minP);
            }

            for (int index = 0; index < topKLogits.Length; index++)
            {
                if (topKLogits[index] < minLogitThreshold)
                {
                    probabilities[index] = 0f;
                    continue;
                }

                float probability = MathF.Exp((topKLogits[index] - maxLogit) / temperature);
                probabilities[index] = probability;
                sum += probability;
            }

            ApplyTopP(probabilities, ref sum);
            Random activeRandom = random ?? Random.Shared;
            double target = activeRandom.NextDouble() * sum;
            double cumulative = 0d;
            for (int index = 0; index < probabilities.Length; index++)
            {
                cumulative += probabilities[index];
                if (target <= cumulative)
                {
                    return (topKTokenIds[index], topKLogits[index]);
                }
            }

            int lastIndex = probabilities.Length - 1;
            return (topKTokenIds[lastIndex], topKLogits[lastIndex]);
        }

        private void ApplyTopP(float[] probabilities, ref double sum)
        {
            if (topP >= 1f || sum <= 0d)
            {
                return;
            }

            double threshold = sum * topP;
            double cumulative = 0d;
            for (int index = 0; index < probabilities.Length; index++)
            {
                float probability = probabilities[index];
                if (probability <= 0f)
                {
                    continue;
                }

                cumulative += probability;
                if (cumulative > threshold)
                {
                    for (int trimIndex = index + 1; trimIndex < probabilities.Length; trimIndex++)
                    {
                        sum -= probabilities[trimIndex];
                        probabilities[trimIndex] = 0f;
                    }

                    return;
                }
            }
        }

        private void ApplyRepeatPenalty(BitNetSession session, float[] adjustedLogits)
        {
            if (repeatLastN == 0 || Math.Abs(repeatPenalty - 1f) < 1e-6f)
            {
                return;
            }

            ReadOnlySpan<int> tokenHistory = session.Tokens.Span;
            int startIndex = Math.Max(0, tokenHistory.Length - repeatLastN);
            HashSet<int> penalizedTokenIds = new();
            for (int index = startIndex; index < tokenHistory.Length; index++)
            {
                int tokenId = tokenHistory[index];
                if ((uint)tokenId >= (uint)adjustedLogits.Length || !penalizedTokenIds.Add(tokenId))
                {
                    continue;
                }

                adjustedLogits[tokenId] = adjustedLogits[tokenId] <= 0f
                    ? adjustedLogits[tokenId] * repeatPenalty
                    : adjustedLogits[tokenId] / repeatPenalty;
            }
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
