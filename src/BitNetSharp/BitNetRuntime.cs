using BitNetSharp.Models;
using BitNetSharp.Nodes;
using BitNetSharp.Core;
using GGUFSharp;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp
{
    public sealed class BitNetRuntime : IDisposable
    {
        private readonly BitNetModel model;
        private readonly BitNetMemoryManager memoryManager;
        private readonly BitNetTokenizer tokenizer;
        private readonly IOPProvider opProvider;
        private readonly bool enableCache;
        private readonly EmbeddingNode embeddingNode;
        private readonly RmsNormNode[] attentionNormNodes;
        private readonly ResidualNode[] residualNodes;
        private readonly FeedForwardNormNode[] feedForwardNormNodes;
        private readonly FeedForwardResidualNode[] feedForwardResidualNodes;
        private readonly FinalNormNode finalNormNode;
        private readonly LmHeadNode lmHeadNode;
        private readonly SamplingNode samplingNode;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedQueryWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedKeyWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedValueWeights;
        private readonly float[][]? cachedAttentionSubNormWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedAttentionOutputWeights;
        private readonly float[][]? cachedAttentionOutputScales;
        private readonly float[][]? cachedAttentionOutputBiases;
        private readonly float[][]? cachedFeedForwardSubNormWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedFeedForwardGateWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedFeedForwardUpWeights;
        private readonly (byte[] PackedWeights, float Scale)[]? cachedFeedForwardDownWeights;
        private BitNetSession? session;
        private bool disposed;

        /// <summary>
        /// Creates a temporary runtime that can execute the current single-token inference chain end to end.
        /// </summary>
        public BitNetRuntime(BitNetModel model, BitNetMemoryManager memoryManager, InferenceConfig? inferenceConfig = null, bool enableCache = true, int topK = 40, bool enableSampling = false, int? samplingSeed = null, float temperature = 0.80f, float topP = 0.95f, float minP = 0.05f, int repeatLastN = 64, float repeatPenalty = 1.00f)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(memoryManager);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            }

            this.model = model;
            this.memoryManager = memoryManager;
            tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            InferenceConfig = inferenceConfig;
            this.enableCache = enableCache;
            opProvider = InferenceConfig.OPProvider;
            embeddingNode = new EmbeddingNode(model, enableCache: enableCache, inferenceConfig: InferenceConfig);
            embeddingNode.Init();

            attentionNormNodes = new RmsNormNode[model.Layers.Count];
            residualNodes = new ResidualNode[model.Layers.Count];
            feedForwardNormNodes = new FeedForwardNormNode[model.Layers.Count];
            feedForwardResidualNodes = new FeedForwardResidualNode[model.Layers.Count];
            cachedQueryWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedKeyWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedValueWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedAttentionSubNormWeights = enableCache ? new float[model.Layers.Count][] : null;
            cachedAttentionOutputWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedAttentionOutputScales = enableCache ? new float[model.Layers.Count][] : null;
            cachedAttentionOutputBiases = enableCache ? new float[model.Layers.Count][] : null;
            cachedFeedForwardSubNormWeights = enableCache ? new float[model.Layers.Count][] : null;
            cachedFeedForwardGateWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedFeedForwardUpWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;
            cachedFeedForwardDownWeights = enableCache ? new (byte[] PackedWeights, float Scale)[model.Layers.Count] : null;

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex] = new RmsNormNode(model, layer.AttentionNorm, enableCache: enableCache, inferenceConfig: InferenceConfig);
                attentionNormNodes[layerIndex].Init();

                residualNodes[layerIndex] = new ResidualNode(model, InferenceConfig);
                residualNodes[layerIndex].Init();

                feedForwardNormNodes[layerIndex] = new FeedForwardNormNode(model, layer.FeedForwardNorm, enableCache: enableCache, inferenceConfig: InferenceConfig);
                feedForwardNormNodes[layerIndex].Init();

                feedForwardResidualNodes[layerIndex] = new FeedForwardResidualNode(model, InferenceConfig);
                feedForwardResidualNodes[layerIndex].Init();

                if (enableCache)
                {
                    cachedQueryWeights![layerIndex] = ReadPackedWeights(layer.AttentionQueryWeight, "QKV query");
                    cachedKeyWeights![layerIndex] = ReadPackedWeights(layer.AttentionKeyWeight, "QKV key");
                    cachedValueWeights![layerIndex] = ReadPackedWeights(layer.AttentionValueWeight, "QKV value");
                    cachedAttentionSubNormWeights![layerIndex] = ReadFloatTensor(layer.AttentionSubNorm, "Attention sub-norm");
                    cachedAttentionOutputWeights![layerIndex] = ReadPackedWeights(layer.AttentionOutputWeight, "Packed attention");
                    cachedAttentionOutputScales![layerIndex] = layer.AttentionOutputScale is null ? [] : ReadFloatTensor(layer.AttentionOutputScale, "Attention output scale");
                    cachedAttentionOutputBiases![layerIndex] = layer.AttentionOutputBias is null ? [] : ReadFloatTensor(layer.AttentionOutputBias, "Attention output bias");
                    cachedFeedForwardSubNormWeights![layerIndex] = ReadFloatTensor(layer.FeedForwardSubNorm, "Feed-forward sub-norm");
                    cachedFeedForwardGateWeights![layerIndex] = ReadPackedWeights(layer.FeedForwardGateWeight, "Feed-forward gate");
                    cachedFeedForwardUpWeights![layerIndex] = ReadPackedWeights(layer.FeedForwardUpWeight, "Feed-forward up");
                    cachedFeedForwardDownWeights![layerIndex] = ReadPackedWeights(layer.FeedForwardDownWeight, "Feed-forward down");
                }
            }

            finalNormNode = new FinalNormNode(model, enableCache: enableCache, inferenceConfig: InferenceConfig);
            finalNormNode.Init();
            lmHeadNode = new LmHeadNode(model, enableCache: enableCache, inferenceConfig: InferenceConfig);
            lmHeadNode.Init();
            samplingNode = new SamplingNode(topK, enableSampling, samplingSeed, temperature, topP, minP, repeatLastN, repeatPenalty);
            samplingNode.Init();
        }

        public InferenceConfig InferenceConfig { get; }

        public int TopK => samplingNode.TopK;

        public bool EnableSampling => samplingNode.EnableSampling;

        public float Temperature => samplingNode.Temperature;

        public float TopP => samplingNode.TopP;

        public float MinP => samplingNode.MinP;

        public int RepeatLastN => samplingNode.RepeatLastN;

        public float RepeatPenalty => samplingNode.RepeatPenalty;

        /// <summary>
        /// Gets the session that holds the runtime state for incremental generation.
        /// </summary>
        public BitNetSession Session => session ?? throw new InvalidOperationException("The runtime does not have an active session.");

        /// <summary>
        /// Executes the current single-token inference chain and returns the decoded next token text.
        /// </summary>
        public string Inference(int tokenId)
        {
            return tokenizer.Decode(new[] { InferenceTokenId(tokenId) });
        }

        /// <summary>
        /// Executes the current single-token inference chain and returns the next token id.
        /// </summary>
        public int InferenceTokenId(int tokenId, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            StartSession();
            PrefillCore(Session, new[] { tokenId }, cancellationToken);
            return Session.NextTokenId;
        }

        /// <summary>
        /// Creates a new session and pre-fills it with the specified prompt tokens.
        /// </summary>
        public void Prefill(ReadOnlyMemory<int> promptTokenIds, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ValidatePromptTokenIds(promptTokenIds, nameof(promptTokenIds));

            StartSession();
            PrefillCore(Session, promptTokenIds, cancellationToken);
        }

        /// <summary>
        /// Appends prompt tokens to the active session and pre-fills them in order.
        /// </summary>
        public void ContinuePrefill(ReadOnlyMemory<int> promptTokenIds, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ValidatePromptTokenIds(promptTokenIds, nameof(promptTokenIds));

            PrefillCore(Session, promptTokenIds, cancellationToken);
        }

        /// <summary>
        /// Creates a new session for the specified prompt tokens and generates multiple next-token ids.
        /// </summary>
        public int[] GenerateTokenIds(ReadOnlyMemory<int> promptTokenIds, int outputTokenCount, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ValidateOutputTokenCount(outputTokenCount);

            Prefill(promptTokenIds, cancellationToken);
            return GenerateTokenIds(outputTokenCount, cancellationToken);
        }

        /// <summary>
        /// Creates a new session for the specified prompt tokens and generates decoded text for the new tokens.
        /// </summary>
        public string Generate(ReadOnlyMemory<int> promptTokenIds, int outputTokenCount, CancellationToken cancellationToken = default)
        {
            return tokenizer.Decode(GenerateTokenIds(promptTokenIds, outputTokenCount, cancellationToken));
        }

        /// <summary>
        /// Continues generation from the current session state and returns the next-token ids.
        /// </summary>
        public int[] GenerateTokenIds(int outputTokenCount, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ValidateOutputTokenCount(outputTokenCount);

            BitNetSession activeSession = Session;
            if (activeSession.Tokens.IsEmpty)
            {
                throw new InvalidOperationException("The session does not contain any prompt tokens. Initialize the session before continuing generation.");
            }

            if (activeSession.CacheLength != activeSession.Tokens.Length)
            {
                throw new InvalidOperationException("The session state is not ready for generation. Prefill the current token before continuing generation.");
            }

            int[] outputTokenIds = new int[outputTokenCount];
            activeSession.BeginOutputRound();

            try
            {
                for (int index = 0; index < outputTokenIds.Length; index++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int nextTokenId = activeSession.NextTokenId;
                    outputTokenIds[index] = nextTokenId;
                    activeSession.AppendOutputToken(nextTokenId);
                    _ = RunInferenceTokenId(activeSession, nextTokenId, cancellationToken);
                }

                return outputTokenIds;
            }
            finally
            {
                activeSession.CompleteOutputRound();
            }
        }

        /// <summary>
        /// Continues generation from the current session state and returns the decoded output text.
        /// </summary>
        public string Generate(int outputTokenCount, CancellationToken cancellationToken = default)
        {
            return tokenizer.Decode(GenerateTokenIds(outputTokenCount, cancellationToken));
        }

        /// <summary>
        /// Creates a new conversation session and pre-fills a user message using the configured chat template.
        /// </summary>
        public void StartConversation(string userMessage, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            IReadOnlyList<int> promptTokenIds = tokenizer.EncodeChatMessageToIds(BitNetChatRole.User, userMessage, isFirstMessage: true);
            Prefill(promptTokenIds.ToArray(), cancellationToken);
        }

        /// <summary>
        /// Appends a user message to the current conversation session using the configured chat template.
        /// </summary>
        public void ContinueConversation(string userMessage, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            bool isFirstMessage = session is null || Session.Tokens.IsEmpty;
            IReadOnlyList<int> promptTokenIds = tokenizer.EncodeChatMessageToIds(BitNetChatRole.User, userMessage, isFirstMessage: isFirstMessage);
            if (isFirstMessage)
            {
                Prefill(promptTokenIds.ToArray(), cancellationToken);
                return;
            }

            ContinuePrefill(promptTokenIds.ToArray(), cancellationToken);
        }

        /// <summary>
        /// Generates an assistant reply from the current conversation session until EOS or the specified token limit.
        /// </summary>
        public string GenerateAssistantReply(int maxNewTokens = 128, CancellationToken cancellationToken = default)
        {
            StringBuilder builder = new();
            foreach ((_, string tokenText) in StreamAssistantReplyWithTokenIds(maxNewTokens, cancellationToken))
            {
                builder.Append(tokenText);
            }

            return builder.ToString();
        }

        /// <summary>
        /// Generates assistant reply tokens one by one from the current conversation session until EOS or the specified token limit.
        /// </summary>
        public IEnumerable<string> StreamAssistantReply(int maxNewTokens = 128, CancellationToken cancellationToken = default)
        {
            foreach ((_, string tokenText) in StreamAssistantReplyWithTokenIds(maxNewTokens, cancellationToken))
            {
                yield return tokenText;
            }
        }

        /// <summary>
        /// Generates assistant reply tokens one by one and returns both token ids and decoded text.
        /// </summary>
        public IEnumerable<(int TokenId, string TokenText)> StreamAssistantReplyWithTokenIds(int maxNewTokens = 128, CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (maxNewTokens <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            }

            BitNetSession activeSession = Session;
            int stopTokenId = tokenizer.GetConversationTurnDelimiterTokenId();
            activeSession.BeginOutputRound();

            try
            {
                for (int index = 0; index < maxNewTokens; index++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int nextTokenId = activeSession.NextTokenId;
                    activeSession.AppendOutputToken(nextTokenId);
                    _ = RunInferenceTokenId(activeSession, nextTokenId, cancellationToken);
                    if (nextTokenId == stopTokenId)
                    {
                        yield break;
                    }

                    yield return (nextTokenId, tokenizer.Decode(new[] { nextTokenId }));
                }
            }
            finally
            {
                activeSession.CompleteOutputRound();
            }
        }

        private void StartSession()
        {
            BitNetSession? previousSession = session;
            session = new BitNetSession(model, memoryManager);
            previousSession?.Dispose();
        }

        private void PrefillCore(BitNetSession activeSession, ReadOnlyMemory<int> promptTokenIds, CancellationToken cancellationToken)
        {
            ReadOnlySpan<int> promptTokenSpan = promptTokenIds.Span;
            for (int index = 0; index < promptTokenSpan.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int tokenId = promptTokenSpan[index];
                activeSession.AppendToken(tokenId);
                _ = RunInferenceTokenId(activeSession, tokenId, cancellationToken);
            }
        }

        private int RunInferenceTokenId(BitNetSession activeSession, int tokenId, CancellationToken cancellationToken)
        {
            long inferenceStartTimestamp = Stopwatch.GetTimestamp();
            cancellationToken.ThrowIfCancellationRequested();
            activeSession.CurrentToken = tokenId;
            activeSession.CacheWritePosition = activeSession.CacheLength;

            embeddingNode.Forward(activeSession);
            // This temporary orchestration remains inside runtime on purpose so the future Graph work can
            // lift the exact layer sequence from one place instead of re-encoding it through OP interfaces.
            for (int layerIndex = 0; layerIndex < attentionNormNodes.Length; layerIndex++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex].Forward(activeSession);
                ExecuteQKVProjection(activeSession, layerIndex, layer);
                ExecuteAttention(activeSession, layerIndex, layer, cancellationToken);
                residualNodes[layerIndex].Forward(activeSession);
                feedForwardNormNodes[layerIndex].Forward(activeSession);
                ExecuteFeedForward(activeSession, layerIndex, layer);
                feedForwardResidualNodes[layerIndex].Forward(activeSession);
            }

            cancellationToken.ThrowIfCancellationRequested();
            finalNormNode.Forward(activeSession);
            lmHeadNode.Forward(activeSession);
            samplingNode.Forward(activeSession);
            activeSession.CacheLength = activeSession.CacheWritePosition + 1;
            activeSession.LastInferenceElapsedMilliseconds = Stopwatch.GetElapsedTime(inferenceStartTimestamp).Milliseconds;
            return activeSession.NextTokenId;
        }

        private static void ValidateOutputTokenCount(int outputTokenCount)
        {
            if (outputTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputTokenCount));
            }
        }

        private static void ValidatePromptTokenIds(ReadOnlyMemory<int> promptTokenIds, string parameterName)
        {
            if (promptTokenIds.IsEmpty)
            {
                throw new ArgumentException("Prompt tokens must not be empty.", parameterName);
            }
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            session?.Dispose();
            disposed = true;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Temporarily keeps QKV orchestration inside runtime so the future graph layer can reuse the same
        /// model-level sequence without depending on the legacy IOPProvider2 composition API.
        /// </summary>
        private void ExecuteQKVProjection(BitNetSession activeSession, int layerIndex, BitNetLayerDefinition layer)
        {
            RuntimeTensor input = activeSession.RmsNormTensor;
            RuntimeTensor query = activeSession.QKVQueryTensor;
            RuntimeTensor key = activeSession.QKVKeyTensor;
            RuntimeTensor value = activeSession.QKVValueTensor;
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            int queryOutputLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueOutputLength = checked((int)model.Config.KeyValueProjectionSize);

            (byte[] PackedWeights, float Scale) queryWeights = GetPackedWeights(cachedQueryWeights, layerIndex, layer.AttentionQueryWeight, "QKV query");
            (byte[] PackedWeights, float Scale) keyWeights = GetPackedWeights(cachedKeyWeights, layerIndex, layer.AttentionKeyWeight, "QKV key");
            (byte[] PackedWeights, float Scale) valueWeights = GetPackedWeights(cachedValueWeights, layerIndex, layer.AttentionValueWeight, "QKV value");

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..inputMemory.Length];
            RuntimeTensor quantizedTensor = RuntimeTensor.CreateWritable("RuntimeQKVQuantized", quantizedValues, [inputMemory.Length]);
            (float activationScale, _) = opProvider.QuantizeBitNetActivations(input, quantizedTensor);

            opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(queryWeights.PackedWeights, "RuntimeQKVQueryWeights"), queryOutputLength, queryWeights.Scale, query);
            opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(keyWeights.PackedWeights, "RuntimeQKVKeyWeights"), keyValueOutputLength, keyWeights.Scale, key);
            opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(valueWeights.PackedWeights, "RuntimeQKVValueWeights"), keyValueOutputLength, valueWeights.Scale, value);

        }

        /// <summary>
        /// Keeps the current single-token attention composition close to inference until the dedicated graph
        /// abstraction is ready to own model-level scheduling.
        /// </summary>
        private void ExecuteAttention(BitNetSession activeSession, int layerIndex, BitNetLayerDefinition layer, CancellationToken cancellationToken)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            int headCount = checked((int)model.Config.AttentionHeadCount);
            int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
            int headDimension = checked((int)model.Config.AttentionHeadDimension);
            ReadOnlyMemory<float> query = activeSession.QKVQuery;
            ReadOnlyMemory<float> key = activeSession.QKVKey;
            ReadOnlyMemory<float> value = activeSession.QKVValue;
            using IMemoryOwner<float> ropeQueryOwner = MemoryPool<float>.Shared.Rent(query.Length);
            using IMemoryOwner<float> ropeKeyOwner = MemoryPool<float>.Shared.Rent(key.Length);
            Memory<float> ropeQuery = ropeQueryOwner.Memory[..query.Length];
            Memory<float> ropeKey = ropeKeyOwner.Memory[..key.Length];
            ApplyRope(query.Span, ropeQuery.Span, activeSession.CacheWritePosition, headCount, headDimension, checked((int)model.Config.RopeDimensionCount), model.Config.RopeFrequencyBase);
            ApplyRope(key.Span, ropeKey.Span, activeSession.CacheWritePosition, keyValueHeadCount, headDimension, checked((int)model.Config.RopeDimensionCount), model.Config.RopeFrequencyBase);
            WriteCurrentKeyValueToCache(activeSession, layerIndex, ropeKey, value);
            float[] subNormWeights = GetFloatTensor(cachedAttentionSubNormWeights, layerIndex, layer.AttentionSubNorm, "Attention sub-norm");
            (byte[] PackedWeights, float Scale) outputWeights = GetPackedWeights(cachedAttentionOutputWeights, layerIndex, layer.AttentionOutputWeight, "Packed attention");
            float[] outputScaleValues = GetOptionalFloatTensor(cachedAttentionOutputScales, layerIndex, layer.AttentionOutputScale, "Attention output scale");
            float[] outputBiasValues = GetOptionalFloatTensor(cachedAttentionOutputBiases, layerIndex, layer.AttentionOutputBias, "Attention output bias");
            RuntimeTensor subNorm = activeSession.AttentionSubNormTensor;
            RuntimeTensor output = activeSession.AttentionOutputTensor;

            ValidateAttentionProjection(query.Span, key.Span, value.Span, embeddingLength, keyValueLength);

            using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
            Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
            BuildCachedAttentionContext(activeSession, layerIndex, ropeQuery, attentionContext, headCount, keyValueHeadCount, headDimension, keyValueLength, cancellationToken);
            RuntimeTensor attentionContextTensor = RuntimeTensor.CreateWritable("RuntimeAttentionContext", attentionContext, [embeddingLength]);
            RuntimeTensor subNormWeightsTensor = RuntimeTensor.CreateReadOnly<float>("RuntimeAttentionSubNormWeights", subNormWeights.AsMemory(0, attentionContext.Length), [attentionContext.Length]);
            opProvider.ForwardRmsNorm(attentionContextTensor, subNormWeightsTensor, model.Config.AttentionLayerNormRmsEpsilon, subNorm);
            opProvider.ProjectBitNetI2(subNorm, CreatePackedWeightTensor(outputWeights.PackedWeights, "RuntimeAttentionOutputWeights"), embeddingLength, outputWeights.Scale, output);
            Memory<float> outputMemory = output.GetMemory<float>();
            ApplyScale(outputMemory[..embeddingLength], outputScaleValues);
            ApplyBias(outputMemory[..embeddingLength], outputBiasValues);
        }

        /// <summary>
        /// Keeps feed-forward orchestration in runtime as a transition step before a future graph-based layer
        /// scheduler replaces this handwritten sequence.
        /// </summary>
        private void ExecuteFeedForward(BitNetSession activeSession, int layerIndex, BitNetLayerDefinition layer)
        {
            RuntimeTensor input = activeSession.FeedForwardNormTensor;
            ReadOnlyMemory<float> inputMemory = input.GetReadOnlyMemory<float>();
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int feedForwardLength = checked((int)model.Config.FeedForwardLength);
            float[] subNormWeights = GetFloatTensor(cachedFeedForwardSubNormWeights, layerIndex, layer.FeedForwardSubNorm, "Feed-forward sub-norm");
            (byte[] PackedWeights, float Scale) gateWeights = GetPackedWeights(cachedFeedForwardGateWeights, layerIndex, layer.FeedForwardGateWeight, "Feed-forward gate");
            (byte[] PackedWeights, float Scale) upWeights = GetPackedWeights(cachedFeedForwardUpWeights, layerIndex, layer.FeedForwardUpWeight, "Feed-forward up");
            (byte[] PackedWeights, float Scale) downWeights = GetPackedWeights(cachedFeedForwardDownWeights, layerIndex, layer.FeedForwardDownWeight, "Feed-forward down");
            RuntimeTensor subNormOutput = activeSession.FeedForwardSubNormTensor;
            RuntimeTensor output = activeSession.FeedForwardOutputTensor;

            using IMemoryOwner<float> upOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<float> gateOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(inputMemory.Length);
            Memory<float> up = upOwner.Memory[..feedForwardLength];
            Memory<float> gate = gateOwner.Memory[..feedForwardLength];
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..inputMemory.Length];
            RuntimeTensor upTensor = RuntimeTensor.CreateWritable("RuntimeFeedForwardUp", up, [feedForwardLength]);
            RuntimeTensor gateTensor = RuntimeTensor.CreateWritable("RuntimeFeedForwardGate", gate, [feedForwardLength]);
            RuntimeTensor quantizedTensor = RuntimeTensor.CreateWritable("RuntimeFeedForwardQuantized", quantizedValues, [inputMemory.Length]);
            RuntimeTensor subNormWeightsTensor = RuntimeTensor.CreateReadOnly<float>("RuntimeFeedForwardSubNormWeights", subNormWeights.AsMemory(0, feedForwardLength), [feedForwardLength]);
            (float activationScale, _) = opProvider.QuantizeBitNetActivations(input, quantizedTensor);

            opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(upWeights.PackedWeights, "RuntimeFeedForwardUpWeights"), feedForwardLength, upWeights.Scale, upTensor);
            opProvider.ProjectBitNetI2(quantizedTensor, activationScale, CreatePackedWeightTensor(gateWeights.PackedWeights, "RuntimeFeedForwardGateWeights"), feedForwardLength, gateWeights.Scale, gateTensor);
            ApplySquaredReluGate(gate.Span, up.Span);
            opProvider.ForwardRmsNorm(upTensor, subNormWeightsTensor, model.Config.AttentionLayerNormRmsEpsilon, subNormOutput);
            opProvider.ProjectBitNetI2(subNormOutput, CreatePackedWeightTensor(downWeights.PackedWeights, "RuntimeFeedForwardDownWeights"), embeddingLength, downWeights.Scale, output);
        }

        private static RuntimeTensor CreatePackedWeightTensor(ReadOnlyMemory<byte> packedWeights, string name)
        {
            return RuntimeTensor.CreateReadOnly<byte>(name, packedWeights, [packedWeights.Length]);
        }

        private static void ApplySquaredReluGate(ReadOnlySpan<float> gate, Span<float> up)
        {
            for (int index = 0; index < up.Length; index++)
            {
                float relu = MathF.Max(gate[index], 0f);
                up[index] *= relu * relu;
            }
        }

        private void BuildCachedAttentionContext(BitNetSession activeSession, int layerIndex, ReadOnlyMemory<float> query, Memory<float> context, int headCount, int keyValueHeadCount, int headDimension, int keyValueLength, CancellationToken cancellationToken)
        {
            if (headCount % keyValueHeadCount != 0)
            {
                throw new InvalidOperationException("Attention head count must be divisible by the key/value head count.");
            }

            int cacheLength = activeSession.CacheWritePosition + 1;
            if (cacheLength <= 0)
            {
                throw new InvalidOperationException("Attention cache length must be positive.");
            }

            RuntimeTensor keyCacheTensor = activeSession.GetOrCreateLayerKeyCacheTensor(layerIndex);
            RuntimeTensor valueCacheTensor = activeSession.GetOrCreateLayerValueCacheTensor(layerIndex);
            ReadOnlyMemory<float> keyCache = keyCacheTensor.GetReadOnlyMemory<float>();
            ReadOnlyMemory<float> valueCache = valueCacheTensor.GetReadOnlyMemory<float>();
            int groupSize = headCount / keyValueHeadCount;
            float scoreScale = 1f / MathF.Sqrt(headDimension);
            using IMemoryOwner<float> attentionScoreOwner = MemoryPool<float>.Shared.Rent(cacheLength);
            using IMemoryOwner<float> attentionWeightOwner = MemoryPool<float>.Shared.Rent(cacheLength);
            Memory<float> attentionScore = attentionScoreOwner.Memory[..cacheLength];
            Memory<float> attentionWeight = attentionWeightOwner.Memory[..cacheLength];
            RuntimeTensor attentionScoreTensor = RuntimeTensor.CreateWritable("RuntimeAttentionScore", attentionScore, [cacheLength]);
            RuntimeTensor attentionWeightTensor = RuntimeTensor.CreateWritable("RuntimeAttentionWeight", attentionWeight, [cacheLength]);
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int sourceHeadIndex = headIndex / groupSize;
                int queryOffset = headIndex * headDimension;
                int outputOffset = headIndex * headDimension;

                for (int tokenIndex = 0; tokenIndex < cacheLength; tokenIndex++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int cacheOffset = tokenIndex * keyValueLength + (sourceHeadIndex * headDimension);
                    attentionScore.Span[tokenIndex] = ComputeScaledAttentionScore(query.Span, queryOffset, keyCache.Span, cacheOffset, headDimension, scoreScale);
                }

                opProvider.ForwardSoftmax(attentionScoreTensor, attentionWeightTensor);

                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    float weightedValue = 0f;
                    for (int tokenIndex = 0; tokenIndex < cacheLength; tokenIndex++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        int cacheOffset = GetValueCacheOffset(activeSession, tokenIndex, sourceHeadIndex, dimensionIndex, headDimension);
                        weightedValue += valueCache.Span[cacheOffset] * attentionWeight.Span[tokenIndex];
                    }

                    context.Span[outputOffset + dimensionIndex] = weightedValue;
                }
            }

        }

        private static void WriteCurrentKeyValueToCache(BitNetSession activeSession, int layerIndex, ReadOnlyMemory<float> key, ReadOnlyMemory<float> value)
        {
            int keyValueLength = key.Length;
            int cacheOffset = activeSession.CacheWritePosition * keyValueLength;
            Memory<float> keyCache = activeSession.GetOrCreateLayerKeyCacheTensor(layerIndex).GetMemory<float>();
            Memory<float> valueCache = activeSession.GetOrCreateLayerValueCacheTensor(layerIndex).GetMemory<float>();

            if (cacheOffset + keyValueLength > keyCache.Length || cacheOffset + value.Length > valueCache.Length)
            {
                throw new InvalidOperationException("The current token exceeds the allocated KV cache capacity.");
            }

            key.Span.CopyTo(keyCache.Span.Slice(cacheOffset, keyValueLength));
            WriteValueCache(activeSession, value.Span, valueCache.Span);
        }

        private static int GetValueCacheOffset(BitNetSession activeSession, int tokenIndex, int kvHeadIndex, int dimensionIndex, int headDimension)
        {
            int nCtx = checked((int)activeSession.Model.Config!.ContextLength);
            return tokenIndex + (nCtx * dimensionIndex) + (nCtx * headDimension * kvHeadIndex);
        }

        private static void WriteValueCache(BitNetSession activeSession, ReadOnlySpan<float> value, Span<float> valueCache)
        {
            int nCtx = checked((int)activeSession.Model.Config!.ContextLength);
            int keyValueHeadCount = checked((int)activeSession.Model.Config.AttentionKeyValueHeadCount);
            int headDimension = checked((int)activeSession.Model.Config.AttentionHeadDimension);
            int tokenIndex = activeSession.CacheWritePosition;
            for (int kvHeadIndex = 0; kvHeadIndex < keyValueHeadCount; kvHeadIndex++)
            {
                int sourceHeadOffset = kvHeadIndex * headDimension;
                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    int sourceOffset = sourceHeadOffset + dimensionIndex;
                    int destinationOffset = tokenIndex + (nCtx * dimensionIndex) + (nCtx * headDimension * kvHeadIndex);
                    valueCache[destinationOffset] = value[sourceOffset];
                }
            }
        }

        private static void ValidateAttentionProjection(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value, int embeddingLength, int keyValueLength)
        {
            if (query.Length != embeddingLength)
            {
                throw new InvalidOperationException("Attention query length does not match the loaded model configuration.");
            }

            if (key.Length != keyValueLength)
            {
                throw new InvalidOperationException("Attention key length does not match the loaded model configuration.");
            }

            if (value.Length != keyValueLength)
            {
                throw new InvalidOperationException("Attention value length does not match the loaded model configuration.");
            }
        }

        private static void ApplyScale(Memory<float> output, ReadOnlySpan<float> scaleValues)
        {
            if (scaleValues.IsEmpty)
            {
                return;
            }

            float outputScale = scaleValues[0];
            for (int index = 0; index < output.Length; index++)
            {
                output.Span[index] *= outputScale;
            }
        }

        private static void ApplyBias(Memory<float> output, ReadOnlySpan<float> biasValues)
        {
            if (biasValues.IsEmpty)
            {
                return;
            }

            for (int index = 0; index < output.Length; index++)
            {
                output.Span[index] += biasValues[index];
            }
        }

        private static float ComputeScaledAttentionScore(ReadOnlySpan<float> query, int queryOffset, ReadOnlySpan<float> key, int keyOffset, int headDimension, float scoreScale)
        {
            float dotProduct = 0f;
            for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
            {
                dotProduct += query[queryOffset + dimensionIndex] * key[keyOffset + dimensionIndex];
            }

            return dotProduct * scoreScale;
        }

        private static float RoundTripThroughHalf(float value)
        {
            return (float)(Half)value;
        }

        private static void ApplyRope(ReadOnlySpan<float> source, Span<float> destination, int positionIndex, int headCount, int headDimension, int ropeDimensionCount, float freqBase)
        {
            source.CopyTo(destination);
            int rotaryDimensions = Math.Min(Math.Min(ropeDimensionCount, headDimension), headDimension - (headDimension % 2));
            if (rotaryDimensions <= 0)
            {
                return;
            }

            int rotaryHalf = rotaryDimensions / 2;
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                int headOffset = headIndex * headDimension;
                for (int rotaryIndex = 0; rotaryIndex < rotaryHalf; rotaryIndex++)
                {
                    float theta = positionIndex / MathF.Pow(freqBase, (2f * rotaryIndex) / rotaryDimensions);
                    float cos = MathF.Cos(theta);
                    float sin = MathF.Sin(theta);
                    int evenOffset = headOffset + rotaryIndex;
                    int oddOffset = headOffset + rotaryIndex + rotaryHalf;
                    float even = source[evenOffset];
                    float odd = source[oddOffset];
                    destination[evenOffset] = (even * cos) - (odd * sin);
                    destination[oddOffset] = (odd * cos) + (even * sin);
                }
            }
        }

        private (byte[] PackedWeights, float Scale) GetPackedWeights((byte[] PackedWeights, float Scale)[]? cache, int layerIndex, BitNetTensorInfo tensor, string tensorLabel)
        {
            return cache is null ? ReadPackedWeights(tensor, tensorLabel) : cache[layerIndex];
        }

        private float[] GetFloatTensor(float[][]? cache, int layerIndex, BitNetTensorInfo tensor, string tensorLabel)
        {
            return cache is null ? ReadFloatTensor(tensor, tensorLabel) : cache[layerIndex];
        }

        private float[] GetOptionalFloatTensor(float[][]? cache, int layerIndex, BitNetTensorInfo? tensor, string tensorLabel)
        {
            if (tensor is null)
            {
                return [];
            }

            return cache is null ? ReadFloatTensor(tensor, tensorLabel) : cache[layerIndex];
        }

        private (byte[] PackedWeights, float Scale) ReadPackedWeights(BitNetTensorInfo tensor, string tensorLabel)
        {
            using IMemoryOwner<byte> tensorData = model.ReadTensorData(tensor);
            int packedWeightByteCount = checked(((int)tensor.Dimensions[0] * (int)tensor.Dimensions[1]) / 4);
            if (tensorData.Memory.Length < packedWeightByteCount + sizeof(float))
            {
                throw new InvalidOperationException($"{tensorLabel} tensor '{tensor.Name}' is incomplete.");
            }

            return (
                tensorData.Memory[..packedWeightByteCount].ToArray(),
                MemoryMarshal.Read<float>(tensorData.Memory.Span.Slice(packedWeightByteCount, sizeof(float))));
        }

        private float[] ReadFloatTensor(BitNetTensorInfo tensor, string tensorLabel)
        {
            using IMemoryOwner<byte> tensorData = model.ReadTensorData(tensor);
            return tensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"{tensorLabel} tensor type '{tensor.TensorType}' is not supported."),
            };
        }

        private static float[] ConvertHalfToSingle(ReadOnlySpan<Half> source)
        {
            float[] values = new float[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                values[index] = (float)source[index];
            }

            return values;
        }
    }
}
