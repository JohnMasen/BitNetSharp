using BitNetSharp.Models;
using BitNetSharp.Nodes;

namespace BitNetSharp
{
    public sealed class BitNetRuntime : IDisposable
    {
        private readonly BitNetModel model;
        private readonly BitNetSession session;
        private readonly BitNetTokenizer tokenizer;
        private readonly EmbeddingNode embeddingNode;
        private readonly RmsNormNode[] attentionNormNodes;
        private readonly QKVProjectionNode[] qkvProjectionNodes;
        private readonly AttentionNode[] attentionNodes;
        private readonly ResidualNode[] residualNodes;
        private readonly FeedForwardNormNode[] feedForwardNormNodes;
        private readonly FeedForwardNode[] feedForwardNodes;
        private readonly FeedForwardResidualNode[] feedForwardResidualNodes;
        private readonly FinalNormNode finalNormNode;
        private readonly LmHeadNode lmHeadNode;
        private readonly SamplingNode samplingNode;
        private bool disposed;

        /// <summary>
        /// Creates a temporary runtime that can execute the current single-token inference chain end to end.
        /// </summary>
        public BitNetRuntime(BitNetModel model, BitNetMemoryManager memoryManager, InferenceConfig? inferenceConfig = null, bool enableCache = true, int topK = 10)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(memoryManager);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            }

            this.model = model;
            tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            InferenceConfig = inferenceConfig ?? new InferenceConfig(InferenceBackend.CPU, 1);
            session = new BitNetSession(model, memoryManager);
            embeddingNode = new EmbeddingNode(model, enableCache: enableCache, inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
            embeddingNode.Init();

            attentionNormNodes = new RmsNormNode[model.Layers.Count];
            qkvProjectionNodes = new QKVProjectionNode[model.Layers.Count];
            attentionNodes = new AttentionNode[model.Layers.Count];
            residualNodes = new ResidualNode[model.Layers.Count];
            feedForwardNormNodes = new FeedForwardNormNode[model.Layers.Count];
            feedForwardNodes = new FeedForwardNode[model.Layers.Count];
            feedForwardResidualNodes = new FeedForwardResidualNode[model.Layers.Count];

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex] = new RmsNormNode(model, layer.AttentionNorm, enableCache: enableCache, inferenceConfig: InferenceConfig);
                attentionNormNodes[layerIndex].Init();

                qkvProjectionNodes[layerIndex] = new QKVProjectionNode(model, layer.AttentionQueryWeight, layer.AttentionKeyWeight, layer.AttentionValueWeight, enableCache: enableCache, inferenceConfig: InferenceConfig);
                qkvProjectionNodes[layerIndex].Init();

                attentionNodes[layerIndex] = new AttentionNode(model, layer.AttentionSubNorm, layer.AttentionOutputWeight, layer.AttentionOutputScale, layer.AttentionOutputBias, enableCache: enableCache, inferenceConfig: InferenceConfig);
                attentionNodes[layerIndex].Init();

                residualNodes[layerIndex] = new ResidualNode(model, InferenceConfig);
                residualNodes[layerIndex].Init();

                feedForwardNormNodes[layerIndex] = new FeedForwardNormNode(model, layer.FeedForwardNorm, enableCache: enableCache, inferenceConfig: InferenceConfig);
                feedForwardNormNodes[layerIndex].Init();

                feedForwardNodes[layerIndex] = new FeedForwardNode(model, layer.FeedForwardSubNorm, layer.FeedForwardGateWeight, layer.FeedForwardUpWeight, layer.FeedForwardDownWeight, enableCache: enableCache, inferenceConfig: InferenceConfig);
                feedForwardNodes[layerIndex].Init();

                feedForwardResidualNodes[layerIndex] = new FeedForwardResidualNode(model, InferenceConfig);
                feedForwardResidualNodes[layerIndex].Init();
            }

            finalNormNode = new FinalNormNode(model, enableCache: enableCache, inferenceConfig: InferenceConfig);
            finalNormNode.Init();
            lmHeadNode = new LmHeadNode(model, enableCache: enableCache, inferenceConfig: InferenceConfig);
            lmHeadNode.Init();
            samplingNode = new SamplingNode(topK);
            samplingNode.Init();
        }

        public InferenceConfig InferenceConfig { get; }

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
        public int InferenceTokenId(int tokenId)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            session.Tokens = new[] { tokenId };
            session.CurrentToken = tokenId;

            embeddingNode.Forward(session);
            for (int layerIndex = 0; layerIndex < attentionNormNodes.Length; layerIndex++)
            {
                attentionNormNodes[layerIndex].Forward(session);
                qkvProjectionNodes[layerIndex].Forward(session);
                attentionNodes[layerIndex].Forward(session);
                residualNodes[layerIndex].Forward(session);
                feedForwardNormNodes[layerIndex].Forward(session);
                feedForwardNodes[layerIndex].Forward(session);
                feedForwardResidualNodes[layerIndex].Forward(session);
            }

            finalNormNode.Forward(session);
            lmHeadNode.Forward(session);
            samplingNode.Forward(session);
            return session.NextTokenId;
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            session.Dispose();
            disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
