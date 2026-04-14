using BitNetSharp.Core;
using BitNetSharp.Models;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Adds the current hidden state and the attention output to produce the feed-forward input stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class ResidualNode
    {
        private readonly BitNetModel model;
        private readonly IOPProvider opProvider;
        private bool isInitialized;

        public ResidualNode(BitNetModel model, Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the residual node can be created.");
            }

            this.model = model;
            InferenceConfig = inferenceConfig;
            opProvider = InferenceConfig.OPProvider;
        }

        public Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            isInitialized = true;
        }

        /// <summary>
        /// Adds the session embedding and attention output buffers into the feed-forward input buffer.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.EmbeddingKey))
            {
                throw new InvalidOperationException("Session does not contain embedding output.");
            }

            if (!session.HasMemory<float>(BitNetSession.AttentionOutputKey))
            {
                throw new InvalidOperationException("Session does not contain attention output.");
            }

            RuntimeTensor input = session.EmbeddingTensor;
            RuntimeTensor residual = session.AttentionOutputTensor;
            RuntimeTensor output = session.FeedForwardInputTensor;
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            if (!input.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> inputMemory) || inputMemory.Length != embeddingLength)
            {
                throw new ArgumentException("Residual input length does not match the model embedding length.", nameof(input));
            }

            if (!residual.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> residualMemory) || residualMemory.Length != embeddingLength)
            {
                throw new ArgumentException("Residual source length does not match the model embedding length.", nameof(residual));
            }

            if (!output.TryGet<Memory<float>>(out Memory<float> outputMemory) || outputMemory.Length < embeddingLength)
            {
                throw new ArgumentException("Residual output length does not match the model embedding length.", nameof(output));
            }

            opProvider.Add(input, residual, output);
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The node must be initialized by calling Init before Forward.");
            }
        }
    }
}
