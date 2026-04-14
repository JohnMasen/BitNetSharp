using BitNetSharp.Core;
using BitNetSharp.Models;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Adds the feed-forward input and feed-forward output to produce the updated hidden state stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class FeedForwardResidualNode
    {
        private readonly BitNetModel model;
        private readonly IOPProvider opProvider;
        private bool isInitialized;

        public FeedForwardResidualNode(BitNetModel model, Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the feed-forward residual node can be created.");
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
        /// Adds the session feed-forward input and feed-forward output buffers into the main hidden-state buffer.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.FeedForwardInputKey))
            {
                throw new InvalidOperationException("Session does not contain feed-forward input.");
            }

            if (!session.HasMemory<float>(BitNetSession.FeedForwardOutputKey))
            {
                throw new InvalidOperationException("Session does not contain feed-forward output.");
            }

            RuntimeTensor input = session.FeedForwardInputTensor;
            RuntimeTensor residual = session.FeedForwardOutputTensor;
            RuntimeTensor output = session.EmbeddingTensor;
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            if (!input.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> inputMemory) || inputMemory.Length != embeddingLength)
            {
                throw new ArgumentException("Feed-forward residual input length does not match the model embedding length.", nameof(input));
            }

            if (!residual.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> residualMemory) || residualMemory.Length != embeddingLength)
            {
                throw new ArgumentException("Feed-forward residual source length does not match the model embedding length.", nameof(residual));
            }

            if (!output.TryGet<Memory<float>>(out Memory<float> outputMemory) || outputMemory.Length < embeddingLength)
            {
                throw new ArgumentException("Feed-forward residual output length does not match the model embedding length.", nameof(output));
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
