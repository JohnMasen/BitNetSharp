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

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the residual node can be created.");
            }

            this.model = model;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
            opProvider = InferenceConfig.Backend switch
            {
                Nodes.InferenceBackend.CPU => new CPUDefaultOPProvider(InferenceConfig.ThreadCount),
                Nodes.InferenceBackend.Tensor => new CPUTensorOPProvider(InferenceConfig.ThreadCount),
                Nodes.InferenceBackend.SIMD => new CPUSimdOPProvider(InferenceConfig.ThreadCount),
                _ => throw new NotSupportedException($"Backend '{InferenceConfig.Backend}' is not implemented yet."),
            };
        }

        public Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            isInitialized = true;
        }

        private static Nodes.InferenceConfig CreateDefaultInferenceConfig()
        {
            return new Nodes.InferenceConfig(Nodes.InferenceBackend.CPU, 1);
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

            ReadOnlyMemory<float> input = session.Embedding;
            ReadOnlyMemory<float> residual = session.AttentionOutput;
            Memory<float> output = session.FeedForwardInput;
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            if (input.Length != embeddingLength)
            {
                throw new ArgumentException("Residual input length does not match the model embedding length.", nameof(input));
            }

            if (residual.Length != embeddingLength)
            {
                throw new ArgumentException("Residual source length does not match the model embedding length.", nameof(residual));
            }

            if (output.Length < embeddingLength)
            {
                throw new ArgumentException("Residual output length does not match the model embedding length.", nameof(output));
            }

            opProvider.Add(input, residual, output[..embeddingLength]);
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
