using BitNetSharp.Core;
using BitNetSharp.Models;
using System;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Adds the current hidden state and the attention output to produce the feed-forward input stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class ResidualLayer
    {
        private readonly BitNetModel model;
        private bool isInitialized;

        public ResidualLayer(BitNetModel model, InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the residual layer can be created.");
            }

            this.model = model;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
        }

        public InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            isInitialized = true;
        }

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.CPU, 1);
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

            ForwardCore(session.Embedding, session.AttentionOutput, session.FeedForwardInput);
        }

        private void ForwardCore(ReadOnlyMemory<float> input, ReadOnlyMemory<float> residual, Memory<float> output)
        {
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

            ExecuteForward(input, residual, output[..embeddingLength]);
        }

        private void ExecuteForward(ReadOnlyMemory<float> input, ReadOnlyMemory<float> residual, Memory<float> output)
        {
            switch (InferenceConfig.Backend)
            {
                case InferenceBackend.CPU:
                    AddCpu(input, residual, output, InferenceConfig.ThreadCount);
                    return;
                case InferenceBackend.Tensor:
                    MathHelper.AddTensor(input, residual, output, InferenceConfig.ThreadCount);
                    return;
                case InferenceBackend.SIMD:
                    MathHelper.AddSimd(input, residual, output, InferenceConfig.ThreadCount);
                    return;
                default:
                    throw new NotSupportedException($"Residual backend '{InferenceConfig.Backend}' is not implemented yet.");
            }
        }

        private static void AddCpu(ReadOnlyMemory<float> input, ReadOnlyMemory<float> residual, Memory<float> output, int threads)
        {
            if (threads == 1 || input.Length <= 1)
            {
                AddCpuRange(input.Span, residual.Span, output.Span, 0, input.Length);
                return;
            }

            ThreadHelper.ForEachRange(input.Length, (startIndex, endIndex) =>
                AddCpuRange(input.Span, residual.Span, output.Span, startIndex, endIndex), threads, sizeof(float));
        }

        private static void AddCpuRange(ReadOnlySpan<float> input, ReadOnlySpan<float> residual, Span<float> output, int startIndex, int endIndex)
        {
            for (int index = startIndex; index < endIndex; index++)
            {
                output[index] = input[index] + residual[index];
            }
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The layer must be initialized by calling Init before Forward.");
            }
        }
    }
}
