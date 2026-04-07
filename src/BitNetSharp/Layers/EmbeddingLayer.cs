using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    /// <summary>
    /// Resolves token embeddings for the current token stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class EmbeddingLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo tokenEmbedding;
        private Half[]? cachedEmbeddingValues;
        private bool isInitialized;

        public EmbeddingLayer(BitNetModel model, bool enableCache = false, InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);

            this.model = model;
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
            tokenEmbedding = model.GlobalTensors?.TokenEmbedding ?? throw new InvalidOperationException("The model must be loaded before the embedding layer can be created.");

            if (InferenceConfig.Backend != InferenceBackend.CPU)
            {
                throw new NotSupportedException($"Embedding backend '{InferenceConfig.Backend}' is not implemented yet.");
            }

            if (tokenEmbedding.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Embedding tensor type '{tokenEmbedding.TensorType}' is not supported.");
            }

            if (tokenEmbedding.Dimensions.Count < 2)
            {
                throw new InvalidOperationException("Embedding tensor dimensions are incomplete.");
            }

        }

        public bool EnableCache { get; }

        public InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedEmbeddingValues();
            }

            isInitialized = true;
        }

        private static InferenceConfig CreateDefaultInferenceConfig()
        {
            return new InferenceConfig(InferenceBackend.CPU, 1);
        }

        /// <summary>
        /// Returns the embedding vector for the current token stored on the session.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before embeddings can be read.");
            }

            int vocabularySize = checked((int)model.Config.VocabularySize);
            if ((uint)session.CurrentToken >= (uint)vocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(session.CurrentToken));
            }

            int embeddingLength = checked((int)model.Config.EmbeddingLength);
            int tensorWidth = checked((int)tokenEmbedding.Dimensions[0]);
            int tensorRows = checked((int)tokenEmbedding.Dimensions[1]);
            if (tensorWidth != embeddingLength || tensorRows != vocabularySize)
            {
                throw new InvalidOperationException("Embedding tensor dimensions do not match the loaded model configuration.");
            }

            ReadOnlySpan<Half> values = EnableCache
                ? EnsureCachedEmbeddingValues()
                : ReadAllEmbeddingValues();

            int rowOffset = checked(session.CurrentToken * embeddingLength);
            ReadOnlySpan<Half> embeddingValues = values.Slice(rowOffset, embeddingLength);
            float[] embedding = new float[embeddingLength];
            for (int index = 0; index < embeddingValues.Length; index++)
            {
                embedding[index] = (float)embeddingValues[index];
            }

            session.Embedding = embedding;
        }

        private Half[] ReadAllEmbeddingValues()
        {
            using var tensorData = model.ReadTensorData(tokenEmbedding);
            return MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span).ToArray();
        }

        private Half[] EnsureCachedEmbeddingValues()
        {
            return cachedEmbeddingValues ??= ReadAllEmbeddingValues();
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
