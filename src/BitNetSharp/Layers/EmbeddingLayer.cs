using BitNetSharp.Models;
using GGUFSharp;
using System;
using System.Runtime.InteropServices;

namespace BitNetSharp.Layers
{
    public sealed class EmbeddingLayer
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo tokenEmbedding;
        private Half[]? cachedEmbeddingValues;

        public EmbeddingLayer(BitNetModel model, bool enableCache = false)
        {
            ArgumentNullException.ThrowIfNull(model);

            this.model = model;
            EnableCache = enableCache;
            tokenEmbedding = model.GlobalTensors?.TokenEmbedding ?? throw new InvalidOperationException("The model must be loaded before the embedding layer can be created.");

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

        /// <summary>
        /// Returns the embedding vector for the specified token id.
        /// </summary>
        public float[] Forward(InferenceContext context)
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before embeddings can be read.");
            }

            int vocabularySize = checked((int)model.Config.VocabularySize);
            if ((uint)context.CurrentToken >= (uint)vocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(context.CurrentToken));
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

            int rowOffset = checked(context.CurrentToken * embeddingLength);
            ReadOnlySpan<Half> embeddingValues = values.Slice(rowOffset, embeddingLength);
            float[] embedding = new float[embeddingLength];
            for (int index = 0; index < embeddingValues.Length; index++)
            {
                embedding[index] = (float)embeddingValues[index];
            }

            return embedding;
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
    }
}
