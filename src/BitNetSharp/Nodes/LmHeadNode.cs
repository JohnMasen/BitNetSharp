using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;

namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Projects the final normalized hidden state stored on a <see cref="BitNetSession"/> into vocabulary logits using the tied token embedding weights.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class LmHeadNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo tokenEmbedding;
        private readonly IOPProvider2 opProvider;
        private byte[]? cachedEmbeddingWeights;
        private bool isInitialized;

        public LmHeadNode(BitNetModel model, bool enableCache = false, global::BitNetSharp.Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the LM head node can be created.");
            }

            this.model = model;
            tokenEmbedding = model.GlobalTensors?.TokenEmbedding ?? throw new InvalidOperationException("The model must be loaded before the LM head node can be created.");
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig ?? CreateDefaultInferenceConfig();
            opProvider = InferenceConfig.Backend switch
            {
                global::BitNetSharp.Nodes.InferenceBackend.CPU => new CPUDefaultOPProvider(InferenceConfig.ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.Tensor => new CPUTensorOPProvider(InferenceConfig.ThreadCount),
                global::BitNetSharp.Nodes.InferenceBackend.SIMD => new CPUSimdOPProvider(InferenceConfig.ThreadCount),
                _ => throw new NotSupportedException($"Backend '{InferenceConfig.Backend}' is not implemented yet."),
            };

            ValidateTensorShape();
            ValidateTensorType();
        }

        public bool EnableCache { get; }

        public global::BitNetSharp.Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedEmbeddingWeights();
            }

            isInitialized = true;
        }

        private static global::BitNetSharp.Nodes.InferenceConfig CreateDefaultInferenceConfig()
        {
            return new global::BitNetSharp.Nodes.InferenceConfig(global::BitNetSharp.Nodes.InferenceBackend.SIMD, global::BitNetSharp.Nodes.InferenceConfig.AutoThreadCount);
        }

        /// <summary>
        /// Projects the final normalized hidden state on the session into vocabulary logits.
        /// </summary>
        public void Forward(BitNetSession session)
        {
            ArgumentNullException.ThrowIfNull(session);
            EnsureInitialized();

            if (!ReferenceEquals(session.Model, model))
            {
                throw new InvalidOperationException("The session was created for a different model instance.");
            }

            if (!session.HasMemory<float>(BitNetSession.FinalNormOutputKey))
            {
                throw new InvalidOperationException("Session does not contain final norm output.");
            }

            ReadOnlyMemory<float> input = session.FinalNormOutput;
            Memory<float> output = session.Logits;
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int vocabularySize = checked((int)model.Config.VocabularySize);
            if (input.Length != embeddingLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            if (output.Length < vocabularySize)
            {
                throw new ArgumentException("Output length does not match the model vocabulary size.", nameof(output));
            }

            if (EnableCache)
            {
                opProvider.ForwardLmHead(input, EnsureCachedEmbeddingWeights(), embeddingLength, vocabularySize, output);
                return;
            }

            using var tensorData = model.ReadTensorData(tokenEmbedding);
            opProvider.ForwardLmHead(input, tensorData.Memory[..GetEmbeddingWeightByteCount()], embeddingLength, vocabularySize, output);
        }

        private void ValidateTensorShape()
        {
            if (tokenEmbedding.Dimensions.Count < 2)
            {
                throw new InvalidOperationException("LM head tensor dimensions are incomplete.");
            }

            int expectedRowLength = checked((int)model.Config!.EmbeddingLength);
            int expectedRowCount = checked((int)model.Config.VocabularySize);
            int actualRowLength = checked((int)tokenEmbedding.Dimensions[0]);
            int actualRowCount = checked((int)tokenEmbedding.Dimensions[1]);
            if (actualRowLength != expectedRowLength || actualRowCount != expectedRowCount)
            {
                throw new InvalidOperationException("LM head tensor dimensions do not match the loaded model configuration.");
            }
        }

        private void ValidateTensorType()
        {
            if (tokenEmbedding.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"LM head tensor type '{tokenEmbedding.TensorType}' is not supported.");
            }
        }

        private int GetEmbeddingWeightByteCount()
        {
            return checked((int)model.Config!.EmbeddingLength * (int)model.Config.VocabularySize * sizeof(ushort));
        }

        private byte[] ReadEmbeddingWeights()
        {
            using var tensorData = model.ReadTensorData(tokenEmbedding);
            return tensorData.Memory[..GetEmbeddingWeightByteCount()].ToArray();
        }

        private byte[] EnsureCachedEmbeddingWeights()
        {
            return cachedEmbeddingWeights ??= ReadEmbeddingWeights();
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
