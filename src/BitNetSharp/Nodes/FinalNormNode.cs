using BitNetSharp.Core;
using BitNetSharp.Models;
using GGUFSharp;
namespace BitNetSharp.Nodes
{
    /// <summary>
    /// Applies RMS normalization to the final hidden state stored on a <see cref="BitNetSession"/>.
    /// Call <see cref="Init"/> before invoking <see cref="Forward(BitNetSession)"/>.
    /// </summary>
    public sealed class FinalNormNode
    {
        private readonly BitNetModel model;
        private readonly BitNetTensorInfo outputNormTensor;
        private readonly IOPProvider opProvider;
        private RuntimeTensor? cachedNormWeights;
        private bool isInitialized;

        public FinalNormNode(BitNetModel model, bool enableCache = false, Nodes.InferenceConfig? inferenceConfig = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the final norm node can be created.");
            }

            this.model = model;
            outputNormTensor = model.GlobalTensors?.OutputNorm ?? throw new InvalidOperationException("The model must be loaded before the final norm node can be created.");
            EnableCache = enableCache;
            InferenceConfig = inferenceConfig;
            opProvider = InferenceConfig.OPProvider;

            ValidateTensorShape();
            ValidateTensorType();
        }

        public bool EnableCache { get; }

        public Nodes.InferenceConfig InferenceConfig { get; }

        public void Init()
        {
            if (EnableCache)
            {
                _ = EnsureCachedNormWeights();
            }

            isInitialized = true;
        }

        /// <summary>
        /// Applies RMSNorm to the final hidden state stored on the session.
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
                throw new InvalidOperationException("Session does not contain final hidden state.");
            }

            RuntimeTensor input = session.EmbeddingTensor;
            RuntimeTensor output = session.FinalNormOutputTensor;
            if (!input.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> inputMemory) || inputMemory.IsEmpty)
            {
                throw new ArgumentException("Input must not be empty.", nameof(input));
            }

            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            if (inputMemory.Length != expectedLength)
            {
                throw new ArgumentException("Input length does not match the model embedding length.", nameof(input));
            }

            RuntimeTensor normWeights = EnableCache
                ? EnsureCachedNormWeights()
                : session.GetWeightTensor(outputNormTensor.Name);
            opProvider.ForwardRmsNorm(input, normWeights, model.Config!.AttentionLayerNormRmsEpsilon, output);
        }

        private void ValidateTensorShape()
        {
            int expectedLength = checked((int)model.Config!.EmbeddingLength);
            int actualLength = GetElementCount(outputNormTensor.Dimensions);
            if (actualLength != expectedLength)
            {
                throw new InvalidOperationException("Final norm tensor dimensions do not match the loaded model configuration.");
            }
        }

        private void ValidateTensorType()
        {
            if (outputNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F32 && outputNormTensor.TensorType != GGUFTensorType.GGML_TYPE_F16)
            {
                throw new NotSupportedException($"Final norm tensor type '{outputNormTensor.TensorType}' is not supported.");
            }
        }

        private RuntimeTensor EnsureCachedNormWeights()
        {
            return cachedNormWeights ??= model.GetWeightTensor(outputNormTensor.Name);
        }

        private void EnsureInitialized()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("The node must be initialized by calling Init before Forward.");
            }
        }

        private static int GetElementCount(IReadOnlyList<ulong> dimensions)
        {
            if (dimensions.Count == 0)
            {
                throw new InvalidOperationException("Final norm tensor dimensions are incomplete.");
            }

            ulong total = 1;
            foreach (ulong dimension in dimensions)
            {
                total = checked(total * dimension);
            }

            return checked((int)total);
        }
    }
}
