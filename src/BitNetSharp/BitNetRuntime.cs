using BitNetSharp.Models;
using BitNetSharp.Nodes;
using BitNetSharp.Core;
using GGUFSharp;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp
{
    public sealed class BitNetRuntime : IDisposable
    {
        private readonly BitNetModel model;
        private readonly BitNetSession session;
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
        private bool disposed;

        /// <summary>
        /// Creates a temporary runtime that can execute the current single-token inference chain end to end.
        /// </summary>
        public BitNetRuntime(BitNetModel model, BitNetMemoryManager memoryManager, InferenceConfig? inferenceConfig = null, bool enableCache = true, int topK = 10)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(memoryManager);
            ArgumentNullException.ThrowIfNull(inferenceConfig);

            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            }

            this.model = model;
            tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The model must be loaded before the runtime can be created.");
            InferenceConfig = inferenceConfig;
            this.enableCache = enableCache;
            opProvider = InferenceConfig.OPProvider;
            session = new BitNetSession(model, memoryManager);
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
            // This temporary orchestration remains inside runtime on purpose so the future Graph work can
            // lift the exact layer sequence from one place instead of re-encoding it through OP interfaces.
            for (int layerIndex = 0; layerIndex < attentionNormNodes.Length; layerIndex++)
            {
                BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex].Forward(session);
                ExecuteQKVProjection(layerIndex, layer);
                ExecuteAttention(layerIndex, layer);
                residualNodes[layerIndex].Forward(session);
                feedForwardNormNodes[layerIndex].Forward(session);
                ExecuteFeedForward(layerIndex, layer);
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

        /// <summary>
        /// Temporarily keeps QKV orchestration inside runtime so the future graph layer can reuse the same
        /// model-level sequence without depending on the legacy IOPProvider2 composition API.
        /// </summary>
        private void ExecuteQKVProjection(int layerIndex, BitNetLayerDefinition layer)
        {
            RuntimeTensor input = session.RmsNormTensor;
            RuntimeTensor query = session.QKVQueryTensor;
            RuntimeTensor key = session.QKVKeyTensor;
            RuntimeTensor value = session.QKVValueTensor;
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
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
        private void ExecuteAttention(int layerIndex, BitNetLayerDefinition layer)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            int headCount = checked((int)model.Config.AttentionHeadCount);
            int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
            int headDimension = checked((int)model.Config.AttentionHeadDimension);
            ReadOnlyMemory<float> query = session.QKVQuery;
            ReadOnlyMemory<float> key = session.QKVKey;
            ReadOnlyMemory<float> value = session.QKVValue;
            float[] subNormWeights = GetFloatTensor(cachedAttentionSubNormWeights, layerIndex, layer.AttentionSubNorm, "Attention sub-norm");
            (byte[] PackedWeights, float Scale) outputWeights = GetPackedWeights(cachedAttentionOutputWeights, layerIndex, layer.AttentionOutputWeight, "Packed attention");
            float[] outputScaleValues = GetOptionalFloatTensor(cachedAttentionOutputScales, layerIndex, layer.AttentionOutputScale, "Attention output scale");
            float[] outputBiasValues = GetOptionalFloatTensor(cachedAttentionOutputBiases, layerIndex, layer.AttentionOutputBias, "Attention output bias");
            RuntimeTensor subNorm = session.AttentionSubNormTensor;
            RuntimeTensor output = session.AttentionOutputTensor;

            ValidateAttentionProjection(query.Span, key.Span, value.Span, embeddingLength, keyValueLength);

            using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
            Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
            BuildSingleTokenAttentionContext(query, key, value, attentionContext, headCount, keyValueHeadCount, headDimension);
            RuntimeTensor attentionContextTensor = RuntimeTensor.CreateWritable("RuntimeAttentionContext", attentionContext, [embeddingLength]);
            RuntimeTensor subNormWeightsTensor = RuntimeTensor.CreateReadOnly<float>("RuntimeAttentionSubNormWeights", subNormWeights.AsMemory(0, attentionContext.Length), [attentionContext.Length]);
            opProvider.ForwardRmsNorm(attentionContextTensor, subNormWeightsTensor, model.Config.AttentionLayerNormRmsEpsilon, subNorm);
            opProvider.ProjectBitNetI2(subNorm, CreatePackedWeightTensor(outputWeights.PackedWeights, "RuntimeAttentionOutputWeights"), embeddingLength, outputWeights.Scale, output);
            Memory<float> outputMemory = RuntimeTensorBufferHelper.GetMemory<float>(output, nameof(output));
            ApplyScale(outputMemory[..embeddingLength], outputScaleValues);
            ApplyBias(outputMemory[..embeddingLength], outputBiasValues);
        }

        /// <summary>
        /// Keeps feed-forward orchestration in runtime as a transition step before a future graph-based layer
        /// scheduler replaces this handwritten sequence.
        /// </summary>
        private void ExecuteFeedForward(int layerIndex, BitNetLayerDefinition layer)
        {
            RuntimeTensor input = session.FeedForwardNormTensor;
            ReadOnlyMemory<float> inputMemory = RuntimeTensorBufferHelper.GetReadOnlyMemory<float>(input, nameof(input));
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int feedForwardLength = checked((int)model.Config.FeedForwardLength);
            float[] subNormWeights = GetFloatTensor(cachedFeedForwardSubNormWeights, layerIndex, layer.FeedForwardSubNorm, "Feed-forward sub-norm");
            (byte[] PackedWeights, float Scale) gateWeights = GetPackedWeights(cachedFeedForwardGateWeights, layerIndex, layer.FeedForwardGateWeight, "Feed-forward gate");
            (byte[] PackedWeights, float Scale) upWeights = GetPackedWeights(cachedFeedForwardUpWeights, layerIndex, layer.FeedForwardUpWeight, "Feed-forward up");
            (byte[] PackedWeights, float Scale) downWeights = GetPackedWeights(cachedFeedForwardDownWeights, layerIndex, layer.FeedForwardDownWeight, "Feed-forward down");
            RuntimeTensor subNormOutput = session.FeedForwardSubNormTensor;
            RuntimeTensor output = session.FeedForwardOutputTensor;

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

        private void BuildSingleTokenAttentionContext(ReadOnlyMemory<float> query, ReadOnlyMemory<float> key, ReadOnlyMemory<float> value, Memory<float> context, int headCount, int keyValueHeadCount, int headDimension)
        {
            if (headCount % keyValueHeadCount != 0)
            {
                throw new InvalidOperationException("Attention head count must be divisible by the key/value head count.");
            }

            int groupSize = headCount / keyValueHeadCount;
            float scoreScale = 1f / MathF.Sqrt(headDimension);
            using IMemoryOwner<float> attentionScoreOwner = MemoryPool<float>.Shared.Rent(1);
            using IMemoryOwner<float> attentionWeightOwner = MemoryPool<float>.Shared.Rent(1);
            Memory<float> attentionScore = attentionScoreOwner.Memory[..1];
            Memory<float> attentionWeight = attentionWeightOwner.Memory[..1];
            RuntimeTensor attentionScoreTensor = RuntimeTensor.CreateWritable("RuntimeAttentionScore", attentionScore, [1]);
            RuntimeTensor attentionWeightTensor = RuntimeTensor.CreateWritable("RuntimeAttentionWeight", attentionWeight, [1]);
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                int sourceHeadIndex = headIndex / groupSize;
                int queryOffset = headIndex * headDimension;
                int keyOffset = sourceHeadIndex * headDimension;
                int sourceOffset = sourceHeadIndex * headDimension;
                int outputOffset = headIndex * headDimension;

                attentionScore.Span[0] = ComputeScaledAttentionScore(query.Span, queryOffset, key.Span, keyOffset, headDimension, scoreScale);
                opProvider.ForwardSoftmax(attentionScoreTensor, attentionWeightTensor);

                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    context.Span[outputOffset + dimensionIndex] = RoundTripThroughHalf(value.Span[sourceOffset + dimensionIndex] * attentionWeight.Span[0]);
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
