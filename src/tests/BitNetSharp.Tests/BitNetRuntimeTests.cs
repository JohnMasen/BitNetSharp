using BitNetSharp;
using BitNetSharp.Core;

using GGUFSharp;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BitNetRuntimeTests
    {
        private static readonly Lazy<RuntimeVectorsDocument> RuntimeVectorsDocumentCache = new(LoadRuntimeVectorsDocument);

        [TestMethod]
        public void Inference_MatchesManualSingleTokenChain()
        {
            using var model = TestModelFactory.LoadModel();
            using var runtimeMemoryManager = new BitNetMemoryManager();
            var inferenceConfig = TestInferenceConfigs.Cpu(1);
            using var runtime = new BitNetRuntime(model, runtimeMemoryManager, inferenceConfig);

            string actual = runtime.Inference(0);
            string expected = RunManualInference(model, 0, inferenceConfig);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        [DynamicData(nameof(GetRuntimeCaseIndices))]
        public void Inference_MatchesBaseline(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Cpu(1));
            RuntimeCase testCase = GetRuntimeCase(caseIndex);

            string actual = runtime.Inference(testCase.TokenId);
            string expected = model.Tokenizer!.Decode(new[] { testCase.LmHead.NextTokenId });

            Assert.AreEqual(expected, actual, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        [TestMethod]
        public void Inference_ThrowsForOutOfRangeToken()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Cpu(1));

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => runtime.Inference((int)model.Config!.VocabularySize));
        }

        private static string RunManualInference(global::BitNetSharp.Models.BitNetModel model, int tokenId, global::BitNetSharp.Nodes.InferenceConfig inferenceConfig)
        {
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager)
            {
                Tokens = new[] { tokenId },
                CurrentToken = tokenId,
            };

            var embeddingNode = new global::BitNetSharp.Nodes.EmbeddingNode(model, enableCache: true, inferenceConfig: TestInferenceConfigs.Cpu(1));
            IOPProvider opProvider = CreateOpProvider(inferenceConfig);
            embeddingNode.Init();
            embeddingNode.Forward(session);

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                var layer = model.Layers[layerIndex];

                var attentionNormNode = new Nodes.RmsNormNode(model, layer.AttentionNorm, enableCache: true, inferenceConfig: inferenceConfig);
                var residualNode = new Nodes.ResidualNode(model, inferenceConfig);
                var feedForwardNormNode = new Nodes.FeedForwardNormNode(model, layer.FeedForwardNorm, enableCache: true, inferenceConfig: inferenceConfig);
                var feedForwardResidualNode = new Nodes.FeedForwardResidualNode(model, inferenceConfig);

                attentionNormNode.Init();
                residualNode.Init();
                feedForwardNormNode.Init();
                feedForwardResidualNode.Init();

                attentionNormNode.Forward(session);
                ExecuteQKVProjection(model, session, opProvider, layer);
                ExecuteAttention(model, session, opProvider, layer);
                residualNode.Forward(session);
                feedForwardNormNode.Forward(session);
                ExecuteFeedForward(model, session, opProvider, layer);
                feedForwardResidualNode.Forward(session);
            }

            var finalNormNode = new Nodes.FinalNormNode(model, enableCache: true, inferenceConfig: inferenceConfig);
            var lmHeadNode = new Nodes.LmHeadNode(model, enableCache: true, inferenceConfig: inferenceConfig);
            var samplingNode = new Nodes.SamplingNode();
            finalNormNode.Init();
            lmHeadNode.Init();
            samplingNode.Init();

            finalNormNode.Forward(session);
            lmHeadNode.Forward(session);
            samplingNode.Forward(session);
            return model.Tokenizer!.Decode(new[] { session.NextTokenId });
        }

        public static IEnumerable<object[]> GetRuntimeCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static RuntimeVectorsDocument LoadRuntimeVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<RuntimeVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load runtime baseline JSON.");
        }

        private static RuntimeCase GetRuntimeCase(int caseIndex)
        {
            return RuntimeVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static IOPProvider CreateOpProvider(global::BitNetSharp.Nodes.InferenceConfig inferenceConfig)
        {
            return inferenceConfig.OPProvider;
        }

        private static void ExecuteQKVProjection(global::BitNetSharp.Models.BitNetModel model, BitNetSession session, IOPProvider opProvider, global::BitNetSharp.Models.BitNetLayerDefinition layer)
        {
            ReadOnlyMemory<float> input = session.RmsNorm;
            Memory<float> query = session.QKVQuery;
            Memory<float> key = session.QKVKey;
            Memory<float> value = session.QKVValue;
            (byte[] PackedWeights, float Scale) queryWeights = ReadPackedWeights(model, layer.AttentionQueryWeight, "QKV query");
            (byte[] PackedWeights, float Scale) keyWeights = ReadPackedWeights(model, layer.AttentionKeyWeight, "QKV key");
            (byte[] PackedWeights, float Scale) valueWeights = ReadPackedWeights(model, layer.AttentionValueWeight, "QKV value");
            int queryOutputLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueOutputLength = checked((int)model.Config.KeyValueProjectionSize);

            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = opProvider.QuantizeBitNetActivations(input, quantizedValues);

            opProvider.ProjectBitNetI2(quantizedValues, activationScale, queryWeights.PackedWeights, queryOutputLength, queryWeights.Scale, query);
            opProvider.ProjectBitNetI2(quantizedValues, activationScale, keyWeights.PackedWeights, keyValueOutputLength, keyWeights.Scale, key);
            opProvider.ProjectBitNetI2(quantizedValues, activationScale, valueWeights.PackedWeights, keyValueOutputLength, valueWeights.Scale, value);
        }

        private static void ExecuteAttention(global::BitNetSharp.Models.BitNetModel model, BitNetSession session, IOPProvider opProvider, global::BitNetSharp.Models.BitNetLayerDefinition layer)
        {
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
            int headCount = checked((int)model.Config.AttentionHeadCount);
            int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
            int headDimension = checked((int)model.Config.AttentionHeadDimension);
            float[] subNormWeights = ReadFloatTensor(model, layer.AttentionSubNorm, "Attention sub-norm");
            (byte[] PackedWeights, float Scale) outputWeights = ReadPackedWeights(model, layer.AttentionOutputWeight, "Packed attention");
            float[] outputScaleValues = layer.AttentionOutputScale is null ? [] : ReadFloatTensor(model, layer.AttentionOutputScale, "Attention output scale");
            float[] outputBiasValues = layer.AttentionOutputBias is null ? [] : ReadFloatTensor(model, layer.AttentionOutputBias, "Attention output bias");
            ReadOnlyMemory<float> query = session.QKVQuery;
            ReadOnlyMemory<float> key = session.QKVKey;
            ReadOnlyMemory<float> value = session.QKVValue;
            Memory<float> subNorm = session.AttentionSubNorm;
            Memory<float> output = session.AttentionOutput;

            ValidateAttentionProjection(query.Span, key.Span, value.Span, embeddingLength, keyValueLength);

            using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
            Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
            BuildSingleTokenAttentionContext(opProvider, query, key, value, attentionContext, headCount, keyValueHeadCount, headDimension);

            opProvider.ForwardRmsNorm(attentionContext, subNormWeights.AsMemory(0, attentionContext.Length), model.Config.AttentionLayerNormRmsEpsilon, subNorm);
            opProvider.ProjectBitNetI2(subNorm, outputWeights.PackedWeights, embeddingLength, outputWeights.Scale, output);
            ApplyScale(output[..embeddingLength], outputScaleValues);
            ApplyBias(output[..embeddingLength], outputBiasValues);
        }

        private static void ExecuteFeedForward(global::BitNetSharp.Models.BitNetModel model, BitNetSession session, IOPProvider opProvider, global::BitNetSharp.Models.BitNetLayerDefinition layer)
        {
            ReadOnlyMemory<float> input = session.FeedForwardNorm;
            int embeddingLength = checked((int)model.Config!.EmbeddingLength);
            int feedForwardLength = checked((int)model.Config.FeedForwardLength);
            float[] subNormWeights = ReadFloatTensor(model, layer.FeedForwardSubNorm, "Feed-forward sub-norm");
            (byte[] PackedWeights, float Scale) gateWeights = ReadPackedWeights(model, layer.FeedForwardGateWeight, "Feed-forward gate");
            (byte[] PackedWeights, float Scale) upWeights = ReadPackedWeights(model, layer.FeedForwardUpWeight, "Feed-forward up");
            (byte[] PackedWeights, float Scale) downWeights = ReadPackedWeights(model, layer.FeedForwardDownWeight, "Feed-forward down");
            Memory<float> subNormOutput = session.FeedForwardSubNorm;
            Memory<float> output = session.FeedForwardOutput;

            using IMemoryOwner<float> upOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<float> gateOwner = MemoryPool<float>.Shared.Rent(feedForwardLength);
            using IMemoryOwner<sbyte> quantizedValuesOwner = MemoryPool<sbyte>.Shared.Rent(input.Length);
            Memory<float> up = upOwner.Memory[..feedForwardLength];
            Memory<float> gate = gateOwner.Memory[..feedForwardLength];
            Memory<sbyte> quantizedValues = quantizedValuesOwner.Memory[..input.Length];
            (float activationScale, _) = opProvider.QuantizeBitNetActivations(input, quantizedValues);

            opProvider.ProjectBitNetI2(quantizedValues, activationScale, upWeights.PackedWeights, feedForwardLength, upWeights.Scale, up);
            opProvider.ProjectBitNetI2(quantizedValues, activationScale, gateWeights.PackedWeights, feedForwardLength, gateWeights.Scale, gate);
            ApplySquaredReluGate(gate.Span, up.Span);
            opProvider.ForwardRmsNorm(up, subNormWeights.AsMemory(0, feedForwardLength), model.Config.AttentionLayerNormRmsEpsilon, subNormOutput);
            opProvider.ProjectBitNetI2(subNormOutput, downWeights.PackedWeights, embeddingLength, downWeights.Scale, output);
        }

        private static (byte[] PackedWeights, float Scale) ReadPackedWeights(global::BitNetSharp.Models.BitNetModel model, global::BitNetSharp.Models.BitNetTensorInfo tensor, string tensorLabel)
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

        private static float[] ReadFloatTensor(global::BitNetSharp.Models.BitNetModel model, global::BitNetSharp.Models.BitNetTensorInfo tensor, string tensorLabel)
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

        private static void BuildSingleTokenAttentionContext(IOPProvider opProvider, ReadOnlyMemory<float> query, ReadOnlyMemory<float> key, ReadOnlyMemory<float> value, Memory<float> context, int headCount, int keyValueHeadCount, int headDimension)
        {
            if (headCount % keyValueHeadCount != 0)
            {
                throw new InvalidOperationException("Attention head count must be divisible by the key/value head count.");
            }

            int groupSize = headCount / keyValueHeadCount;
            float scoreScale = 1f / MathF.Sqrt(headDimension);
            Span<float> attentionScore = stackalloc float[1];
            Span<float> attentionWeight = stackalloc float[1];
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                int sourceHeadIndex = headIndex / groupSize;
                int queryOffset = headIndex * headDimension;
                int keyOffset = sourceHeadIndex * headDimension;
                int sourceOffset = sourceHeadIndex * headDimension;
                int outputOffset = headIndex * headDimension;

                attentionScore[0] = ComputeScaledAttentionScore(query.Span, queryOffset, key.Span, keyOffset, headDimension, scoreScale);
                opProvider.ForwardSoftmax(attentionScore, attentionWeight);

                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    context.Span[outputOffset + dimensionIndex] = RoundTripThroughHalf(value.Span[sourceOffset + dimensionIndex] * attentionWeight[0]);
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

        private static void ApplySquaredReluGate(ReadOnlySpan<float> gate, Span<float> up)
        {
            for (int index = 0; index < up.Length; index++)
            {
                float relu = MathF.Max(gate[index], 0f);
                up[index] *= relu * relu;
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

        internal sealed record RuntimeVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<RuntimeCase> TestCases);

        internal sealed record RuntimeCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("lm_head")] RuntimeLmHeadOutputs LmHead);

        internal sealed record RuntimeLmHeadOutputs(
            [property: JsonPropertyName("next_token_id")] int NextTokenId);
    }
}
