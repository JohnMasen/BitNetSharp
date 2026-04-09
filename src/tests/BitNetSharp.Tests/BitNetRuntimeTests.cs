using BitNetSharp;

using System.Collections.Generic;
using System.IO;
using System.Linq;
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
            var inferenceConfig = new Nodes.InferenceConfig(Nodes.InferenceBackend.CPU, 1);
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
            using var runtime = new BitNetRuntime(model, memoryManager, new Nodes.InferenceConfig(Nodes.InferenceBackend.CPU, 1));
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
            using var runtime = new BitNetRuntime(model, memoryManager, new Nodes.InferenceConfig(Nodes.InferenceBackend.CPU, 1));

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

            var embeddingNode = new global::BitNetSharp.Nodes.EmbeddingNode(model, enableCache: true, inferenceConfig: new global::BitNetSharp.Nodes.InferenceConfig(global::BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            embeddingNode.Init();
            embeddingNode.Forward(session);

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                var layer = model.Layers[layerIndex];

                var attentionNormNode = new Nodes.RmsNormNode(model, layer.AttentionNorm, enableCache: true, inferenceConfig: inferenceConfig);
                var qkvProjectionNode = new Nodes.QKVProjectionNode(model, layer.AttentionQueryWeight, layer.AttentionKeyWeight, layer.AttentionValueWeight, enableCache: true, inferenceConfig: inferenceConfig);
                var attentionNode = new Nodes.AttentionNode(model, layer.AttentionSubNorm, layer.AttentionOutputWeight, layer.AttentionOutputScale, layer.AttentionOutputBias, enableCache: true, inferenceConfig: inferenceConfig);
                var residualNode = new Nodes.ResidualNode(model, inferenceConfig);
                var feedForwardNormNode = new Nodes.FeedForwardNormNode(model, layer.FeedForwardNorm, enableCache: true, inferenceConfig: inferenceConfig);
                var feedForwardNode = new Nodes.FeedForwardNode(model, layer.FeedForwardSubNorm, layer.FeedForwardGateWeight, layer.FeedForwardUpWeight, layer.FeedForwardDownWeight, enableCache: true, inferenceConfig: inferenceConfig);
                var feedForwardResidualNode = new Nodes.FeedForwardResidualNode(model, inferenceConfig);

                attentionNormNode.Init();
                qkvProjectionNode.Init();
                attentionNode.Init();
                residualNode.Init();
                feedForwardNormNode.Init();
                feedForwardNode.Init();
                feedForwardResidualNode.Init();

                attentionNormNode.Forward(session);
                qkvProjectionNode.Forward(session);
                attentionNode.Forward(session);
                residualNode.Forward(session);
                feedForwardNormNode.Forward(session);
                feedForwardNode.Forward(session);
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
            return RuntimeVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
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
