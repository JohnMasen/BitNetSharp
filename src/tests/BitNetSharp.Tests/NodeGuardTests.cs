using System.IO;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class NodeGuardTests
    {
        private static readonly Lazy<AttentionVectorsDocument> AttentionVectorsDocumentCache = new(LoadAttentionVectorsDocument);
        private static readonly Lazy<FeedForwardNormVectorsDocument> FeedForwardNormVectorsDocumentCache = new(LoadFeedForwardNormVectorsDocument);
        private static readonly Lazy<FeedForwardVectorsDocument> FeedForwardVectorsDocumentCache = new(LoadFeedForwardVectorsDocument);
        private static readonly Lazy<FinalNormVectorsDocument> FinalNormVectorsDocumentCache = new(LoadFinalNormVectorsDocument);
        private static readonly Lazy<ResidualVectorsDocument> ResidualVectorsDocumentCache = new(LoadResidualVectorsDocument);
        private static readonly Lazy<FeedForwardResidualVectorsDocument> FeedForwardResidualVectorsDocumentCache = new(LoadFeedForwardResidualVectorsDocument);
        private static readonly Lazy<LmHeadVectorsDocument> LmHeadVectorsDocumentCache = new(LoadLmHeadVectorsDocument);

        [TestMethod]
        public void ProvidedConfig_UsesConfiguredProvider()
        {
            using var model = TestModelFactory.LoadModel();

            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model, inferenceConfig: TestInferenceConfigs.Simd(1));
            var rmsNormNode = new BitNetSharp.Nodes.RmsNormNode(model, model.GetLayer(0).AttentionNorm, inferenceConfig: TestInferenceConfigs.Simd(1));
            var qkvNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                model.GetLayer(0).AttentionQueryWeight,
                model.GetLayer(0).AttentionKeyWeight,
                model.GetLayer(0).AttentionValueWeight,
                inferenceConfig: TestInferenceConfigs.Simd(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount));
            var attentionNode = new BitNetSharp.Nodes.AttentionNode(
                model,
                model.GetLayer(0).AttentionSubNorm,
                model.GetLayer(0).AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Simd(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount));
            var feedForwardNormNode = new BitNetSharp.Nodes.FeedForwardNormNode(model, model.GetLayer(0).FeedForwardNorm, inferenceConfig: TestInferenceConfigs.Simd(1));
            var feedForwardNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                model.GetLayer(0).FeedForwardSubNorm,
                model.GetLayer(0).FeedForwardGateWeight,
                model.GetLayer(0).FeedForwardUpWeight,
                model.GetLayer(0).FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Simd(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount));
            var finalNormNode = new BitNetSharp.Nodes.FinalNormNode(model, inferenceConfig: TestInferenceConfigs.Simd(1));
            var lmHeadNode = new BitNetSharp.Nodes.LmHeadNode(model, inferenceConfig: TestInferenceConfigs.Simd(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount));
            var residualNode = new BitNetSharp.Nodes.ResidualNode(model, TestInferenceConfigs.Cpu(1));
            var feedForwardResidualNode = new BitNetSharp.Nodes.FeedForwardResidualNode(model, TestInferenceConfigs.Cpu(1));

            Assert.AreEqual(TestInferenceConfigs.SimdBackend, embeddingNode.InferenceConfig.Backend);
            Assert.AreEqual(1, embeddingNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, rmsNormNode.InferenceConfig.Backend);
            Assert.AreEqual(1, rmsNormNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, qkvNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, qkvNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, attentionNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, attentionNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, feedForwardNormNode.InferenceConfig.Backend);
            Assert.AreEqual(1, feedForwardNormNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, feedForwardNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, feedForwardNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, finalNormNode.InferenceConfig.Backend);
            Assert.AreEqual(1, finalNormNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.SimdBackend, lmHeadNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, lmHeadNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.CpuBackend, residualNode.InferenceConfig.Backend);
            Assert.AreEqual(1, residualNode.InferenceConfig.ThreadCount);
            Assert.AreEqual(TestInferenceConfigs.CpuBackend, feedForwardResidualNode.InferenceConfig.Backend);
            Assert.AreEqual(1, feedForwardResidualNode.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void NullConfig_Throws()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.EmbeddingNode(model, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.RmsNormNode(model, model.GetLayer(0).AttentionNorm, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                model.GetLayer(0).AttentionQueryWeight,
                model.GetLayer(0).AttentionKeyWeight,
                model.GetLayer(0).AttentionValueWeight,
                inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.AttentionNode(
                model,
                model.GetLayer(0).AttentionSubNorm,
                model.GetLayer(0).AttentionOutputWeight,
                inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.FeedForwardNormNode(model, model.GetLayer(0).FeedForwardNorm, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.FeedForwardNode(
                model,
                model.GetLayer(0).FeedForwardSubNorm,
                model.GetLayer(0).FeedForwardGateWeight,
                model.GetLayer(0).FeedForwardUpWeight,
                model.GetLayer(0).FeedForwardDownWeight,
                inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.FinalNormNode(model, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.LmHeadNode(model, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.ResidualNode(model, inferenceConfig: null));
            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.FeedForwardResidualNode(model, inferenceConfig: null));
        }

        [TestMethod]
        public void ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();

            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var embeddingSession = TestModelFactory.CreateSession(model, token: 0);
            Assert.ThrowsExactly<InvalidOperationException>(() => embeddingNode.Forward(embeddingSession));

            var rmsNormNode = new BitNetSharp.Nodes.RmsNormNode(model, model.GetLayer(0).AttentionNorm, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var rmsNormSession = TestModelFactory.CreateSession(model, token: 0);
            rmsNormSession.Embedding = new float[(int)model.Config!.EmbeddingLength];
            Assert.ThrowsExactly<InvalidOperationException>(() => rmsNormNode.Forward(rmsNormSession));

            var qkvNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                model.GetLayer(0).AttentionQueryWeight,
                model.GetLayer(0).AttentionKeyWeight,
                model.GetLayer(0).AttentionValueWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var qkvSession = TestModelFactory.CreateSession(model, token: 0);
            qkvSession.RmsNorm = new float[(int)model.Config.EmbeddingLength];
            Assert.ThrowsExactly<InvalidOperationException>(() => qkvNode.Forward(qkvSession));

            AttentionCase attentionCase = AttentionVectorsDocumentCache.Value.TestCases[0];
            var attentionNode = new BitNetSharp.Nodes.AttentionNode(
                model,
                model.GetLayer(0).AttentionSubNorm,
                model.GetLayer(0).AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var attentionSession = TestModelFactory.CreateSession(model, token: attentionCase.TokenId);
            attentionSession.QKVQuery = attentionCase.FirstLayerAttnQKV.WQKV.Query.ToArray();
            attentionSession.QKVKey = attentionCase.FirstLayerAttnQKV.WQKV.Key.ToArray();
            attentionSession.QKVValue = attentionCase.FirstLayerAttnQKV.WQKV.Value.ToArray();
            Assert.ThrowsExactly<InvalidOperationException>(() => attentionNode.Forward(attentionSession));

            FeedForwardNormCase feedForwardNormCase = FeedForwardNormVectorsDocumentCache.Value.TestCases[0];
            var feedForwardNormNode = new BitNetSharp.Nodes.FeedForwardNormNode(model, model.GetLayer(0).FeedForwardNorm, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var feedForwardNormSession = TestModelFactory.CreateSession(model, token: feedForwardNormCase.TokenId);
            feedForwardNormCase.FirstLayerFfn.FeedForwardInput.CopyTo(feedForwardNormSession.FeedForwardInput.Span);
            Assert.ThrowsExactly<InvalidOperationException>(() => feedForwardNormNode.Forward(feedForwardNormSession));

            FeedForwardCase feedForwardCase = FeedForwardVectorsDocumentCache.Value.TestCases[0];
            var feedForwardNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                model.GetLayer(0).FeedForwardSubNorm,
                model.GetLayer(0).FeedForwardGateWeight,
                model.GetLayer(0).FeedForwardUpWeight,
                model.GetLayer(0).FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var feedForwardSession = TestModelFactory.CreateSession(model, token: feedForwardCase.TokenId);
            feedForwardCase.FirstLayerFfn.FeedForwardNorm.CopyTo(feedForwardSession.FeedForwardNorm.Span);
            Assert.ThrowsExactly<InvalidOperationException>(() => feedForwardNode.Forward(feedForwardSession));

            FinalNormCase finalNormCase = FinalNormVectorsDocumentCache.Value.TestCases[0];
            var finalNormNode = new BitNetSharp.Nodes.FinalNormNode(model, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var finalNormSession = TestModelFactory.CreateSession(model, token: finalNormCase.TokenId);
            finalNormCase.FinalNormInput.Values.CopyTo(finalNormSession.Embedding.Span);
            Assert.ThrowsExactly<InvalidOperationException>(() => finalNormNode.Forward(finalNormSession));

            LmHeadCase lmHeadCase = LmHeadVectorsDocumentCache.Value.TestCases[0];
            var lmHeadNode = new BitNetSharp.Nodes.LmHeadNode(model, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var lmHeadSession = TestModelFactory.CreateSession(model, token: lmHeadCase.TokenId);
            lmHeadCase.FinalNormOutput.Values.CopyTo(lmHeadSession.FinalNormOutput.Span);
            Assert.ThrowsExactly<InvalidOperationException>(() => lmHeadNode.Forward(lmHeadSession));

            ResidualCase residualCase = ResidualVectorsDocumentCache.Value.TestCases[0];
            var residualNode = new BitNetSharp.Nodes.ResidualNode(model, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var residualSession = TestModelFactory.CreateSession(model, token: residualCase.TokenId);
            residualSession.Embedding = residualCase.Dequantized.Values.ToArray();
            residualSession.AttentionOutput = residualCase.FirstLayerAttnOutput.Values.ToArray();
            Assert.ThrowsExactly<InvalidOperationException>(() => residualNode.Forward(residualSession));

            FeedForwardResidualCase feedForwardResidualCase = FeedForwardResidualVectorsDocumentCache.Value.TestCases[0];
            var feedForwardResidualNode = new BitNetSharp.Nodes.FeedForwardResidualNode(model, inferenceConfig: TestInferenceConfigs.Cpu(1));
            var feedForwardResidualSession = TestModelFactory.CreateSession(model, token: feedForwardResidualCase.TokenId);
            feedForwardResidualCase.FirstLayerFfn.FeedForwardInput.CopyTo(feedForwardResidualSession.FeedForwardInput.Span);
            feedForwardResidualCase.FirstLayerFfn.FeedForwardDown.CopyTo(feedForwardResidualSession.FeedForwardOutput.Span);
            Assert.ThrowsExactly<InvalidOperationException>(() => feedForwardResidualNode.Forward(feedForwardResidualSession));
        }

        private static AttentionVectorsDocument LoadAttentionVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<AttentionVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load attention baseline JSON.");
        }

        private static FeedForwardNormVectorsDocument LoadFeedForwardNormVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardNormVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward norm baseline JSON.");
        }

        private static FeedForwardVectorsDocument LoadFeedForwardVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward baseline JSON.");
        }

        private static FinalNormVectorsDocument LoadFinalNormVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FinalNormVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load final norm baseline JSON.");
        }

        private static ResidualVectorsDocument LoadResidualVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<ResidualVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load residual baseline JSON.");
        }

        private static FeedForwardResidualVectorsDocument LoadFeedForwardResidualVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardResidualVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward residual baseline JSON.");
        }

        private static LmHeadVectorsDocument LoadLmHeadVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<LmHeadVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load lm head baseline JSON.");
        }

        private sealed record AttentionVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<AttentionCase> TestCases);

        private sealed record AttentionCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_attn_qkv")] QKVOutputs FirstLayerAttnQKV);

        private sealed record QKVOutputs(
            [property: JsonPropertyName("wqkv")] PackedQKVValues WQKV);

        private sealed record PackedQKVValues(
            [property: JsonPropertyName("qcur")] float[] Query,
            [property: JsonPropertyName("kcur")] float[] Key,
            [property: JsonPropertyName("vcur")] float[] Value);

        private sealed record FeedForwardNormVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardNormCase> TestCases);

        private sealed record FeedForwardNormCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardNormOutputs FirstLayerFfn);

        private sealed record FeedForwardNormOutputs(
            [property: JsonPropertyName("ffn_inp")] float[] FeedForwardInput);

        private sealed record FeedForwardVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardCase> TestCases);

        private sealed record FeedForwardCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardOutputs FirstLayerFfn);

        private sealed record FeedForwardOutputs(
            [property: JsonPropertyName("ffn_norm")] float[] FeedForwardNorm);

        private sealed record FinalNormVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FinalNormCase> TestCases);

        private sealed record FinalNormCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("final_norm_input")] FinalNormVector FinalNormInput);

        private sealed record FinalNormVector(
            [property: JsonPropertyName("values")] float[] Values);

        private sealed record ResidualVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<ResidualCase> TestCases);

        private sealed record ResidualCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("dequantized")] LayerVectorValues Dequantized,
            [property: JsonPropertyName("first_layer_attn_output")] LayerVectorValues FirstLayerAttnOutput);

        private sealed record LayerVectorValues(
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);

        private sealed record FeedForwardResidualVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardResidualCase> TestCases);

        private sealed record FeedForwardResidualCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardResidualOutputs FirstLayerFfn);

        private sealed record FeedForwardResidualOutputs(
            [property: JsonPropertyName("ffn_inp")] float[] FeedForwardInput,
            [property: JsonPropertyName("ffn_down")] float[] FeedForwardDown);

        private sealed record LmHeadVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<LmHeadCase> TestCases);

        private sealed record LmHeadCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("final_norm_output")] LmHeadVector FinalNormOutput);

        private sealed record LmHeadVector(
            [property: JsonPropertyName("values")] float[] Values);
    }
}
