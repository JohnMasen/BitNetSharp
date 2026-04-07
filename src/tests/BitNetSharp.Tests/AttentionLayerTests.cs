using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class AttentionLayerTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<AttentionVectorsDocument> AttentionVectorsDocumentCache = new(LoadAttentionVectorsDocument);

        [TestMethod]
        public void Attention_DefaultConfig_CpuAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            // The current test GGUF does not contain optional attn_output.scale / attn_output.bias tensors yet,
            // so these tests intentionally omit those constructor arguments until a model with them is available.
            var layer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.CPU, layer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void Attention_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var firstLayer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.CPU, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        public void Attention_SubNormMatchesBaseline_CPU_DebugCase()
        {
            VerifyAttentionSubNormMatchesBaselineCpu(DebugCaseIndex);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_SubNormMatchesBaseline_CPU(int caseIndex)
        {
            VerifyAttentionSubNormMatchesBaselineCpu(caseIndex);
        }

        [TestMethod]
        public void Attention_OutputMatchesBaseline_CPU_DebugCase()
        {
            VerifyAttentionOutputMatchesBaselineCpu(DebugCaseIndex);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_OutputMatchesBaseline_CPU(int caseIndex)
        {
            VerifyAttentionOutputMatchesBaselineCpu(caseIndex);
        }

        private static void VerifyAttentionSubNormMatchesBaselineCpu(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            context.QKVProjection = CreateProjection(testCase);

            layer.Init();
            layer.Forward(context);
            float[] actual = context.AttentionSubNorm!;

            AssertFloatArraysAreClose(testCase.FirstLayerAttnSubNorm.Values.ToArray(), actual, 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyAttentionOutputMatchesBaselineCpu(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            context.QKVProjection = CreateProjection(testCase);

            layer.Init();
            layer.Forward(context);
            float[] actual = context.AttentionOutput!;

            AssertFloatArraysAreClose(testCase.FirstLayerAttnOutput.Values.ToArray(), actual, 1e-4f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        [TestMethod]
        public void AttentionCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(0);
            var layerDefinition = model.GetLayer(0);
            var uncachedLayer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var uncachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            uncachedContext.QKVProjection = CreateProjection(testCase);
            var cachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            cachedContext.QKVProjection = CreateProjection(testCase);

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedContext);
            cachedLayer.Forward(cachedContext);

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncachedContext.AttentionSubNorm!, cachedContext.AttentionSubNorm!, 0f, "attention sub-norm cache");
            AssertFloatArraysAreClose(uncachedContext.AttentionOutput!, cachedContext.AttentionOutput!, 0f, "attention output cache");
        }

        [TestMethod]
        public void Attention_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(0);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.AttentionLayer(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            context.QKVProjection = CreateProjection(testCase);

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(context));
        }

        public static IEnumerable<object[]> GetAttentionCaseIndices()
        {
            return AttentionVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static AttentionVectorsDocument LoadAttentionVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<AttentionVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load attention baseline JSON.");
        }

        private static AttentionCase GetAttentionCase(int caseIndex)
        {
            return AttentionVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static BitNetSharp.Layers.QKVProjectionOutput CreateProjection(AttentionCase testCase)
        {
            return new BitNetSharp.Layers.QKVProjectionOutput(
                testCase.FirstLayerAttnQKV.WQKV.Query.ToArray(),
                testCase.FirstLayerAttnQKV.WQKV.Key.ToArray(),
                testCase.FirstLayerAttnQKV.WQKV.Value.ToArray());
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.HasCount(expected.Count, actual, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        internal sealed record AttentionVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<AttentionCase> TestCases);

        internal sealed record AttentionCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_attn_qkv")] QKVOutputs FirstLayerAttnQKV,
            [property: JsonPropertyName("first_layer_attn_sub_norm")] LayerVectorValues FirstLayerAttnSubNorm,
            [property: JsonPropertyName("first_layer_attn_output")] LayerVectorValues FirstLayerAttnOutput);

        internal sealed record LayerVectorValues(
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);

        internal sealed record QKVOutputs(
            [property: JsonPropertyName("wqkv")] PackedQKVValues WQKV);

        internal sealed record PackedQKVValues(
            [property: JsonPropertyName("qcur")] IReadOnlyList<float> Query,
            [property: JsonPropertyName("kcur")] IReadOnlyList<float> Key,
            [property: JsonPropertyName("vcur")] IReadOnlyList<float> Value);
    }
}
