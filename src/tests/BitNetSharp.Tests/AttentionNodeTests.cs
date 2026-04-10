using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class AttentionNodeTests
    {
        private static readonly Lazy<AttentionVectorsDocument> AttentionVectorsDocumentCache = new(LoadAttentionVectorsDocument);

        [TestMethod]
        public void Attention_ProvidedConfig_UsesConfiguredProvider()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var inferenceConfig = TestInferenceConfigs.Simd(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount);
            // The current test GGUF does not contain optional attn_output.scale / attn_output.bias tensors yet,
            // so these tests intentionally omit those constructor arguments until a model with them is available.
            var node = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: inferenceConfig);

            Assert.AreEqual(TestInferenceConfigs.SimdBackend, node.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void Attention_NullConfig_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);

            Assert.ThrowsExactly<ArgumentNullException>(() => new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: null));
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_SubNormMatchesBaseline_CPU(int caseIndex)
        {
            VerifyAttentionSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_SubNormMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyAttentionSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_SubNormMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyAttentionSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.SimdBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_OutputMatchesBaseline_CPU(int caseIndex)
        {
            VerifyAttentionOutputMatchesBaseline(caseIndex, TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_OutputMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyAttentionOutputMatchesBaseline(caseIndex, TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetAttentionCaseIndices))]
        public void Attention_OutputMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyAttentionOutputMatchesBaseline(caseIndex, TestInferenceConfigs.SimdBackend);
        }

        [TestMethod]
        public void Attention_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyAttentionMultiThreadMatchesSingleThread(TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        public void Attention_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyAttentionMultiThreadMatchesSingleThread(TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        public void Attention_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvx2Supported();
            VerifyAttentionMultiThreadMatchesSingleThread(TestInferenceConfigs.SimdBackend);
        }

        private static void VerifyAttentionSubNormMatchesBaseline(int caseIndex, string backend)
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(context, testCase);

            node.Init();
            node.Forward(context);
            Memory<float> actual = context.AttentionSubNorm;

            AssertFloatArraysAreClose(testCase.FirstLayerAttnSubNorm.Values.ToArray(), actual.Span.ToArray(), 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyAttentionOutputMatchesBaseline(int caseIndex, string backend)
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(context, testCase);

            node.Init();
            node.Forward(context);
            Memory<float> actual = context.AttentionOutput;

            AssertFloatArraysAreClose(testCase.FirstLayerAttnOutput.Values.ToArray(), actual.Span.ToArray(), 1e-4f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyAttentionMultiThreadMatchesSingleThread(string backend)
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(0);
            var layerDefinition = model.GetLayer(0);
            var singleThreadLayer = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var multiThreadLayer = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 2));
            var singleThreadContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(singleThreadContext, testCase);
            SetProjection(multiThreadContext, testCase);

            singleThreadLayer.Init();
            multiThreadLayer.Init();
            singleThreadLayer.Forward(singleThreadContext);
            multiThreadLayer.Forward(multiThreadContext);

            AssertFloatArraysAreClose(singleThreadContext.AttentionSubNorm.Span.ToArray(), multiThreadContext.AttentionSubNorm.Span.ToArray(), 1e-6f, $"{backend} attention sub-norm threading");
            AssertFloatArraysAreClose(singleThreadContext.AttentionOutput.Span.ToArray(), multiThreadContext.AttentionOutput.Span.ToArray(), 1e-4f, $"{backend} attention output threading");
        }

        [TestMethod]
        public void AttentionCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(0);
            var layerDefinition = model.GetLayer(0);
            var uncachedNode = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var cachedNode = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                enableCache: true,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var uncachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(uncachedContext, testCase);
            var cachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(cachedContext, testCase);

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedContext);
            cachedNode.Forward(cachedContext);

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncachedContext.AttentionSubNorm.Span.ToArray(), cachedContext.AttentionSubNorm.Span.ToArray(), 0f, "attention sub-norm cache");
            AssertFloatArraysAreClose(uncachedContext.AttentionOutput.Span.ToArray(), cachedContext.AttentionOutput.Span.ToArray(), 0f, "attention output cache");
        }

        [TestMethod]
        public void Attention_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            AttentionCase testCase = GetAttentionCase(0);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.AttentionNode(
                model,
                layerDefinition.AttentionSubNorm,
                layerDefinition.AttentionOutputWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            SetProjection(context, testCase);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(context));
        }

        public static IEnumerable<object[]> GetAttentionCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
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

        private static void SetProjection(global::BitNetSharp.BitNetSession session, AttentionCase testCase)
        {
            session.QKVQuery = testCase.FirstLayerAttnQKV.WQKV.Query.ToArray();
            session.QKVKey = testCase.FirstLayerAttnQKV.WQKV.Key.ToArray();
            session.QKVValue = testCase.FirstLayerAttnQKV.WQKV.Value.ToArray();
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.HasCount(expected.Count, actual, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        private static void EnsureAvx2Supported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
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
            [property: JsonPropertyName("qcur")] float[] Query,
            [property: JsonPropertyName("kcur")] float[] Key,
            [property: JsonPropertyName("vcur")] float[] Value);
    }
}
