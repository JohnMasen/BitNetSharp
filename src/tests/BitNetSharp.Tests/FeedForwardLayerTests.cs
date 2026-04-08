using System;
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
    public sealed class FeedForwardLayerTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FeedForwardVectorsDocument> FeedForwardVectorsDocumentCache = new(LoadFeedForwardVectorsDocument);

        [TestMethod]
        public void FeedForward_DefaultConfig_SimdAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, layer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void FeedForward_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var firstLayer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        public void FeedForward_SubNormMatchesBaseline_CPU_DebugCase()
        {
            VerifyFeedForwardSubNormMatchesBaseline(DebugCaseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_CPU(int caseIndex)
        {
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForward_OutputMatchesBaseline_CPU_DebugCase()
        {
            VerifyFeedForwardOutputMatchesBaseline(DebugCaseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_CPU(int caseIndex)
        {
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForwardCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(0);
            var layerDefinition = model.GetLayer(0);
            var uncachedLayer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(uncachedSession.FeedForwardNorm.Span);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(cachedSession.FeedForwardNorm.Span);

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedSession);
            cachedLayer.Forward(cachedSession);

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.FeedForwardSubNorm.Span.ToArray(), cachedSession.FeedForwardSubNorm.Span.ToArray(), 0f, "feed-forward sub-norm cache");
            AssertFloatArraysAreClose(uncachedSession.FeedForwardOutput.Span.ToArray(), cachedSession.FeedForwardOutput.Span.ToArray(), 0f, "feed-forward output cache");
        }

        [TestMethod]
        public void FeedForward_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(0);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(session.FeedForwardNorm.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(session));
        }

        public static IEnumerable<object[]> GetFeedForwardCaseIndices()
        {
            return FeedForwardVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static void VerifyFeedForwardSubNormMatchesBaseline(int caseIndex, BitNetSharp.Layers.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(session.FeedForwardNorm.Span);

            layer.Init();
            layer.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardSubNorm, session.FeedForwardSubNorm.Span.ToArray(), 1e-4f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFeedForwardOutputMatchesBaseline(int caseIndex, BitNetSharp.Layers.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.FeedForwardLayer(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(session.FeedForwardNorm.Span);

            layer.Init();
            layer.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardDown, session.FeedForwardOutput.Span.ToArray(), 1e-3f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static FeedForwardVectorsDocument LoadFeedForwardVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward baseline JSON.");
        }

        private static FeedForwardCase GetFeedForwardCase(int caseIndex)
        {
            return FeedForwardVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static void EnsureAvx2Supported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
            }
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.HasCount(expected.Count, actual, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        internal sealed record FeedForwardVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardCase> TestCases);

        internal sealed record FeedForwardCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardOutputs FirstLayerFfn);

        internal sealed record FeedForwardOutputs(
            [property: JsonPropertyName("ffn_norm")] float[] FeedForwardNorm,
            [property: JsonPropertyName("ffn_sub_norm")] float[] FeedForwardSubNorm,
            [property: JsonPropertyName("ffn_down")] float[] FeedForwardDown);
    }
}
