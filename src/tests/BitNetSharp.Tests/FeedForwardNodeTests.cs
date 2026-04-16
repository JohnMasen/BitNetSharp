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
    public sealed class FeedForwardNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FeedForwardVectorsDocument> FeedForwardVectorsDocumentCache = new(LoadFeedForwardVectorsDocument);

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_CPU(int caseIndex)
        {
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_SubNormMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFeedForwardSubNormMatchesBaseline(caseIndex, TestInferenceConfigs.SimdBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_CPU(int caseIndex)
        {
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardCaseIndices))]
        public void FeedForward_OutputMatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFeedForwardOutputMatchesBaseline(caseIndex, TestInferenceConfigs.SimdBackend);
        }

        [TestMethod]
        public void FeedForward_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardMultiThreadMatchesSingleThread(TestInferenceConfigs.CpuBackend);
        }

        [TestMethod]
        public void FeedForward_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardMultiThreadMatchesSingleThread(TestInferenceConfigs.TensorBackend);
        }

        [TestMethod]
        public void FeedForward_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvx2Supported();
            VerifyFeedForwardMultiThreadMatchesSingleThread(TestInferenceConfigs.SimdBackend);
        }

        [TestMethod]
        public void FeedForwardCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(0);
            var layerDefinition = model.GetLayer(0);
            var uncachedNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var cachedNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                enableCache: true,
                inferenceConfig: TestInferenceConfigs.Cpu(1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(uncachedSession.FeedForwardNorm.Span);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(cachedSession.FeedForwardNorm.Span);

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedSession);
            cachedNode.Forward(cachedSession);

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.FeedForwardSubNorm.Span.ToArray(), cachedSession.FeedForwardSubNorm.Span.ToArray(), 0f, "feed-forward sub-norm cache");
            AssertFloatArraysAreClose(uncachedSession.FeedForwardOutput.Span.ToArray(), cachedSession.FeedForwardOutput.Span.ToArray(), 0f, "feed-forward output cache");
        }

        public static IEnumerable<object[]> GetFeedForwardCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static void VerifyFeedForwardSubNormMatchesBaseline(int caseIndex, string backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(session.FeedForwardNorm.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardSubNorm, session.FeedForwardSubNorm.Span.ToArray(), 1e-4f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFeedForwardOutputMatchesBaseline(int caseIndex, string backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(session.FeedForwardNorm.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardDown, session.FeedForwardOutput.Span.ToArray(), 1e-3f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFeedForwardMultiThreadMatchesSingleThread(string backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardCase testCase = GetFeedForwardCase(DebugCaseIndex);
            var layerDefinition = model.GetLayer(0);
            var singleThreadNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.FeedForwardNode(
                model,
                layerDefinition.FeedForwardSubNorm,
                layerDefinition.FeedForwardGateWeight,
                layerDefinition.FeedForwardUpWeight,
                layerDefinition.FeedForwardDownWeight,
                inferenceConfig: TestInferenceConfigs.Create(backend, 2));
            var singleThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(singleThreadSession.FeedForwardNorm.Span);
            testCase.FirstLayerFfn.FeedForwardNorm.CopyTo(multiThreadSession.FeedForwardNorm.Span);

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadSession);
            multiThreadNode.Forward(multiThreadSession);

            AssertFloatArraysAreClose(singleThreadSession.FeedForwardSubNorm.Span.ToArray(), multiThreadSession.FeedForwardSubNorm.Span.ToArray(), 1e-4f, $"{backend} feed-forward sub-norm threading");
            AssertFloatArraysAreClose(singleThreadSession.FeedForwardOutput.Span.ToArray(), multiThreadSession.FeedForwardOutput.Span.ToArray(), 1e-3f, $"{backend} feed-forward output threading");
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
