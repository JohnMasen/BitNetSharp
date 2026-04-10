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
    public sealed class FeedForwardNormNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FeedForwardNormVectorsDocument> FeedForwardNormVectorsDocumentCache = new(LoadFeedForwardNormVectorsDocument);

        [TestMethod]
        public void FeedForwardNorm_DefaultConfig_SimdSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.FeedForwardNormNode(model, layerDefinition.FeedForwardNorm);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, node.InferenceConfig.Backend);
            Assert.AreEqual(1, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void FeedForwardNorm_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var firstNode = new BitNetSharp.Nodes.FeedForwardNormNode(model, layerDefinition.FeedForwardNorm, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.FeedForwardNormNode(model, layerDefinition.FeedForwardNorm, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        public void FeedForwardNorm_BaselineMatch_CPU_DebugCase()
        {
            VerifyFeedForwardNormMatchesBaseline(DebugCaseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardNormCaseIndices))]
        public void FeedForwardNorm_BaselineMatch_CPU(int caseIndex)
        {
            VerifyFeedForwardNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardNormCaseIndices))]
        public void FeedForwardNorm_BaselineMatch_Tensor(int caseIndex)
        {
            VerifyFeedForwardNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardNormCaseIndices))]
        public void FeedForwardNorm_BaselineMatch_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFeedForwardNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForwardNorm_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        public void FeedForwardNorm_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        public void FeedForwardNorm_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvx2Supported();
            VerifyFeedForwardNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForwardNorm_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardNormCase testCase = GetFeedForwardNormCase(0);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.FeedForwardNormNode(
                model,
                layerDefinition.FeedForwardNorm,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(session.FeedForwardInput.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(session));
        }

        public static IEnumerable<object[]> GetFeedForwardNormCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static void VerifyFeedForwardNormMatchesBaseline(int caseIndex, BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardNormCase testCase = GetFeedForwardNormCase(caseIndex);
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.FeedForwardNormNode(
                model,
                layerDefinition.FeedForwardNorm,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(session.FeedForwardInput.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardNorm, session.FeedForwardNorm.Span.ToArray(), 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFeedForwardNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardNormCase testCase = GetFeedForwardNormCase(DebugCaseIndex);
            var layerDefinition = model.GetLayer(0);
            var singleThreadNode = new BitNetSharp.Nodes.FeedForwardNormNode(
                model,
                layerDefinition.FeedForwardNorm,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.FeedForwardNormNode(
                model,
                layerDefinition.FeedForwardNorm,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 2));
            var singleThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(singleThreadSession.FeedForwardInput.Span);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(multiThreadSession.FeedForwardInput.Span);

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadSession);
            multiThreadNode.Forward(multiThreadSession);

            AssertFloatArraysAreClose(singleThreadSession.FeedForwardNorm.Span.ToArray(), multiThreadSession.FeedForwardNorm.Span.ToArray(), 1e-6f, $"{backend} feed-forward norm threading");
        }

        private static FeedForwardNormVectorsDocument LoadFeedForwardNormVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardNormVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward norm baseline JSON.");
        }

        private static FeedForwardNormCase GetFeedForwardNormCase(int caseIndex)
        {
            return FeedForwardNormVectorsDocumentCache.Value.TestCases[caseIndex];
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

        internal sealed record FeedForwardNormVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardNormCase> TestCases);

        internal sealed record FeedForwardNormCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardOutputs FirstLayerFfn);

        internal sealed record FeedForwardOutputs(
            [property: JsonPropertyName("ffn_inp")] float[] FeedForwardInput,
            [property: JsonPropertyName("ffn_norm")] float[] FeedForwardNorm);
    }
}
