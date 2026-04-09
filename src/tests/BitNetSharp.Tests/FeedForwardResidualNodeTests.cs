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
    public sealed class FeedForwardResidualNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FeedForwardResidualVectorsDocument> FeedForwardResidualVectorsDocumentCache = new(LoadFeedForwardResidualVectorsDocument);

        [TestMethod]
        public void FeedForwardResidual_DefaultConfig_CpuSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.FeedForwardResidualNode(model);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.CPU, node.InferenceConfig.Backend);
            Assert.AreEqual(1, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void FeedForwardResidual_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstNode = new BitNetSharp.Nodes.FeedForwardResidualNode(model, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.FeedForwardResidualNode(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.CPU, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        public void FeedForwardResidual_MatchesBaseline_CPU_DebugCase()
        {
            VerifyFeedForwardResidualMatchesBaseline(DebugCaseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardResidualCaseIndices))]
        public void FeedForwardResidual_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyFeedForwardResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardResidualCaseIndices))]
        public void FeedForwardResidual_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFeedForwardResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFeedForwardResidualCaseIndices))]
        public void FeedForwardResidual_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureSimdSupported();
            VerifyFeedForwardResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForwardResidual_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardResidualMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        public void FeedForwardResidual_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyFeedForwardResidualMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        public void FeedForwardResidual_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureSimdSupported();
            VerifyFeedForwardResidualMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FeedForwardResidual_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardResidualCase testCase = GetFeedForwardResidualCase(0);
            var node = new BitNetSharp.Nodes.FeedForwardResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(session.FeedForwardInput.Span);
            testCase.FirstLayerFfn.FeedForwardDown.CopyTo(session.FeedForwardOutput.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(session));
        }

        public static IEnumerable<object[]> GetFeedForwardResidualCaseIndices()
        {
            return FeedForwardResidualVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static void VerifyFeedForwardResidualMatchesBaseline(int caseIndex, BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardResidualCase testCase = GetFeedForwardResidualCase(caseIndex);
            var node = new BitNetSharp.Nodes.FeedForwardResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(session.FeedForwardInput.Span);
            testCase.FirstLayerFfn.FeedForwardDown.CopyTo(session.FeedForwardOutput.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.LayerOutput, session.Embedding.Span.ToArray(), 1e-4f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFeedForwardResidualMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FeedForwardResidualCase testCase = GetFeedForwardResidualCase(DebugCaseIndex);
            var singleThreadNode = new BitNetSharp.Nodes.FeedForwardResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.FeedForwardResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 2));
            var singleThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(singleThreadSession.FeedForwardInput.Span);
            testCase.FirstLayerFfn.FeedForwardInput.CopyTo(multiThreadSession.FeedForwardInput.Span);
            testCase.FirstLayerFfn.FeedForwardDown.CopyTo(singleThreadSession.FeedForwardOutput.Span);
            testCase.FirstLayerFfn.FeedForwardDown.CopyTo(multiThreadSession.FeedForwardOutput.Span);

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadSession);
            multiThreadNode.Forward(multiThreadSession);

            AssertFloatArraysAreClose(singleThreadSession.Embedding.Span.ToArray(), multiThreadSession.Embedding.Span.ToArray(), 1e-4f, $"{backend} feed-forward residual threading");
        }

        private static FeedForwardResidualVectorsDocument LoadFeedForwardResidualVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FeedForwardResidualVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load feed-forward residual baseline JSON.");
        }

        private static FeedForwardResidualCase GetFeedForwardResidualCase(int caseIndex)
        {
            return FeedForwardResidualVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.HasCount(expected.Count, actual, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        private static void EnsureSimdSupported()
        {
            if (!Avx.IsSupported)
            {
                Assert.Inconclusive("AVX is not supported on the current machine.");
            }
        }

        internal sealed record FeedForwardResidualVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FeedForwardResidualCase> TestCases);

        internal sealed record FeedForwardResidualCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardResidualOutputs FirstLayerFfn);

        internal sealed record FeedForwardResidualOutputs(
            [property: JsonPropertyName("ffn_inp")] float[] FeedForwardInput,
            [property: JsonPropertyName("ffn_down")] float[] FeedForwardDown,
            [property: JsonPropertyName("l_out")] float[] LayerOutput);
    }
}
