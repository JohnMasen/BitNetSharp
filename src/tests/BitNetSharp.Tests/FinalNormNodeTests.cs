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
    public sealed class FinalNormNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FinalNormVectorsDocument> FinalNormVectorsDocumentCache = new(LoadFinalNormVectorsDocument);

        [TestMethod]
        public void FinalNorm_DefaultConfig_SimdSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.FinalNormNode(model);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, node.InferenceConfig.Backend);
            Assert.AreEqual(1, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void FinalNorm_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstNode = new BitNetSharp.Nodes.FinalNormNode(model, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.FinalNormNode(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        public void FinalNorm_MatchesBaseline_CPU_DebugCase()
        {
            VerifyFinalNormMatchesBaseline(DebugCaseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FinalNorm_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyFinalNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        public void FinalNorm_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyFinalNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        public void FinalNorm_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvx2Supported();
            VerifyFinalNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FinalNorm_Cache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(DebugCaseIndex);
            var uncachedNode = new BitNetSharp.Nodes.FinalNormNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var cachedNode = new BitNetSharp.Nodes.FinalNormNode(
                model,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(uncachedSession.Embedding.Span);
            testCase.FinalNormInput.Values.CopyTo(cachedSession.Embedding.Span);

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedSession);
            cachedNode.Forward(cachedSession);

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.FinalNormOutput.Span.ToArray(), cachedSession.FinalNormOutput.Span.ToArray(), 0f, "final norm cache");
        }

        [TestMethod]
        public void FinalNorm_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(DebugCaseIndex);
            var node = new BitNetSharp.Nodes.FinalNormNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(session.Embedding.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(session));
        }

        public static IEnumerable<object[]> GetFinalNormCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static void VerifyFinalNormMatchesBaseline(int caseIndex, BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(caseIndex);
            var node = new BitNetSharp.Nodes.FinalNormNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(session.Embedding.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FinalNormOutput.Values, session.FinalNormOutput.Span.ToArray(), 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyFinalNormMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(DebugCaseIndex);
            var singleThreadNode = new BitNetSharp.Nodes.FinalNormNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.FinalNormNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 2));
            var singleThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(singleThreadSession.Embedding.Span);
            testCase.FinalNormInput.Values.CopyTo(multiThreadSession.Embedding.Span);

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadSession);
            multiThreadNode.Forward(multiThreadSession);

            AssertFloatArraysAreClose(singleThreadSession.FinalNormOutput.Span.ToArray(), multiThreadSession.FinalNormOutput.Span.ToArray(), 1e-6f, $"{backend} final norm threading");
        }

        private static FinalNormVectorsDocument LoadFinalNormVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<FinalNormVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load final norm baseline JSON.");
        }

        private static FinalNormCase GetFinalNormCase(int caseIndex)
        {
            return FinalNormVectorsDocumentCache.Value.TestCases[caseIndex];
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

        internal sealed record FinalNormVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<FinalNormCase> TestCases);

        internal sealed record FinalNormCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("final_norm_input")] FinalNormVector FinalNormInput,
            [property: JsonPropertyName("final_norm_output")] FinalNormVector FinalNormOutput);

        internal sealed record FinalNormVector(
            [property: JsonPropertyName("values")] float[] Values);
    }
}
