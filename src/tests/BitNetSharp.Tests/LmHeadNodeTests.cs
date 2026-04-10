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
    public sealed class LmHeadNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<LmHeadVectorsDocument> LmHeadVectorsDocumentCache = new(LoadLmHeadVectorsDocument);

        [TestMethod]
        public void LmHead_DefaultConfig_SimdAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.LmHeadNode(model);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, node.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void LmHead_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstNode = new BitNetSharp.Nodes.LmHeadNode(model, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.LmHeadNode(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvxSupported();
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void LmHead_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyLmHeadMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        public void LmHead_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyLmHeadMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        public void LmHead_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvxSupported();
            VerifyLmHeadMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void LmHead_Cache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(DebugCaseIndex);
            var uncachedNode = new BitNetSharp.Nodes.LmHeadNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var cachedNode = new BitNetSharp.Nodes.LmHeadNode(
                model,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(uncachedSession.FinalNormOutput.Span);
            testCase.FinalNormOutput.Values.CopyTo(cachedSession.FinalNormOutput.Span);

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedSession);
            cachedNode.Forward(cachedSession);

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.Logits.Span.ToArray(), cachedSession.Logits.Span.ToArray(), 0f, "lm head cache");
        }

        [TestMethod]
        public void LmHead_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(DebugCaseIndex);
            var node = new BitNetSharp.Nodes.LmHeadNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(session.FinalNormOutput.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(session));
        }

        public static IEnumerable<object[]> GetLmHeadCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static void VerifyLmHeadMatchesBaseline(int caseIndex, BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(caseIndex);
            var node = new BitNetSharp.Nodes.LmHeadNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(session.FinalNormOutput.Span);

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.LmHead.Logits, session.Logits.Span.ToArray(), 1e-2f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static void VerifyLmHeadMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(DebugCaseIndex);
            var singleThreadNode = new BitNetSharp.Nodes.LmHeadNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.LmHeadNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 2));
            var singleThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(singleThreadSession.FinalNormOutput.Span);
            testCase.FinalNormOutput.Values.CopyTo(multiThreadSession.FinalNormOutput.Span);

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadSession);
            multiThreadNode.Forward(multiThreadSession);

            AssertFloatArraysAreClose(singleThreadSession.Logits.Span.ToArray(), multiThreadSession.Logits.Span.ToArray(), 1e-2f, $"{backend} lm-head threading");
        }

        private static LmHeadVectorsDocument LoadLmHeadVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<LmHeadVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load lm head baseline JSON.");
        }

        private static LmHeadCase GetLmHeadCase(int caseIndex)
        {
            return LmHeadVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static void EnsureAvxSupported()
        {
            if (!Avx.IsSupported)
            {
                Assert.Inconclusive("AVX is not supported on the current machine.");
            }
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.AreEqual(expected.Count, actual.Count, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                if (MathF.Abs(expected[index] - actual[index]) > delta)
                {
                    Assert.Fail($"{caseName} mismatch at index {index}. Expected {expected[index]}, actual {actual[index]}, delta {delta}.");
                }
            }
        }

        internal sealed record LmHeadVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<LmHeadCase> TestCases);

        internal sealed record LmHeadCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("final_norm_output")] LmHeadVector FinalNormOutput,
            [property: JsonPropertyName("lm_head")] LmHeadOutputs LmHead);

        internal sealed record LmHeadVector(
            [property: JsonPropertyName("values")] float[] Values);

        internal sealed record LmHeadOutputs(
            [property: JsonPropertyName("logits")] float[] Logits);
    }
}
