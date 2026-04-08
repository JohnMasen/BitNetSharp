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
    public sealed class LmHeadLayerTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<LmHeadVectorsDocument> LmHeadVectorsDocumentCache = new(LoadLmHeadVectorsDocument);

        [TestMethod]
        public void LmHead_DefaultConfig_SimdAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.LmHeadLayer(model);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, layer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void LmHead_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstLayer = new BitNetSharp.Layers.LmHeadLayer(model, inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.LmHeadLayer(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        public void LmHead_MatchesBaseline_CPU_DebugCase()
        {
            VerifyLmHeadMatchesBaseline(DebugCaseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetLmHeadCaseIndices))]
        public void LmHead_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvxSupported();
            VerifyLmHeadMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void LmHead_Cache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(DebugCaseIndex);
            var uncachedLayer = new BitNetSharp.Layers.LmHeadLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.LmHeadLayer(
                model,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(uncachedSession.FinalNormOutput.Span);
            testCase.FinalNormOutput.Values.CopyTo(cachedSession.FinalNormOutput.Span);

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedSession);
            cachedLayer.Forward(cachedSession);

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.Logits.Span.ToArray(), cachedSession.Logits.Span.ToArray(), 0f, "lm head cache");
        }

        [TestMethod]
        public void LmHead_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(DebugCaseIndex);
            var layer = new BitNetSharp.Layers.LmHeadLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(session.FinalNormOutput.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(session));
        }

        public static IEnumerable<object[]> GetLmHeadCaseIndices()
        {
            return LmHeadVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static void VerifyLmHeadMatchesBaseline(int caseIndex, BitNetSharp.Layers.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            LmHeadCase testCase = GetLmHeadCase(caseIndex);
            var layer = new BitNetSharp.Layers.LmHeadLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormOutput.Values.CopyTo(session.FinalNormOutput.Span);

            layer.Init();
            layer.Forward(session);

            AssertFloatArraysAreClose(testCase.LmHead.Logits, session.Logits.Span.ToArray(), 1e-2f, $"token {testCase.TokenId} ({testCase.TokenText})");
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
