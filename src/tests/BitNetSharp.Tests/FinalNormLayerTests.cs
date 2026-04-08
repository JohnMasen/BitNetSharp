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
    public sealed class FinalNormLayerTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<FinalNormVectorsDocument> FinalNormVectorsDocumentCache = new(LoadFinalNormVectorsDocument);

        [TestMethod]
        public void FinalNorm_DefaultConfig_SimdSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.FinalNormLayer(model);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, layer.InferenceConfig.Backend);
            Assert.AreEqual(1, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void FinalNorm_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstLayer = new BitNetSharp.Layers.FinalNormLayer(model, inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.FinalNormLayer(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(1, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        public void FinalNorm_MatchesBaseline_CPU_DebugCase()
        {
            VerifyFinalNormMatchesBaseline(DebugCaseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.CPU);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetFinalNormCaseIndices))]
        public void FinalNorm_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();
            VerifyFinalNormMatchesBaseline(caseIndex, BitNetSharp.Layers.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void FinalNorm_Cache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(DebugCaseIndex);
            var uncachedLayer = new BitNetSharp.Layers.FinalNormLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.FinalNormLayer(
                model,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var uncachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var cachedSession = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(uncachedSession.Embedding.Span);
            testCase.FinalNormInput.Values.CopyTo(cachedSession.Embedding.Span);

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedSession);
            cachedLayer.Forward(cachedSession);

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncachedSession.FinalNormOutput.Span.ToArray(), cachedSession.FinalNormOutput.Span.ToArray(), 0f, "final norm cache");
        }

        [TestMethod]
        public void FinalNorm_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(DebugCaseIndex);
            var layer = new BitNetSharp.Layers.FinalNormLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(session.Embedding.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(session));
        }

        public static IEnumerable<object[]> GetFinalNormCaseIndices()
        {
            return FinalNormVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static void VerifyFinalNormMatchesBaseline(int caseIndex, BitNetSharp.Layers.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            FinalNormCase testCase = GetFinalNormCase(caseIndex);
            var layer = new BitNetSharp.Layers.FinalNormLayer(
                model,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.FinalNormInput.Values.CopyTo(session.Embedding.Span);

            layer.Init();
            layer.Forward(session);

            AssertFloatArraysAreClose(testCase.FinalNormOutput.Values, session.FinalNormOutput.Span.ToArray(), 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
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
