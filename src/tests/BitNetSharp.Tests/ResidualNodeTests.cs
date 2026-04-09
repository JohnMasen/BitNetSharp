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
    public sealed class ResidualNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<ResidualVectorsDocument> ResidualVectorsDocumentCache = new(LoadResidualVectorsDocument);

        [TestMethod]
        public void Residual_DefaultConfig_CpuSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.ResidualNode(model);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.CPU, node.InferenceConfig.Backend);
            Assert.AreEqual(1, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void Residual_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstNode = new BitNetSharp.Nodes.ResidualNode(model, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.ResidualNode(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.CPU, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        public void Residual_MatchesBaseline_CPU_DebugCase()
        {
            VerifyResidualMatchesBaselineCpu(DebugCaseIndex);
        }

        [TestMethod]
        [DynamicData(nameof(GetResidualCaseIndices))]
        public void Residual_MatchesBaseline_CPU(int caseIndex)
        {
            VerifyResidualMatchesBaselineCpu(caseIndex);
        }

        [TestMethod]
        [DynamicData(nameof(GetResidualCaseIndices))]
        public void Residual_MatchesBaseline_Tensor(int caseIndex)
        {
            VerifyResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        [DynamicData(nameof(GetResidualCaseIndices))]
        public void Residual_MatchesBaseline_SIMD(int caseIndex)
        {
            EnsureSimdSupported();
            VerifyResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void Residual_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            ResidualCase testCase = GetResidualCase(0);
            var node = new BitNetSharp.Nodes.ResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            session.Embedding = testCase.Dequantized.Values.ToArray();
            session.AttentionOutput = testCase.FirstLayerAttnOutput.Values.ToArray();

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(session));
        }

        public static IEnumerable<object[]> GetResidualCaseIndices()
        {
            return ResidualVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        private static void VerifyResidualMatchesBaselineCpu(int caseIndex)
        {
            VerifyResidualMatchesBaseline(caseIndex, BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        private static void VerifyResidualMatchesBaseline(int caseIndex, BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            ResidualCase testCase = GetResidualCase(caseIndex);
            var node = new BitNetSharp.Nodes.ResidualNode(
                model,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            session.Embedding = testCase.Dequantized.Values.ToArray();
            session.AttentionOutput = testCase.FirstLayerAttnOutput.Values.ToArray();

            node.Init();
            node.Forward(session);

            AssertFloatArraysAreClose(testCase.FirstLayerFfn.FeedForwardInput, session.FeedForwardInput.Span.ToArray(), 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText})");
        }

        private static ResidualVectorsDocument LoadResidualVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<ResidualVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load residual baseline JSON.");
        }

        private static ResidualCase GetResidualCase(int caseIndex)
        {
            return ResidualVectorsDocumentCache.Value.TestCases[caseIndex];
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

        internal sealed record ResidualVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<ResidualCase> TestCases);

        internal sealed record ResidualCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("dequantized")] LayerVectorValues Dequantized,
            [property: JsonPropertyName("first_layer_attn_output")] LayerVectorValues FirstLayerAttnOutput,
            [property: JsonPropertyName("first_layer_ffn")] FeedForwardOutputs FirstLayerFfn);

        internal sealed record LayerVectorValues(
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);

        internal sealed record FeedForwardOutputs(
            [property: JsonPropertyName("ffn_inp")] float[] FeedForwardInput);
    }
}
