using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class SamplingNodeTests
    {
        private const int DebugCaseIndex = 0;
        private static readonly Lazy<SamplingVectorsDocument> SamplingVectorsDocumentCache = new(LoadSamplingVectorsDocument);

        [TestMethod]
        public void SamplingNode_DefaultTopK_IsTen()
        {
            var step = new BitNetSharp.Nodes.SamplingNode();

            Assert.AreEqual(10, step.TopK);
        }

        [TestMethod]
        public void SamplingNode_InvalidTopK_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(0));
        }

        [TestMethod]
        [DynamicData(nameof(GetSamplingCaseIndices))]
        public void SamplingNode_MatchesBaseline(int caseIndex)
        {
            VerifySamplingMatchesBaseline(caseIndex);
        }

        [TestMethod]
        public void SamplingNode_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            SamplingCase testCase = GetSamplingCase(DebugCaseIndex);
            var step = new BitNetSharp.Nodes.SamplingNode();
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.LmHead.Logits.CopyTo(session.Logits.Span);

            Assert.ThrowsExactly<InvalidOperationException>(() => step.Forward(session));
        }

        public static IEnumerable<object[]> GetSamplingCaseIndices()
        {
            return new[]
            {
                new object[] { 0 },
            };
        }

        private static void VerifySamplingMatchesBaseline(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            SamplingCase testCase = GetSamplingCase(caseIndex);
            var step = new BitNetSharp.Nodes.SamplingNode(testCase.LmHead.TopKTokenIds.Length);
            var session = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            testCase.LmHead.Logits.CopyTo(session.Logits.Span);

            step.Init();
            step.Forward(session);

            Assert.AreEqual(testCase.LmHead.ArgmaxTokenId, session.ArgmaxTokenId, $"token {testCase.TokenId} ({testCase.TokenText}) argmax token id");
            Assert.AreEqual(testCase.LmHead.ArgmaxLogit, session.ArgmaxLogit, 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText}) argmax logit");
            Assert.AreEqual(testCase.LmHead.NextTokenId, session.NextTokenId, $"token {testCase.TokenId} ({testCase.TokenText}) next token id");
            Assert.AreEqual(testCase.LmHead.ArgmaxLogit, session.NextTokenLogit, 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText}) next token logit");
            Assert.AreEqual(testCase.LmHead.NextTokenStrategy, session.NextTokenStrategy, $"token {testCase.TokenId} ({testCase.TokenText}) next token strategy");
            CollectionAssert.AreEqual(testCase.LmHead.TopKTokenIds, session.TopKTokenIds, $"token {testCase.TokenId} ({testCase.TokenText}) top-k token ids");
            AssertFloatArraysAreClose(testCase.LmHead.TopKLogits, session.TopKLogits, 1e-6f, $"token {testCase.TokenId} ({testCase.TokenText}) top-k logits");
        }

        private static SamplingVectorsDocument LoadSamplingVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<SamplingVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load sampling baseline JSON.");
        }

        private static SamplingCase GetSamplingCase(int caseIndex)
        {
            return SamplingVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.AreEqual(expected.Count, actual.Count, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        internal sealed record SamplingVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<SamplingCase> TestCases);

        internal sealed record SamplingCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("lm_head")] SamplingOutputs LmHead);

        internal sealed record SamplingOutputs(
            [property: JsonPropertyName("logits")] float[] Logits,
            [property: JsonPropertyName("top_k_token_ids")] int[] TopKTokenIds,
            [property: JsonPropertyName("top_k_logits")] float[] TopKLogits,
            [property: JsonPropertyName("argmax_token_id")] int ArgmaxTokenId,
            [property: JsonPropertyName("argmax_logit")] float ArgmaxLogit,
            [property: JsonPropertyName("next_token_id")] int NextTokenId,
            [property: JsonPropertyName("next_token_strategy")] string NextTokenStrategy);
    }
}
