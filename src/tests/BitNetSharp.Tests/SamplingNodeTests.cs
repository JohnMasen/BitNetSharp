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
        public void SamplingNode_InvalidTopK_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(0));
        }

        [TestMethod]
        public void SamplingNode_InvalidTemperature_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(temperature: -0.1f));
        }

        [TestMethod]
        public void SamplingNode_InvalidTopP_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(topP: 0f));
        }

        [TestMethod]
        public void SamplingNode_InvalidMinP_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(minP: 1.1f));
        }

        [TestMethod]
        public void SamplingNode_InvalidRepeatLastN_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(repeatLastN: -1));
        }

        [TestMethod]
        public void SamplingNode_InvalidRepeatPenalty_Throws()
        {
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new BitNetSharp.Nodes.SamplingNode(repeatPenalty: 0f));
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

        [TestMethod]
        public void SamplingNode_WithSamplingEnabled_SelectsTokenWithinTopK()
        {
            using var model = TestModelFactory.LoadModel();
            var step = new BitNetSharp.Nodes.SamplingNode(topK: 3, enableSampling: true, randomSeed: 123);
            var session = TestModelFactory.CreateSession(model, token: 0);
            session.Logits.Span.Fill(float.NegativeInfinity);
            session.Logits.Span[0] = 1f;
            session.Logits.Span[1] = 5f;
            session.Logits.Span[2] = 4f;
            session.Logits.Span[3] = 3f;

            step.Init();
            step.Forward(session);

            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, session.TopKTokenIds);
            CollectionAssert.Contains(session.TopKTokenIds, session.NextTokenId);
            Assert.AreEqual("top_k_sampling", session.NextTokenStrategy);
        }

        [TestMethod]
        public void SamplingNode_WithFixedSeed_UsesStableSequenceAcrossForwardCalls()
        {
            using var model = TestModelFactory.LoadModel();
            var step = new BitNetSharp.Nodes.SamplingNode(topK: 3, enableSampling: true, randomSeed: 123, temperature: 0.8f, topP: 0.95f, minP: 0f);
            var firstSession = TestModelFactory.CreateSession(model, token: 0);
            var secondSession = TestModelFactory.CreateSession(model, token: 0);
            firstSession.Logits.Span.Fill(float.NegativeInfinity);
            secondSession.Logits.Span.Fill(float.NegativeInfinity);
            for (int index = 0; index < 3; index++)
            {
                firstSession.Logits.Span[index] = 3 - index;
                secondSession.Logits.Span[index] = 3 - index;
            }

            step.Init();
            step.Forward(firstSession);
            int firstToken = firstSession.NextTokenId;
            step.Forward(secondSession);
            int secondToken = secondSession.NextTokenId;

            Assert.AreNotEqual(firstToken, secondToken);
        }

        [TestMethod]
        public void SamplingNode_WithRepeatPenalty_PenalizesRecentToken()
        {
            using var model = TestModelFactory.LoadModel();
            using var session = TestModelFactory.CreateSession(model, token: 1);
            session.AppendToken(1);
            session.Logits.Span.Fill(float.NegativeInfinity);
            session.Logits.Span[1] = 5f;
            session.Logits.Span[2] = 4.9f;
            var step = new BitNetSharp.Nodes.SamplingNode(topK: 2, enableSampling: false, repeatLastN: 64, repeatPenalty: 2.0f);

            step.Init();
            step.Forward(session);

            Assert.AreEqual(2, session.NextTokenId);
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
            Assert.HasCount(expected.Count, actual, caseName);
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
