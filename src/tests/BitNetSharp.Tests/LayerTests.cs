using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class LayerTests
    {
        private static readonly Lazy<EmbeddingVectorsDocument> EmbeddingVectorsDocumentCache = new(LoadEmbeddingVectorsDocument);

        [TestMethod]
        public void Embedding_ReturnsExpectedLength()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);

            float[] embedding = layer.Forward(context);

            Assert.AreEqual((int)model.Config!.EmbeddingLength, embedding.Length);
        }

        [TestMethod]
        public void Embedding_ThrowsForOutOfRangeToken()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: (int)model.Config!.VocabularySize);

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => layer.Forward(context));
        }

        [TestMethod]
        public void EmbeddingCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var uncachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var cachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model, enableCache: true);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);

            CollectionAssert.AreEqual(uncachedLayer.Forward(context), cachedLayer.Forward(context));
            Assert.IsTrue(cachedLayer.EnableCache);
        }

        [TestMethod]
        [DynamicData(nameof(GetEmbeddingCases))]
        public void Embedding_MatchesBaseline(string caseName, int tokenId, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, tokenId);

            float[] actualValues = layer.Forward(context);

            Assert.AreEqual(expectedValues.Length, actualValues.Length, caseName);
            CollectionAssert.AreEqual(expectedValues, actualValues, caseName);
        }

        public static IEnumerable<object[]> GetEmbeddingCases()
        {
            return EmbeddingVectorsDocumentCache.Value.TestCases.Select(testCase => new object[]
            {
                $"token {testCase.TokenId} ({testCase.TokenText})",
                testCase.TokenId,
                testCase.Dequantized.Values.ToArray(),
            });
        }

        private static EmbeddingVectorsDocument LoadEmbeddingVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.EmbeddingVectorsPath);
            return JsonSerializer.Deserialize<EmbeddingVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load embedding baseline JSON.");
        }

        internal sealed record EmbeddingVectorsDocument(
            [property: JsonPropertyName("vector_length")] int VectorLength,
            [property: JsonPropertyName("test_cases")] IReadOnlyList<EmbeddingVectorCase> TestCases);

        internal sealed record EmbeddingVectorCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("vector_length")] int VectorLength,
            [property: JsonPropertyName("dequantized")] EmbeddingVectorValues Dequantized);

        internal sealed record EmbeddingVectorValues(
            [property: JsonPropertyName("dtype")] string DType,
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);
    }
}
