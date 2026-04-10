using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class TokenizerTests
    {
        private static readonly Lazy<StandardTokensDocument> StandardTokensDocumentCache = new(LoadStandardTokensDocument);
        private static Models.BitNetModel? sharedModel;

        [ClassInitialize]
        public static void ClassInitialize(TestContext _)
        {
            sharedModel = new Models.BitNetModel();
            sharedModel.Load(TestProjectPaths.ModelPath);
        }

        [ClassCleanup]
        public static void ClassCleanup()
        {
            sharedModel?.Dispose();
            sharedModel = null;
        }

        [TestMethod]
        [DynamicData(nameof(GetEncodingCases))]
        public void WhenEncodingStandardCaseThenMatchesSavedTokenCount(string caseName, string text, bool addBos, int expectedTokenCount)
        {
            var actualTokenCount = GetTokenizer().EncodeToIds(text, addBos: addBos).Count;

            Assert.AreEqual(expectedTokenCount, actualTokenCount, caseName);
        }

        [TestMethod]
        [DynamicData(nameof(GetEncodingCases))]
        public void WhenEncodingStandardCaseThenMatchesSavedTokenIds(string caseName, string text, bool addBos, int expectedTokenCount)
        {
            var testCase = GetCase(caseName);
            var actualTokenIds = GetTokenizer().EncodeToIds(text, addBos: addBos).ToArray();

            CollectionAssert.AreEqual(testCase.TokenIds.ToArray(), actualTokenIds, caseName);
        }

        [TestMethod]
        [DynamicData(nameof(GetDecodingCases))]
        public void WhenDecodingStandardCaseThenMatchesSavedText(string caseName, int[] tokenIds, string reconstructedText)
        {
            Assert.AreEqual(reconstructedText, GetTokenizer().Decode(tokenIds), caseName);
        }

        public static IEnumerable<object[]> GetEncodingCases()
        {
            return StandardTokensDocumentCache.Value.TestCases.Take(1).Select(testCase => new object[]
            {
                testCase.Name,
                testCase.Text,
                testCase.AddBos,
                testCase.TokenCount,
            });
        }

        public static IEnumerable<object[]> GetDecodingCases()
        {
            return StandardTokensDocumentCache.Value.TestCases.Take(1).Select(testCase => new object[]
            {
                testCase.Name,
                testCase.TokenIds.ToArray(),
                testCase.ReconstructedText,
            });
        }

        private static Models.BitNetTokenizer GetTokenizer()
        {
            return sharedModel?.Tokenizer ?? throw new InvalidOperationException("Tokenizer test model is not initialized.");
        }

        private static StandardTokenCase GetCase(string caseName)
        {
            return StandardTokensDocumentCache.Value.TestCases.Single(testCase => testCase.Name == caseName);
        }

        private static StandardTokensDocument LoadStandardTokensDocument()
        {
            var json = File.ReadAllText(TestProjectPaths.StandardTokensPath);
            return JsonSerializer.Deserialize<StandardTokensDocument>(json) ?? throw new InvalidOperationException("Failed to load tokenizer baseline JSON.");
        }

        internal sealed record StandardTokensDocument(
            [property: JsonPropertyName("version")] int Version,
            [property: JsonPropertyName("generated_at")] string GeneratedAt,
            [property: JsonPropertyName("model")] string Model,
            [property: JsonPropertyName("llama_dll")] string LlamaDll,
            [property: JsonPropertyName("generator")] string Generator,
            [property: JsonPropertyName("test_cases")] IReadOnlyList<StandardTokenCase> TestCases);

        internal sealed record StandardTokenCase(
            [property: JsonPropertyName("name")] string Name,
            [property: JsonPropertyName("category")] string Category,
            [property: JsonPropertyName("text")] string Text,
            [property: JsonPropertyName("add_bos")] bool AddBos,
            [property: JsonPropertyName("token_count")] int TokenCount,
            [property: JsonPropertyName("token_ids")] IReadOnlyList<int> TokenIds,
            [property: JsonPropertyName("reconstructed_text")] string ReconstructedText,
            [property: JsonPropertyName("reconstructed_utf8_hex")] string ReconstructedUtf8Hex,
            [property: JsonPropertyName("tokens")] IReadOnlyList<StandardToken> Tokens);

        internal sealed record StandardToken(
            [property: JsonPropertyName("id")] int Id,
            [property: JsonPropertyName("text")] string Text,
            [property: JsonPropertyName("utf8_hex")] string Utf8Hex,
            [property: JsonPropertyName("byte_length")] int ByteLength,
            [property: JsonPropertyName("utf8_valid")] bool Utf8Valid);
    }
}
