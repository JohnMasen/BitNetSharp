using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class HiRopeKCacheDumpTests
    {
        private static readonly Lazy<HiRopeKCacheDumpDocument> HiRopeKCacheDumpDocumentCache = new(LoadHiRopeKCacheDumpDocument);

        [TestMethod]
        public void HiRopeDump_LoadsExpectedPrompt()
        {
            HiRopeKCacheDumpDocument document = HiRopeKCacheDumpDocumentCache.Value;

            Assert.AreEqual("User: hi<|eot_id|>Assistant: ", document.Prompt.Text);
            CollectionAssert.AreEqual(new[] { 128000, 1502, 25, 15960, 128009, 72803, 25, 220 }, document.Prompt.TokenIds.ToArray());
            Assert.AreEqual(7, document.Prompt.AssistantFirstTokenPreSamplingPositionIndex);
        }

        private static HiRopeKCacheDumpDocument LoadHiRopeKCacheDumpDocument()
        {
            string json = File.ReadAllText(TestProjectPaths.HiRopeKCacheDumpPath);
            return JsonSerializer.Deserialize<HiRopeKCacheDumpDocument>(json) ?? throw new InvalidOperationException("Failed to load hi rope/k-cache dump JSON.");
        }

        internal sealed record HiRopeKCacheDumpDocument(
            [property: JsonPropertyName("prompt")] HiRopePrompt Prompt);

        internal sealed record HiRopePrompt(
            [property: JsonPropertyName("text")] string Text,
            [property: JsonPropertyName("token_ids")] IReadOnlyList<int> TokenIds,
            [property: JsonPropertyName("assistant_first_token_pre_sampling_position_index")] int AssistantFirstTokenPreSamplingPositionIndex);
    }
}
