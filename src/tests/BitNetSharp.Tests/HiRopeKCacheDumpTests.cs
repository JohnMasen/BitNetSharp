namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class HiRopeKCacheDumpTests
    {
        [TestMethod]
        public void HiRopeDump_LoadsExpectedPrompt()
        {
            FullDumpBaseline.FullDumpPrompt document = FullDumpBaseline.Manifest.Prompt;

            Assert.AreEqual("User: hi<|eot_id|>Assistant: ", document.Text);
            CollectionAssert.AreEqual(new[] { 128000, 1502, 25, 15960, 128009, 72803, 25, 220 }, document.TokenIds.ToArray());
            Assert.AreEqual(7, document.AssistantFirstTokenPreSamplingPositionIndex);
        }
    }
}
