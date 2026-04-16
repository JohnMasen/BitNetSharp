using System.Linq;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class RuntimeTopCandidateAlignmentTests
    {
        private static Models.BitNetModel? sharedModel;

        [ClassInitialize]
        public static void ClassInitialize(TestContext _)
        {
            sharedModel = TestModelFactory.LoadModel();
        }

        [ClassCleanup]
        public static void ClassCleanup()
        {
            sharedModel?.Dispose();
            sharedModel = null;
        }

        [TestMethod]
        public void HiPrompt_PromptTokenIds_MatchDumpBaseline()
        {
            Models.BitNetTokenizer tokenizer = GetTokenizer();
            int[] expectedPromptTokenIds = FullDumpBaseline.Manifest.Prompt.TokenIds.ToArray();

            int[] actualPromptTokenIds = tokenizer.EncodeChatMessageToIds(Models.BitNetChatRole.User, "hi", isFirstMessage: true).ToArray();

            CollectionAssert.AreEqual(expectedPromptTokenIds, actualPromptTokenIds);
        }

        private static Models.BitNetModel GetModel()
        {
            return sharedModel ?? throw new InvalidOperationException("Runtime top candidate alignment test model is not initialized.");
        }

        private static Models.BitNetTokenizer GetTokenizer()
        {
            return GetModel().Tokenizer ?? throw new InvalidOperationException("Tokenizer is not initialized.");
        }
    }
}
