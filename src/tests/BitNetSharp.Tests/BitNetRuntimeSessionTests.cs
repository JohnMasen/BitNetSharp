namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BitNetRuntimeSessionTests
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
        public void GenerateTokenIds_WithPrompt_MatchesStepwiseSingleTokenInference()
        {
            Models.BitNetModel model = GetModel();
            using var actualMemoryManager = new BitNetMemoryManager();
            using var expectedMemoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, actualMemoryManager, TestInferenceConfigs.Simd(0));
            using var expectedRuntime = new BitNetRuntime(model, expectedMemoryManager, TestInferenceConfigs.Simd(0));

            int firstTokenId = expectedRuntime.InferenceTokenId(0);
            int secondTokenId = expectedRuntime.InferenceTokenId(firstTokenId);

            int[] actual = runtime.GenerateTokenIds(new[] { 0 }, 2);

            CollectionAssert.AreEqual(new[] { firstTokenId, secondTokenId }, actual);
        }

        [TestMethod]
        public void GenerateTokenIds_WithPrompt_AppendsTokensToSessionHistory()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            int[] generated = runtime.GenerateTokenIds(new[] { 0 }, 2);

            CollectionAssert.AreEqual(new[] { 0, generated[0], generated[1] }, runtime.Session.Tokens.ToArray());
        }

        [TestMethod]
        public void GenerateTokenIds_WithPrompt_CreatesNewSession()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.GenerateTokenIds(new[] { 0 }, 1);
            BitNetSession previousSession = runtime.Session;

            runtime.GenerateTokenIds(new[] { 1 }, 1);

            Assert.AreNotSame(previousSession, runtime.Session);
        }

        [TestMethod]
        public void Prefill_WithPrompt_CreatesSessionHistory()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            runtime.Prefill(new[] { 0, 1 });

            CollectionAssert.AreEqual(new[] { 0, 1 }, runtime.Session.Tokens.ToArray());
        }

        [TestMethod]
        public void Prefill_WithPrompt_PopulatesKvCacheLength()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            runtime.Prefill(new[] { 0, 1 });

            Assert.AreEqual(2, runtime.Session.CacheLength);
        }

        [TestMethod]
        public void ContinuePrefill_AppendsTokensToActiveSession()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.Prefill(new[] { 0 });

            runtime.ContinuePrefill(new[] { 1, 2 });

            CollectionAssert.AreEqual(new[] { 0, 1, 2 }, runtime.Session.Tokens.ToArray());
        }

        [TestMethod]
        public void GenerateTokenIds_AfterPrefill_MatchesGenerateTokenIdsWithPrompt()
        {
            Models.BitNetModel model = GetModel();
            using var prefillMemoryManager = new BitNetMemoryManager();
            using var generateMemoryManager = new BitNetMemoryManager();
            using var prefillRuntime = new BitNetRuntime(model, prefillMemoryManager, TestInferenceConfigs.Simd(0));
            using var generateRuntime = new BitNetRuntime(model, generateMemoryManager, TestInferenceConfigs.Simd(0));
            prefillRuntime.Prefill(new[] { 0 });

            int[] prefillThenGenerate = prefillRuntime.GenerateTokenIds(2);
            int[] generateFromPrompt = generateRuntime.GenerateTokenIds(new[] { 0 }, 2);

            CollectionAssert.AreEqual(generateFromPrompt, prefillThenGenerate);
        }

        [TestMethod]
        public void GenerateTokenIds_AfterPrefill_AdvancesKvCacheLength()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.Prefill(new[] { 0 });

            runtime.GenerateTokenIds(2);

            Assert.AreEqual(3, runtime.Session.CacheLength);
        }

        [TestMethod]
        public void GenerateTokenIds_WithoutPrompt_ContinuesFromCurrentSessionToken()
        {
            Models.BitNetModel model = GetModel();
            using var actualMemoryManager = new BitNetMemoryManager();
            using var expectedMemoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, actualMemoryManager, TestInferenceConfigs.Simd(0));
            using var expectedRuntime = new BitNetRuntime(model, expectedMemoryManager, TestInferenceConfigs.Simd(0));

            int firstTokenId = expectedRuntime.InferenceTokenId(0);
            int expectedNextTokenId = expectedRuntime.InferenceTokenId(firstTokenId);
            runtime.GenerateTokenIds(new[] { 0 }, 1);

            int[] actual = runtime.GenerateTokenIds(1);

            CollectionAssert.AreEqual(new[] { expectedNextTokenId }, actual);
        }

        [TestMethod]
        public void GenerateTokenIds_WithoutPrompt_IncrementsOutputRound()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.GenerateTokenIds(new[] { 0 }, 1);

            runtime.GenerateTokenIds(1);

            Assert.AreEqual(2, runtime.Session.OutputRound);
        }

        [TestMethod]
        public void GenerateTokenIds_WithoutPrompt_ThrowsForEmptySession()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            Assert.ThrowsExactly<InvalidOperationException>(() => runtime.GenerateTokenIds(1));
        }

        [TestMethod]
        public void ContinuePrefill_WithoutActiveSession_Throws()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            Assert.ThrowsExactly<InvalidOperationException>(() => runtime.ContinuePrefill(new[] { 0 }));
        }

        [TestMethod]
        public void Session_ThrowsWithoutActiveSession()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            Assert.ThrowsExactly<InvalidOperationException>(() => _ = runtime.Session);
        }

        [TestMethod]
        public void StartConversation_CreatesActiveSession()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));

            runtime.StartConversation("hello");

            Assert.IsFalse(runtime.Session.Tokens.IsEmpty);
        }

        [TestMethod]
        public void GenerateAssistantReply_ReturnsText()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.StartConversation("hello");

            string reply = runtime.GenerateAssistantReply(4);

            Assert.IsNotNull(reply);
        }

        [TestMethod]
        public void StreamAssistantReply_YieldsTokens()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            runtime.StartConversation("hello");

            string[] replyTokens = runtime.StreamAssistantReply(4).ToArray();

            Assert.IsNotNull(replyTokens);
        }

        [TestMethod]
        public void StreamAssistantReply_WhenCanceled_ThrowsOperationCanceledException()
        {
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0));
            using var cancellationTokenSource = new CancellationTokenSource();
            runtime.StartConversation("hello");
            cancellationTokenSource.Cancel();

            Assert.ThrowsExactly<OperationCanceledException>(() => runtime.StreamAssistantReply(4, cancellationTokenSource.Token).ToArray());
        }

        private static Models.BitNetModel GetModel()
        {
            return sharedModel ?? throw new InvalidOperationException("Runtime session test model is not initialized.");
        }
    }
}
