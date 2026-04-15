namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BitNetSessionTests
    {
        [TestMethod]
        public void AppendToken_AppendsHistory()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager, new[] { 1, 2 });

            session.AppendToken(3);

            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, session.Tokens.ToArray());
        }

        [TestMethod]
        public void Constructor_WithInitialTokens_SetsCurrentToken()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager, new[] { 10, 11 });

            Assert.AreEqual(11, session.CurrentToken);
        }

        [TestMethod]
        public void BeginOutputRound_IncrementsRound()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager);

            session.BeginOutputRound();

            Assert.AreEqual(1, session.OutputRound);
        }

        [TestMethod]
        public void AppendOutputToken_ThrowsWithoutRound()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager);

            Assert.ThrowsExactly<InvalidOperationException>(() => session.AppendOutputToken(1));
        }

        [TestMethod]
        public void AppendOutputToken_TracksCurrentOutput()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager, new[] { 10, 11 });
            session.BeginOutputRound();

            session.AppendOutputToken(12);
            session.AppendOutputToken(13);
            session.CompleteOutputRound();

            CollectionAssert.AreEqual(new[] { 12, 13 }, session.CurrentOutputTokens.ToArray());
        }

        [TestMethod]
        public void GetOrCreateLayerKeyCacheTensor_AllocatesContextSizedBuffer()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager);

            RuntimeTensor tensor = session.GetOrCreateLayerKeyCacheTensor(0);

            Assert.IsTrue(tensor.TryGet<Memory<float>>(out Memory<float> buffer));
            Assert.AreEqual(checked((int)(model.Config!.ContextLength * model.Config.KeyValueProjectionSize)), buffer.Length);
        }

        [TestMethod]
        public void GetOrCreateLayerKeyCacheTensor_UsesDifferentKeysPerLayer()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager);

            RuntimeTensor first = session.GetOrCreateLayerKeyCacheTensor(0);
            RuntimeTensor second = session.GetOrCreateLayerKeyCacheTensor(1);

            Assert.AreNotSame(first, second);
            Assert.AreEqual("LayerKeyCache:0", first.Name);
            Assert.AreEqual("LayerKeyCache:1", second.Name);
        }

        [TestMethod]
        public void Constructor_WithoutInitialTokens_UsesEmptyHistory()
        {
            using var model = TestModelFactory.LoadModel();
            using var memoryManager = new BitNetMemoryManager();
            using var session = new BitNetSession(model, memoryManager);

            Assert.IsTrue(session.Tokens.IsEmpty);
        }
    }
}
