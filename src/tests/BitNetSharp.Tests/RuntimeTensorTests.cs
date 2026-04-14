namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class RuntimeTensorTests
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
        public void GetWeightTensor_WhenRequested_ReturnsSharedTensorAcrossSessions()
        {
            var model = GetModel();
            using var firstSession = TestModelFactory.CreateSession(model, token: 0);
            using var secondSession = TestModelFactory.CreateSession(model, token: 1);

            RuntimeTensor first = firstSession.GetWeightTensor(model.GlobalTensors!.TokenEmbedding.Name);
            RuntimeTensor second = secondSession.GetWeightTensor(model.GlobalTensors.TokenEmbedding.Name);

            Assert.AreSame(first, second);
        }

        [TestMethod]
        public void GetWeightTensor_WhenRequested_ExposesReadonlyBytes()
        {
            var model = GetModel();
            using var session = TestModelFactory.CreateSession(model, token: 0);

            RuntimeTensor tensor = session.GetWeightTensor(model.GlobalTensors!.TokenEmbedding.Name);
            bool hasBytes = tensor.TryGet<ReadOnlyMemory<byte>>(out ReadOnlyMemory<byte> bytes);
            bool hasWritableBytes = tensor.TryGet<Memory<byte>>(out _);

            Assert.IsTrue(hasBytes);
            Assert.IsFalse(bytes.IsEmpty);
            Assert.IsTrue(tensor.IsReadOnly);
            Assert.AreEqual(typeof(byte), tensor.ElementType);
            Assert.IsFalse(hasWritableBytes);
        }

        [TestMethod]
        public void GetOrCreateRuntimeTensor_WhenCalledTwice_ReturnsCachedTensor()
        {
            var model = GetModel();
            using var session = TestModelFactory.CreateSession(model, token: 0);

            RuntimeTensor first = session.GetOrCreateRuntimeTensor(nameof(BitNetSession.Embedding));
            RuntimeTensor second = session.GetOrCreateRuntimeTensor(nameof(BitNetSession.Embedding));

            Assert.AreSame(first, second);
            Assert.IsFalse(first.IsReadOnly);
            Assert.AreEqual(typeof(float), first.ElementType);
            Assert.AreEqual((int)model.Config!.EmbeddingLength, first.Shape[0]);
        }

        [TestMethod]
        public void GetOrCreateRuntimeTensor_WhenCopiedFromHost_UpdatesSessionBuffer()
        {
            var model = GetModel();
            using var session = TestModelFactory.CreateSession(model, token: 0);
            RuntimeTensor tensor = session.GetOrCreateRuntimeTensor(nameof(BitNetSession.Embedding));
            float[] values = Enumerable.Range(0, (int)model.Config!.EmbeddingLength)
                .Select(static index => index + 0.5f)
                .ToArray();

            tensor.CopyFrom<float>(values.AsMemory());
            bool hasBuffer = tensor.TryGet<Memory<float>>(out Memory<float> buffer);

            Assert.IsTrue(hasBuffer);
            CollectionAssert.AreEqual(values, buffer.ToArray());
            CollectionAssert.AreEqual(values, session.Embedding.ToArray());
        }

        [TestMethod]
        public void GetOrCreateRuntimeTensor_WhenRequestedFromDifferentSessions_ReturnsDifferentTensorInstances()
        {
            var model = GetModel();
            using var firstSession = TestModelFactory.CreateSession(model, token: 0);
            using var secondSession = TestModelFactory.CreateSession(model, token: 1);

            RuntimeTensor first = firstSession.GetOrCreateRuntimeTensor(nameof(BitNetSession.Embedding));
            RuntimeTensor second = secondSession.GetOrCreateRuntimeTensor(nameof(BitNetSession.Embedding));

            Assert.AreNotSame(first, second);
        }

        private static Models.BitNetModel GetModel()
        {
            return sharedModel ?? throw new InvalidOperationException("RuntimeTensor test model is not initialized.");
        }
    }
}
