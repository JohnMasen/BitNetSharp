namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BitNetMemoryManagerTests
    {
        [TestMethod]
        public void RequestMemory_ReusesBufferForSameSessionAndKey()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();

            Memory<int> buffer = memoryManager.RequestMemory<int>(sessionId, "Tokens", 4);
            buffer.Span[0] = 13;

            Memory<int> fetched = memoryManager.GetMemory<int>(sessionId, "Tokens");

            Assert.AreEqual(4, fetched.Length);
            Assert.AreEqual(13, fetched.Span[0]);
        }

        [TestMethod]
        public void RequestMemory_SameKeyDifferentSessions_ReturnsDifferentBuffers()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid firstSessionId = Guid.NewGuid();
            Guid secondSessionId = Guid.NewGuid();

            Memory<int> firstBuffer = memoryManager.RequestMemory<int>(firstSessionId, "LayerKeyCache:0", 4);
            Memory<int> secondBuffer = memoryManager.RequestMemory<int>(secondSessionId, "LayerKeyCache:0", 4);

            firstBuffer.Span[0] = 11;
            secondBuffer.Span[0] = 29;

            Assert.AreEqual(11, firstBuffer.Span[0]);
            Assert.AreEqual(29, secondBuffer.Span[0]);
        }

        [TestMethod]
        public void TryGetMemory_MissingKey_ReturnsFalse()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();

            bool found = memoryManager.TryGetMemory<float>(sessionId, "LayerValueCache:2", out Memory<float> buffer);

            Assert.IsFalse(found);
            Assert.IsTrue(buffer.IsEmpty);
        }

        [TestMethod]
        public void Release_SpecificKey_PreservesOtherKeys()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();
            memoryManager.RequestMemory<int>(sessionId, "LayerKeyCache:0", 4);
            memoryManager.RequestMemory<int>(sessionId, "LayerKeyCache:1", 4);

            memoryManager.Release(sessionId, "LayerKeyCache:0");

            Assert.IsFalse(memoryManager.TryGetMemory<int>(sessionId, "LayerKeyCache:0", out _));
            Assert.IsTrue(memoryManager.TryGetMemory<int>(sessionId, "LayerKeyCache:1", out Memory<int> remainingBuffer));
            Assert.AreEqual(4, remainingBuffer.Length);
        }
    }
}
