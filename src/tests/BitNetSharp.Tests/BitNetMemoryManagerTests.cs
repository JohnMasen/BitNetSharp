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

            RuntimeTensorLease bufferLease = memoryManager.RequestMemory<int>(sessionId, "Tokens", 4);
            Assert.IsTrue(bufferLease.Tensor.TryGet<Memory<int>>(out Memory<int> buffer));
            buffer.Span[0] = 13;

            RuntimeTensorLease fetchedLease = memoryManager.GetMemory(sessionId, "Tokens");
            Assert.IsTrue(fetchedLease.Tensor.TryGet<Memory<int>>(out Memory<int> fetched));

            Assert.AreEqual(4, fetched.Length);
            Assert.AreEqual(13, fetched.Span[0]);
        }

        [TestMethod]
        public void RequestMemory_SameKeyDifferentSessions_ReturnsDifferentBuffers()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid firstSessionId = Guid.NewGuid();
            Guid secondSessionId = Guid.NewGuid();

            RuntimeTensorLease firstLease = memoryManager.RequestMemory<int>(firstSessionId, "LayerKeyCache:0", 4);
            RuntimeTensorLease secondLease = memoryManager.RequestMemory<int>(secondSessionId, "LayerKeyCache:0", 4);
            Assert.IsTrue(firstLease.Tensor.TryGet<Memory<int>>(out Memory<int> firstBuffer));
            Assert.IsTrue(secondLease.Tensor.TryGet<Memory<int>>(out Memory<int> secondBuffer));

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

            bool found = memoryManager.TryGetMemory(sessionId, "LayerValueCache:2", out RuntimeTensorLease? lease);

            Assert.IsFalse(found);
            Assert.IsNull(lease);
        }

        [TestMethod]
        public void Release_SpecificKey_PreservesOtherKeys()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();
            memoryManager.RequestMemory<int>(sessionId, "LayerKeyCache:0", 4);
            memoryManager.RequestMemory<int>(sessionId, "LayerKeyCache:1", 4);

            memoryManager.Release(sessionId, "LayerKeyCache:0");

            Assert.IsFalse(memoryManager.TryGetMemory(sessionId, "LayerKeyCache:0", out _));
            Assert.IsTrue(memoryManager.TryGetMemory(sessionId, "LayerKeyCache:1", out RuntimeTensorLease? remainingLease));
            Assert.IsNotNull(remainingLease);
            Assert.IsTrue(remainingLease.Tensor.TryGet<Memory<int>>(out Memory<int> remainingBuffer));
            Assert.AreEqual(4, remainingBuffer.Length);
        }
    }
}
