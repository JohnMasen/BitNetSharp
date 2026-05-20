namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class MemoryManagerLeaseTests
    {
        [TestMethod]
        public void GetMemoryLease_ReturnsRequestedLength_ForFloatMemory()
        {
            using var memoryManager = new BitNetMemoryManager();

            using IMemoryLease lease = memoryManager.GetMemoryLease<float>(128);
            Memory<float> buffer = lease.GetMemoryObject<Memory<float>>();

            Assert.AreEqual(128, buffer.Length);
            Assert.AreEqual(128, lease.Length);
            Assert.AreEqual(typeof(float), lease.ElementType);

            buffer.Span[0] = 1.5f;
            buffer.Span[127] = 2.5f;
            Assert.AreEqual(1.5f, buffer.Span[0]);
            Assert.AreEqual(2.5f, buffer.Span[127]);
        }

        [TestMethod]
        public void GetMemoryObject_SupportsReadOnlyMemory()
        {
            using var memoryManager = new BitNetMemoryManager();

            using IMemoryLease lease = memoryManager.GetMemoryLease<int>(16);
            ReadOnlyMemory<int> buffer = lease.GetMemoryObject<ReadOnlyMemory<int>>();

            Assert.AreEqual(16, buffer.Length);
        }

        [TestMethod]
        public void GetMemoryObject_UnsupportedContainerType_Throws()
        {
            using var memoryManager = new BitNetMemoryManager();

            using IMemoryLease lease = memoryManager.GetMemoryLease<float>(4);

            Assert.ThrowsExactly<InvalidOperationException>(() => lease.GetMemoryObject<float[]>());
        }

        [TestMethod]
        public void GetMemoryObject_ElementTypeMismatch_Throws()
        {
            using var memoryManager = new BitNetMemoryManager();

            using IMemoryLease lease = memoryManager.GetMemoryLease<float>(4);

            Assert.ThrowsExactly<InvalidOperationException>(() => lease.GetMemoryObject<Memory<int>>());
        }

        [TestMethod]
        public void Dispose_IsIdempotent()
        {
            using var memoryManager = new BitNetMemoryManager();

            IMemoryLease lease = memoryManager.GetMemoryLease<float>(4);
            lease.Dispose();
            lease.Dispose();
        }

        [TestMethod]
        public void GetMemoryObject_AfterDispose_Throws()
        {
            using var memoryManager = new BitNetMemoryManager();

            IMemoryLease lease = memoryManager.GetMemoryLease<float>(4);
            lease.Dispose();

            Assert.ThrowsExactly<ObjectDisposedException>(() => lease.GetMemoryObject<Memory<float>>());
        }

        [TestMethod]
        public void GetMemoryLease_DoesNotAppearInStatistics()
        {
            using var memoryManager = new BitNetMemoryManager();

            using IMemoryLease lease = memoryManager.GetMemoryLease<float>(64);

            BitNetMemoryStatistics statistics = memoryManager.GetStatistics();
            Assert.AreEqual(0, statistics.AllocationCount);
            Assert.AreEqual(0L, statistics.EstimatedTotalBytes);
        }

        [TestMethod]
        public void GetMemoryLease_InvalidSize_Throws()
        {
            using var memoryManager = new BitNetMemoryManager();

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memoryManager.GetMemoryLease<float>(0));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memoryManager.GetMemoryLease<float>(-1));
        }

        [TestMethod]
        public void GetMemoryLease_AfterManagerDisposed_Throws()
        {
            var memoryManager = new BitNetMemoryManager();
            memoryManager.Dispose();

            Assert.ThrowsExactly<ObjectDisposedException>(() => memoryManager.GetMemoryLease<float>(4));
        }
    }
}
