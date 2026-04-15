namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class ConsoleMemoryReportTests
    {
        [TestMethod]
        public void GetStatistics_ReturnsTrackedAllocations()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();
            memoryManager.RequestMemory<float>(sessionId, "Embedding", 16);
            memoryManager.RequestMemory<int>(sessionId, "Tokens", 4);

            BitNetMemoryStatistics statistics = memoryManager.GetStatistics();

            Assert.AreEqual(2, statistics.AllocationCount);
            Assert.AreEqual(80L, statistics.EstimatedTotalBytes);
        }

        [TestMethod]
        public void GetStatistics_TracksElementTypeAndBytes()
        {
            using var memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();
            memoryManager.RequestMemory<float>(sessionId, "LayerKeyCache:0", 16);

            BitNetMemoryStatistics statistics = memoryManager.GetStatistics();
            BitNetMemoryAllocationSnapshot allocation = statistics.Allocations.Single();

            Assert.AreEqual(typeof(float), allocation.ElementType);
            Assert.AreEqual(64L, allocation.EstimatedBytes);
        }
    }
}
