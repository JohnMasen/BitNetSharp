using BitNetSharp.Hosting;
using BitNetSharp.Hosting.CPU;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class RuntimeMemoryManagerTests
    {
        [TestMethod]
        public void GetOrCreateRuntimeTensor_WhenRequestedAgain_UsesSameSessionStorage()
        {
            using IRuntimeMemoryManager memoryManager = new BitNetMemoryManager();
            Guid sessionId = Guid.NewGuid();

            RuntimeTensor first = memoryManager.GetOrCreateRuntimeTensor<float>(sessionId, "Embedding", 3);
            first.GetMemory<float>().Span[1] = 2.5f;
            RuntimeTensor second = memoryManager.GetOrCreateRuntimeTensor<float>(sessionId, "Embedding", 3);

            Assert.AreEqual(2.5f, second.GetMemory<float>().Span[1]);
        }

        [TestMethod]
        public void GetOrCreateRuntimeTensor_WhenSessionsDiffer_UsesIsolatedStorage()
        {
            using IRuntimeMemoryManager memoryManager = new BitNetMemoryManager();

            RuntimeTensor first = memoryManager.GetOrCreateRuntimeTensor<int>(Guid.NewGuid(), "Embedding", 2);
            RuntimeTensor second = memoryManager.GetOrCreateRuntimeTensor<int>(Guid.NewGuid(), "Embedding", 2);
            first.GetMemory<int>().Span[0] = 11;
            second.GetMemory<int>().Span[0] = 29;

            Assert.AreEqual(11, first.GetMemory<int>().Span[0]);
            Assert.AreEqual(29, second.GetMemory<int>().Span[0]);
        }

        [TestMethod]
        public void RentRuntimeTensor_ReturnsWritableTensorWithRequestedMetadata()
        {
            using IRuntimeMemoryManager memoryManager = new BitNetMemoryManager();
            using IRuntimeTensorLease lease = memoryManager.RentRuntimeTensor<float>("AttentionScore", "Attention", 2, 3);

            RuntimeTensor tensor = lease.Tensor;

            Assert.AreEqual("AttentionScore", tensor.Name);
            Assert.AreEqual(typeof(float), tensor.ElementType);
            CollectionAssert.AreEqual(new[] { 2, 3 }, tensor.Shape.ToArray());
            Assert.IsFalse(tensor.IsReadOnly);
        }

        [TestMethod]
        public void RentRuntimeTensor_WhenWritten_ExposesZeroCopyCpuBuffer()
        {
            using IRuntimeMemoryManager memoryManager = new BitNetMemoryManager();
            using IRuntimeTensorLease lease = memoryManager.RentRuntimeTensor<float>("AttentionScore", "Attention", 3);

            RuntimeTensor tensor = lease.Tensor;
            tensor.CopyFrom(new[] { 1f, 2f, 3f }.AsMemory());

            CollectionAssert.AreEqual(new[] { 1f, 2f, 3f }, tensor.GetMemory<float>().ToArray());
        }

        [TestMethod]
        public void RentRuntimeTensor_WhenDisposed_InvalidatesTensorAccess()
        {
            using IRuntimeMemoryManager memoryManager = new BitNetMemoryManager();
            IRuntimeTensorLease lease = memoryManager.RentRuntimeTensor<float>("AttentionScore", "Attention", 3);
            RuntimeTensor tensor = lease.Tensor;
            lease.Dispose();

            Assert.ThrowsExactly<ObjectDisposedException>(() => tensor.GetMemory<float>());
        }
    }
}
