using BitNetSharp.Core;
using BitNetSharp.Nodes;

namespace BitNetSharp.Tests
{
    [TestClass]
    public sealed class OPProviderBackendTests
    {
        [TestMethod]
        public void CPUDefaultProvider_Backend_ReturnsStringName()
        {
            IOPProvider provider = new CPUDefaultOPProvider();

            Assert.AreEqual(InferenceBackendNames.CPU, provider.Backend);
        }

        [TestMethod]
        public void CPUTensorProvider_Backend_ReturnsStringName()
        {
            IOPProvider provider = new CPUTensorOPProvider();

            Assert.AreEqual(InferenceBackendNames.Tensor, provider.Backend);
        }

        [TestMethod]
        public void CPUSimdProvider_Backend_ReturnsStringName()
        {
            IOPProvider provider = new CPUSimdOPProvider();

            Assert.AreEqual(InferenceBackendNames.SIMD, provider.Backend);
        }
    }
}
