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

            Assert.AreEqual(TestInferenceConfigs.CpuBackend, provider.Backend);
        }

        [TestMethod]
        public void CPUTensorProvider_Backend_ReturnsStringName()
        {
            IOPProvider provider = new CPUTensorOPProvider();

            Assert.AreEqual(TestInferenceConfigs.TensorBackend, provider.Backend);
        }

        [TestMethod]
        public void CPUSimdProvider_Backend_ReturnsStringName()
        {
            IOPProvider provider = new CPUSimdOPProvider();

            Assert.AreEqual(TestInferenceConfigs.SimdBackend, provider.Backend);
        }
    }
}
