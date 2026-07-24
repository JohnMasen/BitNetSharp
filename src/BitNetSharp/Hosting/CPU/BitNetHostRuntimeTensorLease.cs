using BitNetSharp.Hosting;

namespace BitNetSharp.Hosting.CPU
{
    internal sealed class BitNetHostRuntimeTensorLease<T> : IRuntimeTensorLease
        where T : unmanaged
    {
        private readonly IMemoryLease memoryLease;
        private readonly RuntimeTensor tensor;
        private bool disposed;

        public BitNetHostRuntimeTensorLease(IMemoryLease memoryLease, string name, int[] shape)
        {
            ArgumentNullException.ThrowIfNull(memoryLease);
            ArgumentNullException.ThrowIfNull(shape);

            this.memoryLease = memoryLease;
            tensor = memoryLease.AsWritableRuntimeTensor<T>(name, shape);
        }

        public RuntimeTensor Tensor
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return tensor;
            }
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            memoryLease.Dispose();
        }
    }
}
