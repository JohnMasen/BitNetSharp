namespace BitNetSharp
{
    public sealed class RuntimeTensorLease : IDisposable
    {
        private readonly IDisposable leaseHandle;
        private bool disposed;

        public RuntimeTensor Tensor { get; }

        public RuntimeTensorLease(RuntimeTensor tensor, IDisposable leaseHandle)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ArgumentNullException.ThrowIfNull(leaseHandle);

            this.leaseHandle = leaseHandle;
            Tensor = tensor;
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            leaseHandle.Dispose();
            disposed = true;
        }
    }
}
