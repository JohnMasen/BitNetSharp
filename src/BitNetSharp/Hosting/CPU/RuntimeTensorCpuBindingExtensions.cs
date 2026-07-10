using BitNetSharp.Hosting;

namespace BitNetSharp.Hosting.CPU
{
    /// <summary>
    /// Creates zero-copy runtime tensor bindings over CPU memory.
    /// </summary>
    public static class RuntimeTensorCpuBindingExtensions
    {
        /// <summary>
        /// Binds writable CPU memory to a runtime tensor without transferring ownership.
        /// </summary>
        public static RuntimeTensor AsWritableRuntimeTensor<T>(this Memory<T> buffer, string name, params int[] shape)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(shape);
            RuntimeTensor.ValidateBindingShape(buffer.Length, shape);
            return RuntimeTensor.CreateWritable(name, buffer, shape);
        }

        /// <summary>
        /// Binds readonly CPU memory to a runtime tensor without transferring ownership.
        /// </summary>
        public static RuntimeTensor AsReadOnlyRuntimeTensor<T>(this ReadOnlyMemory<T> buffer, string name, params int[] shape)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(shape);
            RuntimeTensor.ValidateBindingShape(buffer.Length, shape);
            return RuntimeTensor.CreateReadOnly(name, buffer, shape);
        }

        /// <summary>
        /// Binds writable CPU memory leased from a memory manager to a runtime tensor without transferring ownership.
        /// </summary>
        public static RuntimeTensor AsWritableRuntimeTensor<T>(this IMemoryLease lease, string name, params int[] shape)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(lease);
            ValidateLeaseElementType<T>(lease);
            return RuntimeTensor.CreateWritableBinding(name, lease.Length, shape, () => lease.GetMemory<T>());
        }

        /// <summary>
        /// Binds readonly CPU memory leased from a memory manager to a runtime tensor without transferring ownership.
        /// </summary>
        public static RuntimeTensor AsReadOnlyRuntimeTensor<T>(this IMemoryLease lease, string name, params int[] shape)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(lease);
            ValidateLeaseElementType<T>(lease);
            return RuntimeTensor.CreateReadOnlyBinding(name, lease.Length, shape, () => lease.GetReadOnlyMemory<T>());
        }

        private static void ValidateLeaseElementType<T>(IMemoryLease lease)
            where T : unmanaged
        {
            if (lease.ElementType != typeof(T))
            {
                throw new InvalidOperationException($"Memory lease '{lease.Tag ?? "<unnamed>"}' stores '{lease.ElementType}', not '{typeof(T)}'.");
            }
        }
    }
}
