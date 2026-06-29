using BitNetSharp.Hosting;
namespace BitNetSharp.Hosting.CPU
{
    internal static class MemoryLeaseExtensions
    {
        public static Memory<T> GetMemory<T>(this IMemoryLease lease)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(lease);

            if (lease.ElementType != typeof(T))
            {
                throw new InvalidOperationException($"Memory lease '{lease.Tag ?? "<unnamed>"}' stores '{lease.ElementType}', not '{typeof(T)}'.");
            }

            return lease.GetMemoryObject<Memory<T>>();
        }

        public static ReadOnlyMemory<T> GetReadOnlyMemory<T>(this IMemoryLease lease)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(lease);

            if (lease.ElementType != typeof(T))
            {
                throw new InvalidOperationException($"Memory lease '{lease.Tag ?? "<unnamed>"}' stores '{lease.ElementType}', not '{typeof(T)}'.");
            }

            return lease.GetMemoryObject<ReadOnlyMemory<T>>();
        }
    }
}

