namespace BitNetSharp.Hosting
{
    /// <summary>
    /// Allocates backend-specific storage for session-owned runtime tensors and temporary workspace tensors.
    /// </summary>
    public interface IRuntimeMemoryManager : IDisposable
    {
        /// <summary>
        /// Gets or creates a writable runtime tensor owned by the specified session.
        /// </summary>
        RuntimeTensor GetOrCreateRuntimeTensor<T>(Guid sessionId, string name, params int[] shape)
            where T : unmanaged;

        /// <summary>
        /// Rents a writable runtime tensor for temporary workspace use.
        /// </summary>
        IRuntimeTensorLease RentRuntimeTensor<T>(string name, string? tag = null, params int[] shape)
            where T : unmanaged;

        /// <summary>
        /// Releases all persistent runtime tensors owned by the specified session.
        /// </summary>
        void Release(Guid sessionId);

        /// <summary>
        /// Gets a readonly snapshot of persistent runtime tensor allocations.
        /// </summary>
        BitNetMemoryStatistics GetStatistics();
    }
}
