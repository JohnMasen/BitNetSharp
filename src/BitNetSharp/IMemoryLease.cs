namespace BitNetSharp
{
    /// <summary>
    /// Represents a one-shot, disposable rental of a memory block managed by a
    /// <see cref="BitNetMemoryManager"/> or a backend-specific memory provider.
    /// Disposing the lease returns the underlying memory to its pool (or otherwise
    /// releases backend resources). The lease is intended to be consumed with a
    /// <c>using</c> statement.
    /// </summary>
    public interface IMemoryLease : IDisposable
    {
        /// <summary>
        /// Gets the element type of the leased memory (e.g. <c>typeof(float)</c>).
        /// </summary>
        Type ElementType { get; }

        /// <summary>
        /// Gets the number of elements (of <see cref="ElementType"/>) covered by this lease.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Returns the underlying memory wrapped as the requested container type
        /// (for example <see cref="Memory{T}"/> on a host backend, or a
        /// backend-specific handle on a GPU/NPU backend).
        /// </summary>
        /// <typeparam name="T">The container type to obtain.</typeparam>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the lease does not support the requested container type.
        /// </exception>
        /// <exception cref="ObjectDisposedException">
        /// Thrown when the lease has already been disposed.
        /// </exception>
        T GetMemoryObject<T>();
    }
}
