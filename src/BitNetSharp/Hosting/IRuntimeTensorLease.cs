namespace BitNetSharp.Hosting
{
    /// <summary>
    /// Represents a disposable rental of a temporary runtime tensor.
    /// </summary>
    public interface IRuntimeTensorLease : IDisposable
    {
        /// <summary>
        /// Gets the temporary tensor while this lease is active.
        /// </summary>
        RuntimeTensor Tensor { get; }
    }
}
