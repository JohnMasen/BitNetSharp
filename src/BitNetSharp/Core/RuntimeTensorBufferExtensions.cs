namespace BitNetSharp.Core
{
    internal static class RuntimeTensorBufferExtensions
    {
        internal static ReadOnlyMemory<T> GetReadOnlyMemory<T>(this RuntimeTensor tensor)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(tensor);

            if (tensor.TryGet<ReadOnlyMemory<T>>(out ReadOnlyMemory<T> buffer))
            {
                return buffer;
            }

            throw new InvalidOperationException($"Runtime tensor '{tensor.Name}' does not expose readable '{typeof(T)}' memory.");
        }

        internal static Memory<T> GetMemory<T>(this RuntimeTensor tensor)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(tensor);

            if (tensor.TryGet<Memory<T>>(out Memory<T> buffer))
            {
                return buffer;
            }

            throw new InvalidOperationException($"Runtime tensor '{tensor.Name}' does not expose writable '{typeof(T)}' memory.");
        }
    }
}
