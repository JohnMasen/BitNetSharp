namespace BitNetSharp.Core
{
    internal static class RuntimeTensorBufferHelper
    {
        internal static ReadOnlyMemory<T> GetReadOnlyMemory<T>(RuntimeTensor tensor, string tensorName)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ArgumentException.ThrowIfNullOrWhiteSpace(tensorName);

            if (tensor.TryGet<ReadOnlyMemory<T>>(out ReadOnlyMemory<T> buffer))
            {
                return buffer;
            }

            throw new InvalidOperationException($"Runtime tensor '{tensorName}' does not expose readable '{typeof(T)}' memory.");
        }

        internal static Memory<T> GetMemory<T>(RuntimeTensor tensor, string tensorName)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ArgumentException.ThrowIfNullOrWhiteSpace(tensorName);

            if (tensor.TryGet<Memory<T>>(out Memory<T> buffer))
            {
                return buffer;
            }

            throw new InvalidOperationException($"Runtime tensor '{tensorName}' does not expose writable '{typeof(T)}' memory.");
        }
    }
}
