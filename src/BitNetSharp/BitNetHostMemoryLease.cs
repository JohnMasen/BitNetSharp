using System.Buffers;

namespace BitNetSharp
{
    /// <summary>
    /// Default host-memory implementation of <see cref="IMemoryLease"/>. Wraps an
    /// <see cref="IMemoryOwner{T}"/> rented from <see cref="MemoryPool{T}.Shared"/>
    /// and returns it to the pool on <see cref="Dispose"/>.
    /// </summary>
    internal sealed class BitNetHostMemoryLease<T> : IMemoryLease
        where T : unmanaged
    {
        private readonly IMemoryOwner<T> memoryOwner;
        private readonly int length;
        private readonly string? tag;
        private bool disposed;

        public BitNetHostMemoryLease(IMemoryOwner<T> memoryOwner, int length, string? tag = null)
        {
            ArgumentNullException.ThrowIfNull(memoryOwner);
            this.memoryOwner = memoryOwner;
            this.length = length;
            this.tag = tag;
        }

        public string? Tag => tag;

        public Type ElementType => typeof(T);

        public int Length => length;

        public TContainer GetMemoryObject<TContainer>()
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (typeof(TContainer) == typeof(Memory<T>))
            {
                Memory<T> memory = memoryOwner.Memory.Slice(0, length);
                return (TContainer)(object)memory;
            }

            if (typeof(TContainer) == typeof(ReadOnlyMemory<T>))
            {
                ReadOnlyMemory<T> memory = memoryOwner.Memory.Slice(0, length);
                return (TContainer)(object)memory;
            }

            throw new InvalidOperationException(
                $"Container type '{typeof(TContainer)}' is not supported by {nameof(BitNetHostMemoryLease<T>)} with element type '{typeof(T)}'.");
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            memoryOwner.Dispose();
        }
    }
}
