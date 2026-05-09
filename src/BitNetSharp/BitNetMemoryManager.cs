using System.Buffers;

namespace BitNetSharp
{
    public class BitNetMemoryManager : IDisposable
    {
        private readonly Dictionary<Guid, Dictionary<string, MemoryEntry>> memorySessions = new();
        private bool disposed;

        private sealed record MemoryEntry(IDisposable Owner, int RequestedLength, Type ElementType, int ElementSizeInBytes);

        /// <summary>
        /// Gets a previously requested memory block for the specified session and key.
        /// </summary>
        public Memory<T> GetMemory<T>(Guid id, string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (TryGetMemory<T>(id, key, out Memory<T> memory))
            {
                return memory;
            }

            if (memorySessions.ContainsKey(id))
            {
                throw new InvalidOperationException($"Requested memory not found, key={key}");
            }

            throw new InvalidOperationException($"Requested session not found, id={id}");
        }

        /// <summary>
        /// Allocates a pooled memory block for the specified session and key.
        /// </summary>
        public Memory<T> RequestMemory<T>(Guid id, string key, int size) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (size <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size));
            }

            if (!memorySessions.TryGetValue(id, out Dictionary<string, MemoryEntry>? session))
            {
                session = new();
                memorySessions.Add(id, session);
            }

            if (session.TryGetValue(key, out MemoryEntry? existingMemoryEntry))
            {
                if (existingMemoryEntry.Owner is IMemoryOwner<T> existingMemoryOwner)
                {
                    if (existingMemoryOwner.Memory.Length >= size)
                    {
                        MemoryEntry updatedEntry = CreateMemoryEntry(existingMemoryOwner, size);
                        session[key] = updatedEntry;
                        return GetMemory(existingMemoryOwner, updatedEntry.RequestedLength);
                    }

                    throw new InvalidOperationException($"Request memory size is larger than existing one,key={key},request size={size},existingSize={existingMemoryOwner.Memory.Length}");
                }

                throw new InvalidOperationException("Invalid memory object type,expected IMemoryOwner<T>");
            }

            IMemoryOwner<T> memoryOwner = MemoryPool<T>.Shared.Rent(size);
            MemoryEntry entry = CreateMemoryEntry(memoryOwner, size);
            session[key] = entry;
            return GetMemory(memoryOwner, entry.RequestedLength);
        }

        /// <summary>
        /// Tries to get a previously requested memory block for the specified session and key.
        /// </summary>
        public bool TryGetMemory<T>(Guid id, string key, out Memory<T> memory) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (!memorySessions.TryGetValue(id, out Dictionary<string, MemoryEntry>? session))
            {
                memory = Memory<T>.Empty;
                return false;
            }

            if (!session.TryGetValue(key, out MemoryEntry? memoryEntry))
            {
                memory = Memory<T>.Empty;
                return false;
            }

            if (memoryEntry.Owner is IMemoryOwner<T> memoryOwner)
            {
                memory = GetMemory(memoryOwner, memoryEntry.RequestedLength);
                return true;
            }

            throw new InvalidOperationException("Invalid memory object type,expected IMemoryOwner<T>");
        }

        /// <summary>
        /// Gets a readonly snapshot of the currently tracked memory allocations.
        /// </summary>
        public BitNetMemoryStatistics GetStatistics()
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            List<BitNetMemoryAllocationSnapshot> allocations = new();
            long estimatedTotalBytes = 0;
            foreach ((Guid sessionId, Dictionary<string, MemoryEntry> session) in memorySessions)
            {
                foreach ((string key, MemoryEntry entry) in session)
                {
                    long estimatedBytes = (long)entry.RequestedLength * entry.ElementSizeInBytes;
                    allocations.Add(new BitNetMemoryAllocationSnapshot(sessionId, key, entry.ElementType, entry.RequestedLength, estimatedBytes));
                    estimatedTotalBytes += estimatedBytes;
                }
            }

            return new BitNetMemoryStatistics(allocations.Count, estimatedTotalBytes, allocations);
        }

        /// <summary>
        /// Releases all memory blocks for the specified session.
        /// </summary>
        public void Release(Guid id)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            ReleaseCore(id);
        }

        /// <summary>
        /// Releases the memory block for the specified session and key.
        /// </summary>
        public void Release(Guid id, string key)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (memorySessions.TryGetValue(id, out Dictionary<string, MemoryEntry>? memorySession)
                && memorySession.Remove(key, out MemoryEntry? memoryEntry))
            {
                memoryEntry.Owner.Dispose();

                if (memorySession.Count == 0)
                {
                    memorySessions.Remove(id);
                }
            }
        }

        /// <summary>
        /// Releases all tracked sessions and pooled memory owned by this manager.
        /// </summary>
        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            foreach (IDisposable disposable in memorySessions.Values
                .SelectMany(static session => session.Values.Select(memoryEntry => memoryEntry.Owner)))
            {
                disposable.Dispose();
            }

            memorySessions.Clear();

            disposed = true;
            GC.SuppressFinalize(this);
        }

        private void ReleaseCore(Guid id)
        {
            if (memorySessions.TryGetValue(id, out Dictionary<string, MemoryEntry>? memorySession))
            {
                foreach (IDisposable disposable in memorySession.Values.Select(memoryEntry => memoryEntry.Owner))
                {
                    disposable.Dispose();
                }

                memorySessions.Remove(id);
            }
        }

        private static Memory<T> GetMemory<T>(IMemoryOwner<T> memoryOwner, int requestedLength)
            where T : unmanaged
        {
            return memoryOwner.Memory.Slice(0, requestedLength);
        }

        private static MemoryEntry CreateMemoryEntry<T>(IMemoryOwner<T> memoryOwner, int requestedLength)
            where T : unmanaged
        {
            return new MemoryEntry(memoryOwner, requestedLength, typeof(T), GetElementSizeInBytes<T>());
        }

        private static int GetElementSizeInBytes<T>() where T : unmanaged
        {
            return typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte) ? sizeof(byte)
                : typeof(T) == typeof(short) || typeof(T) == typeof(ushort) || typeof(T) == typeof(Half) ? sizeof(short)
                : typeof(T) == typeof(int) || typeof(T) == typeof(uint) || typeof(T) == typeof(float) ? sizeof(int)
                : typeof(T) == typeof(long) || typeof(T) == typeof(ulong) || typeof(T) == typeof(double) ? sizeof(long)
                : typeof(T) == typeof(bool) ? sizeof(byte)
                : throw new NotSupportedException($"Element type '{typeof(T)}' is not supported for memory statistics.");
        }
    }
}
