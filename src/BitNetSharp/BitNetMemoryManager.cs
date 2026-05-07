using System.Buffers;

namespace BitNetSharp
{
    public class BitNetMemoryManager : IDisposable
    {
        private readonly Dictionary<Guid, Dictionary<string, MemoryEntry>> memorySessions = new();
        private bool disposed;

        private sealed record MemoryEntry(IDisposable Owner, RuntimeTensor Tensor, int RequestedLength, Type ElementType, int ElementSizeInBytes);

        private sealed class NoOpLeaseHandle : IDisposable
        {
            internal static NoOpLeaseHandle Instance { get; } = new();

            public void Dispose()
            {
            }
        }

        /// <summary>
        /// Gets a previously requested tensor lease for the specified session and key.
        /// </summary>
        public RuntimeTensorLease GetMemory(Guid id, string key)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (TryGetMemory(id, key, out RuntimeTensorLease? lease))
            {
                return lease;
            }

            if (memorySessions.ContainsKey(id))
            {
                throw new InvalidOperationException($"Requested memory not found, key={key}");
            }

            throw new InvalidOperationException($"Requested session not found, id={id}");
        }

        /// <summary>
        /// Allocates a pooled tensor lease for the specified session and key.
        /// </summary>
        public RuntimeTensorLease RequestMemory<T>(Guid id, string key, int size) where T : unmanaged
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
                if (existingMemoryEntry.Owner is IMemoryOwner<T> existingMemoryOwner && existingMemoryOwner.Memory.Length >= size)
                {
                    MemoryEntry updatedEntry = CreateMemoryEntry(existingMemoryOwner, key, size);
                    session[key] = updatedEntry;
                    return CreateLease(updatedEntry);
                }

                existingMemoryEntry.Owner.Dispose();
            }

            IMemoryOwner<T> memoryOwner = MemoryPool<T>.Shared.Rent(size);
            MemoryEntry entry = CreateMemoryEntry(memoryOwner, key, size);
            session[key] = entry;
            return CreateLease(entry);
        }

        /// <summary>
        /// Tries to get a previously requested tensor lease for the specified session and key.
        /// </summary>
        public bool TryGetMemory(Guid id, string key, out RuntimeTensorLease? lease)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            if (!memorySessions.TryGetValue(id, out Dictionary<string, MemoryEntry>? session))
            {
                lease = null;
                return false;
            }

            if (!session.TryGetValue(key, out MemoryEntry? memoryEntry))
            {
                lease = null;
                return false;
            }

            lease = CreateLease(memoryEntry);
            return true;
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

        private static RuntimeTensorLease CreateLease(MemoryEntry entry)
        {
            return new RuntimeTensorLease(entry.Tensor, NoOpLeaseHandle.Instance);
        }

        private static MemoryEntry CreateMemoryEntry<T>(IMemoryOwner<T> memoryOwner, string key, int requestedLength)
            where T : unmanaged
        {
            Memory<T> memory = memoryOwner.Memory.Slice(0, requestedLength);
            RuntimeTensor tensor = RuntimeTensor.CreateWritable(key, memory, [requestedLength]);
            return new MemoryEntry(memoryOwner, tensor, requestedLength, typeof(T), GetElementSizeInBytes<T>());
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
