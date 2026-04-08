using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;

namespace BitNetSharp
{
    public class BitNetMemoryManager : IDisposable
    {
        private readonly Dictionary<Guid, Dictionary<string, (IDisposable obj, int requestedLength)>> memorySessions = new();
        private bool disposed;

        /// <summary>
        /// Gets a previously requested pooled memory block for the specified session and key.
        /// </summary>
        public Memory<T> GetMemory<T>(Guid id, string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (TryGetMemory(id, key, out Memory<T> memory))
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

            if (!memorySessions.TryGetValue(id, out var session))
            {
                session = new();
                memorySessions.Add(id, session);
            }

            if (session.TryGetValue(key, out (IDisposable obj, int requestedLength) existingMemoryEntry))
            {
                if (existingMemoryEntry.obj is IMemoryOwner<T> existingMemoryOwner && existingMemoryOwner.Memory.Length >= size)
                {
                    session[key] = (existingMemoryOwner, size);
                    return existingMemoryOwner.Memory.Slice(0, size);
                }

                existingMemoryEntry.obj.Dispose();
            }

            IMemoryOwner<T> memoryOwner = MemoryPool<T>.Shared.Rent(size);
            session[key] = (memoryOwner, size);
            return memoryOwner.Memory.Slice(0, size);
        }

        /// <summary>
        /// Tries to get a previously requested pooled memory block for the specified session and key.
        /// </summary>
        public bool TryGetMemory<T>(Guid id, string key, out Memory<T> memory) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (!memorySessions.TryGetValue(id, out var session))
            {
                memory = default;
                return false;
            }

            if (!session.TryGetValue(key, out var memoryEntry))
            {
                memory = default;
                return false;
            }

            IMemoryOwner<T>? memoryOwner = memoryEntry.obj as IMemoryOwner<T>;
            if (memoryOwner is null)
            {
                throw new InvalidOperationException($"Memory management error, invalid memory item {key}, type should be IMemoryOwner, current={memoryEntry.obj.GetType()}");
            }

            memory = memoryOwner.Memory.Slice(0, memoryEntry.requestedLength);
            return true;
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

            if (memorySessions.TryGetValue(id, out var memorySession) && memorySession.Remove(key, out (IDisposable obj, int requestedLength) memoryEntry))
            {
                memoryEntry.obj.Dispose();
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
                .SelectMany(session => session.Values.Select(memoryEntry => memoryEntry.obj)))
            {
                disposable.Dispose();
            }

            memorySessions.Clear();

            disposed = true;
            GC.SuppressFinalize(this);
        }

        private void ReleaseCore(Guid id)
        {
            if (memorySessions.TryGetValue(id, out var memorySession))
            {
                foreach (IDisposable disposable in memorySession.Values.Select(memoryEntry => memoryEntry.obj))
                {
                    disposable.Dispose();
                }

                memorySessions.Remove(id);
            }
        }
    }
}
