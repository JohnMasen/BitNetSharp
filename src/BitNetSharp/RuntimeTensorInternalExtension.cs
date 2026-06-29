using BitNetSharp.Hosting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BitNetSharp
{
    internal static class RuntimeTensorInternalExtension
    {
        internal static RuntimeTensor WrapToTensor<T>(this Memory<T> memory,string namePrefix=null,int? length=null)
        {
            string name = $"wrap_{namePrefix??string.Empty}_{Guid.NewGuid()}";
            int memoryLength = length ?? memory.Length;
            if (length.HasValue)
            {
                ArgumentOutOfRangeException.ThrowIfGreaterThan(length.Value,memory.Length,nameof(length));
                memory = memory.Slice(0,length.Value);
            }
            return new RuntimeTensor(name, typeof(T), [length.Value], false,
                requestedType =>
                {
                    if (requestedType == typeof(Memory<T>))
                    {
                        return (true, memory);
                    }

                    if (requestedType == typeof(ReadOnlyMemory<T>))
                    {
                        return (true, (ReadOnlyMemory<T>)memory);
                    }

                    return (false, null);
                },
                (elementType, source) =>
                {
                    if (elementType != typeof(T) || source is not ReadOnlyMemory<T> typedSource)
                    {
                        return false;
                    }

                    if (typedSource.Length > memory.Length)
                    {
                        throw new ArgumentException($"Source length for runtime tensor '{name}' exceeds the allocated buffer.", nameof(source));
                    }

                    typedSource.Span.CopyTo(memory.Span);
                    return true;
                }
                );
        }
        internal static RuntimeTensor WrapToReadonlyTensor<T>(this ReadOnlyMemory<T> memory,string namePrefix= null, int? length = null)
        {
            string name = $"wrap_{namePrefix ?? string.Empty}_{Guid.NewGuid()}";
            int memoryLength = length ?? memory.Length;
            if (length.HasValue)
            {
                memory = memory.Slice(0, length.Value);
            }
            return new RuntimeTensor(name, typeof(T), [length.Value], true,
                requestedType =>
                {
                    if (requestedType == typeof(ReadOnlyMemory<T>))
                    {
                        return (true, memory);
                    }

                    return (false, null);
                }
                );
        }
    }
}

