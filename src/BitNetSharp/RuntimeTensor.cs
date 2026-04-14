namespace BitNetSharp
{
    public sealed class RuntimeTensor
    {
        private readonly Func<Type, (bool Success, object? Value)> tryGetCore;
        private readonly Func<Type, object, bool>? copyFromCore;
        private readonly int[] shape;

        internal RuntimeTensor(
            string name,
            Type elementType,
            IEnumerable<int> shape,
            bool isReadOnly,
            Func<Type, (bool Success, object? Value)> tryGetCore,
            Func<Type, object, bool>? copyFromCore = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            ArgumentNullException.ThrowIfNull(elementType);
            ArgumentNullException.ThrowIfNull(shape);
            ArgumentNullException.ThrowIfNull(tryGetCore);

            this.tryGetCore = tryGetCore;
            this.copyFromCore = copyFromCore;
            this.shape = shape.ToArray();
            Name = name;
            ElementType = elementType;
            IsReadOnly = isReadOnly;
        }

        public string Name { get; }

        public Type ElementType { get; }

        public IReadOnlyList<int> Shape => shape;

        public bool IsReadOnly { get; }

        /// <summary>
        /// Tries to get the requested concrete buffer object from the runtime tensor.
        /// </summary>
        public bool TryGet<T>(out T value)
        {
            (bool success, object? rawValue) = tryGetCore(typeof(T));
            if (success && rawValue is T typedValue)
            {
                value = typedValue;
                return true;
            }

            value = default!;
            return false;
        }

        /// <summary>
        /// Copies host data into the writable runtime tensor.
        /// </summary>
        public void CopyFrom<T>(Memory<T> source)
            where T : unmanaged
        {
            CopyFrom((ReadOnlyMemory<T>)source);
        }

        /// <summary>
        /// Copies host data into the writable runtime tensor.
        /// </summary>
        public void CopyFrom<T>(ReadOnlyMemory<T> source)
            where T : unmanaged
        {
            if (copyFromCore is null || !copyFromCore(typeof(T), source))
            {
                throw new InvalidOperationException($"Runtime tensor '{Name}' does not support copying data from '{typeof(T)}'.");
            }
        }

        /// <summary>
        /// Creates a readonly runtime tensor bound to an existing host buffer.
        /// </summary>
        public static RuntimeTensor CreateReadOnly<T>(string name, ReadOnlyMemory<T> buffer, IEnumerable<int> shape)
            where T : unmanaged
        {
            return new RuntimeTensor(
                name,
                typeof(T),
                shape,
                isReadOnly: true,
                requestedType =>
                {
                    if (requestedType == typeof(ReadOnlyMemory<T>))
                    {
                        return (true, buffer);
                    }

                    return (false, null);
                });
        }

        /// <summary>
        /// Creates a writable runtime tensor bound to an existing host buffer.
        /// </summary>
        public static RuntimeTensor CreateWritable<T>(string name, Memory<T> buffer, IEnumerable<int> shape)
            where T : unmanaged
        {
            return new RuntimeTensor(
                name,
                typeof(T),
                shape,
                isReadOnly: false,
                requestedType =>
                {
                    if (requestedType == typeof(Memory<T>))
                    {
                        return (true, buffer);
                    }

                    if (requestedType == typeof(ReadOnlyMemory<T>))
                    {
                        return (true, (ReadOnlyMemory<T>)buffer);
                    }

                    return (false, null);
                },
                (elementType, source) =>
                {
                    if (elementType != typeof(T) || source is not ReadOnlyMemory<T> typedSource)
                    {
                        return false;
                    }

                    if (typedSource.Length > buffer.Length)
                    {
                        throw new ArgumentException($"Source length for runtime tensor '{name}' exceeds the allocated buffer.", nameof(source));
                    }

                    typedSource.Span.CopyTo(buffer.Span);
                    return true;
                });
        }
    }
}
