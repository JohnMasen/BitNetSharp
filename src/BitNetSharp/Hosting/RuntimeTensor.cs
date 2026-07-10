namespace BitNetSharp.Hosting
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
            this.shape = ValidateShape(shape);
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
        // TODO: Add a backend-neutral factory so non-CPU tensors can expose backend-specific buffer handles.
        public static RuntimeTensor CreateReadOnly<T>(string name, ReadOnlyMemory<T> buffer, IEnumerable<int> shape)
            where T : unmanaged
        {
            int[] dimensions = ValidateShape(shape);

            return new RuntimeTensor(
                name,
                typeof(T),
                dimensions,
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

        internal static RuntimeTensor CreateWritableBinding<T>(string name, int length, IEnumerable<int> shape, Func<Memory<T>> getMemory)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(getMemory);

            int[] dimensions = ValidateShape(shape);
            ValidateBufferLength(length, dimensions);
            return new RuntimeTensor(
                name,
                typeof(T),
                dimensions,
                isReadOnly: false,
                requestedType =>
                {
                    Memory<T> buffer = getMemory();
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

                    Memory<T> buffer = getMemory();
                    if (typedSource.Length > buffer.Length)
                    {
                        throw new ArgumentException($"Source length for runtime tensor '{name}' exceeds the allocated buffer.", nameof(source));
                    }

                    typedSource.Span.CopyTo(buffer.Span);
                    return true;
                });
        }

        internal static RuntimeTensor CreateReadOnlyBinding<T>(string name, int length, IEnumerable<int> shape, Func<ReadOnlyMemory<T>> getMemory)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(getMemory);

            int[] dimensions = ValidateShape(shape);
            ValidateBufferLength(length, dimensions);
            return new RuntimeTensor(
                name,
                typeof(T),
                dimensions,
                isReadOnly: true,
                requestedType => requestedType == typeof(ReadOnlyMemory<T>)
                    ? (true, getMemory())
                    : (false, null));
        }

        internal static void ValidateBindingShape(int bufferLength, IEnumerable<int> shape)
        {
            ArgumentNullException.ThrowIfNull(shape);
            ValidateBufferLength(bufferLength, ValidateShape(shape));
        }

        /// <summary>
        /// Creates a writable runtime tensor bound to an existing host buffer.
        /// </summary>
        public static RuntimeTensor CreateWritable<T>(string name, Memory<T> buffer, IEnumerable<int> shape)
            where T : unmanaged
        {
            int[] dimensions = ValidateShape(shape);

            return new RuntimeTensor(
                name,
                typeof(T),
                dimensions,
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

        private static int[] ValidateShape(IEnumerable<int> shape)
        {
            int[] dimensions = shape.ToArray();
            if (dimensions.Length == 0)
            {
                throw new ArgumentException("Runtime tensor shape must contain at least one dimension.", nameof(shape));
            }

            for (int index = 0; index < dimensions.Length; index++)
            {
                if (dimensions[index] <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(shape), "Runtime tensor dimensions must be positive.");
                }
            }

            return dimensions;
        }

        private static void ValidateBufferLength(int bufferLength, IReadOnlyList<int> dimensions)
        {
            if (bufferLength <= 0)
            {
                throw new ArgumentException("Runtime tensor buffers must not be empty.", nameof(bufferLength));
            }

            int elementCount = 1;
            for (int index = 0; index < dimensions.Count; index++)
            {
                elementCount = checked(elementCount * dimensions[index]);
            }

            if (elementCount != bufferLength)
            {
                throw new ArgumentException("Runtime tensor shape does not match the bound buffer length.", nameof(dimensions));
            }
        }
    }
}

