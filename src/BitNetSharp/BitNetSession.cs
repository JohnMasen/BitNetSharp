using BitNetSharp.Models;

namespace BitNetSharp
{
    public class BitNetSession : IDisposable
    {
        private const string TokensKey = nameof(Tokens);
        internal const string EmbeddingKey = nameof(Embedding);
        internal const string RmsNormKey = nameof(RmsNorm);
        internal const string QKVQueryKey = nameof(QKVQuery);
        internal const string QKVKeyKey = nameof(QKVKey);
        internal const string QKVValueKey = nameof(QKVValue);
        internal const string AttentionSubNormKey = nameof(AttentionSubNorm);
        internal const string AttentionOutputKey = nameof(AttentionOutput);
        internal const string FeedForwardInputKey = nameof(FeedForwardInput);
        internal const string FeedForwardNormKey = nameof(FeedForwardNorm);
        internal const string FeedForwardSubNormKey = nameof(FeedForwardSubNorm);
        internal const string FeedForwardOutputKey = nameof(FeedForwardOutput);
        internal const string FinalNormOutputKey = nameof(FinalNormOutput);
        internal const string LogitsKey = nameof(Logits);

        private readonly BitNetModel model;
        private readonly BitNetMemoryManager memoryManager;
        private readonly Dictionary<string, RuntimeTensor> runtimeTensors = new(StringComparer.Ordinal);
        private int currentToken;
        private bool disposed;

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager)
            : this(model, memoryManager, Guid.NewGuid())
        {
        }

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager, Guid id)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(memoryManager);

            this.model = model;
            this.memoryManager = memoryManager;
            Id = id;
            Tokens = Memory<int>.Empty;
            TopKTokenIds = [];
            TopKLogits = [];
        }

        public BitNetModel Model => model;

        public Guid Id { get; }

        /// <summary>
        /// Releases the session state tracked under this session id from the shared memory manager.
        /// </summary>
        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            memoryManager.Release(Id);
            runtimeTensors.Clear();
            disposed = true;
            GC.SuppressFinalize(this);
        }

        public Memory<int> Tokens
        {
            get => GetMemory<int>(TokensKey);
            set => SetMemory(TokensKey, value);
        }

        public int CurrentToken
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return currentToken;
            }
            set
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                currentToken = value;
            }
        }

        public int NextTokenId { get; set; }

        public int ArgmaxTokenId { get; set; }

        public float NextTokenLogit { get; set; }

        public float ArgmaxLogit { get; set; }

        public string? NextTokenStrategy { get; set; }

        public int[] TopKTokenIds { get; set; }

        public float[] TopKLogits { get; set; }

        public RuntimeTensor EmbeddingTensor => GetOrCreateRuntimeTensor(EmbeddingKey);

        public RuntimeTensor RmsNormTensor => GetOrCreateRuntimeTensor(RmsNormKey);

        public RuntimeTensor QKVQueryTensor => GetOrCreateRuntimeTensor(QKVQueryKey);

        public RuntimeTensor QKVKeyTensor => GetOrCreateRuntimeTensor(QKVKeyKey);

        public RuntimeTensor QKVValueTensor => GetOrCreateRuntimeTensor(QKVValueKey);

        public RuntimeTensor AttentionSubNormTensor => GetOrCreateRuntimeTensor(AttentionSubNormKey);

        public RuntimeTensor AttentionOutputTensor => GetOrCreateRuntimeTensor(AttentionOutputKey);

        public RuntimeTensor FeedForwardInputTensor => GetOrCreateRuntimeTensor(FeedForwardInputKey);

        public RuntimeTensor FeedForwardNormTensor => GetOrCreateRuntimeTensor(FeedForwardNormKey);

        public RuntimeTensor FeedForwardSubNormTensor => GetOrCreateRuntimeTensor(FeedForwardSubNormKey);

        public RuntimeTensor FeedForwardOutputTensor => GetOrCreateRuntimeTensor(FeedForwardOutputKey);

        public RuntimeTensor FinalNormOutputTensor => GetOrCreateRuntimeTensor(FinalNormOutputKey);

        public RuntimeTensor LogitsTensor => GetOrCreateRuntimeTensor(LogitsKey);

        /// <summary>
        /// Gets the shared readonly weight tensor with the specified model tensor name.
        /// </summary>
        public RuntimeTensor GetWeightTensor(string name)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return model.GetWeightTensor(name);
        }

        /// <summary>
        /// Gets or creates the mutable runtime tensor for the specified session tensor name.
        /// </summary>
        public RuntimeTensor GetOrCreateRuntimeTensor(string name)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentException.ThrowIfNullOrWhiteSpace(name);

            if (runtimeTensors.TryGetValue(name, out RuntimeTensor? tensor))
            {
                return tensor;
            }

            tensor = CreateRuntimeTensor(name);
            runtimeTensors.Add(name, tensor);
            return tensor;
        }

        public Memory<float> Embedding
        {
            get => GetRuntimeMemory<float>(EmbeddingKey);
            set => CopyToRuntimeTensor<float>(EmbeddingKey, value);
        }

        public Memory<float> RmsNorm
        {
            get => GetRuntimeMemory<float>(RmsNormKey);
            set => CopyToRuntimeTensor<float>(RmsNormKey, value);
        }

        public Memory<float> QKVQuery
        {
            get => GetRuntimeMemory<float>(QKVQueryKey);
            set => CopyToRuntimeTensor<float>(QKVQueryKey, value);
        }

        public Memory<float> QKVKey
        {
            get => GetRuntimeMemory<float>(QKVKeyKey);
            set => CopyToRuntimeTensor<float>(QKVKeyKey, value);
        }

        public Memory<float> QKVValue
        {
            get => GetRuntimeMemory<float>(QKVValueKey);
            set => CopyToRuntimeTensor<float>(QKVValueKey, value);
        }

        public Memory<float> AttentionSubNorm
        {
            get => GetRuntimeMemory<float>(AttentionSubNormKey);
            set => CopyToRuntimeTensor<float>(AttentionSubNormKey, value);
        }

        public Memory<float> AttentionOutput
        {
            get => GetRuntimeMemory<float>(AttentionOutputKey);
            set => CopyToRuntimeTensor<float>(AttentionOutputKey, value);
        }

        public Memory<float> FeedForwardInput => GetRuntimeMemory<float>(FeedForwardInputKey);

        public Memory<float> FeedForwardNorm => GetRuntimeMemory<float>(FeedForwardNormKey);

        public Memory<float> FeedForwardSubNorm => GetRuntimeMemory<float>(FeedForwardSubNormKey);

        public Memory<float> FeedForwardOutput => GetRuntimeMemory<float>(FeedForwardOutputKey);

        public Memory<float> FinalNormOutput => GetRuntimeMemory<float>(FinalNormOutputKey);

        public Memory<float> Logits => GetRuntimeMemory<float>(LogitsKey);

        internal bool HasMemory<T>(string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            return runtimeTensors.TryGetValue(key, out RuntimeTensor? tensor) && tensor.TryGet<Memory<T>>(out _);
        }

        private Memory<T> GetMemory<T>(string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            return memoryManager.TryGetMemory<T>(Id, key, out Memory<T> memory)
                ? memory
                : Memory<T>.Empty;
        }

        private Memory<T> GetRuntimeMemory<T>(string key)
            where T : unmanaged
        {
            RuntimeTensor tensor = GetOrCreateRuntimeTensor(key);
            if (tensor.TryGet<Memory<T>>(out Memory<T> memory))
            {
                return memory;
            }

            throw new InvalidOperationException($"Runtime tensor '{key}' does not expose '{typeof(T)}' memory.");
        }

        private void SetMemory<T>(string key, Memory<T> value) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            WriteMemory(key, value);
        }

        private void CopyToRuntimeTensor<T>(string key, Memory<T> value)
            where T : unmanaged
        {
            RuntimeTensor tensor = GetOrCreateRuntimeTensor(key);
            tensor.CopyFrom<T>(value);
        }

        private RuntimeTensor CreateRuntimeTensor(string name)
        {
            return name switch
            {
                EmbeddingKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                RmsNormKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                QKVQueryKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                QKVKeyKey => CreateRuntimeTensor<float>(name, GetKeyValueProjectionLength()),
                QKVValueKey => CreateRuntimeTensor<float>(name, GetKeyValueProjectionLength()),
                AttentionSubNormKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                AttentionOutputKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                FeedForwardInputKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                FeedForwardNormKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                FeedForwardSubNormKey => CreateRuntimeTensor<float>(name, GetFeedForwardLength()),
                FeedForwardOutputKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                FinalNormOutputKey => CreateRuntimeTensor<float>(name, GetEmbeddingLength()),
                LogitsKey => CreateRuntimeTensor<float>(name, GetVocabularySize()),
                _ => throw new InvalidOperationException($"Unknown runtime tensor '{name}'."),
            };
        }

        private RuntimeTensor CreateRuntimeTensor<T>(string name, int length)
            where T : unmanaged
        {
            Memory<T> memory = memoryManager.RequestMemory<T>(Id, name, length);
            return new RuntimeTensor(
                name,
                typeof(T),
                [length],
                isReadOnly: false,
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
                });
        }

        private void WriteMemory<T>(string key, Memory<T> value) where T : unmanaged
        {
            if (value.IsEmpty)
            {
                memoryManager.Release(Id, key);
                return;
            }

            Memory<T> memory = memoryManager.RequestMemory<T>(Id, key, value.Length);
            value.Span.CopyTo(memory.Span);
        }

        private int GetEmbeddingLength()
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before session memory can be initialized.");
            }

            return checked((int)model.Config.EmbeddingLength);
        }

        private int GetKeyValueProjectionLength()
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before session memory can be initialized.");
            }

            return checked((int)model.Config.KeyValueProjectionSize);
        }

        private int GetFeedForwardLength()
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before session memory can be initialized.");
            }

            return checked((int)model.Config.FeedForwardLength);
        }

        private int GetVocabularySize()
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before session memory can be initialized.");
            }

            return checked((int)model.Config.VocabularySize);
        }
    }
}
