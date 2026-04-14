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

        public Memory<float> Embedding
        {
            get => GetOrCreateFloatMemory(EmbeddingKey, GetEmbeddingLength());
            set => SetMemory(EmbeddingKey, value);
        }

        public Memory<float> RmsNorm
        {
            get => GetOrCreateFloatMemory(RmsNormKey, GetEmbeddingLength());
            set => SetMemory(RmsNormKey, value);
        }

        public Memory<float> QKVQuery
        {
            get => GetOrCreateFloatMemory(QKVQueryKey, GetEmbeddingLength());
            set => SetMemory(QKVQueryKey, value);
        }

        public Memory<float> QKVKey
        {
            get => GetOrCreateFloatMemory(QKVKeyKey, GetKeyValueProjectionLength());
            set => SetMemory(QKVKeyKey, value);
        }

        public Memory<float> QKVValue
        {
            get => GetOrCreateFloatMemory(QKVValueKey, GetKeyValueProjectionLength());
            set => SetMemory(QKVValueKey, value);
        }

        public Memory<float> AttentionSubNorm
        {
            get => GetOrCreateFloatMemory(AttentionSubNormKey, GetEmbeddingLength());
            set => SetMemory(AttentionSubNormKey, value);
        }

        public Memory<float> AttentionOutput
        {
            get => GetOrCreateFloatMemory(AttentionOutputKey, GetEmbeddingLength());
            set => SetMemory(AttentionOutputKey, value);
        }
        //TODO:hot path, do not need to check if contains key
        public Memory<float> FeedForwardInput => GetOrCreateFloatMemory(FeedForwardInputKey, GetEmbeddingLength());

        public Memory<float> FeedForwardNorm => GetOrCreateFloatMemory(FeedForwardNormKey, GetEmbeddingLength());

        public Memory<float> FeedForwardSubNorm => GetOrCreateFloatMemory(FeedForwardSubNormKey, GetFeedForwardLength());

        public Memory<float> FeedForwardOutput => GetOrCreateFloatMemory(FeedForwardOutputKey, GetEmbeddingLength());

        public Memory<float> FinalNormOutput => GetOrCreateFloatMemory(FinalNormOutputKey, GetEmbeddingLength());

        public Memory<float> Logits => GetOrCreateFloatMemory(LogitsKey, GetVocabularySize());

        internal bool HasMemory<T>(string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return memoryManager.TryGetMemory<T>(Id, key, out _);
        }

        private Memory<T> GetMemory<T>(string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            return memoryManager.TryGetMemory<T>(Id, key, out Memory<T> memory)
                ? memory
                : Memory<T>.Empty;
        }

        private Memory<float> GetOrCreateFloatMemory(string key, int length)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            return memoryManager.TryGetMemory<float>(Id, key, out Memory<float> memory)
                ? memory
                : memoryManager.RequestMemory<float>(Id, key, length);
        }

        private void SetMemory<T>(string key, Memory<T> value) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            WriteMemory(key, value);
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
