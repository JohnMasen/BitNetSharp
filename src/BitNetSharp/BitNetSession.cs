using BitNetSharp.Models;

namespace BitNetSharp
{
    public class BitNetSession : IDisposable
    {
        private const string LayerKeyCachePrefix = "LayerKeyCache:";
        private const string LayerValueCachePrefix = "LayerValueCache:";
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
        private readonly List<int> inputTokens = [];
        private readonly List<int> outputTokens = [];
        private int currentToken;
        private bool disposed;

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager)
            : this(model, memoryManager, Guid.NewGuid())
        {
        }

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager, ReadOnlyMemory<int> tokens)
            : this(model, memoryManager, Guid.NewGuid(), tokens)
        {
        }

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager, Guid id)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(memoryManager);

            this.model = model;
            this.memoryManager = memoryManager;
            Id = id;
            TopKTokenIds = [];
            TopKLogits = [];
        }

        public BitNetSession(BitNetModel model, BitNetMemoryManager memoryManager, Guid id, ReadOnlyMemory<int> tokens)
            : this(model, memoryManager, id)
        {
            InitializeTokens(tokens);
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

            runtimeTensors.Clear();
            inputTokens.Clear();
            outputTokens.Clear();
            memoryManager.Release(Id);
            disposed = true;
            GC.SuppressFinalize(this);
        }

        public IEnumerable<int> Tokens => EnumerateTokens();

        public int TokenCount
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return inputTokens.Count + outputTokens.Count;
            }
        }

        public int CurrentToken
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return currentToken;
            }
            internal set
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                currentToken = value;
            }
        }

        public int OutputRound { get; private set; }

        public bool HasActiveOutputRound { get; private set; }

        public int CurrentOutputTokenCount => outputTokens.Count;

        public int CacheLength { get; set; }

        public int CacheWritePosition { get; set; }

        public int NextTokenId { get; set; }

        public int ArgmaxTokenId { get; set; }

        public float NextTokenLogit { get; set; }

        public float ArgmaxLogit { get; set; }

        public string? NextTokenStrategy { get; set; }

        public long LastInferenceElapsedMilliseconds { get; set; }

        public long LastSamplingElapsedMilliseconds { get; set; }

        public int[] TopKTokenIds { get; set; }

        public float[] TopKLogits { get; set; }

        /// <summary>
        /// Gets the tokens appended during the current output round.
        /// </summary>
        public ReadOnlyMemory<int> CurrentOutputTokens
        {
            get
            {
                ObjectDisposedException.ThrowIf(disposed, this);

                if (CurrentOutputTokenCount == 0)
                {
                    return ReadOnlyMemory<int>.Empty;
                }

                return outputTokens.ToArray();
            }
        }

        public RuntimeTensor Embedding => GetOrCreateRuntimeTensor(EmbeddingKey);

        public RuntimeTensor RmsNorm => GetOrCreateRuntimeTensor(RmsNormKey);

        public RuntimeTensor QKVQuery => GetOrCreateRuntimeTensor(QKVQueryKey);

        public RuntimeTensor QKVKey => GetOrCreateRuntimeTensor(QKVKeyKey);

        public RuntimeTensor QKVValue => GetOrCreateRuntimeTensor(QKVValueKey);

        public RuntimeTensor AttentionSubNorm => GetOrCreateRuntimeTensor(AttentionSubNormKey);

        public RuntimeTensor AttentionOutput => GetOrCreateRuntimeTensor(AttentionOutputKey);

        public RuntimeTensor FeedForwardInput => GetOrCreateRuntimeTensor(FeedForwardInputKey);

        public RuntimeTensor FeedForwardNorm => GetOrCreateRuntimeTensor(FeedForwardNormKey);

        public RuntimeTensor FeedForwardSubNorm => GetOrCreateRuntimeTensor(FeedForwardSubNormKey);

        public RuntimeTensor FeedForwardOutput => GetOrCreateRuntimeTensor(FeedForwardOutputKey);

        public RuntimeTensor FinalNormOutput => GetOrCreateRuntimeTensor(FinalNormOutputKey);

        public RuntimeTensor Logits => GetOrCreateRuntimeTensor(LogitsKey);

        public RuntimeTensor EmbeddingTensor => Embedding;

        public RuntimeTensor RmsNormTensor => RmsNorm;

        public RuntimeTensor QKVQueryTensor => QKVQuery;

        public RuntimeTensor QKVKeyTensor => QKVKey;

        public RuntimeTensor QKVValueTensor => QKVValue;

        public RuntimeTensor AttentionSubNormTensor => AttentionSubNorm;

        public RuntimeTensor AttentionOutputTensor => AttentionOutput;

        public RuntimeTensor FeedForwardInputTensor => FeedForwardInput;

        public RuntimeTensor FeedForwardNormTensor => FeedForwardNorm;

        public RuntimeTensor FeedForwardSubNormTensor => FeedForwardSubNorm;

        public RuntimeTensor FeedForwardOutputTensor => FeedForwardOutput;

        public RuntimeTensor FinalNormOutputTensor => FinalNormOutput;

        public RuntimeTensor LogitsTensor => Logits;

        /// <summary>
        /// Gets or creates the per-layer key cache tensor used by future multi-token decode flows.
        /// </summary>
        public RuntimeTensor GetOrCreateLayerKeyCacheTensor(int layerIndex)
        {
            ValidateLayerIndex(layerIndex);
            return GetOrCreateRuntimeTensor(CreateLayerCacheTensorName(LayerKeyCachePrefix, layerIndex));
        }

        /// <summary>
        /// Gets or creates the per-layer value cache tensor used by future multi-token decode flows.
        /// </summary>
        public RuntimeTensor GetOrCreateLayerValueCacheTensor(int layerIndex)
        {
            ValidateLayerIndex(layerIndex);
            return GetOrCreateRuntimeTensor(CreateLayerCacheTensorName(LayerValueCachePrefix, layerIndex));
        }

        /// <summary>
        /// Appends a token to the session history.
        /// </summary>
        public void AppendToken(int tokenId)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            CommitOutputTokens();
            AppendTokenCore(inputTokens, tokenId);
        }

        /// <summary>
        /// Starts a new output round so subsequent output tokens can be tracked separately from prior history.
        /// </summary>
        public void BeginOutputRound()
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            CommitOutputTokens();
            OutputRound++;
            HasActiveOutputRound = true;
        }

        /// <summary>
        /// Appends an output token to the session history and the current output round view.
        /// </summary>
        public void AppendOutputToken(int tokenId)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (!HasActiveOutputRound)
            {
                throw new InvalidOperationException("Call BeginOutputRound before appending output tokens.");
            }

            AppendTokenCore(outputTokens, tokenId);
        }

        /// <summary>
        /// Marks the current output round as complete while preserving its tracked output tokens.
        /// </summary>
        public void CompleteOutputRound()
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            HasActiveOutputRound = false;
        }

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

        internal bool HasMemory<T>(string key) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            return runtimeTensors.TryGetValue(key, out RuntimeTensor? tensor) && tensor.TryGet<Memory<T>>(out _);
        }

        internal int GetTokenAt(int index)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if ((uint)index >= (uint)TokenCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            if (index < inputTokens.Count)
            {
                return inputTokens[index];
            }

            return outputTokens[index - inputTokens.Count];
        }

        private RuntimeTensor CreateRuntimeTensor(string name)
        {
            if (TryCreateLayerCacheTensor(name, out RuntimeTensor? layerCacheTensor))
            {
                return layerCacheTensor;
            }

            return name switch
            {
                EmbeddingKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                RmsNormKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                QKVQueryKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                QKVKeyKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.KeyValueProjectionSize)),
                QKVValueKey => CreateRuntimeTensor<float>(name, GetConfig(() => model.Config.KeyValueProjectionSize)),
                AttentionSubNormKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                AttentionOutputKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                FeedForwardInputKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                FeedForwardNormKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                FeedForwardSubNormKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.FeedForwardLength)),
                FeedForwardOutputKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                FinalNormOutputKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.EmbeddingLength)),
                LogitsKey => CreateRuntimeTensor<float>(name, GetConfig(()=>model.Config.VocabularySize)),
                _ => throw new InvalidOperationException($"Unknown runtime tensor '{name}'."),
            };
        }

        private bool TryCreateLayerCacheTensor(string name, out RuntimeTensor? tensor)
        {
            if (TryParseLayerCacheTensorName(name, LayerKeyCachePrefix, out int keyLayerIndex)
                || TryParseLayerCacheTensorName(name, LayerValueCachePrefix, out keyLayerIndex))
            {
                ValidateLayerIndex(keyLayerIndex);
                int cacheElementCount = GetConfig(()=>model.Config.ContextLength) * GetConfig(() => model.Config.KeyValueProjectionSize);
                tensor = CreateRuntimeTensor<float>(name, cacheElementCount);
                return true;
            }

            tensor = null;
            return false;
        }

        private RuntimeTensor CreateRuntimeTensor<T>(string name, int length)
            where T : unmanaged
        {
            Memory<T> memory = memoryManager.RequestMemory<T>(Id, name, length);
            RuntimeTensor tensor = RuntimeTensor.CreateWritable(name, memory, [length]);

            if (runtimeTensors.ContainsKey(name))
            {
                runtimeTensors[name] = tensor;
            }

            return tensor;
        }

        private static string CreateLayerCacheTensorName(string prefix, int layerIndex)
        {
            if (layerIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            return $"{prefix}{layerIndex}";
        }

        private static bool TryParseLayerCacheTensorName(string name, string prefix, out int layerIndex)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal))
            {
                layerIndex = -1;
                return false;
            }

            if (!int.TryParse(name.AsSpan(prefix.Length), out layerIndex))
            {
                throw new InvalidOperationException($"Layer cache tensor name '{name}' is invalid.");
            }

            return true;
        }

        private void ValidateLayerIndex(int layerIndex)
        {
            if ((uint)layerIndex >= (uint)model.Layers.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }
        }

        private IEnumerable<int> EnumerateTokens()
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            foreach (int token in inputTokens)
            {
                yield return token;
            }

            foreach (int token in outputTokens)
            {
                yield return token;
            }
        }

        private void AppendTokenCore(List<int> targetTokens, int tokenId)
        {
            targetTokens.Add(tokenId);
            currentToken = tokenId;
        }

        private void CommitOutputTokens()
        {
            if (outputTokens.Count == 0)
            {
                return;
            }

            inputTokens.AddRange(outputTokens);
            outputTokens.Clear();
        }

        private void InitializeTokens(ReadOnlyMemory<int> tokens)
        {
            if (tokens.IsEmpty)
            {
                return;
            }
            inputTokens.AddRange(tokens.ToArray());
            currentToken = tokens.Span[^1];
        }

        
        private int GetConfig(Func<uint> getConfigFunc)
        {
            if (model.Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before session memory can be initialized.");
            }

            return checked((int)getConfigFunc());
        }
    }
}
