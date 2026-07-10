using BitNetSharp.Hosting.CPU;
using BitNetSharp.Hosting;
using GGUFSharp;
using System.Buffers;
using System.Runtime.InteropServices;

namespace BitNetSharp.Models
{
    public class BitNetModel : IDisposable
    {
        private readonly BitNetMemoryManager weightMemoryManager = new();
        private readonly Dictionary<string, RuntimeTensor> weightTensors = new(StringComparer.Ordinal);
        private readonly Guid weightSessionId = Guid.NewGuid();
        private bool disposed;
        private GGUFFile? loadedFile;
        private GGUFReader? loadedReader;
        private IReadOnlyDictionary<string, GGUFTensorInfo> rawTensorIndex = new Dictionary<string, GGUFTensorInfo>(StringComparer.Ordinal);

        public BitNetModelConfig? Config { get; private set; }

        public BitNetTokenizerConfig? TokenizerConfig { get; private set; }

        public BitNetGlobalTensors? GlobalTensors { get; private set; }

        public IReadOnlyList<BitNetLayerDefinition> Layers { get; private set; } = [];

        public IReadOnlyDictionary<string, BitNetTensorInfo> TensorIndex { get; private set; } =
            new Dictionary<string, BitNetTensorInfo>(StringComparer.Ordinal);

        public BitNetTokenizer? Tokenizer { get; private set; }

        public bool UsesTiedEmbeddings => GlobalTensors is not null;

        public BitNetModel()
        {
        }

        internal BitNetModel(
            GGUFReader loadedReader,
            GGUFFile loadedFile,
            BitNetModelConfig config,
            BitNetTokenizerConfig tokenizerConfig,
            BitNetTokenizer tokenizer,
            BitNetGlobalTensors globalTensors,
            IReadOnlyList<BitNetLayerDefinition> layers,
            IReadOnlyDictionary<string, BitNetTensorInfo> tensorIndex,
            IReadOnlyDictionary<string, GGUFTensorInfo> rawTensorIndex)
        {
            ArgumentNullException.ThrowIfNull(loadedReader);
            ArgumentNullException.ThrowIfNull(loadedFile);
            ArgumentNullException.ThrowIfNull(config);
            ArgumentNullException.ThrowIfNull(tokenizerConfig);
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(globalTensors);
            ArgumentNullException.ThrowIfNull(layers);
            ArgumentNullException.ThrowIfNull(tensorIndex);
            ArgumentNullException.ThrowIfNull(rawTensorIndex);

            this.loadedReader = loadedReader;
            this.loadedFile = loadedFile;
            Config = config;
            TokenizerConfig = tokenizerConfig;
            Tokenizer = tokenizer;
            GlobalTensors = globalTensors;
            Layers = layers;
            TensorIndex = tensorIndex;
            this.rawTensorIndex = rawTensorIndex;
        }

        /// <summary>
        /// Loads model metadata, tokenizer data, tensor descriptors, and layer definitions from a GGUF file.
        /// </summary>
        [Obsolete("Use BitNetModelLoader.Load instead.")]
        public void Load(string ggufPath)
        {
            Load(ggufPath, options: null);
        }

        /// <summary>
        /// Loads model metadata, tokenizer data, tensor descriptors, and layer definitions from a GGUF file using the provided load options.
        /// </summary>
        [Obsolete("Use BitNetModelLoader.Load instead.")]
        public void Load(string ggufPath, BitNetModelLoadOptions? options)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            using BitNetModel model = new BitNetModelLoader().Load(ggufPath, options);
            ReplaceLoadedState(model);
        }

        private void ReplaceLoadedState(BitNetModel model)
        {
            ArgumentNullException.ThrowIfNull(model);

            weightMemoryManager.Release(weightSessionId);
            weightTensors.Clear();

            loadedReader = model.loadedReader;
            loadedFile = model.loadedFile;
            Config = model.Config;
            TokenizerConfig = model.TokenizerConfig;
            Tokenizer = model.Tokenizer;
            GlobalTensors = model.GlobalTensors;
            Layers = model.Layers;
            TensorIndex = model.TensorIndex;
            rawTensorIndex = model.rawTensorIndex;

            model.loadedReader = null;
            model.loadedFile = null;
            model.Tokenizer = null;
            model.TokenizerConfig = null;
            model.GlobalTensors = null;
            model.Config = null;
            model.Layers = [];
            model.TensorIndex = new Dictionary<string, BitNetTensorInfo>(StringComparer.Ordinal);
            model.rawTensorIndex = new Dictionary<string, GGUFTensorInfo>(StringComparer.Ordinal);
        }

        /// <summary>
        /// Gets a shared readonly runtime tensor for the requested model weight.
        /// </summary>
        public RuntimeTensor GetWeightTensor(string tensorName)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
            }

            if (weightTensors.TryGetValue(tensorName, out RuntimeTensor? cachedTensor))
            {
                return cachedTensor;
            }

            if (!TensorIndex.TryGetValue(tensorName, out BitNetTensorInfo? tensorInfo))
            {
                throw new InvalidOperationException($"Required tensor '{tensorName}' was not found.");
            }

            RuntimeTensor tensor = CreateWeightTensor(tensorInfo);
            weightTensors.Add(tensorName, tensor);
            return tensor;
        }

        private RuntimeTensor CreateWeightTensor(BitNetTensorInfo tensorInfo)
        {
            using IMemoryOwner<byte> tensorData = ReadTensorData(tensorInfo.Name);
            return ShouldExposeAsFloatTensor(tensorInfo)
                ? CreateSingleWeightTensor(tensorInfo, tensorData.Memory)
                : CreateByteWeightTensor(tensorInfo, tensorData.Memory);
        }

        private RuntimeTensor CreateByteWeightTensor(BitNetTensorInfo tensorInfo, ReadOnlyMemory<byte> tensorData)
        {
            Memory<byte> buffer = weightMemoryManager.RequestMemory<byte>(weightSessionId, tensorInfo.Name, tensorData.Length);

            tensorData.CopyTo(buffer);
            return RuntimeTensor.CreateReadOnly<byte>(
                tensorInfo.Name,
                buffer,
                tensorInfo.Dimensions.Select(static dimension => checked((int)dimension)));
        }

        private RuntimeTensor CreateSingleWeightTensor(BitNetTensorInfo tensorInfo, ReadOnlyMemory<byte> tensorData)
        {
            int elementCount = checked(tensorInfo.Dimensions.Aggregate<ulong, int>(1, static (count, dimension) => checked(count * (int)dimension)));
            Memory<float> buffer = weightMemoryManager.RequestMemory<float>(weightSessionId, tensorInfo.Name, elementCount);

            switch (tensorInfo.TensorType)
            {
                case GGUFTensorType.GGML_TYPE_F32:
                    MemoryMarshal.Cast<byte, float>(tensorData.Span[..checked(elementCount * sizeof(float))]).CopyTo(buffer.Span);
                    break;
                case GGUFTensorType.GGML_TYPE_F16:
                {
                    ReadOnlySpan<Half> source = MemoryMarshal.Cast<byte, Half>(tensorData.Span[..checked(elementCount * sizeof(ushort))]);
                    for (int index = 0; index < elementCount; index++)
                    {
                        buffer.Span[index] = (float)source[index];
                    }

                    break;
                }
                default:
                    throw new NotSupportedException($"Tensor '{tensorInfo.Name}' type '{tensorInfo.TensorType}' cannot be exposed as float weights.");
            }

            return RuntimeTensor.CreateReadOnly<float>(
                tensorInfo.Name,
                buffer,
                tensorInfo.Dimensions.Select(static dimension => checked((int)dimension)));
        }

        private static bool ShouldExposeAsFloatTensor(BitNetTensorInfo tensorInfo)
        {
            return !tensorInfo.IsQuantized && tensorInfo.Role != BitNetTensorRole.TokenEmbedding;
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file.
        /// </summary>
        public IMemoryOwner<byte> ReadTensorData(string tensorName)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
            }

            if (loadedFile is null || loadedReader is null)
            {
                throw new InvalidOperationException("The model must be loaded before tensor data can be read.");
            }

            if (!rawTensorIndex.TryGetValue(tensorName, out GGUFTensorInfo? tensorInfo))
            {
                throw new InvalidOperationException($"Required tensor '{tensorName}' was not found.");
            }

            return loadedReader.ReadTensorData(loadedFile, tensorInfo);
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file directly into a managed session buffer.
        /// </summary>
        public Memory<byte> ReadTensorData(string tensorName, BitNetMemoryManager memoryManager, Guid sessionId, string key)
        {
            ArgumentNullException.ThrowIfNull(memoryManager);
            ArgumentException.ThrowIfNullOrWhiteSpace(key);

            using IMemoryOwner<byte> tensorData = ReadTensorData(tensorName);
            Memory<byte> buffer = memoryManager.RequestMemory<byte>(sessionId, key, tensorData.Memory.Length);

            tensorData.Memory.CopyTo(buffer);
            return buffer;
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file.
        /// </summary>
        public IMemoryOwner<byte> ReadTensorData(BitNetTensorInfo tensorInfo)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ArgumentNullException.ThrowIfNull(tensorInfo);

            return ReadTensorData(tensorInfo.Name);
        }

        /// <summary>
        /// Reads the raw byte payload of a tensor from the loaded GGUF file directly into a managed session buffer.
        /// </summary>
        public Memory<byte> ReadTensorData(BitNetTensorInfo tensorInfo, BitNetMemoryManager memoryManager, Guid sessionId, string key)
        {
            ArgumentNullException.ThrowIfNull(tensorInfo);

            return ReadTensorData(tensorInfo.Name, memoryManager, sessionId, key);
        }

        /// <summary>
        /// Releases the loaded model state so test fixtures and callers can free the model after use.
        /// </summary>
        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            weightMemoryManager.Dispose();
            weightTensors.Clear();
            Tokenizer = null;
            TokenizerConfig = null;
            GlobalTensors = null;
            Config = null;
            Layers = [];
            TensorIndex = new Dictionary<string, BitNetTensorInfo>(StringComparer.Ordinal);
            rawTensorIndex = new Dictionary<string, GGUFTensorInfo>(StringComparer.Ordinal);
            loadedFile = null;
            loadedReader = null;
            disposed = true;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Returns the requested layer definition from the loaded model.
        /// </summary>
        public BitNetLayerDefinition GetLayer(int layerIndex)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (Config is null)
            {
                throw new InvalidOperationException("The model must be loaded before layers can be accessed.");
            }

            if ((uint)layerIndex >= Config.BlockCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            return Layers[layerIndex];
        }

        /// <summary>
        /// Tries to find a tensor by its GGUF name.
        /// </summary>
        public bool TryGetTensor(string tensorName, out BitNetTensorInfo? tensor)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
            }

            return TensorIndex.TryGetValue(tensorName, out tensor);
        }

    }
}


