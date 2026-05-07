using GGUFSharp;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace BitNetSharp.Models
{
    internal sealed class ModelOp
    {
        private readonly BitNetModel model;
        private readonly Dictionary<string, PackedProjectionWeights> packedProjectionWeights = new(StringComparer.Ordinal);
        private readonly Dictionary<string, float[]> floatTensorValues = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Half[]> halfTensorValues = new(StringComparer.Ordinal);

        internal ModelOp(BitNetModel model)
        {
            ArgumentNullException.ThrowIfNull(model);
            this.model = model;
        }

        internal void Reset()
        {
            packedProjectionWeights.Clear();
            floatTensorValues.Clear();
            halfTensorValues.Clear();
        }

        internal PackedProjectionWeights GetPackedProjectionWeights(BitNetTensorInfo tensor, string tensorLabel, bool enableCache)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ArgumentException.ThrowIfNullOrWhiteSpace(tensorLabel);

            if (enableCache && packedProjectionWeights.TryGetValue(tensor.Name, out PackedProjectionWeights cachedWeights))
            {
                return cachedWeights;
            }

            using IMemoryOwner<byte> tensorData = model.ReadTensorData(tensor);
            PackedProjectionWeights weights = ParsePackedProjectionWeights(tensorData.Memory, tensor, tensorLabel);
            if (enableCache)
            {
                packedProjectionWeights[tensor.Name] = weights;
            }

            return weights;
        }

        internal ReadOnlyMemory<float> GetFloatTensorValues(BitNetTensorInfo tensor, string tensorLabel, bool enableCache)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ArgumentException.ThrowIfNullOrWhiteSpace(tensorLabel);

            if (enableCache && floatTensorValues.TryGetValue(tensor.Name, out float[]? cachedValues))
            {
                return cachedValues;
            }

            float[] values = ReadFloatTensorValues(tensor, tensorLabel);
            if (enableCache)
            {
                floatTensorValues[tensor.Name] = values;
            }

            return values;
        }

        internal ReadOnlyMemory<float> GetOptionalFloatTensorValues(BitNetTensorInfo? tensor, string tensorLabel, bool enableCache)
        {
            return tensor is null ? ReadOnlyMemory<float>.Empty : GetFloatTensorValues(tensor, tensorLabel, enableCache);
        }

        internal ReadOnlyMemory<Half> GetEmbeddingValues(BitNetTensorInfo tensor, bool enableCache)
        {
            ArgumentNullException.ThrowIfNull(tensor);

            if (enableCache && halfTensorValues.TryGetValue(tensor.Name, out Half[]? cachedValues))
            {
                return cachedValues;
            }

            using IMemoryOwner<byte> tensorData = model.ReadTensorData(tensor);
            Half[] values = MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span).ToArray();
            if (enableCache)
            {
                halfTensorValues[tensor.Name] = values;
            }

            return values;
        }

        private float[] ReadFloatTensorValues(BitNetTensorInfo tensor, string tensorLabel)
        {
            using IMemoryOwner<byte> tensorData = model.ReadTensorData(tensor);
            return tensor.TensorType switch
            {
                GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"{tensorLabel} tensor type '{tensor.TensorType}' is not supported."),
            };
        }

        private static PackedProjectionWeights ParsePackedProjectionWeights(ReadOnlyMemory<byte> tensorBytes, BitNetTensorInfo tensor, string tensorLabel)
        {
            int packedWeightByteCount = checked(((int)tensor.Dimensions[0] * (int)tensor.Dimensions[1]) / 4);
            if (tensorBytes.Length < packedWeightByteCount + sizeof(float))
            {
                throw new InvalidOperationException($"{tensorLabel} tensor '{tensor.Name}' is incomplete.");
            }

            return new PackedProjectionWeights(
                //TODO:remove ToArray(), should use better cache memory pattern
                tensorBytes[..packedWeightByteCount].ToArray(),
                MemoryMarshal.Read<float>(tensorBytes.Span.Slice(packedWeightByteCount, sizeof(float))));
        }

        private static float[] ConvertHalfToSingle(ReadOnlySpan<Half> source)
        {
            float[] result = new float[source.Length];
            TensorPrimitives.ConvertToSingle(source, result);
            return result;
        }
    }

    internal readonly record struct PackedProjectionWeights(ReadOnlyMemory<byte> PackedWeights, float Scale);
}
