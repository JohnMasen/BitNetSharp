using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    internal static class FullDumpBaseline
    {
        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            PropertyNameCaseInsensitive = true,
        };

        private static readonly Lazy<FullDumpManifest> ManifestCache = new(LoadManifest);
        private static readonly Lazy<byte[]> DataCache = new(LoadData);

        public static FullDumpManifest Manifest => ManifestCache.Value;

        public static FullDumpEntry GetEntry(string entryName)
        {
            return Manifest.Entries.Single(entry => string.Equals(entry.Name, entryName, StringComparison.Ordinal));
        }

        public static float[] ReadFloatValues(string entryName)
        {
            FullDumpEntry entry = GetEntry(entryName);
            if (string.Equals(entry.DType, "F32", StringComparison.Ordinal))
            {
                return ReadF32(entry);
            }

            if (string.Equals(entry.DType, "F16", StringComparison.Ordinal))
            {
                return ReadF16(entry);
            }

            throw new NotSupportedException($"Entry '{entryName}' with dtype '{entry.DType}' is not supported as float tensor.");
        }

        public static T ReadInlineData<T>(string entryName)
        {
            FullDumpEntry entry = GetEntry(entryName);
            if (!string.Equals(entry.DType, "inline_json", StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"Entry '{entryName}' is not inline_json.");
            }

            return entry.InlineData.Deserialize<T>(SerializerOptions) ?? throw new InvalidOperationException($"Failed to deserialize inline data for '{entryName}'.");
        }

        private static FullDumpManifest LoadManifest()
        {
            string json = File.ReadAllText(TestProjectPaths.FullDumpManifestPath);
            return JsonSerializer.Deserialize<FullDumpManifest>(json, SerializerOptions) ?? throw new InvalidOperationException("Failed to load FullDump manifest JSON.");
        }

        private static byte[] LoadData()
        {
            return File.ReadAllBytes(TestProjectPaths.FullDumpDataPath);
        }

        private static float[] ReadF32(FullDumpEntry entry)
        {
            byte[] data = DataCache.Value;
            float[] values = new float[entry.ElementCount];
            for (int index = 0; index < values.Length; index++)
            {
                values[index] = BitConverter.ToSingle(data, entry.Offset + (index * sizeof(float)));
            }

            return values;
        }

        private static float[] ReadF16(FullDumpEntry entry)
        {
            byte[] data = DataCache.Value;
            float[] values = new float[entry.ElementCount];
            for (int index = 0; index < values.Length; index++)
            {
                ushort bits = BitConverter.ToUInt16(data, entry.Offset + (index * sizeof(ushort)));
                values[index] = (float)BitConverter.UInt16BitsToHalf(bits);
            }

            return values;
        }

        internal sealed record FullDumpManifest(
            [property: JsonPropertyName("schema_version")] string SchemaVersion,
            [property: JsonPropertyName("model_path")] string ModelPath,
            [property: JsonPropertyName("prompt")] FullDumpPrompt Prompt,
            [property: JsonPropertyName("entries")] IReadOnlyList<FullDumpEntry> Entries);

        internal sealed record FullDumpPrompt(
            [property: JsonPropertyName("text")] string Text,
            [property: JsonPropertyName("token_ids")] IReadOnlyList<int> TokenIds,
            [property: JsonPropertyName("assistant_first_token_pre_sampling_position_index")] int AssistantFirstTokenPreSamplingPositionIndex);

        internal sealed record FullDumpEntry(
            [property: JsonPropertyName("Name")] string Name,
            [property: JsonPropertyName("category")] string Category,
            [property: JsonPropertyName("layer_index")] int LayerIndex,
            [property: JsonPropertyName("step_name")] string StepName,
            [property: JsonPropertyName("tensor_name")] string TensorName,
            [property: JsonPropertyName("dtype")] string DType,
            [property: JsonPropertyName("shape")] IReadOnlyList<int> Shape,
            [property: JsonPropertyName("offset")] int Offset,
            [property: JsonPropertyName("byte_length")] int ByteLength,
            [property: JsonPropertyName("element_count")] int ElementCount,
            [property: JsonPropertyName("inline_data")] JsonElement InlineData);
    }
}
