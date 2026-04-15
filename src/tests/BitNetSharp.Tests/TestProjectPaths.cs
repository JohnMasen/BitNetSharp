using System.IO;

namespace BitNetSharp.Tests
{
    internal static class TestProjectPaths
    {
        public static string ModelPath => Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "..",
            "..",
            "..",
            "Models",
            "bitnet-b1.58-2B-4T-gguf",
            "ggml-model-i2_s.gguf"));

        public static string StandardTokensPath => Path.Combine(
            AppContext.BaseDirectory,
            "TestData",
            "standard_tokens.json");

        public static string LayerVectorsPath => Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "TestData",
            "layer_vectors_pure.json");

        public static string HiTopCandidatesDumpPath => Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "TestData",
            "hi_top_candidates_dump.json");

        public static string HiChatFullDumpPath => Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "TestData",
            "hi_chat_full_dump.json");

        public static string HiRopeKCacheDumpPath => Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "TestData",
            "hi_rope_k_cache_dump.json");
    }
}
