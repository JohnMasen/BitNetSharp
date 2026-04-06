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
    }
}
