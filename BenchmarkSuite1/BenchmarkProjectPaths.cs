using System;
using System.IO;

namespace BitNetSharp.Benchmarks;

internal static class BenchmarkProjectPaths
{
    private static readonly string[] ModelPathSegments =
    [
        "Models",
        "bitnet-b1.58-2B-4T-gguf",
        "ggml-model-i2_s.gguf",
    ];

    internal static string ModelPath => FindModelPath();

    private static string FindModelPath()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, Path.Combine(ModelPathSegments));
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException(
            $"Could not locate benchmark model file '{Path.Combine(ModelPathSegments)}' from '{AppContext.BaseDirectory}'.");
    }
}
