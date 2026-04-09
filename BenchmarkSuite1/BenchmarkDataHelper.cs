using System;

namespace BitNetSharp.Benchmarks;

internal static class BenchmarkDataHelper
{
    internal static void FillDeterministicValues(Span<float> destination, int seed)
    {
        for (int index = 0; index < destination.Length; index++)
        {
            int primary = ((index + seed) % 17) - 8;
            int secondary = ((index + (seed * 3)) % 5) - 2;
            destination[index] = (primary * 0.125f) + (secondary * 0.03125f);
        }
    }
}
