using BitNetSharp.Core;
using BitNetSharp.Nodes;

namespace BitNetSharp.Benchmarks;

internal static class BenchmarkInferenceConfigs
{
    public static InferenceConfig Cpu(int threadCount)
    {
        return new InferenceConfig(new CPUDefaultOPProvider(threadCount));
    }

    public static InferenceConfig Tensor(int threadCount)
    {
        return new InferenceConfig(new CPUTensorOPProvider(threadCount));
    }

    public static InferenceConfig Simd(int threadCount)
    {
        return new InferenceConfig(new CPUSimdOPProvider(threadCount));
    }
}
