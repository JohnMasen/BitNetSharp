using BitNetSharp.Core;
using BitNetSharp.Nodes;

namespace BitNetSharp.Tests
{
    internal static class TestInferenceConfigs
    {
        public const string CpuBackend = "CPU";

        public const string TensorBackend = "Tensor";

        public const string SimdBackend = "SIMD";

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

        public static InferenceConfig Create(string backend, int threadCount)
        {
            return backend switch
            {
                CpuBackend => Cpu(threadCount),
                TensorBackend => Tensor(threadCount),
                SimdBackend => Simd(threadCount),
                _ => throw new NotSupportedException($"Backend '{backend}' is not supported in tests."),
            };
        }
    }
}
