namespace BitNetSharp.Nodes
{
    public enum InferenceBackend
    {
        CPU,
        Tensor,
        SIMD,
    }

    public static class InferenceBackendNames
    {
        public const string CPU = nameof(InferenceBackend.CPU);
        public const string Tensor = nameof(InferenceBackend.Tensor);
        public const string SIMD = nameof(InferenceBackend.SIMD);
    }

    public static class InferenceBackendExtensions
    {
        public static string ToBackendName(this InferenceBackend backend)
        {
            return backend switch
            {
                InferenceBackend.CPU => InferenceBackendNames.CPU,
                InferenceBackend.Tensor => InferenceBackendNames.Tensor,
                InferenceBackend.SIMD => InferenceBackendNames.SIMD,
                _ => throw new ArgumentOutOfRangeException(nameof(backend), backend, "Unsupported inference backend."),
            };
        }
    }
}
