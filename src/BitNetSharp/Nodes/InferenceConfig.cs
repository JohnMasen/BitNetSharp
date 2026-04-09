namespace BitNetSharp.Nodes
{
    public sealed class InferenceConfig
    {
        public const int AutoThreadCount = 0;

        public InferenceConfig(InferenceBackend backend, int threadCount)
        {
            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            Backend = backend;
            ThreadCount = threadCount;
        }

        public InferenceBackend Backend { get; }

        public int ThreadCount { get; }
    }
}
