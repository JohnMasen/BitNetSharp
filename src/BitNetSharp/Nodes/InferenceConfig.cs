using BitNetSharp.Core;

namespace BitNetSharp.Nodes
{
    public sealed class InferenceConfig
    {
        public const int AutoThreadCount = 0;

        public InferenceConfig(IOPProvider opProvider)
        {
            ArgumentNullException.ThrowIfNull(opProvider);
            OPProvider = opProvider;
        }

        public IOPProvider OPProvider { get; }

        public string Backend => OPProvider.Backend;

        public int ThreadCount => OPProvider.ThreadCount;
    }
}
