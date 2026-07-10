using BitNetSharp.Hosting.CPU;
using BitNetSharp.Hosting;
namespace BitNetSharp.Tests
{
    internal static class TestModelFactory
    {
        private static readonly BitNetMemoryManager SharedMemoryManager = new();

        internal static Models.BitNetModel LoadModel(Models.BitNetModelLoadOptions? options = null)
        {
            Models.BitNetModelLoader loader = new();
            return loader.Load(TestProjectPaths.ModelPath, options);
        }

        internal static BitNetSession CreateSession(Models.BitNetModel model, int token)
        {
            ArgumentNullException.ThrowIfNull(model);

            return new BitNetSession(model, SharedMemoryManager, new[] { token });
        }
    }
}


