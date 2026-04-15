namespace BitNetSharp.Tests
{
    internal static class TestModelFactory
    {
        private static readonly BitNetMemoryManager SharedMemoryManager = new();

        internal static Models.BitNetModel LoadModel(Models.BitNetModelLoadOptions? options = null)
        {
            Models.BitNetModel model = new Models.BitNetModel();
            if (options is null)
            {
                model.Load(TestProjectPaths.ModelPath);
            }
            else
            {
                model.Load(TestProjectPaths.ModelPath, options);
            }

            return model;
        }

        internal static BitNetSession CreateSession(Models.BitNetModel model, int token)
        {
            ArgumentNullException.ThrowIfNull(model);

            return new BitNetSession(model, SharedMemoryManager, new[] { token });
        }
    }
}
