namespace BitNetSharp.Tests
{
    internal static class TestModelFactory
    {
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

        internal static global::BitNetSharp.BitNetSession CreateSession(Models.BitNetModel model, int token)
        {
            ArgumentNullException.ThrowIfNull(model);

            return new global::BitNetSharp.BitNetSession(model)
            {
                Tokens = [token],
                CurrentToken = token,
            };
        }
    }
}
