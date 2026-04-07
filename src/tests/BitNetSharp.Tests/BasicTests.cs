namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BasicTests
    {
        [TestMethod]
        public void LoadModel_CreatesTokenizer()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.IsNotNull(model.Tokenizer);
        }

        [TestMethod]
        public void LoadModel_ReadsArchitectureName()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.AreEqual("bitnet-b1.58", model.Config?.ArchitectureName);
        }

        [TestMethod]
        public void LoadModel_ReadsLayerCount()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.HasCount(30, model.Layers);
        }

        [TestMethod]
        public void LoadModel_FindsKnownTensor()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.IsTrue(model.TryGetTensor("blk.0.attn_q.weight", out _));
        }

        [TestMethod]
        public void GetLayer_ReturnsQueryWeightTensor()
        {
            using var model = TestModelFactory.LoadModel();

            Assert.AreEqual("blk.0.attn_q.weight", model.GetLayer(0).AttentionQueryWeight.Name);
        }

        [TestMethod]
        public void LoadModel_UsesCustomMetadataParser()
        {
            Models.BitNetMetadataParser parser = file =>
            {
                var result = TestMetadataParser.ParseDefaultBitNetMetadata(file);
                return new Models.BitNetMetadataParseResult
                {
                    ModelConfig = result.ModelConfig with { ModelName = "custom-bitnet" },
                    TokenizerConfig = result.TokenizerConfig,
                };
            };

            using var model = TestModelFactory.LoadModel(new Models.BitNetModelLoadOptions { MetadataParser = parser });

            Assert.AreEqual("custom-bitnet", model.Config?.ModelName);
        }

    }
}
