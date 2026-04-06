using GGUFSharp;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class BasicTests
    {
        [TestMethod]
        public void WhenLoadingModelThenTokenizerIsCreated()
        {
            using var model = LoadModel();

            Assert.IsNotNull(model.Tokenizer);
        }

        [TestMethod]
        public void WhenLoadingModelThenArchitectureNameMatchesMetadata()
        {
            using var model = LoadModel();

            Assert.AreEqual("bitnet-b1.58", model.Config?.ArchitectureName);
        }

        [TestMethod]
        public void WhenLoadingModelThenLayerCountMatchesBlockCount()
        {
            using var model = LoadModel();

            Assert.HasCount(30, model.Layers);
        }

        [TestMethod]
        public void WhenLoadingModelThenKnownTensorCanBeFound()
        {
            using var model = LoadModel();

            Assert.IsTrue(model.TryGetTensor("blk.0.attn_q.weight", out _));
        }

        [TestMethod]
        public void WhenGettingFirstLayerThenAttentionQueryWeightMatchesExpectedTensorName()
        {
            using var model = LoadModel();

            Assert.AreEqual("blk.0.attn_q.weight", model.GetLayer(0).AttentionQueryWeight.Name);
        }

        [TestMethod]
        public void WhenLoadingModelWithCustomMetadataParserThenParsedMetadataIsUsed()
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

            using var model = LoadModel(new Models.BitNetModelLoadOptions { MetadataParser = parser });

            Assert.AreEqual("custom-bitnet", model.Config?.ModelName);
        }

        private static Models.BitNetModel LoadModel(Models.BitNetModelLoadOptions? options = null)
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
    }
}
