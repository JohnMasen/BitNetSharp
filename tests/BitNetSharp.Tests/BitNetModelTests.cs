using System;
using Xunit;
using BitNetSharp.Models;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class BitNetModelTests
    {
        private static BitNetConfig SmallConfig() => BitNetConfig.Small();

        [Fact]
        public void Forward_OutputShape()
        {
            var model = new BitNetModel(SmallConfig());
            var tokenIds = new[] { 1, 2, 3 };
            Tensor logits = model.Forward(tokenIds);

            Assert.Equal(new[] { 3, SmallConfig().VocabSize }, logits.Shape);
        }

        [Fact]
        public void Forward_ProducesFiniteValues()
        {
            var model = new BitNetModel(SmallConfig());
            var tokenIds = new[] { 0, 5, 10 };
            Tensor logits = model.Forward(tokenIds);

            for (int i = 0; i < logits.Size; i++)
                Assert.True(float.IsFinite(logits.Data[i]),
                    $"Logit at index {i} is not finite: {logits.Data[i]}");
        }

        [Fact]
        public void PredictNextToken_ReturnsValidTokenId()
        {
            var config = SmallConfig();
            var model = new BitNetModel(config);
            var tokenIds = new[] { 0, 1, 2 };
            int nextToken = model.PredictNextToken(tokenIds);

            Assert.InRange(nextToken, 0, config.VocabSize - 1);
        }

        [Fact]
        public void Forward_EmptyTokenIdsThrows()
        {
            var model = new BitNetModel(SmallConfig());
            Assert.Throws<ArgumentException>(() => model.Forward(Array.Empty<int>()));
        }

        [Fact]
        public void Forward_OutOfRangeTokenIdThrows()
        {
            var config = SmallConfig();
            var model = new BitNetModel(config);
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                model.Forward(new[] { config.VocabSize + 1 }));
        }

        [Fact]
        public void Forward_SingleToken()
        {
            var model = new BitNetModel(SmallConfig());
            Tensor logits = model.Forward(new[] { 0 });
            Assert.Equal(new[] { 1, SmallConfig().VocabSize }, logits.Shape);
        }
    }
}
