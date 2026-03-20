using System;
using Xunit;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class RMSNormTests
    {
        [Fact]
        public void Forward_OutputShape()
        {
            var norm = new RMSNorm(4);
            var input = new Tensor(new[] { 4 }, new float[] { 1f, 2f, 3f, 4f });
            Tensor output = norm.Forward(input);
            Assert.Equal(4, output.Size);
        }

        [Fact]
        public void Forward_NormalizesToUnitRMS_WithUnitGamma()
        {
            var norm = new RMSNorm(4);   // gamma defaults to all-ones
            var input = new Tensor(new[] { 4 }, new float[] { 2f, 2f, 2f, 2f });
            Tensor output = norm.Forward(input);

            // For constant input, each output element should be ≈ 1
            foreach (float v in output.Data)
                Assert.Equal(1f, v, 4);
        }

        [Fact]
        public void Forward_SizeMismatchThrows()
        {
            var norm = new RMSNorm(4);
            var input = new Tensor(new[] { 3 }, new float[] { 1f, 2f, 3f });
            Assert.Throws<ArgumentException>(() => norm.Forward(input));
        }

        [Fact]
        public void ForwardBatch_CorrectShape()
        {
            var norm = new RMSNorm(4);
            var input = new Tensor(new[] { 3, 4 }, new float[] {
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f
            });
            Tensor output = norm.ForwardBatch(input);
            Assert.Equal(new[] { 3, 4 }, output.Shape);
        }

        [Fact]
        public void ForwardBatch_EachRowHasUnitRMS_WithUnitGamma()
        {
            var norm = new RMSNorm(3);
            // Each row has same elements → normalised value should be 1
            var input = new Tensor(new[] { 2, 3 }, new float[] {
                3f, 3f, 3f,
                5f, 5f, 5f
            });
            Tensor output = norm.ForwardBatch(input);

            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(1f, output[r, c], 4);
        }

        [Fact]
        public void GammaScaling_AppliedCorrectly()
        {
            var norm = new RMSNorm(2);
            norm.Gamma[0] = 2f;
            norm.Gamma[1] = 3f;

            // Input with constant values → normalised to 1, then scaled by gamma
            var input = new Tensor(new[] { 2 }, new float[] { 4f, 4f });
            Tensor output = norm.Forward(input);

            Assert.Equal(2f, output.Data[0], 4);
            Assert.Equal(3f, output.Data[1], 4);
        }
    }
}
