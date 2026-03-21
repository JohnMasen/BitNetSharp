using System;
using Xunit;
using BitNetSharp.Layers;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class BitLinearTests
    {
        [Fact]
        public void Forward_OutputSize()
        {
            var layer = new BitLinear(8, 4);
            var input = new Tensor(new[] { 8 }, new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
            Tensor output = layer.Forward(input);
            Assert.Equal(4, output.Size);
        }

        [Fact]
        public void ForwardBatch_OutputShape()
        {
            var layer = new BitLinear(8, 4);
            var input = new Tensor(new[] { 3, 8 });
            // Fill with non-zero values so quantization is well-defined
            for (int i = 0; i < input.Size; i++)
                input.Data[i] = (i + 1) * 0.1f;

            Tensor output = layer.ForwardBatch(input);
            Assert.Equal(new[] { 3, 4 }, output.Shape);
        }

        [Fact]
        public void Forward_WrongInputSizeThrows()
        {
            var layer = new BitLinear(8, 4);
            var input = new Tensor(new[] { 5 });
            Assert.Throws<ArgumentException>(() => layer.Forward(input));
        }

        [Fact]
        public void WeightsAreTernaryAfterQuantization()
        {
            var layer = new BitLinear(4, 4);
            var (quantized, _) = Quantization.QuantizeWeights(layer.Weights);
            foreach (float v in quantized.Data)
                Assert.True(v == -1f || v == 0f || v == 1f,
                    $"Expected ternary value but got {v}");
        }

        [Fact]
        public void Forward_ProducesFiniteValues()
        {
            var layer = new BitLinear(8, 4);
            var input = new Tensor(new[] { 8 }, new float[] { 1f, -1f, 2f, -2f, 0f, 3f, -3f, 4f });
            Tensor output = layer.Forward(input);

            foreach (float v in output.Data)
                Assert.True(float.IsFinite(v), $"Expected finite value but got {v}");
        }
    }
}
