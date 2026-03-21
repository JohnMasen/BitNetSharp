using System;
using Xunit;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class QuantizationTests
    {
        [Fact]
        public void QuantizeWeights_OutputIsTernary()
        {
            var weights = new Tensor(new[] { 4 }, new float[] { 0.5f, -0.3f, 0.001f, -0.9f });
            var (quantized, scale) = Quantization.QuantizeWeights(weights);

            foreach (float v in quantized.Data)
                Assert.True(v == -1f || v == 0f || v == 1f,
                    $"Expected ternary value but got {v}");
        }

        [Fact]
        public void QuantizeWeights_ScaleIsPositive()
        {
            var weights = new Tensor(new[] { 4 }, new float[] { 1f, 2f, -3f, 4f });
            var (_, scale) = Quantization.QuantizeWeights(weights);
            Assert.True(scale > 0f);
        }

        [Fact]
        public void QuantizeActivations_OutputInInt8Range()
        {
            var activations = new Tensor(new[] { 5 }, new float[] { 1f, -1f, 2f, -2f, 0f });
            var (quantized, scale) = Quantization.QuantizeActivations(activations);

            foreach (float v in quantized.Data)
                Assert.InRange(v, -128f, 127f);
        }

        [Fact]
        public void QuantizeActivations_MaxMapsTo127()
        {
            float maxVal = 3.5f;
            var activations = new Tensor(new[] { 3 }, new float[] { maxVal, 0f, -1f });
            var (quantized, scale) = Quantization.QuantizeActivations(activations);

            // The maximum absolute value should quantize to ≈ 127
            Assert.Equal(127f, quantized.Data[0], 0);
        }

        [Fact]
        public void QuantizeWeights_ZeroWeightsQuantizeToZero()
        {
            var weights = new Tensor(new[] { 4 }, new float[] { 0f, 0f, 0f, 0f });
            var (quantized, _) = Quantization.QuantizeWeights(weights);

            foreach (float v in quantized.Data)
                Assert.Equal(0f, v);
        }

        [Fact]
        public void Dequantize_RevertsScale()
        {
            // If activationScale * weightScale = 10, output/10 should recover original
            var output = new Tensor(new[] { 3 }, new float[] { 10f, 20f, 30f });
            Tensor dequantized = Quantization.Dequantize(output, 2f, 5f);

            Assert.Equal(1f, dequantized.Data[0], 4);
            Assert.Equal(2f, dequantized.Data[1], 4);
            Assert.Equal(3f, dequantized.Data[2], 4);
        }
    }
}
