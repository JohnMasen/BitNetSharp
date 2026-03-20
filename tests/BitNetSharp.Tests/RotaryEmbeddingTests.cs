using System;
using Xunit;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class RotaryEmbeddingTests
    {
        [Fact]
        public void Apply_DoesNotChangeShape()
        {
            var rope = new RotaryEmbedding(8, 32);
            var tensor = new Tensor(new[] { 4, 8 });
            for (int i = 0; i < tensor.Size; i++)
                tensor.Data[i] = i + 1f;

            rope.Apply(tensor);

            Assert.Equal(new[] { 4, 8 }, tensor.Shape);
        }

        [Fact]
        public void Apply_PreservesNorm()
        {
            // RoPE rotates pairs of dimensions; it should preserve the L2 norm per position
            var rope = new RotaryEmbedding(4, 32);
            var tensor = new Tensor(new[] { 3, 4 });
            var rng = new Random(1);
            for (int i = 0; i < tensor.Size; i++)
                tensor.Data[i] = (float)(rng.NextDouble() * 2 - 1);

            // Compute norms before
            float[] normsBefore = ComputeRowNorms(tensor);

            rope.Apply(tensor);

            // Compute norms after
            float[] normsAfter = ComputeRowNorms(tensor);

            for (int r = 0; r < 3; r++)
                Assert.Equal(normsBefore[r], normsAfter[r], 4);
        }

        [Fact]
        public void Constructor_OddHeadDimThrows()
        {
            Assert.Throws<ArgumentException>(() => new RotaryEmbedding(3, 32));
        }

        [Fact]
        public void Apply_WrongColumnCountThrows()
        {
            var rope = new RotaryEmbedding(8, 32);
            var tensor = new Tensor(new[] { 4, 4 });   // headDim mismatch
            Assert.Throws<ArgumentException>(() => rope.Apply(tensor));
        }

        private static float[] ComputeRowNorms(Tensor t)
        {
            int rows = t.Rows;
            int cols = t.Cols;
            float[] norms = new float[rows];
            for (int r = 0; r < rows; r++)
            {
                float sum = 0f;
                for (int c = 0; c < cols; c++)
                    sum += t[r, c] * t[r, c];
                norms[r] = MathF.Sqrt(sum);
            }
            return norms;
        }
    }
}
