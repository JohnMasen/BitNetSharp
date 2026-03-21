using System;
using Xunit;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    public class TensorTests
    {
        [Fact]
        public void MatMul_CorrectResult()
        {
            // 2x3 @ 3x2 = 2x2
            var a = new Tensor(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var b = new Tensor(new[] { 3, 2 }, new float[] { 7, 8, 9, 10, 11, 12 });

            Tensor c = Tensor.MatMul(a, b);

            Assert.Equal(new[] { 2, 2 }, c.Shape);
            Assert.Equal(1*7 + 2*9 + 3*11, c[0, 0], 4);
            Assert.Equal(1*8 + 2*10 + 3*12, c[0, 1], 4);
            Assert.Equal(4*7 + 5*9 + 6*11, c[1, 0], 4);
            Assert.Equal(4*8 + 5*10 + 6*12, c[1, 1], 4);
        }

        [Fact]
        public void Transpose_CorrectShape()
        {
            var t = new Tensor(new[] { 3, 4 }, new float[12]);
            Tensor transposed = t.Transpose();
            Assert.Equal(new[] { 4, 3 }, transposed.Shape);
        }

        [Fact]
        public void Add_ElementWise()
        {
            var a = new Tensor(new[] { 3 }, new float[] { 1, 2, 3 });
            var b = new Tensor(new[] { 3 }, new float[] { 10, 20, 30 });
            Tensor c = Tensor.Add(a, b);
            Assert.Equal(new float[] { 11, 22, 33 }, c.Data);
        }

        [Fact]
        public void Reshape_SameElements()
        {
            var t = new Tensor(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            Tensor r = t.Reshape(3, 2);
            Assert.Equal(new[] { 3, 2 }, r.Shape);
            Assert.Equal(t.Data, r.Data);
        }

        [Fact]
        public void Reshape_InvalidSizeThrows()
        {
            var t = new Tensor(new[] { 2, 3 });
            Assert.Throws<ArgumentException>(() => t.Reshape(2, 2));
        }

        [Fact]
        public void MatMul_ShapeMismatchThrows()
        {
            var a = new Tensor(new[] { 2, 3 });
            var b = new Tensor(new[] { 4, 2 });
            Assert.Throws<ArgumentException>(() => Tensor.MatMul(a, b));
        }
    }
}
