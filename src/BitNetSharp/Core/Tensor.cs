using System;
using System.Runtime.CompilerServices;

namespace BitNetSharp.Core
{
    /// <summary>
    /// A lightweight multi-dimensional tensor backed by a flat float array.
    /// Supports 1-D and 2-D operations required by BitNet layers.
    /// </summary>
    public sealed class Tensor
    {
        public float[] Data { get; }
        public int[] Shape { get; }

        public int Rows => Shape.Length >= 2 ? Shape[0] : 1;
        public int Cols => Shape.Length >= 2 ? Shape[1] : Shape[0];
        public int Size => Data.Length;

        public Tensor(int[] shape)
        {
            Shape = shape;
            int size = 1;
            foreach (int d in shape)
                size *= d;
            Data = new float[size];
        }

        public Tensor(int[] shape, float[] data)
        {
            Shape = shape;
            Data = data;
        }

        public float this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data[row * Cols + col];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Data[row * Cols + col] = value;
        }

        public float this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data[index];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Data[index] = value;
        }

        /// <summary>Performs matrix multiplication C = A * B where A is (M, K) and B is (K, N).</summary>
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            int m = a.Rows;
            int k = a.Cols;
            int n = b.Cols;

            if (k != b.Rows)
                throw new ArgumentException($"Shape mismatch: ({m},{k}) x ({b.Rows},{n})");

            var result = new Tensor(new[] { m, n });

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                        sum += a[i, p] * b[p, j];
                    result[i, j] = sum;
                }
            }

            return result;
        }

        /// <summary>Element-wise addition of two tensors with identical shapes.</summary>
        public static Tensor Add(Tensor a, Tensor b)
        {
            if (a.Size != b.Size)
                throw new ArgumentException("Tensor sizes must match for element-wise addition.");

            var result = new Tensor(a.Shape, new float[a.Size]);
            for (int i = 0; i < a.Size; i++)
                result.Data[i] = a.Data[i] + b.Data[i];
            return result;
        }

        /// <summary>Element-wise multiplication (Hadamard product).</summary>
        public static Tensor Multiply(Tensor a, Tensor b)
        {
            if (a.Size != b.Size)
                throw new ArgumentException("Tensor sizes must match for element-wise multiplication.");

            var result = new Tensor(a.Shape, new float[a.Size]);
            for (int i = 0; i < a.Size; i++)
                result.Data[i] = a.Data[i] * b.Data[i];
            return result;
        }

        /// <summary>Returns a transposed copy (2-D only).</summary>
        public Tensor Transpose()
        {
            var result = new Tensor(new[] { Cols, Rows });
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[j, i] = this[i, j];
            return result;
        }

        /// <summary>Applies a scalar function to every element and returns a new tensor.</summary>
        public Tensor Map(Func<float, float> fn)
        {
            var result = new Tensor(Shape, new float[Size]);
            for (int i = 0; i < Size; i++)
                result.Data[i] = fn(Data[i]);
            return result;
        }

        /// <summary>Creates a deep copy.</summary>
        public Tensor Clone()
        {
            var data = new float[Size];
            Data.CopyTo(data, 0);
            return new Tensor((int[])Shape.Clone(), data);
        }

        /// <summary>Reshapes to a new shape with the same number of elements.</summary>
        public Tensor Reshape(params int[] newShape)
        {
            int newSize = 1;
            foreach (int d in newShape)
                newSize *= d;
            if (newSize != Size)
                throw new ArgumentException("Total number of elements must not change during reshape.");
            var data = new float[Size];
            Data.CopyTo(data, 0);
            return new Tensor(newShape, data);
        }
    }
}
