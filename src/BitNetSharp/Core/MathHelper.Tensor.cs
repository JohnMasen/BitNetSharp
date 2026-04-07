using System;
using System.Numerics.Tensors;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormTensor(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon)
        {
            float sumSquares = TensorPrimitives.Dot(input, input);
            float meanSquare = sumSquares / input.Length;
            float inverseRootMeanSquare = 1f / MathF.Sqrt(meanSquare + epsilon);

            float[] normalized = new float[input.Length];
            TensorPrimitives.Multiply(input, inverseRootMeanSquare, normalized);

            float[] output = new float[input.Length];
            TensorPrimitives.Multiply(normalized, normWeights, output);
            return output;
        }
    }
}
