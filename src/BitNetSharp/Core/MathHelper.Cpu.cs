using System;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static float[] ForwardRmsNormCpuStandard(ReadOnlySpan<float> input, ReadOnlySpan<float> normWeights, float epsilon)
        {
            double sumSquares = 0;
            for (int index = 0; index < input.Length; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            double meanSquare = sumSquares / input.Length;
            double inverseRootMeanSquare = 1d / Math.Sqrt(meanSquare + epsilon);
            float[] output = new float[input.Length];
            for (int index = 0; index < input.Length; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * normWeights[index]);
            }

            return output;
        }
    }
}
