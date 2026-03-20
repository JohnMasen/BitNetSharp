using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace BitNetKernelTest;

public static class TensorHelper
{
    extension<T>(Tensor<T> source)
    {
        public string Print()
        {
            StringBuilder sb = new StringBuilder();
            int pos = 0;
            dumpTensorDimention(source, 0, ref pos, sb);
            return sb.ToString();
        }

    }

    private static void dumpTensorDimention<T>(Tensor<T> data, int dimention, ref int pos, StringBuilder sb)
    {
        sb.Append("[");
        if (dimention == data.Rank - 1) //last layer
        {
            for (int i = 0; i < data.Lengths[dimention]; i++)
            {
                sb.Append(data.ElementAt(pos++));
                sb.Append(",");
            }
            sb.Length--;//remove last comma
        }
        else
        {
            for (int i = 0; i < data.Lengths[dimention]; i++)
            {
                dumpTensorDimention(data, dimention + 1, ref pos, sb);
            }
        }
        sb.Append("]");
    }
}
