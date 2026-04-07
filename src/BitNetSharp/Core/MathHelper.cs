using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core
{
    public partial class MathHelper
    {
        public static  int VectorProcessOne(scoped ReadOnlySpan<sbyte> dataBlock, scoped ReadOnlySpan<byte> weightBlock)
        {
            if (dataBlock.Length != 128)
            {
                throw new ArgumentException("dataBlock length must be 128");
            }
            if (weightBlock.Length != 32)
            {
                throw new ArgumentException("weightBlock length must be 32");
            }
            //load data
            var data0 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock));
            var data1 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(32)));
            var data2 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(64)));
            var data3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(dataBlock.Slice(96)));
            var bitMask = Vector256.Create<byte>(0b_0000_0011);
            //load weight
            var weight3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weightBlock));

            var weight0 = Vector256.ShiftRightLogical(weight3, 6);
            weight0 = Vector256.BitwiseAnd(weight0, bitMask);

            var weight1 = Vector256.ShiftRightLogical(weight3, 4);
            weight1 = Vector256.BitwiseAnd(weight1, bitMask);

            var weight2 = Vector256.ShiftRightLogical(weight3, 2);
            weight2 = Vector256.BitwiseAnd(weight2, bitMask);

            weight3 = Vector256.BitwiseAnd(weight3, bitMask);

            //fast compress test



            return sum4(
                processBlock(data0, weight0),
                 processBlock(data1, weight1),
                 processBlock(data2, weight2),
                 processBlock(data3, weight3));
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static Vector256<int> processBlock(Vector256<sbyte> data, Vector256<byte> weight)
            {
                //result=data * block - data 
                //map logic
                // x*(-1|0|1)=x*[(0,1,2)-1]
                var result0 = Avx2.MultiplyAddAdjacent(weight, data);
                var sData = Avx2.MultiplyAddAdjacent(Vector256<byte>.One, data);//expand sbyte to short
                var w = Avx2.Subtract(result0, sData);

                //expand to int32 with adjacent add, prepare for sum
                return Avx2.MultiplyAddAdjacent(w, Vector256<short>.One);
            }

            //sums values in each Vector 256
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static int sum4(Vector256<int> v1, Vector256<int> v2, Vector256<int> v3, Vector256<int> v4)
            {

                var r = Avx2.Add(Avx2.Add(v1, v2), Avx2.Add(v3, v4));
                r = Avx2.HorizontalAdd(r, r);
                r = Avx2.HorizontalAdd(r, r);
                Vector256<int> r1 = Avx2.Permute2x128(r, r, 0b_0000_0001);//switch lanes
                return Avx2.Add(r, r1).ToScalar();
            }



        }
    }
}
