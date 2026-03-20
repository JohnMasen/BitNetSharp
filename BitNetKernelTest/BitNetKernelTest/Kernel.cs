using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using Zyl.VectorTraits;

namespace BitNetKernelTest;

public class Kernel
{
    //private int[] lookupMap = new int[256];

    //public Kernel()
    //{
    //    buildLookupTable();
    //}

    public int BasicTest(ReadOnlySpan<int> left, ReadOnlySpan<byte> right)
    {
        int result = 0;
        for (int i = 0; i < left.Length; i++)
        {
            result += left[i] * right[i] switch
            {
                0 => 0,
                0b_0000_0001 => 1,
                _ => -1
            };
        }
        return result;
    }

    public int BasicProcessOne(scoped ReadOnlySpan<sbyte> dataBlock, scoped ReadOnlySpan<byte> weightBlock)
    {
        if (dataBlock.Length != weightBlock.Length * 4)
        {
            throw new ArgumentException("weightBlock.Length should be 4 * dataBlock.Length");
        }
        int idx = 0;
        int result = 0;
        for (int i = 0; i < 4; i++) //group 0 - group 4
        {
            for (var j = 0; j < weightBlock.Length; j++) //byte 0 - byte 32
            {
                int w = (weightBlock[j] >> (6 - i * 2) & 0b_0000_0011);
                int v = dataBlock[idx++];
                switch (w)
                {
                    case 0b_0000:
                        result -= v;
                        break;
                    case 0b_0001:
                        break;
                    case 0b_0010:
                        result += v;
                        break;
                    default:
                        throw new ArgumentException("Invalid bitmask [0x11] in weightBlock");
                }
            }
        }
        return result;


    }

    public int TensorProcessOne(scoped ReadOnlySpan<sbyte> dataBlock, scoped ReadOnlySpan<byte> weightBlock)
    {
        if (dataBlock.Length != weightBlock.Length * 4 )
        {
            throw new ArgumentException("weightBlock.Length should be 4 * dataBlock.Length");
        }

        TensorSpan<short> dataSpan = Tensor.ConvertTruncating<sbyte, short>(new ReadOnlyTensorSpan<sbyte>(dataBlock, [4, weightBlock.Length]));
        TensorSpan<short> weightSpan = Tensor.ConvertTruncating<byte, short>(new ReadOnlyTensorSpan<byte>(weightBlock));//load 32 bytes packed weight
        TensorSpan<short> wt = Tensor.CreateFromShapeUninitialized<short>([4, weightBlock.Length]);

        //map logic
        // x*(-1|0|1)=x*[(0,1,2)-1]
        Tensor.BitwiseAnd<short>(weightSpan >> 6, 0b_0000_0011, wt[0..1, ..]);  //group 0
        Tensor.BitwiseAnd<short>(weightSpan >> 4, 0b_0000_0011, wt[1..2, ..]);  //group 1
        Tensor.BitwiseAnd<short>(weightSpan >> 2, 0b_0000_0011, wt[2..3, ..]);  //group 2
        Tensor.BitwiseAnd<short>(weightSpan, 0b_0000_0011, wt[3..4, ..]);      //group 3
        wt -= (short)1;

        return Tensor.Dot<short>(dataSpan, wt);
    }

    public int VectorProcessOne(scoped ReadOnlySpan<sbyte> dataBlock, scoped ReadOnlySpan<byte> weightBlock)
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
        var bitMask = Vector256.Create<byte>(3);
        //load weight
        var weight3 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weightBlock));

        var weight0 = Vector256.ShiftRightLogical(weight3, 6);
        weight0 = Vector256.BitwiseAnd(weight0, bitMask);

        var weight1 = Vector256.ShiftRightLogical(weight3, 4);
        weight1 = Vector256.BitwiseAnd(weight1, bitMask);

        var weight2 = Vector256.ShiftRightLogical(weight3, 2);
        weight2 = Vector256.BitwiseAnd(weight2, bitMask);

        weight3 = Vector256.BitwiseAnd(weight3, bitMask);
        return 
            processBlock(ref data0, ref weight0) +
             processBlock(ref data1, ref weight1)+
             processBlock(ref data2, ref weight2)+
             processBlock(ref data3, ref weight3);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int processBlock(ref Vector256<sbyte> data, ref Vector256<byte> weight)
        {
            //result=data * block - data 
            //map logic
            // x*(-1|0|1)=x*[(0,1,2)-1]
            var result0 = Avx2.MultiplyAddAdjacent(weight, data);
            var sData = Avx2.MultiplyAddAdjacent(Vector256<byte>.One, data);
            var w = Avx2.Subtract(result0, sData);

            //sum all elements
            var wr = Avx2.MultiplyAddAdjacent(w, Vector256<short>.One);
            wr = Avx2.HorizontalAdd(wr, wr);
            wr = Avx2.HorizontalAdd(wr, wr);
            //wr0 = Avx2.HorizontalAdd(wr0, wr0);
            return wr[0] + wr[4];//special process, Avx2 will broadcast data to fill 256 bit, this causes position gap
        }



    }






}
