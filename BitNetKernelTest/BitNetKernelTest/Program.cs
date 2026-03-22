using System.Numerics.Tensors;
using System.Text;
using BitNetKernelTest;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

var kernel = new Kernel();
int itemsCount = 100;
int[] data=new int[itemsCount];
data.AsSpan().Fill(1);
byte[] weights= Random.Shared.GetItems<byte>([ 0, 0b_0000_0001 ,0b_0000_0011], itemsCount);


Console.WriteLine($"Basic Test {kernel.BasicTest(data, weights)}");
testBlockAligned();
testBlockNotAligned();


void testBlockAligned()
{
    sbyte[] b1 = Enumerable.Repeat<sbyte>(1, 128).ToArray();
    Random.Shared.NextBytes(MemoryMarshal.AsBytes(b1.AsSpan()));
    byte[] b2 = Enumerable.Repeat<byte>(0b_1010_0001, 32).ToArray();
    Span<sbyte> tb1 = new Span<sbyte>(b1);
    Span<byte> tb2 = new Span<byte>(b2);
    Console.WriteLine($"BasicProcessOne:{kernel.BasicProcessOne(tb1, tb2)}");
    Console.WriteLine($"TensorProcessOne:{kernel.TensorProcessOne(b1, b2)}");
    Console.WriteLine($"VectorProcessOne:{kernel.VectorProcessOne(b1, b2)}");
    Console.WriteLine($"VectorProcessOne_Test2:{kernel.VectorProcessOne_Test2(b1, b2)}");
}
void testBlockNotAligned()
{
    sbyte[] b1 = Enumerable.Repeat<sbyte>(1, 32).ToArray();
    Random.Shared.NextBytes(MemoryMarshal.AsBytes(b1.AsSpan()));
    byte[] b2 = Enumerable.Repeat<byte>(0b_1010_0001, 8).ToArray();
    Span<sbyte> tb1 = new Span<sbyte>(b1);
    Span<byte> tb2 = new Span<byte>(b2);
    Console.WriteLine($"BasicProcessOne:{kernel.BasicProcessOne(tb1, tb2)}");
    Console.WriteLine($"TensorProcessOne:{kernel.TensorProcessOne(b1, b2)}");
}