using BenchmarkDotNet.Running;

namespace BitNetSharp.Benchmarks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            BenchmarkRunner.Run(typeof(Program).Assembly, args: args);
        }
    }
}
