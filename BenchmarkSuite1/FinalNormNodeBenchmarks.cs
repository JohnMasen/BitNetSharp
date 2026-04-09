using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Models;
using BitNetSharp.Nodes;
using Microsoft.VSDiagnostics;
using System;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Benchmarks;

[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class FinalNormNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private FinalNormNode? cpuSingleThreadNode;
    private FinalNormNode? cpuMultiThreadNode;
    private FinalNormNode? tensorSingleThreadNode;
    private FinalNormNode? tensorMultiThreadNode;
    private FinalNormNode? simdSingleThreadNode;
    private FinalNormNode? simdMultiThreadNode;

    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        session = new BitNetSession(model, memoryManager)
        {
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        BenchmarkDataHelper.FillDeterministicValues(session.Embedding.Span, 17);

        cpuSingleThreadNode = CreateNode(InferenceBackend.CPU, 1);
        cpuMultiThreadNode = CreateNode(InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
        tensorSingleThreadNode = CreateNode(InferenceBackend.Tensor, 1);
        tensorMultiThreadNode = CreateNode(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount);
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = CreateNode(InferenceBackend.SIMD, 1);
            simdMultiThreadNode = CreateNode(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
        }
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        memoryManager?.Dispose();
        memoryManager = null;
        model?.Dispose();
        model = null;
        session = null;
        cpuSingleThreadNode = null;
        cpuMultiThreadNode = null;
        tensorSingleThreadNode = null;
        tensorMultiThreadNode = null;
        simdSingleThreadNode = null;
        simdMultiThreadNode = null;
    }

    [Benchmark(Baseline = true)]
    public Memory<float> FinalNorm_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> FinalNorm_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> FinalNorm_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> FinalNorm_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> FinalNorm_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Final norm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> FinalNorm_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Final norm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private FinalNormNode CreateNode(InferenceBackend backend, int threadCount)
    {
        var node = new FinalNormNode(
            model!,
            enableCache: true,
            inferenceConfig: new InferenceConfig(backend, threadCount));
        node.Init();
        return node;
    }

    private Memory<float> Run(FinalNormNode node)
    {
        node.Forward(session!);
        return session!.FinalNormOutput;
    }

}
