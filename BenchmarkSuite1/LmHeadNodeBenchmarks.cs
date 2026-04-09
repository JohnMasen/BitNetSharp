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
public class LmHeadNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private LmHeadNode? cpuSingleThreadNode;
    private LmHeadNode? cpuMultiThreadNode;
    private LmHeadNode? tensorSingleThreadNode;
    private LmHeadNode? tensorMultiThreadNode;
    private LmHeadNode? simdSingleThreadNode;
    private LmHeadNode? simdMultiThreadNode;

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
        BenchmarkDataHelper.FillDeterministicValues(session.FinalNormOutput.Span, 19);

        cpuSingleThreadNode = CreateNode(InferenceBackend.CPU, 1);
        cpuMultiThreadNode = CreateNode(InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
        tensorSingleThreadNode = CreateNode(InferenceBackend.Tensor, 1);
        tensorMultiThreadNode = CreateNode(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount);
        if (Avx.IsSupported)
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
    public Memory<float> LmHead_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> LmHead_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> LmHead_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> LmHead_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> LmHead_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("LM head SIMD benchmark requires AVX support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> LmHead_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("LM head SIMD benchmark requires AVX support.");
        }

        return Run(simdMultiThreadNode);
    }

    private LmHeadNode CreateNode(InferenceBackend backend, int threadCount)
    {
        var node = new LmHeadNode(
            model!,
            enableCache: true,
            inferenceConfig: new InferenceConfig(backend, threadCount));
        node.Init();
        return node;
    }

    private Memory<float> Run(LmHeadNode node)
    {
        node.Forward(session!);
        return session!.Logits;
    }

}
