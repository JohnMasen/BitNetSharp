using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Core;
using BitNetSharp.Nodes;
using BitNetSharp.Models;
using Microsoft.VSDiagnostics;
using System;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Benchmarks;

[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class ResidualNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private ResidualNode? cpuSingleThreadNode;
    private ResidualNode? cpuMultiThreadNode;
    private ResidualNode? tensorSingleThreadNode;
    private ResidualNode? tensorMultiThreadNode;
    private ResidualNode? simdSingleThreadNode;
    private ResidualNode? simdMultiThreadNode;

    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        session = new BitNetSession(model, memoryManager, new[] { 0 });
        BenchmarkDataHelper.FillDeterministicValues(session.Embedding.Span, 37);
        BenchmarkDataHelper.FillDeterministicValues(session.AttentionOutput.Span, 41);

        cpuSingleThreadNode = CreateNode(BenchmarkInferenceConfigs.Cpu(1));
        cpuMultiThreadNode = CreateNode(BenchmarkInferenceConfigs.Cpu(InferenceConfig.AutoThreadCount));
        tensorSingleThreadNode = CreateNode(BenchmarkInferenceConfigs.Tensor(1));
        tensorMultiThreadNode = CreateNode(BenchmarkInferenceConfigs.Tensor(InferenceConfig.AutoThreadCount));
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = CreateNode(BenchmarkInferenceConfigs.Simd(1));
            simdMultiThreadNode = CreateNode(BenchmarkInferenceConfigs.Simd(InferenceConfig.AutoThreadCount));
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
    public Memory<float> Residual_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> Residual_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> Residual_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> Residual_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> Residual_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Residual SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> Residual_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Residual SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private ResidualNode CreateNode(InferenceConfig inferenceConfig)
    {
        var node = new ResidualNode(model!, inferenceConfig);
        node.Init();
        return node;
    }

    private Memory<float> Run(ResidualNode node)
    {
        node.Forward(session!);
        return session!.FeedForwardInput;
    }
}
