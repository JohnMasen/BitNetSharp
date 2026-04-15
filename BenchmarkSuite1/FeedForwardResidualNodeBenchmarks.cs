using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Nodes;
using BitNetSharp.Models;
using Microsoft.VSDiagnostics;
using System;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Benchmarks;

[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class FeedForwardResidualNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private FeedForwardResidualNode? cpuSingleThreadNode;
    private FeedForwardResidualNode? cpuMultiThreadNode;
    private FeedForwardResidualNode? tensorSingleThreadNode;
    private FeedForwardResidualNode? tensorMultiThreadNode;
    private FeedForwardResidualNode? simdSingleThreadNode;
    private FeedForwardResidualNode? simdMultiThreadNode;

    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        session = new BitNetSession(model, memoryManager, new[] { 0 });
        BenchmarkDataHelper.FillDeterministicValues(session.FeedForwardInput.Span, 11);
        BenchmarkDataHelper.FillDeterministicValues(session.FeedForwardOutput.Span, 13);

        cpuSingleThreadNode = CreateNode(BenchmarkInferenceConfigs.Cpu(1));
        cpuMultiThreadNode = CreateNode(BenchmarkInferenceConfigs.Cpu(InferenceConfig.AutoThreadCount));
        tensorSingleThreadNode = CreateNode(BenchmarkInferenceConfigs.Tensor(1));
        tensorMultiThreadNode = CreateNode(BenchmarkInferenceConfigs.Tensor(InferenceConfig.AutoThreadCount));
        if (Avx.IsSupported)
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
    public Memory<float> FeedForwardResidual_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardResidual_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardResidual_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardResidual_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardResidual_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward residual SIMD benchmark requires AVX support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> FeedForwardResidual_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward residual SIMD benchmark requires AVX support.");
        }

        return Run(simdMultiThreadNode);
    }

    private FeedForwardResidualNode CreateNode(InferenceConfig inferenceConfig)
    {
        var node = new FeedForwardResidualNode(model!, inferenceConfig);
        node.Init();
        return node;
    }

    private Memory<float> Run(FeedForwardResidualNode node)
    {
        node.Forward(session!);
        return session!.Embedding;
    }

}
