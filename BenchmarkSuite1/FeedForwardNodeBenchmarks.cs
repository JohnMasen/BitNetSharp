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
public class FeedForwardNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private FeedForwardNode? cpuSingleThreadNode;
    private FeedForwardNode? cpuMultiThreadNode;
    private FeedForwardNode? tensorSingleThreadNode;
    private FeedForwardNode? tensorMultiThreadNode;
    private FeedForwardNode? simdSingleThreadNode;
    private FeedForwardNode? simdMultiThreadNode;

    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        var layerDefinition = model.GetLayer(0);
        session = new BitNetSession(model, memoryManager)
        {
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        BenchmarkDataHelper.FillDeterministicValues(session.FeedForwardNorm.Span, 5);

        cpuSingleThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Cpu(1));
        cpuMultiThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Cpu(InferenceConfig.AutoThreadCount));
        tensorSingleThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Tensor(1));
        tensorMultiThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Tensor(InferenceConfig.AutoThreadCount));
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Simd(1));
            simdMultiThreadNode = CreateNode(layerDefinition, BenchmarkInferenceConfigs.Simd(InferenceConfig.AutoThreadCount));
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
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) FeedForward_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private FeedForwardNode CreateNode(BitNetLayerDefinition layerDefinition, InferenceConfig inferenceConfig)
    {
        var node = new FeedForwardNode(
            model!,
            layerDefinition.FeedForwardSubNorm,
            layerDefinition.FeedForwardGateWeight,
            layerDefinition.FeedForwardUpWeight,
            layerDefinition.FeedForwardDownWeight,
            enableCache: true,
            inferenceConfig: inferenceConfig);
        node.Init();
        return node;
    }

    private (Memory<float> SubNorm, Memory<float> Output) Run(FeedForwardNode node)
    {
        node.Forward(session!);
        return (session!.FeedForwardSubNorm, session.FeedForwardOutput);
    }

}
