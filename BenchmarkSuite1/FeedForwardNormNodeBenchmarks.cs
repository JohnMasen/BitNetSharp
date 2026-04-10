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
public class FeedForwardNormNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private FeedForwardNormNode? cpuSingleThreadNode;
    private FeedForwardNormNode? cpuMultiThreadNode;
    private FeedForwardNormNode? tensorSingleThreadNode;
    private FeedForwardNormNode? tensorMultiThreadNode;
    private FeedForwardNormNode? simdSingleThreadNode;
    private FeedForwardNormNode? simdMultiThreadNode;

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
        BenchmarkDataHelper.FillDeterministicValues(session.FeedForwardInput.Span, 7);

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
    public Memory<float> FeedForwardNorm_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardNorm_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardNorm_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardNorm_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> FeedForwardNorm_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward norm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> FeedForwardNorm_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Feed-forward norm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private FeedForwardNormNode CreateNode(BitNetLayerDefinition layerDefinition, InferenceConfig inferenceConfig)
    {
        var node = new FeedForwardNormNode(
            model!,
            layerDefinition.FeedForwardNorm,
            enableCache: true,
            inferenceConfig: inferenceConfig);
        node.Init();
        return node;
    }

    private Memory<float> Run(FeedForwardNormNode node)
    {
        node.Forward(session!);
        return session!.FeedForwardNorm;
    }

}
