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
public class AttentionNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private AttentionNode? cpuSingleThreadNode;
    private AttentionNode? cpuMultiThreadNode;
    private AttentionNode? tensorSingleThreadNode;
    private AttentionNode? tensorMultiThreadNode;
    private AttentionNode? simdSingleThreadNode;
    private AttentionNode? simdMultiThreadNode;
    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        var layerDefinition = model.GetLayer(0);
        session = new BitNetSession(model, memoryManager, new[] { 0 });
        BenchmarkDataHelper.FillDeterministicValues(session.QKVQuery.Span, 23);
        BenchmarkDataHelper.FillDeterministicValues(session.QKVKey.Span, 29);
        BenchmarkDataHelper.FillDeterministicValues(session.QKVValue.Span, 31);
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
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("Attention SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("Attention SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private AttentionNode CreateNode(BitNetLayerDefinition layerDefinition, InferenceConfig inferenceConfig)
    {
        var node = new AttentionNode(
            model!,
            layerDefinition.AttentionSubNorm,
            layerDefinition.AttentionOutputWeight,
            layerDefinition.AttentionOutputScale,
            layerDefinition.AttentionOutputBias,
            enableCache: true,
            inferenceConfig: inferenceConfig);
        node.Init();
        return node;
    }

    private (Memory<float> SubNorm, Memory<float> Output) Run(AttentionNode node)
    {
        node.Forward(session!);
        return (session!.AttentionSubNorm, session.AttentionOutput);
    }
}