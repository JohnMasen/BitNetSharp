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
        var embeddingNode = new EmbeddingNode(model, enableCache: true);
        embeddingNode.Init();
        var setupSession = new BitNetSession(model, memoryManager)
        {
            Tokens = new[]
            {
                0
            },
            CurrentToken = 0,
        };
        session = setupSession;
        embeddingNode.Forward(setupSession);
        var rmsNormNode = new RmsNormNode(model, layerDefinition.AttentionNorm, enableCache: true, inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormNode.Init();
        rmsNormNode.Forward(setupSession);
        var qkvProjectionNode = new QKVProjectionNode(model, layerDefinition.AttentionQueryWeight, layerDefinition.AttentionKeyWeight, layerDefinition.AttentionValueWeight, enableCache: true, inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        qkvProjectionNode.Init();
        qkvProjectionNode.Forward(setupSession);
        cpuSingleThreadNode = CreateNode(layerDefinition, InferenceBackend.CPU, 1);
        cpuMultiThreadNode = CreateNode(layerDefinition, InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
        tensorSingleThreadNode = CreateNode(layerDefinition, InferenceBackend.Tensor, 1);
        tensorMultiThreadNode = CreateNode(layerDefinition, InferenceBackend.Tensor, InferenceConfig.AutoThreadCount);
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = CreateNode(layerDefinition, InferenceBackend.SIMD, 1);
            simdMultiThreadNode = CreateNode(layerDefinition, InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
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
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_SingleThread()
    {
        return Run(cpuSingleThreadNode!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_MultiThread()
    {
        return Run(cpuMultiThreadNode!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_SingleThread()
    {
        return Run(tensorSingleThreadNode!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_MultiThread()
    {
        return Run(tensorMultiThreadNode!);
    }

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

    private AttentionNode CreateNode(BitNetLayerDefinition layerDefinition, InferenceBackend backend, int threadCount)
    {
        var node = new AttentionNode(model!, layerDefinition.AttentionSubNorm, layerDefinition.AttentionOutputWeight, layerDefinition.AttentionOutputScale, layerDefinition.AttentionOutputBias, enableCache: true, inferenceConfig: new InferenceConfig(backend, threadCount));
        node.Init();
        return node;
    }

    private (Memory<float> SubNorm, Memory<float> Output) Run(AttentionNode node)
    {
        node.Forward(session!);
        return (session!.AttentionSubNorm, session.AttentionOutput);
    }
}