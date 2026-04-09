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
public class QKVProjectionNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private QKVProjectionNode? cpuSingleThreadNode;
    private QKVProjectionNode? cpuMultiThreadNode;
    private QKVProjectionNode? tensorSingleThreadNode;
    private QKVProjectionNode? tensorMultiThreadNode;
    private QKVProjectionNode? simdSingleThreadNode;
    private QKVProjectionNode? simdMultiThreadNode;

    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        var layerDefinition = model.GetLayer(0);
        var embeddingNode = new EmbeddingNode(model, enableCache: true);
        embeddingNode.Init();
        var session = new BitNetSession(model, memoryManager)
        {
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        this.session = session;

        embeddingNode.Forward(session);
        var rmsNormNode = new RmsNormNode(
            model,
            layerDefinition.AttentionNorm,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormNode.Init();
        rmsNormNode.Forward(session);
        cpuSingleThreadNode = new QKVProjectionNode(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        cpuSingleThreadNode.Init();
        cpuMultiThreadNode = new QKVProjectionNode(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, InferenceConfig.AutoThreadCount));
        cpuMultiThreadNode.Init();
        tensorSingleThreadNode = new QKVProjectionNode(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, 1));
        tensorSingleThreadNode.Init();
        tensorMultiThreadNode = new QKVProjectionNode(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount));
        tensorMultiThreadNode.Init();
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = new QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, 1));
            simdSingleThreadNode.Init();
            simdMultiThreadNode = new QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount));
            simdMultiThreadNode.Init();
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
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_CPU_SingleThread()
    {
        return Run(cpuSingleThreadNode!);
    }

    [Benchmark]
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_CPU_MultiThread()
    {
        return Run(cpuMultiThreadNode!);
    }

    [Benchmark]
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_Tensor_SingleThread()
    {
        return Run(tensorSingleThreadNode!);
    }

    [Benchmark]
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_Tensor_MultiThread()
    {
        return Run(tensorMultiThreadNode!);
    }

    [Benchmark]
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("QKV SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public (Memory<float> Query, Memory<float> Key, Memory<float> Value) QKV_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("QKV SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private (Memory<float> Query, Memory<float> Key, Memory<float> Value) Run(QKVProjectionNode node)
    {
        node.Forward(session!);
        return (session!.QKVQuery, session.QKVKey, session.QKVValue);
    }
}
