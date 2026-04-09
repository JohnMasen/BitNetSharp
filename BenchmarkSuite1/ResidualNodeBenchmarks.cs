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

        var layerDefinition = model.GetLayer(0);
        var embeddingNode = new EmbeddingNode(model, enableCache: true);
        embeddingNode.Init();
        var setupSession = new BitNetSession(model, memoryManager)
        {
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        session = setupSession;

        embeddingNode.Forward(setupSession);
        var rmsNormNode = new RmsNormNode(
            model,
            layerDefinition.AttentionNorm,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormNode.Init();
        rmsNormNode.Forward(setupSession);

        var qkvProjectionNode = new QKVProjectionNode(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        qkvProjectionNode.Init();
        qkvProjectionNode.Forward(setupSession);

        var attentionNode = new AttentionNode(
            model,
            layerDefinition.AttentionSubNorm,
            layerDefinition.AttentionOutputWeight,
            layerDefinition.AttentionOutputScale,
            layerDefinition.AttentionOutputBias,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        attentionNode.Init();
        attentionNode.Forward(setupSession);

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

    private ResidualNode CreateNode(InferenceBackend backend, int threadCount)
    {
        var node = new ResidualNode(model!, new InferenceConfig(backend, threadCount));
        node.Init();
        return node;
    }

    private Memory<float> Run(ResidualNode node)
    {
        node.Forward(session!);
        return session!.FeedForwardInput;
    }
}
