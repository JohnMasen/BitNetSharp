using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Core;
using BitNetSharp.Layers;
using BitNetSharp.Models;
using Microsoft.VSDiagnostics;
using System;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Benchmarks;
[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class AttentionLayerBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private AttentionLayer? cpuSingleThreadLayer;
    private AttentionLayer? cpuMultiThreadLayer;
    private AttentionLayer? tensorSingleThreadLayer;
    private AttentionLayer? tensorMultiThreadLayer;
    private AttentionLayer? simdSingleThreadLayer;
    private AttentionLayer? simdMultiThreadLayer;
    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        var layerDefinition = model.GetLayer(0);
        var embeddingLayer = new EmbeddingLayer(model, enableCache: true);
        embeddingLayer.Init();
        var setupSession = new BitNetSession(model, memoryManager)
        {
            Tokens = new[]
            {
                0
            },
            CurrentToken = 0,
        };
        session = setupSession;
        embeddingLayer.Forward(setupSession);
        var rmsNormLayer = new RmsNormLayer(model, layerDefinition.AttentionNorm, enableCache: true, inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormLayer.Init();
        rmsNormLayer.Forward(setupSession);
        var qkvProjectionLayer = new QKVProjectionLayer(model, layerDefinition.AttentionQueryWeight, layerDefinition.AttentionKeyWeight, layerDefinition.AttentionValueWeight, enableCache: true, inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        qkvProjectionLayer.Init();
        qkvProjectionLayer.Forward(setupSession);
        cpuSingleThreadLayer = CreateLayer(layerDefinition, InferenceBackend.CPU, 1);
        cpuMultiThreadLayer = CreateLayer(layerDefinition, InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
        tensorSingleThreadLayer = CreateLayer(layerDefinition, InferenceBackend.Tensor, 1);
        tensorMultiThreadLayer = CreateLayer(layerDefinition, InferenceBackend.Tensor, InferenceConfig.AutoThreadCount);
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadLayer = CreateLayer(layerDefinition, InferenceBackend.SIMD, 1);
            simdMultiThreadLayer = CreateLayer(layerDefinition, InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
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
        cpuSingleThreadLayer = null;
        cpuMultiThreadLayer = null;
        tensorSingleThreadLayer = null;
        tensorMultiThreadLayer = null;
        simdSingleThreadLayer = null;
        simdMultiThreadLayer = null;
    }

    [Benchmark(Baseline = true)]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_SingleThread()
    {
        return Run(cpuSingleThreadLayer!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_CPU_MultiThread()
    {
        return Run(cpuMultiThreadLayer!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_SingleThread()
    {
        return Run(tensorSingleThreadLayer!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_Tensor_MultiThread()
    {
        return Run(tensorMultiThreadLayer!);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_SIMD_SingleThread()
    {
        if (simdSingleThreadLayer is null)
        {
            throw new PlatformNotSupportedException("Attention SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadLayer);
    }

    [Benchmark]
    public (Memory<float> SubNorm, Memory<float> Output) Attention_SIMD_MultiThread()
    {
        if (simdMultiThreadLayer is null)
        {
            throw new PlatformNotSupportedException("Attention SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadLayer);
    }

    private AttentionLayer CreateLayer(BitNetLayerDefinition layerDefinition, InferenceBackend backend, int threadCount)
    {
        var layer = new AttentionLayer(model!, layerDefinition.AttentionSubNorm, layerDefinition.AttentionOutputWeight, layerDefinition.AttentionOutputScale, layerDefinition.AttentionOutputBias, enableCache: true, inferenceConfig: new InferenceConfig(backend, threadCount));
        layer.Init();
        return layer;
    }

    private (Memory<float> SubNorm, Memory<float> Output) Run(AttentionLayer layer)
    {
        layer.Forward(session!);
        return (session!.AttentionSubNorm, session.AttentionOutput);
    }
}