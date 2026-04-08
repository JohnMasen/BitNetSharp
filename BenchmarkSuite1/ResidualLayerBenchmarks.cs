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
public class ResidualLayerBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private ResidualLayer? cpuSingleThreadLayer;
    private ResidualLayer? cpuMultiThreadLayer;
    private ResidualLayer? tensorSingleThreadLayer;
    private ResidualLayer? tensorMultiThreadLayer;
    private ResidualLayer? simdSingleThreadLayer;
    private ResidualLayer? simdMultiThreadLayer;

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
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        session = setupSession;

        embeddingLayer.Forward(setupSession);
        var rmsNormLayer = new RmsNormLayer(
            model,
            layerDefinition.AttentionNorm,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormLayer.Init();
        rmsNormLayer.Forward(setupSession);

        var qkvProjectionLayer = new QKVProjectionLayer(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        qkvProjectionLayer.Init();
        qkvProjectionLayer.Forward(setupSession);

        var attentionLayer = new AttentionLayer(
            model,
            layerDefinition.AttentionSubNorm,
            layerDefinition.AttentionOutputWeight,
            layerDefinition.AttentionOutputScale,
            layerDefinition.AttentionOutputBias,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        attentionLayer.Init();
        attentionLayer.Forward(setupSession);

        cpuSingleThreadLayer = CreateLayer(InferenceBackend.CPU, 1);
        cpuMultiThreadLayer = CreateLayer(InferenceBackend.CPU, InferenceConfig.AutoThreadCount);
        tensorSingleThreadLayer = CreateLayer(InferenceBackend.Tensor, 1);
        tensorMultiThreadLayer = CreateLayer(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount);
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadLayer = CreateLayer(InferenceBackend.SIMD, 1);
            simdMultiThreadLayer = CreateLayer(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount);
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
    public Memory<float> Residual_CPU_SingleThread() => Run(cpuSingleThreadLayer!);

    [Benchmark]
    public Memory<float> Residual_CPU_MultiThread() => Run(cpuMultiThreadLayer!);

    [Benchmark]
    public Memory<float> Residual_Tensor_SingleThread() => Run(tensorSingleThreadLayer!);

    [Benchmark]
    public Memory<float> Residual_Tensor_MultiThread() => Run(tensorMultiThreadLayer!);

    [Benchmark]
    public Memory<float> Residual_SIMD_SingleThread()
    {
        if (simdSingleThreadLayer is null)
        {
            throw new PlatformNotSupportedException("Residual SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadLayer);
    }

    [Benchmark]
    public Memory<float> Residual_SIMD_MultiThread()
    {
        if (simdMultiThreadLayer is null)
        {
            throw new PlatformNotSupportedException("Residual SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadLayer);
    }

    private ResidualLayer CreateLayer(InferenceBackend backend, int threadCount)
    {
        var layer = new ResidualLayer(model!, new InferenceConfig(backend, threadCount));
        layer.Init();
        return layer;
    }

    private Memory<float> Run(ResidualLayer layer)
    {
        layer.Forward(session!);
        return session!.FeedForwardInput;
    }
}
