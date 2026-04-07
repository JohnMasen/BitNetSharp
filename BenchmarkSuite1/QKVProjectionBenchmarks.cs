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
public class QKVProjectionBenchmarks
{
    private BitNetModel? model;
    private BitNetSession? session;
    private QKVProjectionLayer? cpuSingleThreadLayer;
    private QKVProjectionLayer? cpuMultiThreadLayer;
    private QKVProjectionLayer? tensorSingleThreadLayer;
    private QKVProjectionLayer? tensorMultiThreadLayer;
    private QKVProjectionLayer? simdSingleThreadLayer;
    private QKVProjectionLayer? simdMultiThreadLayer;

    [GlobalSetup]
    public void GlobalSetup()
    {
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);

        var layerDefinition = model.GetLayer(0);
        var embeddingLayer = new EmbeddingLayer(model, enableCache: true);
        embeddingLayer.Init();
        var session = new BitNetSession(model)
        {
            Tokens = [0],
            CurrentToken = 0,
        };
        this.session = session;

        embeddingLayer.Forward(session);
        var rmsNormLayer = new RmsNormLayer(
            model,
            layerDefinition.AttentionNorm,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        rmsNormLayer.Init();
        rmsNormLayer.Forward(session);
        cpuSingleThreadLayer = new QKVProjectionLayer(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        cpuSingleThreadLayer.Init();
        cpuMultiThreadLayer = new QKVProjectionLayer(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, InferenceConfig.AutoThreadCount));
        cpuMultiThreadLayer.Init();
        tensorSingleThreadLayer = new QKVProjectionLayer(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, 1));
        tensorSingleThreadLayer.Init();
        tensorMultiThreadLayer = new QKVProjectionLayer(
            model,
            layerDefinition.AttentionQueryWeight,
            layerDefinition.AttentionKeyWeight,
            layerDefinition.AttentionValueWeight,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount));
        tensorMultiThreadLayer.Init();
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadLayer = new QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, 1));
            simdSingleThreadLayer.Init();
            simdMultiThreadLayer = new QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount));
            simdMultiThreadLayer.Init();
        }
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
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
    public QKVProjectionOutput QKV_CPU_SingleThread()
    {
        return Run(cpuSingleThreadLayer!);
    }

    [Benchmark]
    public QKVProjectionOutput QKV_CPU_MultiThread()
    {
        return Run(cpuMultiThreadLayer!);
    }

    [Benchmark]
    public QKVProjectionOutput QKV_Tensor_SingleThread()
    {
        return Run(tensorSingleThreadLayer!);
    }

    [Benchmark]
    public QKVProjectionOutput QKV_Tensor_MultiThread()
    {
        return Run(tensorMultiThreadLayer!);
    }

    [Benchmark]
    public QKVProjectionOutput QKV_SIMD_SingleThread()
    {
        if (simdSingleThreadLayer is null)
        {
            throw new PlatformNotSupportedException("QKV SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadLayer);
    }

    [Benchmark]
    public QKVProjectionOutput QKV_SIMD_MultiThread()
    {
        if (simdMultiThreadLayer is null)
        {
            throw new PlatformNotSupportedException("QKV SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadLayer);
    }

    private QKVProjectionOutput Run(QKVProjectionLayer layer)
    {
        layer.Forward(session!);
        return session!.QKVProjection!;
    }
}
