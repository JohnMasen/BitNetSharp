using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Core;
using BitNetSharp.Layers;
using BitNetSharp.Models;
using System;
using System.IO;
using System.Runtime.Intrinsics.X86;
using Microsoft.VSDiagnostics;

namespace BitNetSharp.Benchmarks;
[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class RmsNormBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private RmsNormLayer? cpuSingleThreadLayer;
    private RmsNormLayer? cpuMultiThreadLayer;
    private RmsNormLayer? tensorSingleThreadLayer;
    private RmsNormLayer? tensorMultiThreadLayer;
    private RmsNormLayer? simdSingleThreadLayer;
    private RmsNormLayer? simdMultiThreadLayer;
    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        var normTensor = model.GetLayer(0).AttentionNorm;
        var embeddingLayer = new EmbeddingLayer(model, enableCache: true);
        embeddingLayer.Init();
        var session = new BitNetSession(model, memoryManager)
        {
            Tokens = new[] { 0 },
            CurrentToken = 0,
        };
        this.session = session;
        embeddingLayer.Forward(session);
        cpuSingleThreadLayer = new RmsNormLayer(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, 1));
        cpuSingleThreadLayer.Init();
        cpuMultiThreadLayer = new RmsNormLayer(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.CPU, InferenceConfig.AutoThreadCount));
        cpuMultiThreadLayer.Init();
        tensorSingleThreadLayer = new RmsNormLayer(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, 1));
        tensorSingleThreadLayer.Init();
        tensorMultiThreadLayer = new RmsNormLayer(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: new InferenceConfig(InferenceBackend.Tensor, InferenceConfig.AutoThreadCount));
        tensorMultiThreadLayer.Init();
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadLayer = new RmsNormLayer(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, 1));
            simdSingleThreadLayer.Init();
            simdMultiThreadLayer = new RmsNormLayer(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: new InferenceConfig(InferenceBackend.SIMD, InferenceConfig.AutoThreadCount));
            simdMultiThreadLayer.Init();
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
    public Memory<float> RmsNorm_CPU_SingleThread() => Run(cpuSingleThreadLayer!);

    [Benchmark]
    public Memory<float> RmsNorm_CPU_MultiThread() => Run(cpuMultiThreadLayer!);

    [Benchmark]
    public Memory<float> RmsNorm_Tensor_SingleThread() => Run(tensorSingleThreadLayer!);

    [Benchmark]
    public Memory<float> RmsNorm_Tensor_MultiThread() => Run(tensorMultiThreadLayer!);

    [Benchmark]
    public Memory<float> RmsNorm_SIMD_SingleThread()
    {
        if (simdSingleThreadLayer is null)
        {
            throw new PlatformNotSupportedException("RMSNorm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadLayer);
    }

    [Benchmark]
    public Memory<float> RmsNorm_SIMD_MultiThread()
    {
        if (simdMultiThreadLayer is null)
        {
            throw new PlatformNotSupportedException("RMSNorm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadLayer);
    }

    private Memory<float> Run(RmsNormLayer layer)
    {
        layer.Forward(session!);
        return session!.RmsNorm;
    }
}

internal static class BenchmarkProjectPaths
{
    private static readonly string[] ModelPathSegments =
    [
        "Models",
        "bitnet-b1.58-2B-4T-gguf",
        "ggml-model-i2_s.gguf",
    ];

    public static string ModelPath => FindModelPath();

    private static string FindModelPath()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, Path.Combine(ModelPathSegments));
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException(
            $"Could not locate benchmark model file '{Path.Combine(ModelPathSegments)}' from '{AppContext.BaseDirectory}'.");
    }
}