using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
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
    private BitNetModel? model;
    private float[]? input;
    private RmsNormLayer? cpuLayer;
    private RmsNormLayer? tensorLayer;
    private RmsNormLayer? simdLayer;
    [GlobalSetup]
    public void GlobalSetup()
    {
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        var normTensor = model.GetLayer(0).AttentionNorm;
        var embeddingLayer = new EmbeddingLayer(model, enableCache: true);
        var context = new InferenceContext(model)
        {
            Tokens = [0],
            CurrentToken = 0,
        };
        input = embeddingLayer.Forward(context);
        cpuLayer = new RmsNormLayer(model, normTensor, RmsNormBackend.CPUStandard, enableCache: true);
        tensorLayer = new RmsNormLayer(model, normTensor, RmsNormBackend.Tensor, enableCache: true);
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdLayer = new RmsNormLayer(model, normTensor, RmsNormBackend.SIMD, enableCache: true);
        }
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        model?.Dispose();
        model = null;
        input = null;
        cpuLayer = null;
        tensorLayer = null;
        simdLayer = null;
    }

    [Benchmark(Baseline = true)]
    public float[] CPUStandard() => cpuLayer!.Forward(input!);
    [Benchmark]
    public float[] Tensor() => tensorLayer!.Forward(input!);
    [Benchmark]
    public float[] SIMD()
    {
        if (simdLayer is null)
        {
            throw new PlatformNotSupportedException("RMSNorm SIMD benchmark requires AVX2 support.");
        }

        return simdLayer.Forward(input!);
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