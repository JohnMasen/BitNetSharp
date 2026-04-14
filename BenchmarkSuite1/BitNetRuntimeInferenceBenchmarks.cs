using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using BitNetSharp;
using BitNetSharp.Models;
using BitNetSharp.Nodes;
using Microsoft.VSDiagnostics;
using System;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Benchmarks;
[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[SimpleJob(RuntimeMoniker.Net90)]
[SimpleJob(RuntimeMoniker.Net10_0)]
[CPUUsageDiagnoser]
public class BitNetRuntimeInferenceBenchmarks
{
    private const int InputTokenId = 0;
    private BitNetModel? model;
    private BitNetMemoryManager? memoryManager;
    private BitNetRuntime? cpuSingleThreadRuntime;
    private BitNetRuntime? cpuMultiThreadRuntime;
    private BitNetRuntime? tensorSingleThreadRuntime;
    private BitNetRuntime? tensorMultiThreadRuntime;
    private BitNetRuntime? simdSingleThreadRuntime;
    private BitNetRuntime? simdMultiThreadRuntime;
    [GlobalSetup]
    public void GlobalSetup()
    {
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        memoryManager = new BitNetMemoryManager();
        cpuSingleThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Cpu(1));
        cpuMultiThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Cpu(InferenceConfig.AutoThreadCount));
        tensorSingleThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Tensor(1));
        tensorMultiThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Tensor(InferenceConfig.AutoThreadCount));
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Simd(1));
            simdMultiThreadRuntime = CreateRuntime(BenchmarkInferenceConfigs.Simd(InferenceConfig.AutoThreadCount));
        }
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        cpuSingleThreadRuntime?.Dispose();
        cpuMultiThreadRuntime?.Dispose();
        tensorSingleThreadRuntime?.Dispose();
        tensorMultiThreadRuntime?.Dispose();
        simdSingleThreadRuntime?.Dispose();
        simdMultiThreadRuntime?.Dispose();
        cpuSingleThreadRuntime = null;
        cpuMultiThreadRuntime = null;
        tensorSingleThreadRuntime = null;
        tensorMultiThreadRuntime = null;
        simdSingleThreadRuntime = null;
        simdMultiThreadRuntime = null;
        memoryManager?.Dispose();
        memoryManager = null;
        model?.Dispose();
        model = null;
    }

    [Benchmark(Baseline = true)]
    public int Inference_CPU_SingleThread() => cpuSingleThreadRuntime!.InferenceTokenId(InputTokenId);
    [Benchmark]
    public int Inference_CPU_MultiThread() => cpuMultiThreadRuntime!.InferenceTokenId(InputTokenId);
    [Benchmark]
    public int Inference_Tensor_SingleThread() => tensorSingleThreadRuntime!.InferenceTokenId(InputTokenId);
    [Benchmark]
    public int Inference_Tensor_MultiThread() => tensorMultiThreadRuntime!.InferenceTokenId(InputTokenId);
    [Benchmark]
    public int Inference_SIMD_SingleThread()
    {
        if (simdSingleThreadRuntime is null)
        {
            throw new PlatformNotSupportedException("Inference SIMD benchmark requires AVX2 support.");
        }

        return simdSingleThreadRuntime.InferenceTokenId(InputTokenId);
    }

    [Benchmark]
    public int Inference_SIMD_MultiThread()
    {
        if (simdMultiThreadRuntime is null)
        {
            throw new PlatformNotSupportedException("Inference SIMD benchmark requires AVX2 support.");
        }

        return simdMultiThreadRuntime.InferenceTokenId(InputTokenId);
    }

    private BitNetRuntime CreateRuntime(InferenceConfig inferenceConfig)
    {
        return new BitNetRuntime(model!, memoryManager!, inferenceConfig);
    }
}