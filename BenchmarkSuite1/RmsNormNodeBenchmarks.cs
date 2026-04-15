using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using BitNetSharp.Core;
using BitNetSharp.Nodes;
using BitNetSharp.Models;
using System;
using System.Runtime.Intrinsics.X86;
using Microsoft.VSDiagnostics;

namespace BitNetSharp.Benchmarks;
[HideColumns("Error", "StdDev", "Median", "RatioSD")]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[CPUUsageDiagnoser]
public class RmsNormNodeBenchmarks
{
    private BitNetMemoryManager? memoryManager;
    private BitNetModel? model;
    private BitNetSession? session;
    private RmsNormNode? cpuSingleThreadNode;
    private RmsNormNode? cpuMultiThreadNode;
    private RmsNormNode? tensorSingleThreadNode;
    private RmsNormNode? tensorMultiThreadNode;
    private RmsNormNode? simdSingleThreadNode;
    private RmsNormNode? simdMultiThreadNode;
    [GlobalSetup]
    public void GlobalSetup()
    {
        memoryManager = new BitNetMemoryManager();
        model = new BitNetModel();
        model.Load(BenchmarkProjectPaths.ModelPath);
        var normTensor = model.GetLayer(0).AttentionNorm;
        session = new BitNetSession(model, memoryManager, new[] { 0 });
        BenchmarkDataHelper.FillDeterministicValues(session.Embedding.Span, 2);
        cpuSingleThreadNode = new RmsNormNode(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: BenchmarkInferenceConfigs.Cpu(1));
        cpuSingleThreadNode.Init();
        cpuMultiThreadNode = new RmsNormNode(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: BenchmarkInferenceConfigs.Cpu(InferenceConfig.AutoThreadCount));
        cpuMultiThreadNode.Init();
        tensorSingleThreadNode = new RmsNormNode(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: BenchmarkInferenceConfigs.Tensor(1));
        tensorSingleThreadNode.Init();
        tensorMultiThreadNode = new RmsNormNode(
            model,
            normTensor,
            enableCache: true,
            inferenceConfig: BenchmarkInferenceConfigs.Tensor(InferenceConfig.AutoThreadCount));
        tensorMultiThreadNode.Init();
        if (Avx.IsSupported && Avx2.IsSupported)
        {
            simdSingleThreadNode = new RmsNormNode(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: BenchmarkInferenceConfigs.Simd(1));
            simdSingleThreadNode.Init();
            simdMultiThreadNode = new RmsNormNode(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: BenchmarkInferenceConfigs.Simd(InferenceConfig.AutoThreadCount));
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
    public Memory<float> RmsNorm_CPU_SingleThread() => Run(cpuSingleThreadNode!);

    [Benchmark]
    public Memory<float> RmsNorm_CPU_MultiThread() => Run(cpuMultiThreadNode!);

    [Benchmark]
    public Memory<float> RmsNorm_Tensor_SingleThread() => Run(tensorSingleThreadNode!);

    [Benchmark]
    public Memory<float> RmsNorm_Tensor_MultiThread() => Run(tensorMultiThreadNode!);

    [Benchmark]
    public Memory<float> RmsNorm_SIMD_SingleThread()
    {
        if (simdSingleThreadNode is null)
        {
            throw new PlatformNotSupportedException("RMSNorm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdSingleThreadNode);
    }

    [Benchmark]
    public Memory<float> RmsNorm_SIMD_MultiThread()
    {
        if (simdMultiThreadNode is null)
        {
            throw new PlatformNotSupportedException("RMSNorm SIMD benchmark requires AVX2 support.");
        }

        return Run(simdMultiThreadNode);
    }

    private Memory<float> Run(RmsNormNode node)
    {
        node.Forward(session!);
        return session!.RmsNorm;
    }
}
