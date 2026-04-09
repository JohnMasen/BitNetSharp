using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Serialization;
using BitNetSharp.Core;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class QKVNodeTests
    {
        private static readonly Lazy<QKVVectorsDocument> QKVVectorsDocumentCache = new(LoadQKVVectorsDocument);

        [TestMethod]
        public void QKV_DefaultConfig_SimdAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, node.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void QKV_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var firstNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Nodes.InferenceConfig.AutoThreadCount, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_CPU(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Nodes.InferenceBackend.CPU, GetQKVCase(caseIndex));
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_Tensor(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Nodes.InferenceBackend.Tensor, GetQKVCase(caseIndex));
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();

            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Nodes.InferenceBackend.SIMD, GetQKVCase(caseIndex));
        }

        [TestMethod]
        public void QKV_CPU_MultiThreadMatchesSingleThread()
        {
            VerifyQKVMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.CPU);
        }

        [TestMethod]
        public void QKV_Tensor_MultiThreadMatchesSingleThread()
        {
            VerifyQKVMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.Tensor);
        }

        [TestMethod]
        public void QKV_SIMD_MultiThreadMatchesSingleThread()
        {
            EnsureAvx2Supported();
            VerifyQKVMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend.SIMD);
        }

        [TestMethod]
        public void QKVCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            QKVCase testCase = QKVVectorsDocumentCache.Value.TestCases[0];
            var layerDefinition = model.GetLayer(0);
            var uncachedNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var cachedNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var uncachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            uncachedContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();
            var cachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            cachedContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedContext);
            cachedNode.Forward(cachedContext);

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncachedContext.QKVQuery.Span.ToArray(), cachedContext.QKVQuery.Span.ToArray(), 0f, "query cache");
            AssertFloatArraysAreClose(uncachedContext.QKVKey.Span.ToArray(), cachedContext.QKVKey.Span.ToArray(), 0f, "key cache");
            AssertFloatArraysAreClose(uncachedContext.QKVValue.Span.ToArray(), cachedContext.QKVValue.Span.ToArray(), 0f, "value cache");
        }

        [TestMethod]
        [DynamicData(nameof(GetSmallProjectionCaseIndices))]
        public void BitNetProjection_CPU_Matchs_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();

            (string caseName, int inputLength, int outputLength, int seed, float weightScale) = GetSmallProjectionCase(caseIndex);
            AssertSmallProjectionCpuMatchesSimd(caseName, inputLength, outputLength, seed, weightScale);
        }

        [TestMethod]
        [DynamicData(nameof(GetSmallProjectionCaseIndices))]
        public void BitNetProjection_CPU_Matches_Tensor(int caseIndex)
        {
            EnsureAvx2Supported();

            (string caseName, int inputLength, int outputLength, int seed, float weightScale) = GetSmallProjectionCase(caseIndex);
            AssertSmallProjectionTensorMatchesSimd(caseName, inputLength, outputLength, seed, weightScale);
        }

        public static IEnumerable<object[]> GetQKVCaseIndices()
        {
            return QKVVectorsDocumentCache.Value.TestCases.Select((_, caseIndex) => new object[] { caseIndex });
        }

        public static IEnumerable<object[]> GetSmallProjectionCaseIndices()
        {
            yield return new object[] { 0 };
            yield return new object[] { 1 };
        }

        private static QKVVectorsDocument LoadQKVVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<QKVVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load QKV baseline JSON.");
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta, string caseName)
        {
            Assert.HasCount(expected.Count, actual, caseName);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"{caseName} mismatch at index {index}.");
            }
        }

        private static void EnsureAvx2Supported()
        {
            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
            }
        }

        private static QKVCase GetQKVCase(int caseIndex)
        {
            return QKVVectorsDocumentCache.Value.TestCases[caseIndex];
        }

        private static (string CaseName, int InputLength, int OutputLength, int Seed, float WeightScale) GetSmallProjectionCase(int caseIndex)
        {
            return caseIndex switch
            {
                0 => ("single block", 128, 2, 3, 0.25f),
                1 => ("two blocks", 256, 3, 11, 0.5f),
                _ => throw new ArgumentOutOfRangeException(nameof(caseIndex)),
            };
        }

        private static void AssertQKVMatchesBaseline(BitNetSharp.Models.BitNetModel model, BitNetSharp.Nodes.InferenceBackend backend, QKVCase testCase)
        {
            string caseName = $"token {testCase.TokenId} ({testCase.TokenText})";
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            context.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();

            node.Init();
            node.Forward(context);

            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Query.ToArray(), context.QKVQuery.Span.ToArray(), 1e-4f, caseName + " query");
            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Key.ToArray(), context.QKVKey.Span.ToArray(), 1e-4f, caseName + " key");
            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Value.ToArray(), context.QKVValue.Span.ToArray(), 1e-4f, caseName + " value");
        }

        private static void VerifyQKVMultiThreadMatchesSingleThread(BitNetSharp.Nodes.InferenceBackend backend)
        {
            using var model = TestModelFactory.LoadModel();
            QKVCase testCase = GetQKVCase(0);
            var layerDefinition = model.GetLayer(0);
            var singleThreadNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 1));
            var multiThreadNode = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(backend, 2));
            var singleThreadContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            var multiThreadContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            singleThreadContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();
            multiThreadContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();

            singleThreadNode.Init();
            multiThreadNode.Init();
            singleThreadNode.Forward(singleThreadContext);
            multiThreadNode.Forward(multiThreadContext);

            AssertFloatArraysAreClose(singleThreadContext.QKVQuery.Span.ToArray(), multiThreadContext.QKVQuery.Span.ToArray(), 1e-4f, $"{backend} QKV query threading");
            AssertFloatArraysAreClose(singleThreadContext.QKVKey.Span.ToArray(), multiThreadContext.QKVKey.Span.ToArray(), 1e-4f, $"{backend} QKV key threading");
            AssertFloatArraysAreClose(singleThreadContext.QKVValue.Span.ToArray(), multiThreadContext.QKVValue.Span.ToArray(), 1e-4f, $"{backend} QKV value threading");
        }

        [TestMethod]
        public void QKV_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var node = new BitNetSharp.Nodes.QKVProjectionNode(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.RmsNorm = new float[(int)model.Config!.EmbeddingLength];

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(context));
        }

        private static void AssertSmallProjectionCpuMatchesSimd(string caseName, int inputLength, int outputLength, int seed, float weightScale)
        {
            float[] input = CreateSmallProjectionInput(inputLength, seed);
            byte[] packedWeights = CreatePackedProjectionWeights(inputLength, outputLength, seed);
            float[] expected = ProjectWithProvider(new CPUSimdOPProvider(), input, packedWeights, outputLength, weightScale);
            float[] actual = ProjectWithProvider(new CPUDefaultOPProvider(), input, packedWeights, outputLength, weightScale);

            AssertFloatArraysAreClose(expected, actual, 1e-5f, caseName);
        }

        private static void AssertSmallProjectionTensorMatchesSimd(string caseName, int inputLength, int outputLength, int seed, float weightScale)
        {
            float[] input = CreateSmallProjectionInput(inputLength, seed);
            byte[] packedWeights = CreatePackedProjectionWeights(inputLength, outputLength, seed);
            float[] expected = ProjectWithProvider(new CPUSimdOPProvider(), input, packedWeights, outputLength, weightScale);
            float[] actual = ProjectWithProvider(new CPUTensorOPProvider(), input, packedWeights, outputLength, weightScale);

            AssertFloatArraysAreClose(expected, actual, 1e-5f, caseName);
        }

        private static float[] ProjectWithProvider(IOPProvider1 provider, ReadOnlyMemory<float> input, ReadOnlyMemory<byte> packedWeights, int outputLength, float weightScale)
        {
            float[] output = new float[outputLength];
            provider.ProjectBitNetI2(input, packedWeights, outputLength, weightScale, output);
            return output;
        }

        private static float[] CreateSmallProjectionInput(int inputLength, int seed)
        {
            float[] input = new float[inputLength];
            for (int index = 0; index < input.Length; index++)
            {
                int primary = ((index + seed) % 17) - 8;
                int secondary = ((index + (seed * 2)) % 5) - 2;
                input[index] = (primary * 0.125f) + (secondary * 0.03125f);
            }

            return input;
        }

        private static byte[] CreatePackedProjectionWeights(int inputLength, int outputLength, int seed)
        {
            const int ActivationBlockWidth = 128;
            const int PackedGroupWidth = 32;

            int packedRowByteCount = (inputLength * outputLength) / 4 / outputLength;
            byte[] packedWeights = new byte[packedRowByteCount * outputLength];
            for (int outputIndex = 0; outputIndex < outputLength; outputIndex++)
            {
                int rowOffset = outputIndex * packedRowByteCount;
                int packedIndex = rowOffset;
                for (int activationBlockStart = 0; activationBlockStart < inputLength; activationBlockStart += ActivationBlockWidth)
                {
                    for (int groupPosition = 0; groupPosition < PackedGroupWidth; groupPosition++)
                    {
                        byte packedValue = 0;
                        for (int groupIndex = 0; groupIndex < 4; groupIndex++)
                        {
                            int activationIndex = activationBlockStart + (groupIndex * PackedGroupWidth) + groupPosition;
                            int weight = CreateTernaryWeight(outputIndex, activationIndex, seed);
                            packedValue |= (byte)(EncodeWeight(weight) << (6 - (groupIndex * 2)));
                        }

                        packedWeights[packedIndex++] = packedValue;
                    }
                }
            }

            return packedWeights;
        }

        private static int CreateTernaryWeight(int outputIndex, int activationIndex, int seed)
        {
            int pattern = (activationIndex + (outputIndex * 7) + seed) % 9;
            return pattern switch
            {
                0 => -1,
                1 => 0,
                2 => 1,
                3 => 0,
                4 => 1,
                5 => -1,
                6 => 1,
                7 => 0,
                _ => -1,
            };
        }

        private static int EncodeWeight(int weight)
        {
            return weight switch
            {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => throw new InvalidOperationException("Only ternary weights are supported."),
            };
        }

        internal sealed record QKVVectorsDocument(
            [property: JsonPropertyName("test_cases")] IReadOnlyList<QKVCase> TestCases);

        internal sealed record QKVCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("first_layer_rms_norm")] LayerVectorValues FirstLayerRmsNorm,
            [property: JsonPropertyName("first_layer_attn_qkv")] QKVOutputs FirstLayerAttnQKV);

        internal sealed record LayerVectorValues(
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);

        internal sealed record QKVOutputs(
            [property: JsonPropertyName("wqkv")] PackedQKVValues WQKV,
            [property: JsonPropertyName("runtime_source")] string RuntimeSource);

        internal sealed record PackedQKVValues(
            [property: JsonPropertyName("qcur")] float[] Query,
            [property: JsonPropertyName("kcur")] float[] Key,
            [property: JsonPropertyName("vcur")] float[] Value);
    }
}
