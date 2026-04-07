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
    public sealed class QKVLayerTests
    {
        private static readonly Lazy<QKVVectorsDocument> QKVVectorsDocumentCache = new(LoadQKVVectorsDocument);

        [TestMethod]
        public void QKV_DefaultConfig_SimdAutoThreads()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, layer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void QKV_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var firstLayer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(BitNetSharp.Layers.InferenceConfig.AutoThreadCount, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_CPU(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Layers.InferenceBackend.CPU, GetQKVCase(caseIndex));
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_Tensor(int caseIndex)
        {
            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Layers.InferenceBackend.Tensor, GetQKVCase(caseIndex));
        }

        [TestMethod]
        [DynamicData(nameof(GetQKVCaseIndices))]
        public void QKV_BaselineMatch_SIMD(int caseIndex)
        {
            EnsureAvx2Supported();

            using var model = TestModelFactory.LoadModel();
            AssertQKVMatchesBaseline(model, BitNetSharp.Layers.InferenceBackend.SIMD, GetQKVCase(caseIndex));
        }

        [TestMethod]
        public void QKVCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            QKVCase testCase = QKVVectorsDocumentCache.Value.TestCases[0];
            var layerDefinition = model.GetLayer(0);
            var uncachedLayer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var uncachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            uncachedContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();
            var cachedContext = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            cachedContext.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedContext);
            cachedLayer.Forward(cachedContext);
            BitNetSharp.Layers.QKVProjectionOutput uncached = uncachedContext.QKVProjection!;
            BitNetSharp.Layers.QKVProjectionOutput cached = cachedContext.QKVProjection!;

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncached.Query, cached.Query, 0f, "query cache");
            AssertFloatArraysAreClose(uncached.Key, cached.Key, 0f, "key cache");
            AssertFloatArraysAreClose(uncached.Value, cached.Value, 0f, "value cache");
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

        private static void AssertQKVMatchesBaseline(BitNetSharp.Models.BitNetModel model, BitNetSharp.Layers.InferenceBackend backend, QKVCase testCase)
        {
            string caseName = $"token {testCase.TokenId} ({testCase.TokenText})";
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(backend, 1));
            var context = TestModelFactory.CreateSession(model, token: testCase.TokenId);
            context.RmsNorm = testCase.FirstLayerRmsNorm.Values.ToArray();

            layer.Init();
            layer.Forward(context);
            BitNetSharp.Layers.QKVProjectionOutput actual = context.QKVProjection!;

            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Query.ToArray(), actual.Query, 1e-4f, caseName + " query");
            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Key.ToArray(), actual.Key, 1e-4f, caseName + " key");
            AssertFloatArraysAreClose(testCase.FirstLayerAttnQKV.WQKV.Value.ToArray(), actual.Value, 1e-4f, caseName + " value");
        }

        [TestMethod]
        public void QKV_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var layerDefinition = model.GetLayer(0);
            var layer = new BitNetSharp.Layers.QKVProjectionLayer(
                model,
                layerDefinition.AttentionQueryWeight,
                layerDefinition.AttentionKeyWeight,
                layerDefinition.AttentionValueWeight,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.RmsNorm = new float[(int)model.Config!.EmbeddingLength];

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(context));
        }

        private static void AssertSmallProjectionCpuMatchesSimd(string caseName, int inputLength, int outputLength, int seed, float weightScale)
        {
            float[] input = CreateSmallProjectionInput(inputLength, seed);
            byte[] packedWeights = CreatePackedProjectionWeights(inputLength, outputLength, seed);
            float[] expected = MathHelper.ProjectBitNetI2Simd(input, packedWeights, outputLength, weightScale);
            float[] actual = MathHelper.ProjectBitNetI2Cpu(input, packedWeights, outputLength, weightScale);

            AssertFloatArraysAreClose(expected, actual, 1e-5f, caseName);
        }

        private static void AssertSmallProjectionTensorMatchesSimd(string caseName, int inputLength, int outputLength, int seed, float weightScale)
        {
            float[] input = CreateSmallProjectionInput(inputLength, seed);
            byte[] packedWeights = CreatePackedProjectionWeights(inputLength, outputLength, seed);
            float[] expected = MathHelper.ProjectBitNetI2Simd(input, packedWeights, outputLength, weightScale);
            float[] actual = MathHelper.ProjectBitNetI2Tensor(input, packedWeights, outputLength, weightScale);

            AssertFloatArraysAreClose(expected, actual, 1e-5f, caseName);
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
            [property: JsonPropertyName("qcur")] IReadOnlyList<float> Query,
            [property: JsonPropertyName("kcur")] IReadOnlyList<float> Key,
            [property: JsonPropertyName("vcur")] IReadOnlyList<float> Value);
    }
}
