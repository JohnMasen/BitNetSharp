using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class LayerTests
    {
        private static readonly Lazy<LayerVectorsDocument> LayerVectorsDocumentCache = new(LoadLayerVectorsDocument);

        [TestMethod]
        public void Embedding_ReturnsExpectedLength()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);

            layer.Init();
            layer.Forward(context);
            float[] embedding = context.Embedding!;

            Assert.HasCount((int)model.Config!.EmbeddingLength, embedding);
        }

        [TestMethod]
        public void Embedding_ThrowsForOutOfRangeToken()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: (int)model.Config!.VocabularySize);

            layer.Init();
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => layer.Forward(context));
        }

        [TestMethod]
        public void Embedding_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(context));
        }

        [TestMethod]
        public void EmbeddingCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var uncachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var cachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model, enableCache: true);
            var uncachedContext = TestModelFactory.CreateSession(model, token: 0);
            var cachedContext = TestModelFactory.CreateSession(model, token: 0);

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(uncachedContext);
            cachedLayer.Forward(cachedContext);

            CollectionAssert.AreEqual(uncachedContext.Embedding, cachedContext.Embedding);
            Assert.IsTrue(cachedLayer.EnableCache);
        }

        [TestMethod]
        public void Embedding_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstLayer = new BitNetSharp.Layers.EmbeddingLayer(model, inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.EmbeddingLayer(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.CPU, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(1, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        [DynamicData(nameof(GetEmbeddingCases))]
        public void Embedding_MatchesBaseline(string caseName, int tokenId, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, tokenId);

            layer.Init();
            layer.Forward(context);
            float[] actualValues = context.Embedding!;

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            CollectionAssert.AreEqual(expectedValues, actualValues, caseName);
        }

        [TestMethod]
        public void RmsNorm_EpsilonMatchesDump()
        {
            using var model = TestModelFactory.LoadModel();
            float expectedEpsilon = LayerVectorsDocumentCache.Value.FirstLayerRmsNormMetadata.Epsilon;

            Assert.AreEqual(expectedEpsilon, model.Config!.AttentionLayerNormRmsEpsilon, 1e-12f);
        }

        [TestMethod]
        public void RmsNorm_DefaultConfig_SimdSingleThread()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, layer.InferenceConfig.Backend);
            Assert.AreEqual(1, layer.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void RmsNorm_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var firstLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, inferenceConfig: null);
            var secondLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Layers.InferenceBackend.SIMD, firstLayer.InferenceConfig.Backend);
            Assert.AreEqual(1, firstLayer.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstLayer.InferenceConfig, secondLayer.InferenceConfig);
        }

        [TestMethod]
        public void RmsNorm_MatchesManualFormula()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingLayer.Init();
            embeddingLayer.Forward(context);
            float[] input = context.Embedding!;
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));

            layer.Init();
            layer.Forward(context);
            float[] actual = context.RmsNorm!;
            float[] weights = ReadTensorValues(model, normTensor);
            float[] expected = ApplyManualRmsNorm(input, weights, model.Config!.AttentionLayerNormRmsEpsilon);

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_MatchesBaseline(string caseName, float[] input, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            layer.Init();
            layer.Forward(context);
            float[] actualValues = context.RmsNorm!;

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_BaselineMatch_Tensor(string caseName, float[] input, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.Tensor, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            layer.Init();
            layer.Forward(context);
            float[] actualValues = context.RmsNorm!;

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_BaselineMatch_SIMD(string caseName, float[] input, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.SIMD, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
            }

            layer.Init();
            layer.Forward(context);
            float[] actualValues = context.RmsNorm!;

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        public void RmsNormCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingLayer.Init();
            embeddingLayer.Forward(context);
            var normTensor = model.GetLayer(0).AttentionNorm;
            var uncachedLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var cachedContext = TestModelFactory.CreateSession(model, token: 0);
            cachedContext.Embedding = context.Embedding!.ToArray();

            uncachedLayer.Init();
            cachedLayer.Init();
            uncachedLayer.Forward(context);
            cachedLayer.Forward(cachedContext);
            float[] uncached = context.RmsNorm!;
            float[] cached = cachedContext.RmsNorm!;

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncached, cached, 0f);
        }

        [TestMethod]
        public void RmsNorm_CPU_Matches_Tensor()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingLayer.Init();
            embeddingLayer.Forward(context);
            var cpuLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var tensorLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.Tensor, 1));
            var tensorContext = TestModelFactory.CreateSession(model, token: 0);
            tensorContext.Embedding = context.Embedding!.ToArray();

            cpuLayer.Init();
            tensorLayer.Init();
            cpuLayer.Forward(context);
            tensorLayer.Forward(tensorContext);
            float[] expected = context.RmsNorm!;
            float[] actual = tensorContext.RmsNorm!;

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        public void RmsNorm_CPU_Matchs_SIMD()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingLayer.Init();
            embeddingLayer.Forward(context);
            var cpuLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var simdLayer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.SIMD, 1));
            var simdContext = TestModelFactory.CreateSession(model, token: 0);
            simdContext.Embedding = context.Embedding!.ToArray();

            cpuLayer.Init();
            simdLayer.Init();

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.ThrowsExactly<NotSupportedException>(() => simdLayer.Forward(simdContext));
                return;
            }

            cpuLayer.Forward(context);
            simdLayer.Forward(simdContext);
            float[] expected = context.RmsNorm!;
            float[] actual = simdContext.RmsNorm!;

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        public void RmsNorm_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Layers.InferenceConfig(BitNetSharp.Layers.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = new float[(int)model.Config!.EmbeddingLength];

            Assert.ThrowsExactly<InvalidOperationException>(() => layer.Forward(context));
        }

        public static IEnumerable<object[]> GetEmbeddingCases()
        {
            return LayerVectorsDocumentCache.Value.TestCases.Select(testCase => new object[]
            {
                $"token {testCase.TokenId} ({testCase.TokenText})",
                testCase.TokenId,
                testCase.Dequantized.Values.ToArray(),
            });
        }

        public static IEnumerable<object[]> GetRmsNormCases()
        {
            return LayerVectorsDocumentCache.Value.TestCases.Select(testCase => new object[]
            {
                $"token {testCase.TokenId} ({testCase.TokenText})",
                testCase.Dequantized.Values.ToArray(),
                testCase.FirstLayerRmsNorm.Values.ToArray(),
            });
        }

        private static LayerVectorsDocument LoadLayerVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<LayerVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load layer baseline JSON.");
        }

        private static float[] ApplyManualRmsNorm(IReadOnlyList<float> input, IReadOnlyList<float> weights, float epsilon)
        {
            double sumSquares = 0;
            for (int index = 0; index < input.Count; index++)
            {
                double value = input[index];
                sumSquares += value * value;
            }

            double meanSquare = sumSquares / input.Count;
            double inverseRootMeanSquare = 1d / Math.Sqrt(meanSquare + epsilon);
            float[] output = new float[input.Count];
            for (int index = 0; index < input.Count; index++)
            {
                output[index] = (float)(input[index] * inverseRootMeanSquare * weights[index]);
            }

            return output;
        }

        private static float[] ReadTensorValues(BitNetSharp.Models.BitNetModel model, BitNetSharp.Models.BitNetTensorInfo tensorInfo)
        {
            using var tensorData = model.ReadTensorData(tensorInfo);
            return tensorInfo.TensorType switch
            {
                GGUFSharp.GGUFTensorType.GGML_TYPE_F32 => MemoryMarshal.Cast<byte, float>(tensorData.Memory.Span).ToArray(),
                GGUFSharp.GGUFTensorType.GGML_TYPE_F16 => ConvertHalfToSingle(MemoryMarshal.Cast<byte, Half>(tensorData.Memory.Span)),
                _ => throw new NotSupportedException($"Unsupported tensor type '{tensorInfo.TensorType}' in test helper."),
            };
        }

        private static float[] ConvertHalfToSingle(ReadOnlySpan<Half> source)
        {
            float[] values = new float[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                values[index] = (float)source[index];
            }

            return values;
        }

        private static void AssertFloatArraysAreClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float delta)
        {
            Assert.HasCount(expected.Count, actual);
            for (int index = 0; index < expected.Count; index++)
            {
                Assert.AreEqual(expected[index], actual[index], delta, $"Mismatch at index {index}.");
            }
        }

        internal sealed record LayerVectorsDocument(
            [property: JsonPropertyName("vector_length")] int VectorLength,
            [property: JsonPropertyName("first_layer_rms_norm_metadata")] RmsNormMetadata FirstLayerRmsNormMetadata,
            [property: JsonPropertyName("test_cases")] IReadOnlyList<LayerVectorCase> TestCases);

        internal sealed record LayerVectorCase(
            [property: JsonPropertyName("token_id")] int TokenId,
            [property: JsonPropertyName("token_text")] string TokenText,
            [property: JsonPropertyName("vector_length")] int VectorLength,
            [property: JsonPropertyName("dequantized")] LayerVectorValues Dequantized,
            [property: JsonPropertyName("first_layer_rms_norm")] RmsNormOutput FirstLayerRmsNorm);

        internal sealed record LayerVectorValues(
            [property: JsonPropertyName("dtype")] string DType,
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);

        internal sealed record RmsNormMetadata(
            [property: JsonPropertyName("eps")] float Epsilon);

        internal sealed record RmsNormOutput(
            [property: JsonPropertyName("eps")] float Epsilon,
            [property: JsonPropertyName("values")] IReadOnlyList<float> Values);
    }
}
