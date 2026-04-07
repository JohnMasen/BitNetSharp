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
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);

            float[] embedding = layer.Forward(context);

            Assert.AreEqual((int)model.Config!.EmbeddingLength, embedding.Length);
        }

        [TestMethod]
        public void Embedding_ThrowsForOutOfRangeToken()
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: (int)model.Config!.VocabularySize);

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => layer.Forward(context));
        }

        [TestMethod]
        public void EmbeddingCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var uncachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var cachedLayer = new BitNetSharp.Layers.EmbeddingLayer(model, enableCache: true);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);

            CollectionAssert.AreEqual(uncachedLayer.Forward(context), cachedLayer.Forward(context));
            Assert.IsTrue(cachedLayer.EnableCache);
        }

        [TestMethod]
        [DynamicData(nameof(GetEmbeddingCases))]
        public void Embedding_MatchesBaseline(string caseName, int tokenId, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var layer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, tokenId);

            float[] actualValues = layer.Forward(context);

            Assert.AreEqual(expectedValues.Length, actualValues.Length, caseName);
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
        public void RmsNorm_DefaultsToCpuStandard()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);

            Assert.AreEqual(BitNetSharp.Layers.RmsNormBackend.CPUStandard, layer.Backend);
        }

        [TestMethod]
        public void RmsNorm_MatchesManualFormula()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);
            float[] input = embeddingLayer.Forward(context);
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);

            float[] actual = layer.Forward(input);
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
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);

            float[] actualValues = layer.Forward(input);

            Assert.AreEqual(expectedValues.Length, actualValues.Length, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_TensorBackendMatchesBaseline(string caseName, float[] input, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, BitNetSharp.Layers.RmsNormBackend.Tensor);

            float[] actualValues = layer.Forward(input);

            Assert.AreEqual(expectedValues.Length, actualValues.Length, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_SimdBackendMatchesBaseline(string caseName, float[] input, float[] expectedValues)
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var layer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, BitNetSharp.Layers.RmsNormBackend.SIMD);

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
            }

            float[] actualValues = layer.Forward(input);

            Assert.AreEqual(expectedValues.Length, actualValues.Length, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        public void RmsNormCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);
            float[] input = embeddingLayer.Forward(context);
            var normTensor = model.GetLayer(0).AttentionNorm;
            var uncachedLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);
            var cachedLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, enableCache: true);

            float[] uncached = uncachedLayer.Forward(input);
            float[] cached = cachedLayer.Forward(input);

            Assert.IsTrue(cachedLayer.EnableCache);
            AssertFloatArraysAreClose(uncached, cached, 0f);
        }

        [TestMethod]
        public void RmsNorm_TensorBackendMatchesCpuStandard()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);
            float[] input = embeddingLayer.Forward(context);
            var cpuLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);
            var tensorLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, BitNetSharp.Layers.RmsNormBackend.Tensor);

            float[] expected = cpuLayer.Forward(input);
            float[] actual = tensorLayer.Forward(input);

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        public void RmsNorm_SimdBackendMatchesCpuStandard()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingLayer = new BitNetSharp.Layers.EmbeddingLayer(model);
            var context = TestModelFactory.CreateInferenceContext(model, token: 0);
            float[] input = embeddingLayer.Forward(context);
            var cpuLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor);
            var simdLayer = new BitNetSharp.Layers.RmsNormLayer(model, normTensor, BitNetSharp.Layers.RmsNormBackend.SIMD);

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.ThrowsExactly<NotSupportedException>(() => simdLayer.Forward(input));
                return;
            }

            float[] expected = cpuLayer.Forward(input);
            float[] actual = simdLayer.Forward(input);

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
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
            Assert.AreEqual(expected.Count, actual.Count);
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
