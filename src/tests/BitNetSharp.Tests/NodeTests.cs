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
    public sealed class NodeTests
    {
        private static readonly Lazy<LayerVectorsDocument> LayerVectorsDocumentCache = new(LoadLayerVectorsDocument);

        [TestMethod]
        public void Embedding_ReturnsExpectedLength()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);

            node.Init();
            node.Forward(context);
            Memory<float> embedding = context.Embedding;

            Assert.AreEqual((int)model.Config!.EmbeddingLength, embedding.Length);
        }

        [TestMethod]
        public void Embedding_ThrowsForOutOfRangeToken()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: (int)model.Config!.VocabularySize);

            node.Init();
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => node.Forward(context));
        }

        [TestMethod]
        public void Embedding_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(context));
        }

        [TestMethod]
        public void EmbeddingCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var uncachedNode = new BitNetSharp.Nodes.EmbeddingNode(model);
            var cachedNode = new BitNetSharp.Nodes.EmbeddingNode(model, enableCache: true);
            var uncachedContext = TestModelFactory.CreateSession(model, token: 0);
            var cachedContext = TestModelFactory.CreateSession(model, token: 0);

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(uncachedContext);
            cachedNode.Forward(cachedContext);

            CollectionAssert.AreEqual(uncachedContext.Embedding.Span.ToArray(), cachedContext.Embedding.Span.ToArray());
            Assert.IsTrue(cachedNode.EnableCache);
        }

        [TestMethod]
        public void Embedding_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var firstNode = new BitNetSharp.Nodes.EmbeddingNode(model, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.EmbeddingNode(model, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.CPU, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        [DynamicData(nameof(GetEmbeddingCases))]
        public void Embedding_MatchesBaseline(int caseId)
        {
            LayerVectorCase testCase = GetLayerCase(caseId);
            string caseName = GetCaseName(testCase);
            float[] expectedValues = testCase.Dequantized.Values.ToArray();
            using var model = TestModelFactory.LoadModel();
            var node = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, testCase.TokenId);

            node.Init();
            node.Forward(context);
            float[] actualValues = context.Embedding.Span.ToArray();

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
            var node = new BitNetSharp.Nodes.RmsNormNode(model, normTensor);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, node.InferenceConfig.Backend);
            Assert.AreEqual(1, node.InferenceConfig.ThreadCount);
        }

        [TestMethod]
        public void RmsNorm_NullConfig_CreatesNewInstance()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var firstNode = new BitNetSharp.Nodes.RmsNormNode(model, normTensor, inferenceConfig: null);
            var secondNode = new BitNetSharp.Nodes.RmsNormNode(model, normTensor, inferenceConfig: null);

            Assert.AreEqual(BitNetSharp.Nodes.InferenceBackend.SIMD, firstNode.InferenceConfig.Backend);
            Assert.AreEqual(1, firstNode.InferenceConfig.ThreadCount);
            Assert.AreNotSame(firstNode.InferenceConfig, secondNode.InferenceConfig);
        }

        [TestMethod]
        public void RmsNorm_MatchesManualFormula()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingNode.Init();
            embeddingNode.Forward(context);
            float[] input = context.Embedding.Span.ToArray();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var node = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));

            node.Init();
            node.Forward(context);
            float[] actual = context.RmsNorm.Span.ToArray();
            float[] weights = ReadTensorValues(model, normTensor);
            float[] expected = ApplyManualRmsNorm(input, weights, model.Config!.AttentionLayerNormRmsEpsilon);

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_MatchesBaseline(int caseId)
        {
            LayerVectorCase testCase = GetLayerCase(caseId);
            string caseName = GetCaseName(testCase);
            float[] input = testCase.Dequantized.Values.ToArray();
            float[] expectedValues = testCase.FirstLayerRmsNorm.Values.ToArray();
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var node = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            node.Init();
            node.Forward(context);
            float[] actualValues = context.RmsNorm.Span.ToArray();

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_BaselineMatch_Tensor(int caseId)
        {
            LayerVectorCase testCase = GetLayerCase(caseId);
            string caseName = GetCaseName(testCase);
            float[] input = testCase.Dequantized.Values.ToArray();
            float[] expectedValues = testCase.FirstLayerRmsNorm.Values.ToArray();
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var node = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.Tensor, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            node.Init();
            node.Forward(context);
            float[] actualValues = context.RmsNorm.Span.ToArray();

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        [DynamicData(nameof(GetRmsNormCases))]
        public void RmsNorm_BaselineMatch_SIMD(int caseId)
        {
            LayerVectorCase testCase = GetLayerCase(caseId);
            string caseName = GetCaseName(testCase);
            float[] input = testCase.Dequantized.Values.ToArray();
            float[] expectedValues = testCase.FirstLayerRmsNorm.Values.ToArray();
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var node = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.SIMD, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = input.ToArray();

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.Inconclusive("AVX2 is not supported on the current machine.");
            }

            node.Init();
            node.Forward(context);
            float[] actualValues = context.RmsNorm.Span.ToArray();

            Assert.HasCount(expectedValues.Length, actualValues, caseName);
            AssertFloatArraysAreClose(expectedValues, actualValues, 1e-6f);
        }

        [TestMethod]
        public void RmsNormCache_MatchesUncachedReads()
        {
            using var model = TestModelFactory.LoadModel();
            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingNode.Init();
            embeddingNode.Forward(context);
            var normTensor = model.GetLayer(0).AttentionNorm;
            var uncachedNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var cachedNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                enableCache: true,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var cachedContext = TestModelFactory.CreateSession(model, token: 0);
            cachedContext.Embedding = context.Embedding.Span.ToArray();

            uncachedNode.Init();
            cachedNode.Init();
            uncachedNode.Forward(context);
            cachedNode.Forward(cachedContext);
            float[] uncached = context.RmsNorm.Span.ToArray();
            float[] cached = cachedContext.RmsNorm.Span.ToArray();

            Assert.IsTrue(cachedNode.EnableCache);
            AssertFloatArraysAreClose(uncached, cached, 0f);
        }

        [TestMethod]
        public void RmsNorm_CPU_Matches_Tensor()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingNode.Init();
            embeddingNode.Forward(context);
            var cpuNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var tensorNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.Tensor, 1));
            var tensorContext = TestModelFactory.CreateSession(model, token: 0);
            tensorContext.Embedding = context.Embedding.Span.ToArray();

            cpuNode.Init();
            tensorNode.Init();
            cpuNode.Forward(context);
            tensorNode.Forward(tensorContext);
            float[] expected = context.RmsNorm.Span.ToArray();
            float[] actual = tensorContext.RmsNorm.Span.ToArray();

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        public void RmsNorm_CPU_Matchs_SIMD()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var embeddingNode = new BitNetSharp.Nodes.EmbeddingNode(model);
            var context = TestModelFactory.CreateSession(model, token: 0);
            embeddingNode.Init();
            embeddingNode.Forward(context);
            var cpuNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var simdNode = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.SIMD, 1));
            var simdContext = TestModelFactory.CreateSession(model, token: 0);
            simdContext.Embedding = context.Embedding.Span.ToArray();

            cpuNode.Init();
            simdNode.Init();

            if (!Avx.IsSupported || !Avx2.IsSupported)
            {
                Assert.ThrowsExactly<NotSupportedException>(() => simdNode.Forward(simdContext));
                return;
            }

            cpuNode.Forward(context);
            simdNode.Forward(simdContext);
            float[] expected = context.RmsNorm.Span.ToArray();
            float[] actual = simdContext.RmsNorm.Span.ToArray();

            AssertFloatArraysAreClose(expected, actual, 1e-6f);
        }

        [TestMethod]
        public void RmsNorm_ForwardWithoutInit_Throws()
        {
            using var model = TestModelFactory.LoadModel();
            var normTensor = model.GetLayer(0).AttentionNorm;
            var node = new BitNetSharp.Nodes.RmsNormNode(
                model,
                normTensor,
                inferenceConfig: new BitNetSharp.Nodes.InferenceConfig(BitNetSharp.Nodes.InferenceBackend.CPU, 1));
            var context = TestModelFactory.CreateSession(model, token: 0);
            context.Embedding = new float[(int)model.Config!.EmbeddingLength];

            Assert.ThrowsExactly<InvalidOperationException>(() => node.Forward(context));
        }

        public static IEnumerable<object[]> GetEmbeddingCases()
        {
            return Enumerable.Range(0, LayerVectorsDocumentCache.Value.TestCases.Count)
                .Select(caseId => new object[] { caseId });
        }

        public static IEnumerable<object[]> GetRmsNormCases()
        {
            return Enumerable.Range(0, LayerVectorsDocumentCache.Value.TestCases.Count)
                .Select(caseId => new object[] { caseId });
        }

        private static LayerVectorCase GetLayerCase(int caseId)
        {
            return LayerVectorsDocumentCache.Value.TestCases[caseId];
        }

        private static string GetCaseName(LayerVectorCase testCase)
        {
            return $"token {testCase.TokenId} ({testCase.TokenText})";
        }

        private static LayerVectorsDocument LoadLayerVectorsDocument()
        {
            using var stream = File.OpenRead(TestProjectPaths.LayerVectorsPath);
            return JsonSerializer.Deserialize<LayerVectorsDocument>(stream) ?? throw new InvalidOperationException("Failed to load node baseline JSON.");
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
