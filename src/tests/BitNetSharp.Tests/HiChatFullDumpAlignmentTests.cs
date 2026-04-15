using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Buffers;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    public sealed class HiChatFullDumpAlignmentTests
    {
        private const float LayerTolerance = 1e-3f;
        private static readonly Lazy<HiChatFullDumpDocument> HiChatFullDumpDocumentCache = new(LoadHiChatFullDumpDocument);
        private static readonly Lazy<HiRopeKCacheDumpDocument> HiRopeKCacheDumpDocumentCache = new(LoadHiRopeKCacheDumpDocument);
        private static Models.BitNetModel? sharedModel;

        [ClassInitialize]
        public static void ClassInitialize(TestContext _)
        {
            sharedModel = TestModelFactory.LoadModel();
        }

        [ClassCleanup]
        public static void ClassCleanup()
        {
            sharedModel?.Dispose();
            sharedModel = null;
        }

        [TestMethod]
        public void HiChatPrompt_FullRuntimeChain_MatchesDumpUntilFirstMismatch()
        {
            HiChatFullDumpDocument document = HiChatFullDumpDocumentCache.Value;
            HiRopeKCacheDumpDocument ropeDocument = HiRopeKCacheDumpDocumentCache.Value;
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0), enableSampling: false);
            int[] promptTokenIds = document.Prompt.TokenIds.ToArray();
            int[] promptPrefixTokenIds = promptTokenIds[..^1];
            int finalPromptTokenId = promptTokenIds[^1];

            runtime.Prefill(promptPrefixTokenIds);
            BitNetSession session = runtime.Session;
            session.AppendToken(finalPromptTokenId);
            session.CacheWritePosition = session.CacheLength;

            SetPrivateProperty(session, nameof(BitNetSession.CurrentToken), finalPromptTokenId);

            BitNetSharp.Nodes.EmbeddingNode embeddingNode = GetPrivateField<BitNetSharp.Nodes.EmbeddingNode>(runtime, "embeddingNode");
            BitNetSharp.Nodes.RmsNormNode[] attentionNormNodes = GetPrivateField<BitNetSharp.Nodes.RmsNormNode[]>(runtime, "attentionNormNodes");
            BitNetSharp.Nodes.ResidualNode[] residualNodes = GetPrivateField<BitNetSharp.Nodes.ResidualNode[]>(runtime, "residualNodes");
            BitNetSharp.Nodes.FeedForwardNormNode[] feedForwardNormNodes = GetPrivateField<BitNetSharp.Nodes.FeedForwardNormNode[]>(runtime, "feedForwardNormNodes");
            BitNetSharp.Nodes.FeedForwardResidualNode[] feedForwardResidualNodes = GetPrivateField<BitNetSharp.Nodes.FeedForwardResidualNode[]>(runtime, "feedForwardResidualNodes");
            BitNetSharp.Nodes.FinalNormNode finalNormNode = GetPrivateField<BitNetSharp.Nodes.FinalNormNode>(runtime, "finalNormNode");
            BitNetSharp.Nodes.LmHeadNode lmHeadNode = GetPrivateField<BitNetSharp.Nodes.LmHeadNode>(runtime, "lmHeadNode");

            embeddingNode.Forward(session);
            AssertVectorPrefix("embedding_output", document.PromptCompleteTensorDumps.EmbeddingOutput.First32Values, session.Embedding.Span);

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                Models.BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_attention_norm_output", document.PromptCompleteTensorDumps.FirstLayerAttentionNormOutput.First32Values, session.RmsNorm.Span);
                }

                InvokePrivateMethod(runtime, "ExecuteQKVProjection", session, layerIndex, layer);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_qkv_query_output", document.PromptCompleteTensorDumps.FirstLayerQkvQueryOutput.First32Values, session.QKVQuery.Span);
                    AssertVectorPrefix("first_layer_qkv_key_output", document.PromptCompleteTensorDumps.FirstLayerQkvKeyOutput.First32Values, session.QKVKey.Span);
                    AssertVectorPrefix("first_layer_qkv_value_output", document.PromptCompleteTensorDumps.FirstLayerQkvValueOutput.First32Values, session.QKVValue.Span);
                    AssertVectorPrefix("qcur_before_rope", ropeDocument.RuntimeAttentionDump.QcurBeforeRope.First32Values, session.QKVQuery.Span);
                    AssertVectorPrefix("kcur_before_rope", ropeDocument.RuntimeAttentionDump.KcurBeforeRope.First32Values, session.QKVKey.Span);

                    int embeddingLength = checked((int)model.Config!.EmbeddingLength);
                    int headCount = checked((int)model.Config.AttentionHeadCount);
                    int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
                    int headDimension = checked((int)model.Config.AttentionHeadDimension);
                    int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
                    using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
                    Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
                    ReadOnlyMemory<float> queryMemory = session.QKVQuery;
                    InvokePrivateMethod(runtime, "BuildCachedAttentionContext", session, layerIndex, queryMemory, attentionContextOwner.Memory[..embeddingLength], headCount, keyValueHeadCount, headDimension, keyValueLength, CancellationToken.None);
                    AssertVectorPrefix("first_layer_attention_context_output_extra", document.PromptCompleteTensorDumps.FirstLayerAttentionContextOutputExtra.First32Values, attentionContext.Span);
                }

                InvokePrivateMethod(runtime, "ExecuteAttention", session, layerIndex, layer, CancellationToken.None);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_attention_output", document.PromptCompleteTensorDumps.FirstLayerAttentionOutput.First32Values, session.AttentionOutput.Span);
                }

                residualNodes[layerIndex].Forward(session);
                feedForwardNormNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_norm_output", document.PromptCompleteTensorDumps.FirstLayerFeedForwardNormOutput.First32Values, session.FeedForwardNorm.Span);
                }

                InvokePrivateMethod(runtime, "ExecuteFeedForward", session, layerIndex, layer);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_output", document.PromptCompleteTensorDumps.FirstLayerFeedForwardOutput.First32Values, session.FeedForwardOutput.Span);
                }

                feedForwardResidualNodes[layerIndex].Forward(session);
            }

            finalNormNode.Forward(session);
            AssertPromptTokenIds(document, session);
            AssertVectorPrefix("final_norm_output", document.PromptCompleteTensorDumps.FinalNormOutput.First32Values, session.FinalNormOutput.Span);

            lmHeadNode.Forward(session);
            AssertVectorPrefix("lm_head_output_logits", document.PromptCompleteTensorDumps.LmHeadOutputLogits.First32Values, session.Logits.Span);
        }

        private static void AssertPromptTokenIds(HiChatFullDumpDocument document, BitNetSession session)
        {
            CollectionAssert.AreEqual(document.Prompt.TokenIds.ToArray(), session.Tokens.ToArray(), "Prompt token ids mismatch before layer comparison.");
        }

        private static void AssertVectorPrefix(string vectorName, IReadOnlyList<float> expectedPrefix, ReadOnlySpan<float> actual)
        {
            Assert.IsTrue(actual.Length >= expectedPrefix.Count, $"{vectorName} actual length {actual.Length} is shorter than expected prefix length {expectedPrefix.Count}.");
            for (int index = 0; index < expectedPrefix.Count; index++)
            {
                float expected = expectedPrefix[index];
                float actualValue = actual[index];
                if (MathF.Abs(expected - actualValue) > LayerTolerance)
                {
                    Assert.Fail($"{vectorName} mismatch at index {index}. Expected {expected}, actual {actualValue}, tolerance {LayerTolerance}.");
                }
            }
        }

        private static Models.BitNetModel GetModel()
        {
            return sharedModel ?? throw new InvalidOperationException("Hi chat full dump alignment test model is not initialized.");
        }

        private static T GetPrivateField<T>(object instance, string fieldName)
        {
            FieldInfo? field = instance.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field?.GetValue(instance) is not T value)
            {
                throw new InvalidOperationException($"Failed to get private field '{fieldName}'.");
            }

            return value;
        }

        private static void InvokePrivateMethod(object instance, string methodName, params object[] args)
        {
            MethodInfo? method = instance.GetType()
                .GetMethods(BindingFlags.Instance | BindingFlags.NonPublic)
                .SingleOrDefault(candidate => candidate.Name == methodName && candidate.GetParameters().Length == args.Length);
            if (method is null)
            {
                throw new InvalidOperationException($"Failed to get private method '{methodName}'.");
            }

            _ = method.Invoke(instance, args);
        }

        private static void SetPrivateProperty<T>(object instance, string propertyName, T value)
        {
            PropertyInfo? property = instance.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            MethodInfo? setter = property?.SetMethod;
            if (setter is null)
            {
                throw new InvalidOperationException($"Failed to get writable property '{propertyName}'.");
            }

            setter.Invoke(instance, new object?[] { value });
        }

        private static HiChatFullDumpDocument LoadHiChatFullDumpDocument()
        {
            string json = File.ReadAllText(TestProjectPaths.HiChatFullDumpPath);
            return JsonSerializer.Deserialize<HiChatFullDumpDocument>(json) ?? throw new InvalidOperationException("Failed to load hi chat full dump JSON.");
        }

        private static HiRopeKCacheDumpDocument LoadHiRopeKCacheDumpDocument()
        {
            string json = File.ReadAllText(TestProjectPaths.HiRopeKCacheDumpPath);
            return JsonSerializer.Deserialize<HiRopeKCacheDumpDocument>(json) ?? throw new InvalidOperationException("Failed to load hi rope/k-cache dump JSON.");
        }

        internal sealed record HiChatFullDumpDocument(
            [property: JsonPropertyName("prompt")] HiChatPrompt Prompt,
            [property: JsonPropertyName("prompt_complete_tensor_dumps")] HiChatTensorDumps PromptCompleteTensorDumps);

        internal sealed record HiChatPrompt(
            [property: JsonPropertyName("text")] string Text,
            [property: JsonPropertyName("token_ids")] IReadOnlyList<int> TokenIds);

        internal sealed record HiChatTensorDumps(
            [property: JsonPropertyName("embedding_output")] HiChatTensorDump EmbeddingOutput,
            [property: JsonPropertyName("first_layer_attention_norm_output")] HiChatTensorDump FirstLayerAttentionNormOutput,
            [property: JsonPropertyName("first_layer_qkv_query_output")] HiChatTensorDump FirstLayerQkvQueryOutput,
            [property: JsonPropertyName("first_layer_qkv_key_output")] HiChatTensorDump FirstLayerQkvKeyOutput,
            [property: JsonPropertyName("first_layer_qkv_value_output")] HiChatTensorDump FirstLayerQkvValueOutput,
            [property: JsonPropertyName("first_layer_attention_context_output_extra")] HiChatTensorDump FirstLayerAttentionContextOutputExtra,
            [property: JsonPropertyName("first_layer_attention_output")] HiChatTensorDump FirstLayerAttentionOutput,
            [property: JsonPropertyName("first_layer_feedforward_norm_output")] HiChatTensorDump FirstLayerFeedForwardNormOutput,
            [property: JsonPropertyName("first_layer_feedforward_output")] HiChatTensorDump FirstLayerFeedForwardOutput,
            [property: JsonPropertyName("final_norm_output")] HiChatTensorDump FinalNormOutput,
            [property: JsonPropertyName("lm_head_output_logits")] HiChatTensorDump LmHeadOutputLogits);

        internal sealed record HiChatTensorDump(
            [property: JsonPropertyName("first_32_values")] IReadOnlyList<float> First32Values);

        internal sealed record HiRopeKCacheDumpDocument(
            [property: JsonPropertyName("runtime_attention_dump")] HiRopeRuntimeAttentionDump RuntimeAttentionDump);

        internal sealed record HiRopeRuntimeAttentionDump(
            [property: JsonPropertyName("qcur_before_rope")] HiChatTensorDump QcurBeforeRope,
            [property: JsonPropertyName("kcur_before_rope")] HiChatTensorDump KcurBeforeRope);
    }
}
