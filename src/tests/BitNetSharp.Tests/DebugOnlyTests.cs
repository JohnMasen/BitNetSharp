using System.Buffers;
using System.Linq;
using System.Reflection;
using BitNetSharp.Core;
using BitNetSharp.Nodes;

namespace BitNetSharp.Tests
{
    [TestClass]
    [DoNotParallelize]
    [Ignore("exclude debug tests on normal run")]
    public sealed class DebugOnlyTests
    {
        private const float LayerTolerance = 1e-3f;
        private const float AttentionContextTolerance = 1e-2f;
        private const float AttentionOutputTolerance = 3e-3f;
        private const float FeedForwardInputTolerance = 3e-3f;
        private const float FeedForwardNormTolerance = 4e-3f;
        private const float StandaloneFeedForwardOutputTolerance = 4e-1f;
        private const float RopeTolerance = 1e-3f;
        private const float KvCacheReadTolerance = 5e-2f;
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
        [Ignore("Need further investigate")]
        public void HiChatPrompt_FullRuntimeChain_MatchesDumpUntilFirstMismatch()
        {
            FullDumpBaseline.FullDumpPrompt prompt = FullDumpBaseline.Manifest.Prompt;
            Models.BitNetModel model = GetModel();
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, TestInferenceConfigs.Simd(0), enableSampling: false);
            int[] promptTokenIds = prompt.TokenIds.ToArray();
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
            AssertVectorPrefix("embedding_output", FullDumpBaseline.ReadFloatValues("embedding_output_first_32"), session.Embedding.Span);

            for (int layerIndex = 0; layerIndex < model.Layers.Count; layerIndex++)
            {
                Models.BitNetLayerDefinition layer = model.Layers[layerIndex];

                attentionNormNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_attention_norm_output", FullDumpBaseline.ReadFloatValues("first_layer_attention_norm_output_first_32"), session.RmsNorm.Span);
                }

                InvokePrivateMethod(runtime, "ExecuteQKVProjection", session, layerIndex, layer);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_qkv_query_output", FullDumpBaseline.ReadFloatValues("first_layer_qcur_before_rope_first_32"), session.QKVQuery.Span);
                    AssertVectorPrefix("first_layer_qkv_key_output", FullDumpBaseline.ReadFloatValues("first_layer_kcur_before_rope_first_32"), session.QKVKey.Span);
                    AssertVectorPrefix("first_layer_qkv_value_output", FullDumpBaseline.ReadFloatValues("first_layer_vcur_first_32"), session.QKVValue.Span);
                    AssertVectorPrefix("qcur_before_rope", FullDumpBaseline.ReadFloatValues("first_layer_qcur_before_rope_first_32"), session.QKVQuery.Span);
                    AssertVectorPrefix("kcur_before_rope", FullDumpBaseline.ReadFloatValues("first_layer_kcur_before_rope_first_32"), session.QKVKey.Span);

                    int embeddingLength = checked((int)model.Config!.EmbeddingLength);
                    int headCount = checked((int)model.Config.AttentionHeadCount);
                    int keyValueHeadCount = checked((int)model.Config.AttentionKeyValueHeadCount);
                    int headDimension = checked((int)model.Config.AttentionHeadDimension);
                    int keyValueLength = checked((int)model.Config.KeyValueProjectionSize);
                    int ropeDimensionCount = checked((int)model.Config.RopeDimensionCount);
                    float ropeFrequencyBase = model.Config.RopeFrequencyBase;
                    int positionIndex = session.CacheWritePosition;
                    float[] ropeQueryValues = ApplyRopeForTest(session.QKVQuery.Span, positionIndex, headCount, headDimension, ropeDimensionCount, ropeFrequencyBase);
                    float[] ropeKeyValues = ApplyRopeForTest(session.QKVKey.Span, positionIndex, keyValueHeadCount, headDimension, ropeDimensionCount, ropeFrequencyBase);
                    ReadOnlyMemory<float> ropeQueryMemory = ropeQueryValues;
                    AssertVectorPrefix("first_layer_qcur_after_rope", FullDumpBaseline.ReadFloatValues("first_layer_qcur_after_rope_first_32"), ropeQueryValues, RopeTolerance);
                    AssertVectorPrefix("first_layer_kcur_after_rope", FullDumpBaseline.ReadFloatValues("first_layer_kcur_after_rope_first_32"), ropeKeyValues, RopeTolerance);
                    AssertVectorPrefix("first_layer_k_before_kv_cache_write", FullDumpBaseline.ReadFloatValues("first_layer_k_before_kv_cache_write_first_32"), ropeKeyValues, RopeTolerance);
                    AssertVectorPrefix("first_layer_v_before_kv_cache_write", FullDumpBaseline.ReadFloatValues("first_layer_v_before_kv_cache_write_first_32"), session.QKVValue.Span, RopeTolerance);

                    WriteKeyCacheForTest(session, layerIndex, ropeKeyValues);
                    WriteValueCacheForTest(session, layerIndex, session.QKVValue.Span, keyValueHeadCount, headDimension);
                    float[] keyCacheReadValues = ReadKeyCachePrefix(session, layerIndex, keyValueHeadCount, headDimension, 32);
                    float[] valueCacheReadValues = ReadValueCachePrefix(session, layerIndex, keyValueHeadCount, headDimension, 32);
                    AssertVectorPrefix("first_layer_k_after_kv_cache_read", FullDumpBaseline.ReadFloatValues("first_layer_k_after_kv_cache_read_first_32"), keyCacheReadValues, KvCacheReadTolerance);
                    AssertVectorPrefix("first_layer_v_after_kv_cache_read", FullDumpBaseline.ReadFloatValues("first_layer_v_after_kv_cache_read_first_32"), valueCacheReadValues, KvCacheReadTolerance);

                    using IMemoryOwner<float> attentionContextOwner = MemoryPool<float>.Shared.Rent(embeddingLength);
                    Memory<float> attentionContext = attentionContextOwner.Memory[..embeddingLength];
                    InvokePrivateMethod(runtime, "BuildCachedAttentionContext", session, layerIndex, ropeQueryMemory, attentionContextOwner.Memory[..embeddingLength], headCount, keyValueHeadCount, headDimension, keyValueLength, CancellationToken.None);
                    AssertVectorPrefix("first_layer_attention_context_output_extra", FullDumpBaseline.ReadFloatValues("first_layer_kqv_out_0_first_32"), attentionContext.Span);
                }

                InvokePrivateMethod(runtime, "ExecuteAttention", session, layerIndex, layer, CancellationToken.None);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_attention_output", FullDumpBaseline.ReadFloatValues("first_layer_attn_o_out_0_first_32"), session.AttentionOutput.Span);
                }

                residualNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_input", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_input_first_32"), session.FeedForwardInput.Span);
                }

                feedForwardNormNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_norm_output", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_norm_output_first_32"), session.FeedForwardNorm.Span);
                }

                if (layerIndex == 0)
                {
                    VerifyStandaloneFeedForwardNode(model, layer, session);
                }

                InvokePrivateMethod(runtime, "ExecuteFeedForward", session, layerIndex, layer);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_sub_norm_output", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_sub_norm_output_first_32"), session.FeedForwardSubNorm.Span);
                    AssertVectorPrefix("first_layer_feedforward_output", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_down_output_first_32"), session.FeedForwardOutput.Span);
                }

                feedForwardResidualNodes[layerIndex].Forward(session);
                if (layerIndex == 0)
                {
                    AssertVectorPrefix("first_layer_feedforward_output_runtime_semantic", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_output_runtime_semantic_first_32"), session.Embedding.Span);
                }
            }

            finalNormNode.Forward(session);
            AssertPromptTokenIds(prompt, session);
            AssertVectorPrefix("final_norm_output", FullDumpBaseline.ReadFloatValues("final_norm_output_first_32"), session.FinalNormOutput.Span);

            lmHeadNode.Forward(session);
            AssertVectorPrefix("lm_head_output_logits", FullDumpBaseline.ReadFloatValues("lm_head_output_logits_first_32"), session.Logits.Span);
        }

        private static void VerifyStandaloneFeedForwardNode(Models.BitNetModel model, Models.BitNetLayerDefinition layer, BitNetSession session)
        {
            _ = RunStandaloneFeedForwardNode(model, layer, session, TestInferenceConfigs.Simd(0), "standalone_feedforward_output");
        }

        private static float[] RunStandaloneFeedForwardNode(Models.BitNetModel model, Models.BitNetLayerDefinition layer, BitNetSession session, InferenceConfig inferenceConfig, string outputVectorName)
        {
            using var standaloneMemoryManager = new BitNetMemoryManager();
            using var standaloneSession = new BitNetSession(model, standaloneMemoryManager);
            session.FeedForwardNorm.Span.CopyTo(standaloneSession.FeedForwardNorm.Span);

            var node = new FeedForwardNode(
                model,
                layer.FeedForwardSubNorm,
                layer.FeedForwardGateWeight,
                layer.FeedForwardUpWeight,
                layer.FeedForwardDownWeight,
                inferenceConfig: inferenceConfig);

            node.Init();
            node.Forward(standaloneSession);

            AssertVectorPrefix("standalone_feedforward_sub_norm_output", FullDumpBaseline.ReadFloatValues("first_layer_feedforward_sub_norm_output_first_32"), standaloneSession.FeedForwardSubNorm.Span);
            AssertVectorPrefix(outputVectorName, FullDumpBaseline.ReadFloatValues("first_layer_feedforward_down_output_first_32"), standaloneSession.FeedForwardOutput.Span);
            return standaloneSession.FeedForwardOutput.Span.ToArray();
        }

        private static void AssertPromptTokenIds(FullDumpBaseline.FullDumpPrompt prompt, BitNetSession session)
        {
            CollectionAssert.AreEqual(prompt.TokenIds.ToArray(), session.Tokens.ToArray(), "Prompt token ids mismatch before layer comparison.");
        }

        private static void AssertVectorPrefix(string vectorName, IReadOnlyList<float> expectedPrefix, ReadOnlySpan<float> actual)
        {
            float tolerance = string.Equals(vectorName, "first_layer_attention_context_output_extra", StringComparison.Ordinal)
                ? AttentionContextTolerance
                : string.Equals(vectorName, "first_layer_attention_output", StringComparison.Ordinal)
                    ? AttentionOutputTolerance
                    : string.Equals(vectorName, "first_layer_feedforward_input", StringComparison.Ordinal)
                        ? FeedForwardInputTolerance
                    : string.Equals(vectorName, "first_layer_feedforward_norm_output", StringComparison.Ordinal)
                        ? FeedForwardNormTolerance
                    : vectorName.StartsWith("standalone_feedforward_output", StringComparison.Ordinal)
                        ? StandaloneFeedForwardOutputTolerance
                    : LayerTolerance;
            AssertVectorPrefix(vectorName, expectedPrefix, actual, tolerance);
        }

        private static void AssertVectorPrefix(string vectorName, IReadOnlyList<float> expectedPrefix, ReadOnlySpan<float> actual, float tolerance)
        {
            Assert.IsTrue(actual.Length >= expectedPrefix.Count, $"{vectorName} actual length {actual.Length} is shorter than expected prefix length {expectedPrefix.Count}.");
            float maxError = float.NegativeInfinity;
            int maxErrorIndex = -1;
            float expectedAtMaxError = 0f;
            float actualAtMaxError = 0f;
            for (int index = 0; index < expectedPrefix.Count; index++)
            {
                float expected = expectedPrefix[index];
                float actualValue = actual[index];
                float error = MathF.Abs(expected - actualValue);
                if (error > maxError)
                {
                    maxError = error;
                    maxErrorIndex = index;
                    expectedAtMaxError = expected;
                    actualAtMaxError = actualValue;
                }
            }

            if (maxError > tolerance)
            {
                Assert.Fail($"{vectorName} max mismatch at index {maxErrorIndex}. Expected {expectedAtMaxError}, actual {actualAtMaxError}, max error {maxError}, tolerance {tolerance}.");
            }
        }

        private static float[] ApplyRopeForTest(ReadOnlySpan<float> source, int positionIndex, int headCount, int headDimension, int ropeDimensionCount, float freqBase)
        {
            float[] destination = source.ToArray();
            int rotaryDimensions = Math.Min(Math.Min(ropeDimensionCount, headDimension), headDimension - (headDimension % 2));
            if (rotaryDimensions <= 0)
            {
                return destination;
            }

            int rotaryHalf = rotaryDimensions / 2;
            for (int headIndex = 0; headIndex < headCount; headIndex++)
            {
                int headOffset = headIndex * headDimension;
                for (int rotaryIndex = 0; rotaryIndex < rotaryHalf; rotaryIndex++)
                {
                    float theta = positionIndex / MathF.Pow(freqBase, (2f * rotaryIndex) / rotaryDimensions);
                    float cos = MathF.Cos(theta);
                    float sin = MathF.Sin(theta);
                    int evenOffset = headOffset + rotaryIndex;
                    int oddOffset = headOffset + rotaryIndex + rotaryHalf;
                    float even = source[evenOffset];
                    float odd = source[oddOffset];
                    destination[evenOffset] = (even * cos) - (odd * sin);
                    destination[oddOffset] = (odd * cos) + (even * sin);
                }
            }

            return destination;
        }

        private static float[] ReadKeyCachePrefix(BitNetSession session, int layerIndex, int keyValueHeadCount, int headDimension, int elementCount)
        {
            RuntimeTensor keyCacheTensor = session.GetOrCreateLayerKeyCacheTensor(layerIndex);
            if (!keyCacheTensor.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> keyCacheMemory))
            {
                throw new InvalidOperationException("Failed to read key cache tensor as float memory.");
            }

            ReadOnlySpan<float> keyCache = keyCacheMemory.Span;
            float[] values = new float[elementCount];
            int valueIndex = 0;
            for (int dimensionIndex = 0; dimensionIndex < headDimension && valueIndex < values.Length; dimensionIndex++)
            {
                for (int kvHeadIndex = 0; kvHeadIndex < keyValueHeadCount && valueIndex < values.Length; kvHeadIndex++)
                {
                    int cacheOffset = (dimensionIndex * keyValueHeadCount) + kvHeadIndex;
                    values[valueIndex++] = (float)(Half)keyCache[cacheOffset];
                }
            }

            return values;
        }

        private static void WriteKeyCacheForTest(BitNetSession session, int layerIndex, ReadOnlySpan<float> key)
        {
            RuntimeTensor keyCacheTensor = session.GetOrCreateLayerKeyCacheTensor(layerIndex);
            if (!keyCacheTensor.TryGet<Memory<float>>(out Memory<float> keyCacheMemory))
            {
                throw new InvalidOperationException("Failed to access key cache tensor as writable float memory.");
            }

            int keyValueLength = key.Length;
            int cacheOffset = session.CacheWritePosition * keyValueLength;
            key.CopyTo(keyCacheMemory.Span.Slice(cacheOffset, keyValueLength));
        }

        private static float[] ReadValueCachePrefix(BitNetSession session, int layerIndex, int keyValueHeadCount, int headDimension, int elementCount)
        {
            RuntimeTensor valueCacheTensor = session.GetOrCreateLayerValueCacheTensor(layerIndex);
            if (!valueCacheTensor.TryGet<ReadOnlyMemory<float>>(out ReadOnlyMemory<float> valueCacheMemory))
            {
                throw new InvalidOperationException("Failed to read value cache tensor as float memory.");
            }

            ReadOnlySpan<float> valueCache = valueCacheMemory.Span;
            float[] values = new float[elementCount];
            for (int index = 0; index < values.Length; index++)
            {
                values[index] = (float)(Half)valueCache[index];
            }

            return values;
        }

        private static void WriteValueCacheForTest(BitNetSession session, int layerIndex, ReadOnlySpan<float> value, int keyValueHeadCount, int headDimension)
        {
            RuntimeTensor valueCacheTensor = session.GetOrCreateLayerValueCacheTensor(layerIndex);
            if (!valueCacheTensor.TryGet<Memory<float>>(out Memory<float> valueCacheMemory))
            {
                throw new InvalidOperationException("Failed to access value cache tensor as writable float memory.");
            }

            int nCtx = checked((int)session.Model.Config!.ContextLength);
            int tokenIndex = session.CacheWritePosition;
            for (int kvHeadIndex = 0; kvHeadIndex < keyValueHeadCount; kvHeadIndex++)
            {
                int sourceHeadOffset = kvHeadIndex * headDimension;
                for (int dimensionIndex = 0; dimensionIndex < headDimension; dimensionIndex++)
                {
                    int sourceOffset = sourceHeadOffset + dimensionIndex;
                    int destinationOffset = tokenIndex + (nCtx * dimensionIndex) + (nCtx * headDimension * kvHeadIndex);
                    valueCacheMemory.Span[destinationOffset] = value[sourceOffset];
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
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.NonPublic)
                .SingleOrDefault(candidate => candidate.Name == methodName && candidate.GetParameters().Length == args.Length && ParametersMatch(candidate.GetParameters(), args));
            if (method is null)
            {
                throw new InvalidOperationException($"Failed to get private method '{methodName}'.");
            }

            object? target = method.IsStatic ? null : instance;
            _ = method.Invoke(target, args);
        }

        private static bool ParametersMatch(ParameterInfo[] parameters, object[] args)
        {
            for (int index = 0; index < parameters.Length; index++)
            {
                object? argument = args[index];
                Type parameterType = parameters[index].ParameterType;
                if (argument is null)
                {
                    if (parameterType.IsValueType && Nullable.GetUnderlyingType(parameterType) is null)
                    {
                        return false;
                    }

                    continue;
                }

                Type argumentType = argument.GetType();
                if (!parameterType.IsAssignableFrom(argumentType))
                {
                    return false;
                }
            }

            return true;
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
    }
}
