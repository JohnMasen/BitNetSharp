namespace BitNetSharp.Models
{
    public sealed record BitNetModelConfig(
        string ArchitectureName,
        string ModelName,
        uint VocabularySize,
        uint ContextLength,
        uint EmbeddingLength,
        uint BlockCount,
        uint FeedForwardLength,
        uint AttentionHeadCount,
        uint AttentionKeyValueHeadCount,
        uint RopeDimensionCount,
        float RopeFrequencyBase,
        float AttentionLayerNormRmsEpsilon,
        uint FileType,
        uint QuantizationVersion)
    {
        public uint AttentionHeadDimension => EmbeddingLength / AttentionHeadCount;

        public uint KeyValueProjectionSize => AttentionHeadDimension * AttentionKeyValueHeadCount;
    }
}
