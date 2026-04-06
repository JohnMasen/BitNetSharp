namespace BitNetSharp.Models
{
    public enum BitNetTensorRole
    {
        Unknown,
        TokenEmbedding,
        OutputNorm,
        AttentionNorm,
        AttentionSubNorm,
        AttentionQueryWeight,
        AttentionKeyWeight,
        AttentionValueWeight,
        AttentionOutputWeight,
        FeedForwardNorm,
        FeedForwardSubNorm,
        FeedForwardGateWeight,
        FeedForwardUpWeight,
        FeedForwardDownWeight,
    }
}
