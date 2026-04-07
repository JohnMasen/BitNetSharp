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
        AttentionOutputScale,
        AttentionOutputBias,
        FeedForwardNorm,
        FeedForwardSubNorm,
        FeedForwardGateWeight,
        FeedForwardUpWeight,
        FeedForwardDownWeight,
    }
}
