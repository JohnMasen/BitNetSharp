namespace BitNetSharp.Models
{
    public sealed record BitNetLayerDefinition(
        int LayerIndex,
        BitNetTensorInfo AttentionNorm,
        BitNetTensorInfo AttentionSubNorm,
        BitNetTensorInfo AttentionQueryWeight,
        BitNetTensorInfo AttentionKeyWeight,
        BitNetTensorInfo AttentionValueWeight,
        BitNetTensorInfo AttentionOutputWeight,
        BitNetTensorInfo? AttentionOutputScale,
        BitNetTensorInfo? AttentionOutputBias,
        BitNetTensorInfo FeedForwardNorm,
        BitNetTensorInfo FeedForwardSubNorm,
        BitNetTensorInfo FeedForwardGateWeight,
        BitNetTensorInfo FeedForwardUpWeight,
        BitNetTensorInfo FeedForwardDownWeight);
}
