namespace BitNetSharp.Models
{
    /// <summary>
    /// Configuration for a BitNet transformer model.
    /// </summary>
    public sealed class BitNetConfig
    {
        /// <summary>Size of the vocabulary.</summary>
        public int VocabSize { get; set; } = 32000;

        /// <summary>Embedding (model) dimension.</summary>
        public int EmbedDim { get; set; } = 512;

        /// <summary>Number of transformer blocks (layers).</summary>
        public int NumLayers { get; set; } = 6;

        /// <summary>Number of attention heads per block.</summary>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Hidden size of the feed-forward network.
        /// Typically 4× the embedding dimension.
        /// </summary>
        public int FFNDim { get; set; } = 2048;

        /// <summary>Maximum supported sequence length.</summary>
        public int MaxSequenceLength { get; set; } = 2048;

        /// <summary>Creates a small model configuration suitable for testing.</summary>
        public static BitNetConfig Small() =>
            new BitNetConfig
            {
                VocabSize = 1000,
                EmbedDim = 64,
                NumLayers = 2,
                NumHeads = 4,
                FFNDim = 256,
                MaxSequenceLength = 128
            };

        /// <summary>Creates a medium model configuration.</summary>
        public static BitNetConfig Medium() =>
            new BitNetConfig
            {
                VocabSize = 32000,
                EmbedDim = 512,
                NumLayers = 6,
                NumHeads = 8,
                FFNDim = 2048,
                MaxSequenceLength = 2048
            };

        /// <summary>Creates a large model configuration.</summary>
        public static BitNetConfig Large() =>
            new BitNetConfig
            {
                VocabSize = 32000,
                EmbedDim = 1024,
                NumLayers = 24,
                NumHeads = 16,
                FFNDim = 4096,
                MaxSequenceLength = 2048
            };
    }
}
