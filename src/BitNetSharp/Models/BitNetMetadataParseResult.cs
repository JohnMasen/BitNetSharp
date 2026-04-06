namespace BitNetSharp.Models
{
    /// <summary>
    /// Represents the structured metadata produced by a GGUF metadata parser for model loading.
    /// </summary>
    public sealed record BitNetMetadataParseResult
    {
        /// <summary>
        /// Gets the parsed static model configuration.
        /// </summary>
        public required BitNetModelConfig ModelConfig { get; init; }

        /// <summary>
        /// Gets the parsed tokenizer configuration.
        /// </summary>
        public required BitNetTokenizerConfig TokenizerConfig { get; init; }
    }
}
