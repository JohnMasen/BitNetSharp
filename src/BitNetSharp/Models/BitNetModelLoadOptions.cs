namespace BitNetSharp.Models
{
    /// <summary>
    /// Provides optional model-loading behaviors for <see cref="BitNetModel.Load(string, BitNetModelLoadOptions?)"/>.
    /// </summary>
    public sealed class BitNetModelLoadOptions
    {
        /// <summary>
        /// Gets or sets the callback used to parse GGUF metadata into a structured model configuration.
        /// Implementations must not retain references to the supplied <c>GGUFFile</c> instance after the callback returns.
        /// </summary>
        public BitNetMetadataParser? MetadataParser { get; init; }
    }
}
