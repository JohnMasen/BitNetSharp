using GGUFSharp;

namespace BitNetSharp.Models
{
    /// <summary>
    /// Parses GGUF metadata into the structured model metadata required by <see cref="BitNetModel"/>.
    /// Implementations must only read the supplied <see cref="GGUFFile"/> during callback execution and must not retain references to it after returning.
    /// </summary>
    /// <param name="file">The GGUF file whose metadata should be parsed for model loading.</param>
    /// <returns>A structured metadata result used to initialize the model configuration and tokenizer configuration.</returns>
    public delegate BitNetMetadataParseResult BitNetMetadataParser(GGUFFile file);
}
