namespace BitNetSharp.Models
{
    public sealed record BitNetTokenizerConfig(
        string TokenizerModelType,
        bool AddBosToken,
        uint BosTokenId,
        uint EosTokenId,
        uint PaddingTokenId,
        string ChatTemplate,
        IReadOnlyList<string> Tokens,
        IReadOnlyList<string> Merges,
        IReadOnlyList<float> Scores,
        IReadOnlyList<int> TokenTypes);
}
