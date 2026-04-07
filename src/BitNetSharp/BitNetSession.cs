using BitNetSharp.Layers;
using BitNetSharp.Models;

namespace BitNetSharp
{
    public class BitNetSession(BitNetModel model)
    {
        public BitNetModel Model => model;

        public int[] Tokens { get; set; } = [];

        public int CurrentToken { get; set; }

        public float[]? Embedding { get; set; }

        public float[]? RmsNorm { get; set; }

        public QKVProjectionOutput? QKVProjection { get; set; }

        public float[]? AttentionSubNorm { get; set; }

        public float[]? AttentionOutput { get; set; }
    }
}
