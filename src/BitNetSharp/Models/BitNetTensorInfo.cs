using GGUFSharp;
using System.Collections.Generic;

namespace BitNetSharp.Models
{
    public sealed record BitNetTensorInfo(
        string Name,
        int? LayerIndex,
        BitNetTensorRole Role,
        GGUFTensorType TensorType,
        IReadOnlyList<ulong> Dimensions,
        ulong Offset,
        ulong ByteSize,
        bool IsQuantized,
        bool IsGlobal);
}
