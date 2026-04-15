namespace BitNetSharp
{
    public sealed record BitNetMemoryAllocationSnapshot(
        Guid SessionId,
        string Key,
        Type ElementType,
        int RequestedLength,
        long EstimatedBytes);
}
