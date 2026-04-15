namespace BitNetSharp
{
    public sealed record BitNetMemoryStatistics(
        int AllocationCount,
        long EstimatedTotalBytes,
        IReadOnlyList<BitNetMemoryAllocationSnapshot> Allocations);
}
