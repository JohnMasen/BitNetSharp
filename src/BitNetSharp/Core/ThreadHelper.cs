using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace BitNetSharp.Core
{
    internal static class ThreadHelper
    {
        private const int GroupMultiplier = 2;

        internal static int DefaultGroupCount { get; }= Environment.ProcessorCount * GroupMultiplier;

        internal static void ForEachRange(int itemCount, Action<int, int> rangeAction, int threadCount = 0, int itemByteLength = 1, int? alignmentByteLength = null)
        {
            ArgumentNullException.ThrowIfNull(rangeAction);

            WorkRange[] ranges = CreateRanges(itemCount, threadCount, itemByteLength, alignmentByteLength);
            if (ranges.Length == 0)
            {
                return;
            }

            if (ranges.Length == 1)
            {
                rangeAction(ranges[0].StartIndex, ranges[0].EndIndex);
                return;
            }

            Parallel.ForEach(
                ranges,
                new ParallelOptions { MaxDegreeOfParallelism = ranges.Length },
                range => rangeAction(range.StartIndex, range.EndIndex));
        }

        internal static void ForEachRange<T>(ReadOnlySpan<T> items, Action<int, int> rangeAction, int threadCount = 0, int? alignmentByteLength = null)
            where T : unmanaged
        {

            ForEachRange(items.Length, rangeAction, threadCount, Unsafe.SizeOf<T>(), alignmentByteLength);
        }

        internal static void ForEachRange<T>(Span<T> items, Action<int, int> rangeAction, int threadCount = 0, int? alignmentByteLength = null)
            where T : unmanaged
        {
            ForEachRange((ReadOnlySpan<T>)items, rangeAction, threadCount, alignmentByteLength);
        }

        internal static WorkRange[] CreateRanges(int itemCount, int threadCount = 0, int itemByteLength = 1, int? alignmentByteLength = null)
        {
            if (itemCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(itemCount));
            }

            if (threadCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            }

            if (itemByteLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(itemByteLength));
            }

            if (alignmentByteLength is <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(alignmentByteLength));
            }

            if (itemCount == 0)
            {
                return [];
            }

            int groupCount = Math.Min(GetRequestedGroupCount(threadCount), itemCount);
            int alignmentItemCount = GetAlignmentItemCount(itemByteLength, alignmentByteLength);
            WorkRange[] ranges = new WorkRange[groupCount];
            int startIndex = 0;
            for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
            {
                int remainingItems = itemCount - startIndex;
                int remainingGroups = groupCount - groupIndex;
                int rangeLength = (remainingItems + remainingGroups - 1) / remainingGroups;
                if (alignmentItemCount > 1 && remainingGroups > 1)
                {
                    rangeLength = GetAlignedRangeLength(startIndex, remainingItems, remainingGroups, rangeLength, alignmentItemCount);
                }

                int endIndex = startIndex + rangeLength;
                ranges[groupIndex] = new WorkRange(startIndex, endIndex);
                startIndex = endIndex;
            }

            return ranges;
        }

        internal static WorkRange[] CreateRanges<T>(ReadOnlySpan<T> items, int threadCount = 0, int? alignmentByteLength = null)
            where T : unmanaged
        {
            return CreateRanges(items.Length, threadCount, Marshal.SizeOf<T>(), alignmentByteLength);
        }

        internal static WorkRange[] CreateRanges<T>(Span<T> items, int threadCount = 0, int? alignmentByteLength = null)
            where T : unmanaged
        {
            return CreateRanges((ReadOnlySpan<T>)items, threadCount, alignmentByteLength);
        }

        private static int GetRequestedGroupCount(int threadCount)
        {
            return threadCount == 0 ? DefaultGroupCount : threadCount;
        }

        private static int GetAlignedRangeLength(int startIndex, int remainingItems, int remainingGroups, int requestedLength, int alignmentItemCount)
        {
            int minLength = 1;
            int maxLength = remainingItems - (remainingGroups - 1);
            int alignedEndIndex = AlignUp(startIndex + requestedLength, alignmentItemCount);
            int alignedLength = alignedEndIndex - startIndex;
            if (alignedLength >= minLength && alignedLength <= maxLength)
            {
                return alignedLength;
            }

            int fallbackEndIndex = AlignDown(startIndex + maxLength, alignmentItemCount);
            int fallbackLength = fallbackEndIndex - startIndex;
            if (fallbackLength >= minLength)
            {
                return fallbackLength;
            }

            return maxLength;
        }

        private static int GetAlignmentItemCount(int itemByteLength, int? alignmentByteLength)
        {
            if (!alignmentByteLength.HasValue)
            {
                return 1;
            }

            int greatestCommonDivisor = GetGreatestCommonDivisor(itemByteLength, alignmentByteLength.Value);
            return alignmentByteLength.Value / greatestCommonDivisor;
        }

        private static int GetGreatestCommonDivisor(int left, int right)
        {
            while (right != 0)
            {
                int remainder = left % right;
                left = right;
                right = remainder;
            }

            return Math.Abs(left);
        }

        private static int AlignUp(int value, int alignment)
        {
            int remainder = value % alignment;
            return remainder == 0 ? value : value + (alignment - remainder);
        }

        private static int AlignDown(int value, int alignment)
        {
            return value - (value % alignment);
        }

        internal readonly record struct WorkRange(int StartIndex, int EndIndex);
    }
}
