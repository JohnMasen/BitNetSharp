using BitNetSharp.Models;
using System.Globalization;
using System.Text;

namespace BitNetSharp.Console
{
    internal static class ConsoleMemoryReportHelper
    {
        public static long GetActualKvCacheBytes(BitNetRuntime runtime)
        {
            if (!HasActiveSession(runtime))
            {
                return 0;
            }

            BitNetSession session = runtime.Session;
            BitNetModelConfig? config = session.Model.Config;
            if (config is null)
            {
                return 0;
            }

            return (long)session.CacheLength
                * (long)config.KeyValueProjectionSize
                * session.Model.Layers.Count
                * 2L
                * sizeof(float);
        }

        public static long GetAllocatedKvCacheBytes(BitNetRuntime runtime)
        {
            if (!HasActiveSession(runtime))
            {
                return 0;
            }

            BitNetSession session = runtime.Session;
            BitNetModelConfig? config = session.Model.Config;
            if (config is null)
            {
                return 0;
            }

            return (long)config.ContextLength
                * (long)config.KeyValueProjectionSize
                * session.Model.Layers.Count
                * 2L
                * sizeof(float);
        }

        public static string BuildMemoryCsv(BitNetMemoryStatistics statistics, long actualKvCacheBytes, long allocatedKvCacheBytes)
        {
            StringBuilder builder = new();
            builder.AppendLine("Category,SessionId,Key,ElementType,RequestedLength,EstimatedBytes,ActualKvCacheBytes,AllocatedKvCacheBytes");
            builder.AppendLine($"Summary,,,,,{statistics.EstimatedTotalBytes.ToString(CultureInfo.InvariantCulture)},{actualKvCacheBytes.ToString(CultureInfo.InvariantCulture)},{allocatedKvCacheBytes.ToString(CultureInfo.InvariantCulture)}");
            foreach (BitNetMemoryAllocationSnapshot allocation in statistics.Allocations.OrderBy(static allocation => allocation.SessionId).ThenBy(static allocation => allocation.Key, StringComparer.Ordinal))
            {
                builder.Append("Allocation,");
                builder.Append(allocation.SessionId.ToString());
                builder.Append(',');
                builder.Append(EscapeCsv(allocation.Key));
                builder.Append(',');
                builder.Append(EscapeCsv(allocation.ElementType.Name));
                builder.Append(',');
                builder.Append(allocation.RequestedLength.ToString(CultureInfo.InvariantCulture));
                builder.Append(',');
                builder.Append(allocation.EstimatedBytes.ToString(CultureInfo.InvariantCulture));
                builder.AppendLine(",,");
            }

            return builder.ToString();
        }

        public static string FormatBytes(long bytes)
        {
            const double KiloByte = 1024d;
            const double MegaByte = KiloByte * 1024d;

            if (bytes >= MegaByte)
            {
                return $"{bytes / MegaByte:F2} MB";
            }

            if (bytes >= KiloByte)
            {
                return $"{bytes / KiloByte:F2} KB";
            }

            return $"{bytes} B";
        }

        private static bool HasActiveSession(BitNetRuntime runtime)
        {
            try
            {
                return !runtime.Session.Tokens.IsEmpty;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
        }

        private static string EscapeCsv(string value)
        {
            return value.Contains(',', StringComparison.Ordinal) || value.Contains('"', StringComparison.Ordinal) || value.Contains('\n', StringComparison.Ordinal)
                ? $"\"{value.Replace("\"", "\"\"", StringComparison.Ordinal)}\""
                : value;
        }
    }
}
