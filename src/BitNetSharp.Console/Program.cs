using BitNetSharp;
using BitNetSharp.Core;
using BitNetSharp.Models;
using BitNetSharp.Nodes;
using System.CommandLine;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace BitNetSharp.Console
{
    internal static class Program
    {
        private const string ExitCommand = "/exit";
        private const string MaxNewTokensArgument = "--max-new-tokens";
        private const string TopKArgument = "--top-k";
        private const string ShowMemoryArgument = "--show-memory";
        private const string MemoryCsvArgument = "--memory-csv";
        private const string EnableSamplingArgument = "--enable-sampling";
        private const string SamplingSeedArgument = "--sampling-seed";
        private const string TemperatureArgument = "--temperature";
        private const string TopPArgument = "--top-p";
        private const string MinPArgument = "--min-p";
        private const string RepeatLastNArgument = "--repeat-last-n";
        private const string RepeatPenaltyArgument = "--repeat-penalty";
        private const string PromptArgument = "--prompt";
        private const string ShowTokenIdsArgument = "--show-token-ids";

        private static int Main(string[] args)
        {
            ConfigureConsoleEncoding();
            RootCommand rootCommand = CreateRootCommand();
            return rootCommand.Parse(args).Invoke();
        }

        private static RootCommand CreateRootCommand()
        {
            Argument<string> modelPathArgument = new("model-path")
            {
                Description = "Path to the GGUF model file."
            };
            modelPathArgument.Validators.Add(result =>
            {
                if (result.Tokens.Count > 0 && string.IsNullOrWhiteSpace(result.Tokens[0].Value))
                {
                    result.AddError("A GGUF model path is required.");
                }
            });

            Option<int> maxNewTokensOption = new(MaxNewTokensArgument)
            {
                Description = "Maximum number of assistant tokens to generate. Default: 128",
                DefaultValueFactory = _ => 128
            };
            maxNewTokensOption.Validators.Add(result =>
            {
                if (result.GetValue(maxNewTokensOption) <= 0)
                {
                    result.AddError("--max-new-tokens requires a positive integer value.");
                }
            });

            Option<int> topKOption = new(TopKArgument)
            {
                Description = "Top-K sampling candidate count. Default: 40",
                DefaultValueFactory = _ => 40
            };
            topKOption.Validators.Add(result =>
            {
                if (result.GetValue(topKOption) <= 0)
                {
                    result.AddError("--top-k requires a positive integer value.");
                }
            });

            Option<bool> enableSamplingOption = new(EnableSamplingArgument)
            {
                Description = "Enable Top-K probabilistic sampling instead of default greedy decoding."
            };
            Option<int?> samplingSeedOption = new(SamplingSeedArgument)
            {
                Description = "Use a fixed sampling seed for reproducible sampling output."
            };
            samplingSeedOption.Validators.Add(result =>
            {
                int? value = result.GetValue(samplingSeedOption);
                if (value <= 0)
                {
                    result.AddError("--sampling-seed requires a positive integer value.");
                }
            });

            Option<float> temperatureOption = new(TemperatureArgument)
            {
                Description = "Sampling temperature. Default: 0.8",
                DefaultValueFactory = _ => 0.80f
            };
            temperatureOption.Validators.Add(result =>
            {
                if (result.GetValue(temperatureOption) < 0f)
                {
                    result.AddError("--temperature requires a non-negative numeric value.");
                }
            });

            Option<float> topPOption = new(TopPArgument)
            {
                Description = "Nucleus sampling threshold. Default: 0.95",
                DefaultValueFactory = _ => 0.95f
            };
            topPOption.Validators.Add(result =>
            {
                float value = result.GetValue(topPOption);
                if (value <= 0f || value > 1f)
                {
                    result.AddError("--top-p requires a numeric value in the range (0, 1].");
                }
            });

            Option<float> minPOption = new(MinPArgument)
            {
                Description = "Minimum probability threshold. Default: 0.05",
                DefaultValueFactory = _ => 0.05f
            };
            minPOption.Validators.Add(result =>
            {
                float value = result.GetValue(minPOption);
                if (value < 0f || value > 1f)
                {
                    result.AddError("--min-p requires a numeric value in the range [0, 1].");
                }
            });

            Option<int> repeatLastNOption = new(RepeatLastNArgument)
            {
                Description = "Number of recent tokens considered for repeat penalty. Default: 64",
                DefaultValueFactory = _ => 64
            };
            repeatLastNOption.Validators.Add(result =>
            {
                if (result.GetValue(repeatLastNOption) < 0)
                {
                    result.AddError("--repeat-last-n requires a non-negative integer value.");
                }
            });

            Option<float> repeatPenaltyOption = new(RepeatPenaltyArgument)
            {
                Description = "Repeat penalty multiplier. Default: 1.0",
                DefaultValueFactory = _ => 1.00f
            };
            repeatPenaltyOption.Validators.Add(result =>
            {
                if (result.GetValue(repeatPenaltyOption) <= 0f)
                {
                    result.AddError("--repeat-penalty requires a positive numeric value.");
                }
            });

            Option<string?> promptOption = new(PromptArgument)
            {
                Description = "Run a single prompt, print the assistant reply, then exit."
            };
            promptOption.Validators.Add(result =>
            {
                if (result.Tokens.Count > 0 && string.IsNullOrWhiteSpace(result.GetValue(promptOption)))
                {
                    result.AddError("--prompt requires a text value.");
                }
            });

            Option<bool> showTokenIdsOption = new(ShowTokenIdsArgument)
            {
                Description = "Show token ids alongside assistant output."
            };
            Option<bool> showMemoryOption = new(ShowMemoryArgument)
            {
                Description = "Show MemoryManager allocation summary and KV cache usage."
            };
            Option<string?> memoryCsvOption = new(MemoryCsvArgument)
            {
                Description = "Export memory allocation details to CSV."
            };
            memoryCsvOption.Validators.Add(result =>
            {
                if (result.Tokens.Count > 0 && string.IsNullOrWhiteSpace(result.GetValue(memoryCsvOption)))
                {
                    result.AddError("--memory-csv requires a file path value.");
                }
            });

            RootCommand rootCommand = new("Run BitNetSharp GGUF models from the console.")
            {
                modelPathArgument,
                maxNewTokensOption,
                topKOption,
                enableSamplingOption,
                samplingSeedOption,
                temperatureOption,
                topPOption,
                minPOption,
                repeatLastNOption,
                repeatPenaltyOption,
                promptOption,
                showTokenIdsOption,
                showMemoryOption,
                memoryCsvOption
            };
            rootCommand.SetAction(parseResult => Run(new ConsoleOptions(
                parseResult.GetValue(modelPathArgument),
                parseResult.GetValue(maxNewTokensOption),
                parseResult.GetValue(topKOption),
                IsSamplingEnabled(parseResult, enableSamplingOption, samplingSeedOption, temperatureOption, topPOption, minPOption, repeatLastNOption, repeatPenaltyOption),
                parseResult.GetValue(samplingSeedOption),
                parseResult.GetValue(temperatureOption),
                parseResult.GetValue(topPOption),
                parseResult.GetValue(minPOption),
                parseResult.GetValue(repeatLastNOption),
                parseResult.GetValue(repeatPenaltyOption),
                parseResult.GetValue(promptOption),
                parseResult.GetValue(showTokenIdsOption),
                parseResult.GetValue(showMemoryOption) || parseResult.GetResult(memoryCsvOption) is not null,
                parseResult.GetValue(memoryCsvOption))));
            return rootCommand;
        }

        private static bool IsSamplingEnabled(
            ParseResult parseResult,
            Option<bool> enableSamplingOption,
            Option<int?> samplingSeedOption,
            Option<float> temperatureOption,
            Option<float> topPOption,
            Option<float> minPOption,
            Option<int> repeatLastNOption,
            Option<float> repeatPenaltyOption)
        {
            return parseResult.GetValue(enableSamplingOption)
                || parseResult.GetResult(samplingSeedOption) is not null
                || parseResult.GetResult(temperatureOption) is not null
                || parseResult.GetResult(topPOption) is not null
                || parseResult.GetResult(minPOption) is not null
                || parseResult.GetResult(repeatLastNOption) is not null
                || parseResult.GetResult(repeatPenaltyOption) is not null;
        }

        private static int Run(ConsoleOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);

            using var model = new BitNetModel();
            model.Load(options.ModelPath);
            using var memoryManager = new BitNetMemoryManager();
            using var runtime = new BitNetRuntime(model, memoryManager, new InferenceConfig(new CPUSimdOPProvider(0)), topK: options.TopK, enableSampling: options.EnableSampling, samplingSeed: options.SamplingSeed, temperature: options.Temperature, topP: options.TopP, minP: options.MinP, repeatLastN: options.RepeatLastN, repeatPenalty: options.RepeatPenalty);
            bool exitRequested = false;
            CancellationTokenSource? generationCts = null;
            ConsoleCancelEventHandler cancelHandler = (_, eventArgs) =>
            {
                eventArgs.Cancel = true;
                exitRequested = true;
                generationCts?.Cancel();
                System.Console.WriteLine();
                System.Console.WriteLine("Cancellation Requested. Exiting...");
            };
            System.Console.CancelKeyPress += cancelHandler;

            try
            {
                PrintModelInfo(model, runtime, options);
                WriteMemoryReport(memoryManager, runtime, options, reportLabel: "Startup");
                if (!string.IsNullOrWhiteSpace(options.Prompt))
                {
                    return RunSinglePrompt(model, runtime, memoryManager, options);
                }

                System.Console.WriteLine($"Type {ExitCommand}, press Ctrl+C, or submit an empty line to exit.");
                while (true)
                {
                    if (exitRequested)
                    {
                        return 0;
                    }

                    System.Console.Write("You: ");
                    string? userInput = System.Console.ReadLine();
                    if (exitRequested
                        || string.IsNullOrWhiteSpace(userInput)
                        || string.Equals(userInput.Trim(), ExitCommand, StringComparison.OrdinalIgnoreCase))
                    {
                        return 0;
                    }

                    if (runtimeHasActiveSession(runtime))
                    {
                        runtime.ContinueConversation(userInput);
                    }
                    else
                    {
                        runtime.StartConversation(userInput);
                    }

                    generationCts = new CancellationTokenSource();
                    bool wroteAssistantPrefix = false;
                    bool wroteReplyToken = false;
                    List<TokenPerformanceSample> tokenPerformanceSamples = [];
                    Stopwatch totalStopwatch = Stopwatch.StartNew();
                    long firstTokenElapsedMilliseconds = -1;
                    try
                    {
                        foreach ((int tokenId, string tokenText) in runtime.StreamAssistantReplyWithTokenIds(options.MaxNewTokens, generationCts.Token))
                        {
                            if (!wroteAssistantPrefix)
                            {
                                System.Console.Write("Assistant: ");
                                wroteAssistantPrefix = true;
                            }

                            WriteAssistantToken(tokenId, tokenText, options.ShowTokenIds);
                            tokenPerformanceSamples.Add(new TokenPerformanceSample(tokenId, tokenText, runtime.Session.LastInferenceElapsedMilliseconds, runtime.Session.LastSamplingElapsedMilliseconds));
                            wroteReplyToken = true;
                            if (firstTokenElapsedMilliseconds < 0)
                            {
                                firstTokenElapsedMilliseconds = totalStopwatch.ElapsedMilliseconds;
                            }
                        }
                    }
                    catch (OperationCanceledException) when (exitRequested)
                    {
                        return 0;
                    }
                    finally
                    {
                        generationCts.Dispose();
                        generationCts = null;
                    }

                    if (!wroteAssistantPrefix)
                    {
                        System.Console.Write("Assistant: ");
                    }

                    if (!wroteReplyToken)
                    {
                        System.Console.Write("[empty reply before EOS]");
                    }

                    WriteStopTokenDebugOutput(runtime, model, options, isSinglePrompt: false);

                    System.Console.WriteLine();
                    PrintTiming(runtime, firstTokenElapsedMilliseconds, totalStopwatch.ElapsedMilliseconds);
                    WritePerformanceTable(tokenPerformanceSamples, runtime, firstTokenElapsedMilliseconds, totalStopwatch.ElapsedMilliseconds);
                    PrintSessionState(runtime);
                    WriteMemoryReport(memoryManager, runtime, options, reportLabel: "Per Turn");
                }
            }
            finally
            {
                generationCts?.Dispose();
                System.Console.CancelKeyPress -= cancelHandler;
            }
        }

        private static void PrintModelInfo(BitNetModel model, BitNetRuntime runtime, ConsoleOptions options)
        {
            System.Console.WriteLine($"Loaded Model: {model.Config?.ModelName ?? "unknown"}");
            System.Console.WriteLine($"Architecture: {model.Config?.ArchitectureName ?? "unknown"}");
            System.Console.WriteLine($"Context Length: {model.Config?.ContextLength.ToString() ?? "unknown"}");
            System.Console.WriteLine($"Tokenizer: {model.TokenizerConfig?.TokenizerModelType ?? "unknown"}");
            System.Console.WriteLine($"Max New Tokens: {options.MaxNewTokens}");
            System.Console.WriteLine($"Top K: {runtime.TopK}");
            System.Console.WriteLine($"Enable Sampling: {runtime.EnableSampling}");
            System.Console.WriteLine($"Temperature: {runtime.Temperature.ToString(CultureInfo.InvariantCulture)}");
            System.Console.WriteLine($"Top P: {runtime.TopP.ToString(CultureInfo.InvariantCulture)}");
            System.Console.WriteLine($"Min P: {runtime.MinP.ToString(CultureInfo.InvariantCulture)}");
            System.Console.WriteLine($"Repeat Last N: {runtime.RepeatLastN}");
            System.Console.WriteLine($"Repeat Penalty: {runtime.RepeatPenalty.ToString(CultureInfo.InvariantCulture)}");
            System.Console.WriteLine();
        }

        private static void PrintSessionState(BitNetRuntime runtime)
        {
            if (!runtimeHasActiveSession(runtime))
            {
                return;
            }

            System.Console.WriteLine($"[Session Tokens={runtime.Session.Tokens.Length}, Cache Length={runtime.Session.CacheLength}]");
        }

        private static void PrintTiming(BitNetRuntime runtime, long firstTokenElapsedMilliseconds, long totalElapsedMilliseconds)
        {
            string firstTokenText = firstTokenElapsedMilliseconds >= 0
                ? $"{firstTokenElapsedMilliseconds} ms"
                : "N/A (EOS Before Visible Token)";

            int generatedTokenCount = runtimeHasActiveSession(runtime)
                ? runtime.Session.CurrentOutputTokens.Length
                : 0;
            int postFirstVisibleTokenCount = firstTokenElapsedMilliseconds >= 0
                ? Math.Max(0, generatedTokenCount - 1)
                : generatedTokenCount;
            long postFirstVisibleTokenElapsedMilliseconds = firstTokenElapsedMilliseconds >= 0
                ? Math.Max(0, totalElapsedMilliseconds - firstTokenElapsedMilliseconds)
                : totalElapsedMilliseconds;
            string tokensPerSecondText = postFirstVisibleTokenCount > 0 && postFirstVisibleTokenElapsedMilliseconds > 0
                ? (postFirstVisibleTokenCount / (postFirstVisibleTokenElapsedMilliseconds / 1000d)).ToString("F2", CultureInfo.InvariantCulture)
                : "N/A";

            System.Console.WriteLine($"[Timing First Token={firstTokenText}, Total={totalElapsedMilliseconds} ms, Tokens/s Excluding First={tokensPerSecondText}]");
        }

        private static void ConfigureConsoleEncoding()
        {
            Encoding utf8WithoutBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
            System.Console.InputEncoding = utf8WithoutBom;
            System.Console.OutputEncoding = utf8WithoutBom;
        }

        private static void WriteMemoryReport(BitNetMemoryManager memoryManager, BitNetRuntime runtime, ConsoleOptions options, string reportLabel)
        {
            if (!options.ShowMemory)
            {
                return;
            }

            BitNetMemoryStatistics statistics = memoryManager.GetStatistics();
            long actualKvCacheBytes = ConsoleMemoryReportHelper.GetActualKvCacheBytes(runtime);
            long allocatedKvCacheBytes = ConsoleMemoryReportHelper.GetAllocatedKvCacheBytes(runtime);
            System.Console.WriteLine($"[Memory {reportLabel} Allocation Count={statistics.AllocationCount}, Estimated Total={ConsoleMemoryReportHelper.FormatBytes(statistics.EstimatedTotalBytes)}, Actual KV Cache={ConsoleMemoryReportHelper.FormatBytes(actualKvCacheBytes)}, Allocated KV Cache={ConsoleMemoryReportHelper.FormatBytes(allocatedKvCacheBytes)}]");

            if (!string.IsNullOrWhiteSpace(options.MemoryCsvPath))
            {
                File.WriteAllText(options.MemoryCsvPath, ConsoleMemoryReportHelper.BuildMemoryCsv(statistics, actualKvCacheBytes, allocatedKvCacheBytes), Encoding.UTF8);
                System.Console.WriteLine($"[Memory Csv Path={options.MemoryCsvPath}]");
            }
        }

        private static bool runtimeHasActiveSession(BitNetRuntime runtime)
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

        private static int RunSinglePrompt(BitNetModel model, BitNetRuntime runtime, BitNetMemoryManager memoryManager, ConsoleOptions options)
        {
            runtime.StartConversation(options.Prompt!);

            Stopwatch totalStopwatch = Stopwatch.StartNew();
            long firstTokenElapsedMilliseconds = -1;
            bool wroteReplyToken = false;
            List<TokenPerformanceSample> tokenPerformanceSamples = [];
            System.Console.Write("Assistant: ");
            foreach ((int tokenId, string tokenText) in runtime.StreamAssistantReplyWithTokenIds(options.MaxNewTokens))
            {
                WriteAssistantToken(tokenId, tokenText, options.ShowTokenIds);
                tokenPerformanceSamples.Add(new TokenPerformanceSample(tokenId, tokenText, runtime.Session.LastInferenceElapsedMilliseconds, runtime.Session.LastSamplingElapsedMilliseconds));
                wroteReplyToken = true;
                if (firstTokenElapsedMilliseconds < 0)
                {
                    firstTokenElapsedMilliseconds = totalStopwatch.ElapsedMilliseconds;
                }
            }

            if (!wroteReplyToken)
            {
                System.Console.Write("[empty reply before EOS]");
            }

            WriteStopTokenDebugOutput(runtime, model, options, isSinglePrompt: true);

            System.Console.WriteLine();
            PrintTiming(runtime, firstTokenElapsedMilliseconds, totalStopwatch.ElapsedMilliseconds);
            WritePerformanceTable(tokenPerformanceSamples, runtime, firstTokenElapsedMilliseconds, totalStopwatch.ElapsedMilliseconds);
            PrintSessionState(runtime);
            WriteMemoryReport(memoryManager, runtime, options, reportLabel: "Single Prompt");
            return 0;
        }

        private static void WriteAssistantToken(int tokenId, string tokenText, bool showTokenIds)
        {
            if (!showTokenIds)
            {
                System.Console.Write(tokenText);
                return;
            }

            string visibleText = tokenText
                .Replace("\r", "\\r", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal)
                .Replace("\t", "\\t", StringComparison.Ordinal);
            System.Console.Write($"{visibleText}[{tokenId}]");
        }

        private static void WritePerformanceTable(IReadOnlyList<TokenPerformanceSample> tokenPerformanceSamples, BitNetRuntime runtime, long firstTokenElapsedMilliseconds, long totalElapsedMilliseconds)
        {
            if (tokenPerformanceSamples.Count == 0)
            {
                return;
            }

            System.Console.WriteLine();
            System.Console.WriteLine("Performance Table:");
            string[] indexValues = tokenPerformanceSamples.Select((_, index) => (index + 1).ToString(CultureInfo.InvariantCulture)).ToArray();
            string[] tokenIdValues = tokenPerformanceSamples.Select(sample => sample.TokenId.ToString(CultureInfo.InvariantCulture)).ToArray();
            string[] textValues = tokenPerformanceSamples.Select(sample => sample.TokenText
                .Replace("\r", "\\r", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal)
                .Replace("\t", "\\t", StringComparison.Ordinal)).ToArray();
            string[] inferValues = tokenPerformanceSamples.Select(sample => sample.InferenceElapsedMilliseconds.ToString(CultureInfo.InvariantCulture)).ToArray();
            string[] sampleValues = tokenPerformanceSamples.Select(sample => sample.SamplingElapsedMilliseconds.ToString(CultureInfo.InvariantCulture)).ToArray();

            int indexWidth = GetColumnWidth("#", indexValues);
            int tokenIdWidth = GetColumnWidth("TokenId", tokenIdValues);
            int textWidth = GetColumnWidth("Text", textValues);
            int inferWidth = GetColumnWidth("Infer (ms)", inferValues);
            int sampleWidth = GetColumnWidth("Sample (ms)", sampleValues);

            string header = string.Format(CultureInfo.InvariantCulture,
                "{0}  {1}  {2}  {3}  {4}",
                "#".PadLeft(indexWidth),
                "TokenId".PadLeft(tokenIdWidth),
                "Text".PadRight(textWidth),
                "Infer (ms)".PadLeft(inferWidth),
                "Sample (ms)".PadLeft(sampleWidth));
            string separator = string.Format(CultureInfo.InvariantCulture,
                "{0}  {1}  {2}  {3}  {4}",
                new string('-', indexWidth),
                new string('-', tokenIdWidth),
                new string('-', textWidth),
                new string('-', inferWidth),
                new string('-', sampleWidth));
            System.Console.WriteLine(header);
            System.Console.WriteLine(separator);
            for (int index = 0; index < tokenPerformanceSamples.Count; index++)
            {
                System.Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0}  {1}  {2}  {3}  {4}",
                    indexValues[index].PadLeft(indexWidth),
                    tokenIdValues[index].PadLeft(tokenIdWidth),
                    textValues[index].PadRight(textWidth),
                    inferValues[index].PadLeft(inferWidth),
                    sampleValues[index].PadLeft(sampleWidth)));
            }

            long totalInferenceMilliseconds = tokenPerformanceSamples.Sum(sample => sample.InferenceElapsedMilliseconds);
            long totalSamplingMilliseconds = tokenPerformanceSamples.Sum(sample => sample.SamplingElapsedMilliseconds);
            int generatedTokenCount = runtimeHasActiveSession(runtime)
                ? runtime.Session.CurrentOutputTokens.Length
                : tokenPerformanceSamples.Count;
            int postFirstVisibleTokenCount = firstTokenElapsedMilliseconds >= 0
                ? Math.Max(0, generatedTokenCount - 1)
                : generatedTokenCount;
            long postFirstVisibleTokenElapsedMilliseconds = firstTokenElapsedMilliseconds >= 0
                ? Math.Max(0, totalElapsedMilliseconds - firstTokenElapsedMilliseconds)
                : totalElapsedMilliseconds;
            string tokensPerSecondText = postFirstVisibleTokenCount > 0 && postFirstVisibleTokenElapsedMilliseconds > 0
                ? (postFirstVisibleTokenCount / (postFirstVisibleTokenElapsedMilliseconds / 1000d)).ToString("F2", CultureInfo.InvariantCulture)
                : "N/A";

            string generatedTokenCountText = generatedTokenCount.ToString(CultureInfo.InvariantCulture);
            string totalInferenceText = totalInferenceMilliseconds.ToString(CultureInfo.InvariantCulture);
            string totalSamplingText = totalSamplingMilliseconds.ToString(CultureInfo.InvariantCulture);
            int summaryLabelWidth = "Summary".Length;
            int summaryTokenWidth = GetColumnWidth("Tokens", [generatedTokenCountText]);
            int summaryInferWidth = GetColumnWidth("Infer Total (ms)", [totalInferenceText]);
            int summarySampleWidth = GetColumnWidth("Sample Total (ms)", [totalSamplingText]);
            int summaryTpsWidth = GetColumnWidth("Tokens/s Excluding First", [tokensPerSecondText]);

            System.Console.WriteLine();
            System.Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0}  {1}  {2}  {3}  {4}",
                "Summary".PadRight(summaryLabelWidth),
                "Tokens".PadLeft(summaryTokenWidth),
                "Infer Total (ms)".PadLeft(summaryInferWidth),
                "Sample Total (ms)".PadLeft(summarySampleWidth),
                "Tokens/s Excluding First".PadLeft(summaryTpsWidth)));
            System.Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0}  {1}  {2}  {3}  {4}",
                new string('-', summaryLabelWidth),
                new string('-', summaryTokenWidth),
                new string('-', summaryInferWidth),
                new string('-', summarySampleWidth),
                new string('-', summaryTpsWidth)));
            System.Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0}  {1}  {2}  {3}  {4}",
                "Total".PadRight(summaryLabelWidth),
                generatedTokenCountText.PadLeft(summaryTokenWidth),
                totalInferenceText.PadLeft(summaryInferWidth),
                totalSamplingText.PadLeft(summarySampleWidth),
                tokensPerSecondText.PadLeft(summaryTpsWidth)));
        }

        private static int GetColumnWidth(string header, IReadOnlyList<string> values)
        {
            int width = header.Length;
            for (int index = 0; index < values.Count; index++)
            {
                width = Math.Max(width, values[index].Length);
            }

            return width;
        }

        private static void WriteStopTokenDebugOutput(BitNetRuntime runtime, BitNetModel model, ConsoleOptions options, bool isSinglePrompt)
        {
            if (!runtimeHasActiveSession(runtime))
            {
                return;
            }

            ReadOnlyMemory<int> outputTokens = runtime.Session.CurrentOutputTokens;
            if (outputTokens.IsEmpty)
            {
                return;
            }

            int stopTokenId = outputTokens.Span[^1];
            if (!TryGetEndOfGenerationTokenText(model, stopTokenId, out string? stopTokenText))
            {
                return;
            }

            if (options.ShowTokenIds)
            {
                WriteAssistantToken(stopTokenId, stopTokenText, showTokenIds: true);
            }

            if (isSinglePrompt)
            {
                System.Console.Write("[End Of Text]");
            }
        }

        private static bool TryGetEndOfGenerationTokenText(BitNetModel model, int tokenId, out string? tokenText)
        {
            tokenText = null;
            IReadOnlyList<string>? tokens = model.TokenizerConfig?.Tokens;
            if (tokens is null || (uint)tokenId >= (uint)tokens.Count)
            {
                return false;
            }

            string candidate = tokens[tokenId];
            int eosTokenId = checked((int)(model.TokenizerConfig?.EosTokenId ?? uint.MaxValue));
            if (tokenId != eosTokenId
                && !string.Equals(candidate, "<|eot_id|>", StringComparison.Ordinal)
                && !string.Equals(candidate, "<|end_of_text|>", StringComparison.Ordinal))
            {
                return false;
            }

            tokenText = candidate
                .Replace("\r", "\\r", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal)
                .Replace("\t", "\\t", StringComparison.Ordinal);
            return true;
        }

        private sealed record ConsoleOptions(string ModelPath, int MaxNewTokens, int TopK, bool EnableSampling, int? SamplingSeed, float Temperature, float TopP, float MinP, int RepeatLastN, float RepeatPenalty, string? Prompt, bool ShowTokenIds, bool ShowMemory, string? MemoryCsvPath);

        private sealed record TokenPerformanceSample(int TokenId, string TokenText, long InferenceElapsedMilliseconds, long SamplingElapsedMilliseconds);
    }
}
