# BitNetSharp

#### This project is still under active development, and this document may not reflect the latest changes.

`BitNetSharp` is a .NET project for loading and running BitNet GGUF models. The repository is no longer an early simplified Transformer demo. Its current implementation is centered around the real execution path for **GGUF models, tokenization, runtime inference, session state, KV cache, and console entry points**.

## Current Status

The codebase currently includes the following core capabilities:

- loading GGUF models, metadata, tensor indexes, and layer definitions
- parsing GGUF tokenizer data and running GPT-2 style BPE encode/decode
- incremental inference through `BitNetRuntime`
- append-only session state through `BitNetSession`
- reading and writing per-layer KV cache in the attention path
- console entry points for chat and one-shot prompt inference
- memory statistics and CSV export
- CPU, Tensor, and SIMD OP providers

## Target Frameworks

- `net9.0`
- `net10.0`

The main test project currently targets:

- `net10.0`

## Repository Layout

```text
BitNetSharp/
├─ src/
│  ├─ BitNetSharp/                 # Core library
│  ├─ BitNetSharp.Console/         # Console entry point
│  └─ tests/
│     └─ BitNetSharp.Tests/        # MSTest project
├─ BenchmarkSuite1/                # BenchmarkDotNet benchmarks
├─ doc/                            # Progress and architecture notes
├─ Models/                         # Local model files
└─ README.md
```

## Main Components

| Component | Purpose |
|---|---|
| `BitNetModel` | Loads GGUF metadata, tokenizer config, tensor indexes, and layer definitions |
| `BitNetTokenizer` | Encodes and decodes text and token ids, including chat-template-related helpers |
| `BitNetRuntime` | Hosts the current end-to-end runtime inference chain |
| `BitNetSession` | Stores token history, output rounds, runtime tensors, and KV cache tensors |
| `BitNetMemoryManager` | Manages pooled memory and provides allocation snapshots |
| `InferenceConfig` | Provides the active `IOPProvider` |
| `CPUDefaultOPProvider` | Standard CPU implementation |
| `CPUTensorOPProvider` | Tensor-style CPU implementation |
| `CPUSimdOPProvider` | SIMD-optimized CPU implementation |
| `BitNetSharp.Console` | Console chat and inference application |

## Current Runtime Flow

The current runtime is an incremental generation pipeline built around session state:

```text
Prompt Text
  -> Tokenizer
  -> Prefill / ContinuePrefill
  -> Embedding
  -> per-layer:
       RMSNorm
       QKV projection
       Attention (with KV cache read/write)
       Residual
       Feed-forward norm
       Feed-forward
       Feed-forward residual
  -> FinalNorm
  -> LM Head
  -> Sampling
  -> Decode
```

## Current Capabilities

### Model

`BitNetModel` currently handles:

- reading models from GGUF files
- parsing model config and tokenizer config
- building the tensor index
- exposing shared read-only weight tensors

### Tokenizer

`BitNetTokenizer` currently supports:

- encoding based on GGUF vocabulary and merges
- decoding token ids back to text
- special token recognition
- chat message encoding helpers

Current limitation:

- tokenizer model type is currently limited to `gpt2`

### Runtime / Session

`BitNetRuntime` and `BitNetSession` currently support:

- `Prefill`
- `ContinuePrefill`
- `GenerateTokenIds`
- `Generate`
- `StartConversation`
- `ContinueConversation`
- `GenerateAssistantReply`
- `StreamAssistantReply`
- `StreamAssistantReplyWithTokenIds`

`BitNetSession` follows an append-only design:

- token history is only appended, never rolled back
- a new conversation requires a new session
- each layer KV cache is managed through dedicated runtime tensors

## Quick Start

### Load a model and generate a reply

```csharp
using BitNetSharp;
using BitNetSharp.Core;
using BitNetSharp.Models;

using var model = new BitNetModel();
model.Load(@"D:\Models\ggml-model-i2_s.gguf");

using var memoryManager = new BitNetMemoryManager();
var inferenceConfig = new InferenceConfig(new CPUSimdOPProvider(0));

using var runtime = new BitNetRuntime(
    model,
    memoryManager,
    inferenceConfig,
    topK: 40,
    enableSampling: false);

runtime.StartConversation("Hello");
string reply = runtime.GenerateAssistantReply(maxNewTokens: 64);

Console.WriteLine(reply);
```

### Generate from prompt token ids

```csharp
using BitNetSharp;
using BitNetSharp.Core;
using BitNetSharp.Models;

using var model = new BitNetModel();
model.Load(@"D:\Models\ggml-model-i2_s.gguf");

using var memoryManager = new BitNetMemoryManager();
var inferenceConfig = new InferenceConfig(new CPUSimdOPProvider(0));

using var runtime = new BitNetRuntime(model, memoryManager, inferenceConfig);

runtime.Prefill(new[] { 1, 123, 456 });
int[] outputTokenIds = runtime.GenerateTokenIds(outputTokenCount: 16);

Console.WriteLine(string.Join(", ", outputTokenIds));
```

## Console Application

`BitNetSharp.Console` currently provides:

- interactive multi-turn chat
- one-shot prompt inference
- streaming output
- optional sampling
- token id display
- memory statistics output
- memory CSV export
- Ctrl+C generation cancellation

### Build the console project

```bash
dotnet build .\src\BitNetSharp.Console\BitNetSharp.Console.csproj -f net10.0
```

### Run a single prompt

```bash
dotnet run --project .\src\BitNetSharp.Console\BitNetSharp.Console.csproj -f net10.0 -- "D:\Models\ggml-model-i2_s.gguf" --prompt "Explain BitNet in one paragraph." --max-new-tokens 128
```

### Run interactive chat

```bash
dotnet run --project .\src\BitNetSharp.Console\BitNetSharp.Console.csproj -f net10.0 -- "D:\Models\ggml-model-i2_s.gguf"
```

### Common options

- `--max-new-tokens`
- `--top-k`
- `--enable-sampling`
- `--sampling-seed`
- `--temperature`
- `--top-p`
- `--min-p`
- `--repeat-last-n`
- `--repeat-penalty`
- `--prompt`
- `--show-token-ids`
- `--show-memory`
- `--memory-csv`

## Build

### Build the library

```bash
dotnet build .\src\BitNetSharp\BitNetSharp.csproj
```

### Build the console project

```bash
dotnet build .\src\BitNetSharp.Console\BitNetSharp.Console.csproj
```

### Build benchmarks

```bash
dotnet build .\BenchmarkSuite1\BenchmarkSuite1.csproj
```

## Test

```bash
dotnet test .\src\tests\BitNetSharp.Tests\BitNetSharp.Tests.csproj
```

## Benchmark

`BenchmarkSuite1` contains BenchmarkDotNet benchmarks.

Example:

```bash
dotnet run --project .\BenchmarkSuite1\BenchmarkSuite1.csproj -f net10.0 -c Release
```

## Design Notes

There are a few important points to keep in mind about the current codebase:

- the runtime is still a transitional implementation rather than the final graph runtime
- `BitNetSession` is append-only
- model weights are shared at the model level
- runtime tensors belong to session-level state
- KV cache currently uses a static allocation strategy
- memory statistics are available through `BitNetMemoryManager.GetStatistics()`

For implementation progress and design notes, see:

- `doc/ImplementProgress.md`
- `doc/archdesign/`

## License

See [MIT LICENSE](LICENSE).
