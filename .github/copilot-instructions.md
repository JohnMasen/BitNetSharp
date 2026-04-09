# Copilot Instructions

## Project Guidelines
- The user wants scope changes to focus on adding new tokenizer unit tests rather than unrelated structural refactors when they explicitly ask for tests.
- For BitNet tokenizer alignment, only consider the authoritative non-C# source in the BitNet repo; do not use the deleted `D:\GithubRoot\BitNet\src\csharp` directory as a reference.
- For BitNet QKV work, use `MathHelper.VectorProcessOne()` only as a reference example and validate the algorithm primarily against the authoritative BitNet source to avoid carrying over mistakes from the example implementation. Use `QKV` with all three letters capitalized in naming, including in file names across the whole solution, such as `QKVLayerTests.cs`.
- When investigating algorithm issues in this repo, avoid relying on large test output; prefer targeted test runs and concise summaries over large raw logs.
- The user plans to optimize the layer structure in the future using abstractions like `ICacheableLayer` and `LayerBase`, but does not want those structural changes made yet.
- Do not modify the user's self-written test files unless they explicitly ask for changes to those tests.
- Continue with straightforward repo changes without waiting for extra confirmation.
- Do not proactively fix warnings in this repo. Only address warnings when the user explicitly asks. The user accepts that some warnings may be unavoidable.
- Eliminate `InferenceContext` and migrate its contents into `BitNetSession`, avoiding separate runtime-context abstractions for this area of the repo.
- For layers in this repo, `Forward` must not be allowed before an explicit `Init` call. `Init` performs internal initialization, currently only eager cache loading.
- For this repo's transformer pipeline, the attention layer input should be the QKV projection output, not the RMSNorm output directly.
- For session state in this repo, do not construct `BitNetMemoryManager` inside `BitNetSession`; pass it in during construction so tests can instantiate a shared manager for current development and future code can use unified memory management.
- For BitNetSession output state in this repo, expose lazy-initialized get-only buffer properties and write into them in place instead of using session buffer request APIs or setter-based copies.
- Simple scalar values such as `CurrentToken` should be ordinary properties and not be managed by `BitNetMemoryManager`, which should be reserved for large memory blocks.
- Defer implementing true multi-token KV-cache/runtime work in this repo until after the single-token inference pipeline is fully working end-to-end.
- Keep `doc/ImplementProgress.md` updated whenever layers or their test files are added or changed.
- For this repo's runtime design, do not introduce `IRuntimeNode`. Keep `RuntimeGraph` layer-only, and handle any non-layer operations by intercepting calls inside `Runtime` during execution instead of modeling them as graph nodes.
- For this repo's runtime graph design, do not use DI to instantiate layers from graph deserialization. DI is not intended as the graph-deserialization mechanism here.
- For this repo's runtime architecture, prefer `Layer` types to have parameterless constructors so they can be safely dynamically instantiated and deserialized from graph definitions.
- Prefer `BitNetRuntime` itself to act as the layer-wrapping factory for logging and performance-monitoring wrappers, keeping wrappers scoped to runtime-only observability rather than general runtime capabilities.
- For this repo's runtime roadmap, do not implement aggregated/fused layers yet. Treat layer aggregation as a future optimization path, especially for SIMD/GPU backends, and prefer a future `GraphOptimizer` to analyze and rewrite the layer-only graph for performance.
- Treat `SamplingStep` as a `Layer` so `BitNetRuntime` stays model-agnostic; the graph should ultimately output a string. Prefer exposing configurable layer properties through layer-level attributes rather than a runtime-wide property bag unless a later cross-layer need proves necessary.
- For this repo's memory tracking design, do not implement versioning yet. Prefer a non-generic `MemoryOwner` base abstraction that exposes `BeginTrace` and `EndTrace` to record read/write requests during a traced interval, leaving future extension points for version management and other metadata.
- For this repo's memory abstraction, treat `IMemoryOwner` as a minimal disposable ownership object and place the primary design emphasis on `IMemoryView<T>` as the key typed access abstraction.
- Simplify `MemoryManager` around two core purposes: shared memory management for multiple inference sessions, and serving as the destination for model weight loading. Keep `MemoryManager` as the base abstraction with a concrete built-in host-memory implementation, avoiding naming that concrete type as `DefaultMemoryManager`; prefer a name that describes its semantics rather than 'defaultness'.
- Remove `CPUBaseOPProvider`; nodes should instantiate the concrete operation provider directly from configuration instead of using a shared base provider abstraction.
- Move pure orchestration logic into `IOPProvider2` default interface implementations first, then consider further architecture simplification afterward.
- For this repo, when implementing OP operators, multithreading support is required rather than optional for large-buffer processing paths.

## QKV Parallel Work Instructions
- For QKV parallel work, `ThreadHelper` should support optional block-aligned splitting. Default splitting should not enforce alignment; only SIMD callers should pass an alignment parameter based on the required data byte length.

## Code Style
- Prefer `for` loops to initialize the index variable inside the `for` statement rather than using a prior assignment like `int index = 0; for (; index < ...; ... )`.

## Tokenizer Testing Instructions
- Separate tokenizer tests into `TokenizerTests.cs`.
- Use a test-project-level GGUF path configuration for tokenizer tests.
- Use class-level test initialization to load the GGUF model once, reuse it across tests in the class, and release it in class cleanup.
- Load the GGUF model in shared test initialization and ensure proper release after the test scope ends.
- Use shorter, clearer test names that are easy to scan at a glance.

## Layer Testing Instructions
- Keep `EmbeddingLayer` tests in a separate `LayerTests.cs` file.
- For QKV tests, prefer data-driven tests that pass only a `CaseId` or sequence number, and load the full test data inside the test method to avoid oversized test output.
- For debugging large data-driven tests in this repo, prefer running a single representative case or adding a dedicated single-case test instead of executing many similar cases.

## GGUF Metadata Mapping Instructions
- For GGUF metadata mapping design, keep `LoadOptions` minimal: retain only a metadata parser callback and pass GGUF meta information to a caller-provided parser, allowing that parser to return a structured result instead of using complex per-field schema/default mapping inside `SchemaMapping`.
- Do not use a factory pattern; put a `SchemaMapping` struct directly on load options.
- Use the built-in default mapping when `SchemaMapping` is null, and always parse through `SchemaMapping`.
- Parser implementations must not retain references to `GGUFFile` internally to avoid memory leaks.
- For `EmbeddingLayer`, make embedding reads optionally cached via a configurable `EnableCache` setting instead of re-reading tensor data on every `Forward` call.

## RMSNorm Implementation Instructions
- Prefer a correct pure CPU implementation of RMSNorm first, with a constructor selector defaulting to `CPUStandard` and future Tensor/SIMD backends left as placeholders.
- For SIMD implementations, reference `MathHelper.sum4` for fast summation patterns.
- When refactoring performance-sensitive helpers such as RMSNorm internals, apply `MethodImplOptions.AggressiveInlining` where appropriate.

## CSV Output Instructions
- When the user asks for CSV output, preserve actual line breaks clearly, preferably in a fenced CSV block to avoid collapsing into one line.
