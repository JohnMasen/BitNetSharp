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
- For session state in this repo, do not construct `BitNetMemoryManager` inside `BitNetSession`; pass it in during construction so tests can instantiate a shared manager for current development and future code can use unified memory management. **Benchmark code should reuse `BitNetMemoryManager` instead of creating many managers, because the manager is designed for reuse and excess instances add unnecessary memory pressure.** The user prefers to first enable index-based access in `BitNetMemoryManager` (id, key, [index], default index=0) to better support KV cache layered/segmented writes.
- For BitNetSession output state in this repo, expose lazy-initialized get-only buffer properties and write into them in place instead of using session buffer request APIs or setter-based copies.
- Simple scalar values such as `CurrentToken` should be ordinary properties and not be managed by `BitNetMemoryManager`, which should be reserved for large memory blocks.
- Defer implementing true multi-token KV-cache/runtime work in this repo until after the single-token inference pipeline is fully working end-to-end.
- Keep `doc/ImplementProgress.md` updated whenever layers or their test files are added or changed.
- For this repo's runtime design, do not introduce `IRuntimeNode`. Keep `RuntimeGraph` layer-only, and handle any non-layer operations by intercepting calls inside `Runtime` during execution instead of modeling them as graph nodes.
- For this repo's runtime graph design, do not use DI to instantiate layers from graph deserialization. DI is not intended as the graph-deserialization mechanism here.
- For this repo's runtime architecture, prefer `Layer` types to have parameterless constructors so they can be safely dynamically instantiated and deserialized from graph definitions. The user is considering keeping the graph as a pure computation graph, exposing semantics and execution views through `GraphView`, and passing `RuntimeTensor` objects to operators as the semantic tensor access abstraction over `MemoryManager`, including shape metadata.
- Prefer `BitNetRuntime` itself to act as the layer-wrapping factory for logging and performance-monitoring wrappers, keeping wrappers scoped to runtime-only observability rather than general runtime capabilities.
- For this repo's runtime roadmap, do not implement aggregated/fused layers yet. Treat layer aggregation as a future optimization path, especially for SIMD/GPU backends, and prefer a future `GraphOptimizer` to analyze and rewrite the layer-only graph for performance.
- Treat `SamplingStep` as a `Layer` so `BitNetRuntime` stays model-agnostic; the graph should ultimately output a string. Prefer exposing configurable layer properties through layer-level attributes rather than a runtime-wide property bag unless a later cross-layer need proves necessary.
- For this repo's memory tracking design, do not implement versioning yet. Prefer a non-generic `MemoryOwner` base abstraction that exposes `BeginTrace` and `EndTrace` to record read/write requests during a traced interval, leaving future extension points for version management and other metadata.
- For this repo's memory abstraction, treat `IMemoryOwner` as a minimal disposable ownership object and place the primary design emphasis on `IMemoryView<T>` as the key typed access abstraction.
- Simplify `MemoryManager` around two core purposes: shared memory management for multiple inference sessions, and serving as the destination for model weight loading. Keep `MemoryManager` as the base abstraction with a concrete built-in host-memory implementation, avoiding naming that concrete type as `DefaultMemoryManager`; prefer a name that describes its semantics rather than 'defaultness'.
- Remove `CPUBaseOPProvider`; nodes should instantiate the concrete operation provider directly from configuration instead of using a shared base provider abstraction.
- Move pure orchestration logic into `IOPProvider2` default interface implementations first, then consider further architecture simplification afterward.
- For this repo, when implementing OP operators, multithreading support is required for CPU backends only; GPU/NPU-style device backends do not need CPU-side multithreaded paths.
- Remove no-op `operationName` parameters from OP APIs when they are not used anywhere and do not carry real semantics.
- Avoid writing pure wrapper methods with no logic or semantics; inline them unless they add real validation, branching, adaptation, or abstraction value.
- For this repo, any temporary `BitNetRuntime.Inference` method added only to validate the single-token runtime chain, along with its tests, is expected to be deleted later after the runtime architecture evolves.
- Do not save "currently only running unit tests, not performance tests" as a long-term preference; this is a temporary situation while the user's machine is busy.
- For `IOPProvider`, use `string` type for `Backend` instead of `char` or enum to allow for future expansion.
- For this repo's OP provider design, prefer `CPUSimdOPProvider` to override `CreateRanges` for alignment-aware partitioning rather than pushing alignment choice into per-call parameters, to keep call sites simpler and SIMD-specific policy localized.
- For this repo's OP provider design, avoid adding `IOPProvider2.CreateRanges` solely for SIMD-specific needs because that couples the interface to one implementation detail; prefer extension points that CPU/Tensor also meaningfully reuse.
- For this repo's OP provider refactor, `IOPProvider2` still needs changes, but implementation work should follow an explicit modification strategy and future architecture plan rather than ad-hoc edits.
- **Plan to move multithreading implementation back from `IOPProvider2` into `IOPProvider`, prepare to deprecate `IOPProvider2`, design `Graph` next, and update current inference tests to call `IOPProvider` only.**
- When migrating inference-related implementations, include sufficient comments to facilitate future design reference for the Graph.
- In this repo's runtime/session design, consider removing semantic session buffer properties like `FeedForwardInput` and `FeedForwardNorm` and exposing only `MemoryManager` for operators to access memory blocks directly. **Weight tensors do not need Memory<T>-based copy-in because weights are loaded during model read. Session should expose `GetWeightTensor(string name)` for shared model weights and `GetRuntimeTensor(string name)` for runtime tensors, while external session properties may cache these tensors and assembly should pass those cached tensor properties directly into operators.**
- For this repo's runtime tensor design, prefer a single non-generic `RuntimeTensor` with a generic `TryGet<T>` access API, with access implemented through a bound delegate rather than passing `MemoryManager` directly into operators. Operators should compute against `RuntimeTensor` without binding to a specific memory type, to support future GPU/NPU backends. **Additionally, CPU-bound tensors can implement host-side typed access interfaces like `IRuntimeTensorHost<float>` and `IRuntimeTensorHost<byte>` instead of introducing more complex backend abstractions.** Because tensors and operators may be dynamically assembled in the future, avoid generic-based core runtime abstractions where possible.
- For this repo's runtime tensor design, provide a generic TryGetHostBuffer-style API so operators can access the actual underlying memory object, including future GPU memory, because operators ultimately need to manipulate concrete memory objects internally.
- For this repo's runtime tensor discussion, consider a simpler model where a tensor acts as data description plus a MemoryManager identifier/key, while operators access underlying memory through that indirection.
- For this repo's managed runtime design, avoid object-centric core tensor storage to reduce casting boilerplate associated with passing raw memory block objects or GPU memory objects in C#.
- For this repo's runtime tensor design, the user is considering a two-level model with non-generic `RuntimeTensor` plus derived `RuntimeTensor<T>`, where the generic layer may provide implicit conversions to Memory-like types, and the previous `StorageHandle` indirection is replaced by a bound delegate-based access path.
- For this repo's runtime tensor design, do not pass `MemoryManager` into operators and avoid adding another `StorageHandle` layer, while still providing a generic Get-style API that allows operators to access concrete memory elegantly.
- For this repo's runtime design, keep only non-generic `RuntimeTensor` inside graph/runtime because dynamic assembly is required; avoid relying on `RuntimeTensor<Memory<float>>` in operator signatures since that blocks dynamic assembly.
- **Consider whether `RuntimeTensor` could hold the concrete data object directly to avoid `MemoryManager` dictionary lookups, while recognizing lifecycle management as the key risk.**
- For this repo's architecture discussions, prefer the latest consolidated design summary over intermediate discussion fragments; avoid relying on partial earlier memory snippets when reasoning about current runtime design.
- For architecture discussion summaries, write them under the `doc/archdesign` path as markdown documents.
- For this repo's runtime design, `RuntimeTensor` should support copying data in from `Memory<T>` for CPU-origin initialization. The user prefers `Session` to create tensors via its internal `MemoryManager`, while multiple sessions should share a single immutable weight copy instead of duplicating model weights per session.
- Defer optimization work for the current RuntimeTensor/IOPProvider performance regression for now; revisit later with architecture simplification aimed at reducing lookup overhead such as extra list/dictionary access.
- **In this repo, retain the current static allocation strategy for token/KV cache; do not alter the overall architecture for dynamic expansion on CPU, as future GPU/NPU devices may not support dynamic memory expansion.**

## QKV Parallel Work Instructions
- For QKV parallel work, `ThreadHelper` should support optional block-aligned splitting. Default splitting should not enforce alignment; only SIMD callers should pass an alignment parameter based on the required data byte length.

## Code Style
- Prefer `for` loops to initialize the index variable inside the `for` statement rather than using a prior assignment like `int index = 0; for (; index < ...; ... )`.
- Use English for code comments/XML documentation when adding interface documentation in this repo.

## Tokenizer Testing Instructions
- Separate tokenizer tests into `TokenizerTests.cs`.
- Use a test-project-level GGUF path configuration for tokenizer tests.
- Use class-level test initialization to load the GGUF model once, reuse it across tests in the class, and release it in class cleanup.
- Load the GGUF model in shared test initialization and ensure proper release after the test scope ends.
- Use shorter, clearer test names that are easy to scan at a glance.
- In this repo, tokenizer-related test data should be based on the authoritative source results provided by the dump agent. **When requesting data from the dump agent, only list the required data without explaining the purpose or background.** 
- If test data already contains verifiable data, use it directly; only stop self-advancing and inform the user when data is missing, needing to ask the dump agent, encountering loops, resolving issues, or determining that issues cannot be resolved.
- For subsequent hi tests in this repo, set max tokens to 32 to save time.

## Layer Testing Instructions
- Keep `EmbeddingLayer` tests in a separate `LayerTests.cs` file.
- For QKV tests, prefer data-driven tests that pass only a `CaseId` or sequence number, and load the full test data inside the test method to avoid oversized test output. **To speed up development, all data-driven multi-case tests should default to only running the first case; expand coverage later by restoring multi-case enumeration.**
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

## Console Performance Output Instructions
- In this repo, Console performance statistics output should not use Markdown tables; use fixed-width space-aligned plain text tables instead.

## Communication Preferences
- Follow-up communication for this repo discussion should be conducted in Chinese.

## RuntimeTensor Access Instructions
- Prefer `RuntimeTensor` buffer access via extension methods like `GetMemory` and `GetReadOnlyMemory` on a static extension class to simplify call sites.

## Future Runtime Design Considerations
- Consider an assembly approach that converts graph node calls into an action list so actions can access concrete memory objects directly; this is only a recorded direction for later research, not current implementation work.

## CPUSimdOPProvider Instructions
- For `CPUSimdOPProvider`, capability checks like AVX2 support should be performed once in the constructor rather than per method, because SIMD is required for the provider to function correctly.

## Session Management Instructions
- In this repo, a new `BitNetSession` must be created for each session; a single session cannot be rolled back, reset, or modify history, and only forward state additions are allowed.
