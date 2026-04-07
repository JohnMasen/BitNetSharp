# Copilot Instructions

## Project Guidelines
- The user wants scope changes to focus on adding new tokenizer unit tests rather than unrelated structural refactors when they explicitly ask for tests.
- For BitNet tokenizer alignment, only consider the authoritative non-C# source in the BitNet repo; do not use the deleted `D:\GithubRoot\BitNet\src\csharp` directory as a reference.
- The user plans to optimize the layer structure in the future using abstractions like `ICacheableLayer` and `LayerBase`, but does not want those structural changes made yet.
- Do not modify the user's self-written test files unless they explicitly ask for changes to those tests.

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
