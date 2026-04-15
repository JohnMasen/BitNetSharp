# MemoryManager and KV Cache Design Notes

## Background

The repository previously introduced a `slot` dimension into `BitNetMemoryManager`, using an `id + slot + key` addressing model.

That design was later reverted after clarifying the intended meaning of `slot` and reviewing the current runtime architecture.

## Why `slot` was removed

The original conceptual purpose of `slot` was to represent a token-position slot inside the KV cache.
In other words, the idea was that each generated or prefetched token could map to a distinct KV cache slot.

However, the actual runtime implementation does not organize KV cache that way.
The current runtime writes KV cache by:

- keeping one key-cache buffer per layer,
- keeping one value-cache buffer per layer,
- using `CacheWritePosition` as the token-position offset inside those per-layer buffers.

That means the real model is:

- `session id` identifies the inference session,
- `key` identifies the concrete runtime buffer,
- token position is handled as an offset inside a buffer,
- not as a separate `slot` entry in `BitNetMemoryManager`.

Because of that, the `slot` dimension no longer matched the actual meaning of KV cache writes.
In practice it had been repurposed to represent `layerIndex`, which introduced semantic confusion:

- expected meaning: `slot = token position`
- actual temporary implementation: `slot = layer index`

After that mismatch became clear, keeping `slot` would only add indirection without expressing a real architectural need.

## Why the repository now uses `id + key`

The current `BitNetMemoryManager` intentionally uses the simpler `id + key` model because it better matches the runtime's actual memory layout:

- `id` scopes memory to a session,
- `key` names the owned buffer,
- layered KV cache buffers are distinguished by key names such as `LayerKeyCache:0` and `LayerValueCache:0`.

This keeps the memory manager aligned with its real responsibilities:

1. shared memory ownership across sessions,
2. runtime buffer lookup by semantic key,
3. model-weight loading destination.

It also avoids one more dictionary nesting level and one more lookup step on hot paths.

## Why KV cache remains statically allocated

The repository currently keeps the static KV cache allocation model.
Each per-layer KV cache buffer is allocated against the configured model context length.

This is intentional.
The codebase should not redesign memory architecture solely for CPU-side dynamic growth, because future GPU/NPU-style devices may not support the same dynamic expansion model cleanly.

Under the current design, static allocation has several advantages:

- predictable buffer layout,
- simple offset-based addressing,
- contiguous per-layer cache memory,
- easier backend portability,
- fewer runtime reallocations and copy paths.

## Current KV cache model

The current runtime model is:

- one session owns one set of runtime buffers,
- each transformer layer owns one key-cache buffer and one value-cache buffer,
- token position is represented by offset arithmetic using `CacheWritePosition`,
- `CacheLength` tracks the number of valid cached token positions.

In short:

- layer is encoded in the key name,
- token position is encoded in the offset,
- `BitNetMemoryManager` only needs `id + key`.

## Resulting design decision

The repository therefore adopts the following design decision for now:

- remove `slot` from `BitNetMemoryManager`,
- keep `BitNetMemoryManager` as `id + key`,
- keep KV cache statically allocated,
- keep token position management inside runtime/session metadata and offset calculations.

This matches the current implementation more closely and avoids carrying a misleading abstraction into later GPU/NPU-oriented work.
