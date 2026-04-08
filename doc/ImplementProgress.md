# Implement Progress

> 维护约定：后续每次新增或修改 layer 实现文件、对应测试文件、或 baseline 测试数据覆盖情况时，同步更新本文件。

## Layer Progress

| Layer | 当前状态 | 代码文件 | 测试文件 | 是否有对应 baseline 测试数据 | 说明 |
|---|---|---|---|---|---|
| `EmbeddingLayer` | 已实现 | `src/BitNetSharp/Layers/EmbeddingLayer.cs` | `src/tests/BitNetSharp.Tests/LayerTests.cs` | 是 | 单 token embedding 已对齐 baseline |
| `RmsNormLayer` | 已实现 | `src/BitNetSharp/Layers/RmsNormLayer.cs` | `src/tests/BitNetSharp.Tests/LayerTests.cs` | 是 | 当前用于 attention 前 norm |
| `QKVProjectionLayer` | 已实现 | `src/BitNetSharp/Layers/QKVProjectionLayer.cs` | `src/tests/BitNetSharp.Tests/QKVLayerTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` |
| `AttentionLayer` | 已实现 | `src/BitNetSharp/Layers/AttentionLayer.cs` | `src/tests/BitNetSharp.Tests/AttentionLayerTests.cs` | 是 | 已覆盖 `sub-norm` 与 `output` baseline |
| `ResidualLayer` | 已实现 | `src/BitNetSharp/Layers/ResidualLayer.cs` | `src/tests/BitNetSharp.Tests/ResidualLayerTests.cs` | 是 | 语义：`Embedding + AttentionOutput -> FeedForwardInput` |
| `FeedForwardNormLayer` | 已实现 | `src/BitNetSharp/Layers/FeedForwardNormLayer.cs` | `src/tests/BitNetSharp.Tests/FeedForwardNormLayerTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` |
| `FeedForwardLayer` | 已实现 | `src/BitNetSharp/Layers/FeedForwardLayer.cs` | `src/tests/BitNetSharp.Tests/FeedForwardLayerTests.cs` | 是 | 已覆盖 `ffn_sub_norm` 与 `ffn_down` baseline |
| `FeedForwardResidualLayer` | 已实现 | `src/BitNetSharp/Layers/FeedForwardResidualLayer.cs` | `src/tests/BitNetSharp.Tests/FeedForwardResidualLayerTests.cs` | 是 | 语义：`FeedForwardInput + FeedForwardOutput -> Embedding` |
| `FinalNormLayer` | 已实现 | `src/BitNetSharp/Layers/FinalNormLayer.cs` | `src/tests/BitNetSharp.Tests/FinalNormLayerTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` |
| `LmHeadLayer` | 已实现 | `src/BitNetSharp/Layers/LmHeadLayer.cs` | `src/tests/BitNetSharp.Tests/LmHeadLayerTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` logits baseline |

## Runtime Progress

| 组件 | 当前状态 | 是否有测试数据 | 说明 |
|---|---|---|---|
| `SamplingStep` / next-token 选择 | 已实现 | 是 | 已覆盖 greedy `argmax`、`top-k`、`next_token_id` baseline |
| 单 token 端到端 runtime 编排 | 未完成 | 是 | 各核心 step 已具备单测，`BitNetRuntime` 仍未接成完整推理链路 |

## Summary

- 已实现并验证的 layer：10 个
- 未实现的核心 layer：无
- 当前主要阻塞：单 token runtime 串联尚未实现
