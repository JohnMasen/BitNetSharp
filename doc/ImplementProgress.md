# Implement Progress

> 维护约定：后续每次新增或修改 node 实现文件、对应测试文件、或 baseline 测试数据覆盖情况时，同步更新本文件。

## Node Progress

| Node | 当前状态 | 代码文件 | 测试文件 | 是否有对应 baseline 测试数据 | 说明 |
|---|---|---|---|---|---|
| `EmbeddingNode` | 已实现 | `src/BitNetSharp/Nodes/EmbeddingNode.cs` | `src/tests/BitNetSharp.Tests/NodeTests.cs` | 是 | 单 token embedding 已对齐 baseline |
| `RmsNormNode` | 已实现 | `src/BitNetSharp/Nodes/RmsNormNode.cs` | `src/tests/BitNetSharp.Tests/NodeTests.cs` | 是 | 当前用于 attention 前 norm；已迁移为依赖 `IOPProvider2` |
| `QKVProjectionNode` | 已实现 | `src/BitNetSharp/Nodes/QKVProjectionNode.cs` | `src/tests/BitNetSharp.Tests/QKVNodeTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD`；当前 node 仅依赖 `IOPProvider2` |
| `AttentionNode` | 已实现 | `src/BitNetSharp/Nodes/AttentionNode.cs` | `src/tests/BitNetSharp.Tests/AttentionNodeTests.cs` | 是 | 已覆盖 `sub-norm` 与 `output` baseline；当前前向主流程已迁移为依赖 `IOPProvider2` |
| `ResidualNode` | 已实现 | `src/BitNetSharp/Nodes/ResidualNode.cs` | `src/tests/BitNetSharp.Tests/ResidualNodeTests.cs` | 是 | 语义：`Embedding + AttentionOutput -> FeedForwardInput`；已迁移为依赖 `IOPProvider2` |
| `FeedForwardNormNode` | 已实现 | `src/BitNetSharp/Nodes/FeedForwardNormNode.cs` | `src/tests/BitNetSharp.Tests/FeedForwardNormNodeTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD`；已迁移为依赖 `IOPProvider2` |
| `FeedForwardNode` | 已实现 | `src/BitNetSharp/Nodes/FeedForwardNode.cs` | `src/tests/BitNetSharp.Tests/FeedForwardNodeTests.cs` | 是 | 已覆盖 `ffn_sub_norm` 与 `ffn_down` baseline；当前 node 仅依赖 `IOPProvider2` |
| `FeedForwardResidualNode` | 已实现 | `src/BitNetSharp/Nodes/FeedForwardResidualNode.cs` | `src/tests/BitNetSharp.Tests/FeedForwardResidualNodeTests.cs` | 是 | 语义：`FeedForwardInput + FeedForwardOutput -> Embedding`；已迁移为依赖 `IOPProvider2` |
| `FinalNormNode` | 已实现 | `src/BitNetSharp/Nodes/FinalNormNode.cs` | `src/tests/BitNetSharp.Tests/FinalNormNodeTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD`；已迁移为依赖 `IOPProvider2` |
| `LmHeadNode` | 已实现 | `src/BitNetSharp/Nodes/LmHeadNode.cs` | `src/tests/BitNetSharp.Tests/LmHeadNodeTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` logits baseline；已迁移为依赖 `IOPProvider2` |

## Runtime Progress

| 组件 | 当前状态 | 是否有测试数据 | 说明 |
|---|---|---|---|
| `SamplingNode` / next-token 选择 | 已实现 | 是 | 已覆盖 greedy `argmax`、`top-k`、`next_token_id` baseline |
| 单 token 端到端 runtime 编排 | 未完成 | 是 | 各核心 node 已具备单测，`BitNetRuntime` 仍未接成完整推理链路 |

## Summary

- 已实现并验证的 node：10 个
- 未实现的核心 node：无
- 当前主要阻塞：单 token runtime 串联尚未实现
- 已引入 `IOPProvider1` / `IOPProvider2` 以及 `CPUDefaultOPProvider` / `CPUTensorOPProvider` / `CPUSimdOPProvider`
- 已移除 `CPUBaseOPProvider` 与 `OPProviderFactory`；当前 `Node` 层按 `InferenceConfig` 直接实例化具体 provider
- `MathHelper` 已彻底移除；高层复合逻辑与低层 backend kernel 现均由 provider 与 `OPProviderCommon` 承载
