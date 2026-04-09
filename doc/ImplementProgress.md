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
| `LmHeadNode` | 已实现 | `src/BitNetSharp/Nodes/LmHeadNode.cs` | `src/tests/BitNetSharp.Tests/LmHeadNodeTests.cs` | 是 | 已覆盖 `CPU` / `Tensor` / `SIMD` logits baseline；已迁移为依赖 `IOPProvider2`，`SIMD` 现使用独立实现；非缓存路径已去除逐次 `Half[]` 权重复制 |

## Runtime Progress

| 组件 | 当前状态 | 是否有测试数据 | 说明 |
|---|---|---|---|
| `SamplingNode` / next-token 选择 | 已实现 | 是 | 已覆盖 greedy `argmax`、`top-k`、`next_token_id` baseline |
| 单 token 端到端 runtime 编排 | 已实现（仅测试用途） | 是 | `BitNetRuntime` 已接起当前单 token 完整推理链路，但该入口仅用于链路验证，后续会连同测试一起删除 |

## Summary

- 已实现并验证的 node：10 个
- 未实现的核心 node：无
- 当前主要阻塞：runtime 正式架构仍未完成；当前单 token 端到端入口仅用于链路验证，后续会删除
- 已引入 `IOPProvider1` / `IOPProvider2` 以及 `CPUDefaultOPProvider` / `CPUTensorOPProvider` / `CPUSimdOPProvider`
- 已移除 `CPUBaseOPProvider` 与 `OPProviderFactory`；当前 `Node` 层按 `InferenceConfig` 直接实例化具体 provider
- `MathHelper` 已彻底移除；高层复合逻辑与低层 backend kernel 现均由 provider 承载
- 各 concrete provider 已回收为单文件实现，避免额外的 `*.Kernels.cs` 分层
- `IOPProvider2` 现已承载纯编排默认实现；共享参数校验已独立收敛到 `ValidationHelper`
- `LmHead` 的 `CPU` / `Tensor` 实现与 CPU-style packed-weight fallback 已回收到具体 provider；共享 helper 不再承载这类 backend 专属实现
- `QuantizeBitNetActivations` 已按 `CPU` / `Tensor` / `SIMD` 分别落到具体 provider；共享 helper 不再承载这类批量量化实现
- `GetBitNetPackedWeightByteCount` 与 `FinalizeBitNetMappedProjection` 已在使用点直接内联；共享 helper 仅保留参数校验
- 已完成 OP 多线程覆盖审计：当前仍存在量化、RMSNorm 归约、Tensor 量化后类型转换、以及 `IOPProvider2` 中 `ApplyScale` / `ApplyBias` 的单线程缺口
- `QuantizeBitNetActivations` 已补齐 `CPU` / `Tensor` / `SIMD` 的多线程路径；当前剩余的大块数据单线程缺口主要是 RMSNorm 归约、Tensor 量化后类型转换、以及 `IOPProvider2` 中 `ApplyScale` / `ApplyBias`
- `RMSNorm` 的归约阶段已补齐 `CPU` / `Tensor` / `SIMD` 的多线程路径；当前剩余的大块数据单线程缺口主要是 Tensor 量化后类型转换，以及 `IOPProvider2` 中 `ApplyScale` / `ApplyBias`
- `CPUTensorOPProvider` 中量化后的 `sbyte -> float` 转换已补齐多线程路径；当前剩余的大块数据单线程缺口主要是 `IOPProvider2` 中 `ApplyScale` / `ApplyBias`
- `IOPProvider2` 中的 `ApplyScale` / `ApplyBias` 已补齐多线程路径；当前 OP 内面对大块数据的主要公共处理环节已具备 thread-aware 实现
- backend 驱动的 node 单元测试已补齐 `CPU` / `Tensor` / `SIMD` 的单线程基线与多线程一致性覆盖；缺失的 `BenchmarkSuite1` 6 档矩阵也已补齐到 `FeedForward` / `FeedForwardNorm` / `FeedForwardResidual` / `FinalNorm` / `LmHead`
- `BenchmarkSuite1` 的 node benchmark 风格已进一步统一：公共输入填充与模型路径查找已提取，node 输入准备改为直接填充目标 session buffer，避免通过上游 node 预热来混入额外 setup 语义
- `CPUDefaultOPProvider` / `CPUTensorOPProvider` / `CPUSimdOPProvider` 中仅做直接转发的 `Execute*` wrapper 已移除，公共入口直接承载实现，减少无意义的方法层级
- `IOPProvider1` 中无实际用途的 `operationName` 参数已移除；`IOPProvider2` 默认实现、node 调用点与相关测试已同步清理这类无效标签参数
- `RmsNormNode` / `FeedForwardNormNode` / `FinalNormNode` / `FeedForwardResidualNode` 中仅做直接转发的 `ExecuteForward` wrapper 也已移除；当前 node/OP 主链路中的无意义方法层级已继续压缩
- `LmHeadNode` / `QKVProjectionNode` / `ResidualNode` / `FeedForwardNode` / `AttentionNode` 以及剩余 norm/residual node 中仅作为 `Forward` 延续的 `ForwardCore` 已内联回 `Forward`；当前 node 主链路中的 trivial wrapper 基本已清掉
- 已为 `BitNetRuntime` 添加临时 `Inference(int tokenId)` 入口，用于验证当前单 token 完整链路：`Embedding -> per-layer(attn norm / QKV / Attention / Residual / FFN norm / FFN / FFN residual) -> FinalNorm -> LmHead -> Sampling -> Decode`；该方法及其测试后续将随 runtime 正式架构演进一并删除
- 多余的 `global::BitNetSharp.` 资格限定已继续清理；当前剩余测试代码已改为优先使用普通命名空间引用，仅在未来确有命名冲突时才保留 `global::`
- 临时 `BitNetRuntime.Inference` 测试已补上对 `layer_vectors_pure.json` 中 `next_token_id` 的直接断言，当前既验证 runtime 编排链路一致性，也验证最终 decode 输出与 baseline 数据一致
