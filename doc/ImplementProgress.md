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
| 单 token 端到端 runtime 编排 | 已实现（过渡版） | 是 | `BitNetRuntime` 已接起当前端到端链路，并新增 `Prefill` / `ContinuePrefill` 与连续生成入口；runtime 路径现已按层写入并读取 `KV cache` 参与 attention，但当前仍是运行时内联编排的过渡实现，尚未下沉为正式 graph/runtime 结构 |
| 文本对话 runtime / console 入口 | 已实现（增强版） | 是 | 已新增基于 GGUF/chat runtime 行为对齐的最小 user/assistant 文本对话入口，可通过 `StartConversation` / `ContinueConversation` / `GenerateAssistantReply` / `StreamAssistantReply` / `StreamAssistantReplyWithTokenIds` 驱动会话，并提供 `BitNetSharp.Console` 控制台项目；当前控制台已支持 `--max-new-tokens`、`--top-k`、`--temperature`、`--top-p`、`--min-p`、`--prompt`、`--show-token-ids`、模型信息输出、流式 token 输出、Ctrl+C 中断生成，以及显式启用的可选采样调试 |
| Console 内存统计与 CSV 导出 | 已实现（第一版） | 是 | `BitNetSharp.Console` 已支持 `--show-memory` 与 `--memory-csv`，可显示 `MemoryManager` 跟踪片段总览，并在外围推导 `Actual KV Cache` / `Allocated KV Cache` 后导出 CSV 明细 |
| `RuntimeTensor` / Session 张量访问入口 | 已实现（第一版） | 是 | 已引入非泛型 `RuntimeTensor`，并新增 `GetWeightTensor(string)` 与 `GetOrCreateRuntimeTensor(string)`；权重 tensor 由 `BitNetModel` 共享缓存，runtime tensor 由 `BitNetSession` 私有创建 |
| `BitNetSession` 多轮输出状态 / `KV cache` 容器 | 已实现（第一版） | 是 | 已支持 append-only token 历史、输出轮次跟踪、当前轮输出视图，以及按层 `K/V cache` tensor 容器；当前 `KV cache` 继续使用按层 key 前缀区分的静态分配存储，并已接入 runtime 过渡链路 |
| `IOPProvider` RuntimeTensor 迁移 | 已实现（第一版） | 是 | `IOPProvider` 的主要张量输入输出已从 `Memory<T>` / `Span<T>` 迁移到 `RuntimeTensor`；CPU / Tensor / SIMD provider、主要 node、以及运行时过渡编排路径已同步切换 |

## Summary

- 已实现并验证的 node：10 个
- 未实现的核心 node：无
- 当前主要阻塞：runtime 正式架构仍未完成；当前单 token 端到端入口仅用于链路验证，后续会删除
- 已引入 `IOPProvider` 以及 `CPUDefaultOPProvider` / `CPUTensorOPProvider` / `CPUSimdOPProvider`
- 已移除 `CPUBaseOPProvider`、`OPProviderFactory` 与 `InferenceBackend`；当前 `InferenceConfig` 直接持有 `IOPProvider`，`Node` / `Runtime` 不再通过 enum 分支选择 backend/provider
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
- `IOPProvider` 中无实际用途的 `operationName` 参数已移除；相关调用点与测试已同步清理这类无效标签参数
- `RmsNormNode` / `FeedForwardNormNode` / `FinalNormNode` / `FeedForwardResidualNode` 中仅做直接转发的 `ExecuteForward` wrapper 也已移除；当前 node/OP 主链路中的无意义方法层级已继续压缩
- `LmHeadNode` / `QKVProjectionNode` / `ResidualNode` / `FeedForwardNode` / `AttentionNode` 以及剩余 norm/residual node 中仅作为 `Forward` 延续的 `ForwardCore` 已内联回 `Forward`；当前 node 主链路中的 trivial wrapper 基本已清掉
- 已为 `BitNetRuntime` 添加临时 `Inference(int tokenId)` 入口，用于验证当前单 token 完整链路：`Embedding -> per-layer(attn norm / QKV / Attention / Residual / FFN norm / FFN / FFN residual) -> FinalNorm -> LmHead -> Sampling -> Decode`；该方法及其测试后续将随 runtime 正式架构演进一并删除
- 多余的 `global::BitNetSharp.` 资格限定已继续清理；当前剩余测试代码已改为优先使用普通命名空间引用，仅在未来确有命名冲突时才保留 `global::`
- 临时 `BitNetRuntime.Inference` 测试已补上对 `layer_vectors_pure.json` 中 `next_token_id` 的直接断言，当前既验证 runtime 编排链路一致性，也验证最终 decode 输出与 baseline 数据一致
- `OPProviderBackendTests` 中针对 `IOPProvider2` 默认编排的临时测试已移除；当前 OP 相关测试聚焦具体 provider 的 `IOPProvider` 行为与实际 node 路径
- 默认开发节奏下，所有多 case 的数据驱动测试当前均只枚举第一个 case，以缩短回归时间；保留原有按 caseId 取数逻辑，后续需要扩大覆盖时可直接恢复 provider 枚举
- 与首 case 数据驱动测试重复的专用 `DebugCase` 基线测试入口已移除；当前调试默认直接复用只跑首 case 的数据驱动测试
- `InferenceConfig` 中硬编码 `CPU` / `Tensor` / `SIMD` 三类 provider 的 `Create*` 工厂已移除；核心层不再通过这些静态入口假设固定 provider 集合，调用方侧改为显式构造并传入 `IOPProvider`
- `OPProviderBackendNames` 也已从主代码移除；provider 自身直接声明 backend 字符串，测试侧固定标识改收敛到测试辅助，避免主代码继续暴露固定 provider 集合假设
- 已引入第一版 `RuntimeTensor`：`BitNetSession` 可通过 `GetWeightTensor(string)` 获取模型级共享只读权重 tensor，并通过 `GetOrCreateRuntimeTensor(string)` 获取会话私有 runtime tensor；现有会话缓冲区属性已改为基于 `RuntimeTensor` 的统一创建与访问
- `IOPProvider` 的主要张量边界也已迁移到 `RuntimeTensor`；`ForwardSoftmax` 也已从 `Span` 公开签名切到 `RuntimeTensor`。当前 provider 内部仍解析回 `Memory/Span` 以复用现有 CPU/Tensor/SIMD 实现，但 node/runtime/测试辅助层已不再直接把 `Memory<T>` 或 `Span<T>` 作为 OP API 传递对象
- `BitNetSession` 已补上多轮输出状态层第一版：现支持 append-only token 历史、输出轮次跟踪、当前轮输出切片视图、以及按层 `K/V cache` tensor 容器；单个 `session` 不再允许通过公开 API 重置历史，重新开始生成需要创建新 `session`
- `BitNetRuntime` 已补上 `Prefill` / `ContinuePrefill` 与连续生成入口；当前 runtime 过渡链路现已在 QKV 阶段按层写入 `K/V cache`，并在 attention 阶段读取历史 `KV cache` 参与上下文计算，使历史 token 开始真正参与 runtime 推理
- 已补上最小文本对话路径：`BitNetTokenizer` 现支持当前 GGUF 模板下的 user/assistant 聊天编码，`BitNetRuntime` 新增 `StartConversation` / `ContinueConversation` / `GenerateAssistantReply`，并已创建 `BitNetSharp.Console` 作为最小交互式对话程序入口
- `BitNetRuntime` 已新增带 `CancellationToken` 的流式回复路径 `StreamAssistantReply`，控制台程序现按 token 即时打印 assistant 输出，并把 Ctrl+C 直接接到当前生成的取消逻辑，避免只能在整轮生成结束后才退出
- `SamplingNode` 当前已向 llama.cpp 默认采样链做最小对齐：同一轮生成共享随机源，并新增 `temperature` / `top-p` / `min-p`；`BitNetRuntime` / `BitNetSharp.Console` 也已暴露这些参数，同时保留显式启用采样的入口，避免破坏现有 greedy baseline 测试
- `BitNetMemoryManager` 已恢复为 `id + key` 模型；当前不再保留 `slot` 维度，按层 `KV cache` 继续通过 key 前缀编码层号，避免额外的字典查找开销
- 关于移除 `slot` 并保留静态 `KV cache` 分配的设计说明，已补充到 `doc/archdesign/MemoryManager-KVCache-Design.md`
- `BitNetMemoryManager` 已补上只读统计快照 `GetStatistics()`；控制台可选择显示已跟踪内存片段总览，并额外在外围推导 `Actual KV Cache` 与 `Allocated KV Cache`。同时支持导出 CSV，便于后续用透视表分析
- 聊天 prompt 已开始按 dump 出的 BitNet 运行时行为对齐：当前 user 轮改为 `User: <content><|eot_id|>Assistant: `，assistant 轮结束优先使用 `EOT` 分隔 token 结束，而不是继续固定使用 `Human/BITNETAssistant + EOS`
- 为便于定位不可见字符与采样异常，控制台已新增 `--show-token-ids` 调试开关，可将每个生成 token 以 `文本[tokenId]` 形式打印出来
- `BitNetRuntime` 已新增 `Prefill` 与 `ContinuePrefill`：现在可先用 prompt token 建立 active session，再衔接后续 `GenerateTokenIds(count)`；当前 `Prefill` 仍是逐 token 复用现有单 token 推理链路，尚未让 attention 真正读取历史 `KV cache`
