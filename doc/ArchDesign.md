# BitNetSharp 架构设计草案

## 1. 文档目标

本文档用于整理 `BitNetSharp` 的第一版推理架构设计，目标是支持在 `CPU` 上执行 `BitNet` 模型推理，并为后续的正确性实现、性能优化和 `prefill` / `decode` 分化提供稳定骨架。

当前设计原则是：

- 以 `BitNetSession` 作为统一状态载体
- 每个推理步骤只接收一个 `BitNetSession`
- 步骤自行从上下文读取输入，并将结果写回上下文
- 推理流程、模型定义、底层算子三者解耦
- 优先保证流程清晰和可扩展，再逐步替换热点步骤为 `BitNet` 优化实现

---

## 2. 总体设计思路

整体建议分为五层：

### 2.1 Model 层

职责：保存静态模型信息。

建议承载内容：

- 模型超参数
- 词表与嵌入权重
- 各层权重
- 量化参数
- `BitNet` 线性层需要的权重布局信息

建议由 `BitNetModel` 作为核心入口。

### 2.2 Session 层

职责：表示一次推理会话的长期状态。

建议承载内容：

- 已输入和已生成的 token
- 当前序列长度
- 当前 position
- `KV Cache`
- 采样配置
- 会话级内存池

建议由 `BitNetSession` 作为会话对象。

### 2.3 Runtime 层

职责：驱动一次推理执行。

建议承载内容：

- 创建和准备 `BitNetSession`
- 组织顶层推理流水线
- 控制 `prefill` / `decode` 执行模式
- 按顺序调度各步骤

建议由 `BitNetRuntime` 作为推理入口。

当前补充约定：

- 现阶段先保持当前 `CPU` 路径的内存管理方式，不提前把额外内存管理职责压到 `BitNetSession`
- 当前分片逻辑主要服务 `CPU` 计算，不强行抽象为适配 `GPU` 的统一模型
- 后续如引入 `GPU` 推理，再一并调整 `BitNetRuntime` 与 `BitNetSession` 的职责边界，并视需要引入独立内存管理器

### 2.4 Step 层

职责：以统一协议组织推理步骤。

每个步骤：

- 只接收 `BitNetSession`
- 从上下文读取所需输入
- 将自己的结果写回上下文
- 不直接依赖调用者传递零散参数

建议包含：

- 顶层流程步骤
- 每层 block 步骤
- attention / MLP 子步骤

### 2.5 Kernel 层

职责：实现底层数值算子。

建议包含：

- `RmsNorm`
- `Softmax`
- `Attention Score`
- `Attention Reduce`
- `BitLinear`
- 其他矩阵向量或矩阵矩阵计算

`BitNet` 的低比特特性主要放在这一层，而不是放在步骤调度层。

### 2.6 Layer 推理配置约定

当前各个 layer 构造函数统一接收可选的 `InferenceConfig`。

- `InferenceConfig` 只包含 `Backend` 与 `ThreadCount`
- 该配置会以只读属性形式暴露在 layer 上
- 如果构造时传入 `null`，layer 会按各自默认策略创建一个新的默认配置实例
- 运行时始终读取该只读属性，不再额外推断计算后端或线程数

当前默认策略：

- `EmbeddingLayer`：`CPU` + 单线程
- `RmsNormLayer`：`SIMD` + 单线程
- `QKVProjectionLayer`：`SIMD` + 自动线程数

---

## 3. 核心对象职责

### 3.1 `BitNetModel`

定位为静态模型定义。

建议负责：

- 暴露模型结构信息
- 暴露每层权重访问入口
- 管理嵌入层、输出层和层级权重集合
- 保存与 `BitNet` 量化相关的元数据

不建议负责：

- 会话状态
- 推理过程中的中间张量
- `KV Cache`

### 3.2 `BitNetSession`

定位为一次交互会话的长期状态容器。

建议负责：

- 保存输入 prompt 和生成历史
- 保存 `KV Cache`
- 保存生成相关配置
- 记录当前位置和序列长度
- 提供会话级资源复用入口

不建议负责：

- 执行具体算子
- 决定层内执行顺序

### 3.3 `BitNetRuntime`

定位为推理执行协调器。

建议负责：

- 将 `BitNetModel` 和 `BitNetSession` 绑定到一次执行中
- 初始化本轮执行所需的 `BitNetSession` 状态
- 驱动推理步骤树
- 区分 `prefill` 与 `decode`
- 控制一轮或多轮 token 生成

### 3.4 `BitNetSession`（执行期视角）

定位为当前实现中的统一数据交换中心。

这是整个设计的核心。

建议原则：

- 所有步骤共享同一个上下文实例
- 步骤之间不直接传递零散参数
- 在同一个 `BitNetSession` 中显式区分长期状态与短期状态
- 关键槽位尽量强类型化，避免完全依赖字符串字典

---

## 4. `BitNetSession` 结构设计

建议将 `BitNetSession` 分为以下区域。

### 4.1 执行元信息区

建议包含：

- 当前执行模式：`Prefill` / `Decode`
- 当前层索引
- 当前 token 范围
- 当前 batch 大小
- 当前 sequence length
- 当前 position
- 是否启用 cache

用途：让步骤根据当前模式选择不同内部策略。

### 4.2 输入输出区

建议包含：

- 输入 token ids
- 当前 `Hidden`
- 最终 `Logits`
- 采样后的 `NextTokenId`

用途：承载顶层流程的主要输入和输出。

### 4.3 中间张量区

建议包含固定槽位，例如：

- `AttentionInput`
- `Q`
- `KCurrent`
- `VCurrent`
- `QPositioned`
- `KPositioned`
- `AttentionScores`
- `AttentionWeights`
- `AttentionContext`
- `AttentionProjected`
- `MlpInput`
- `MlpUp`
- `MlpGate`
- `MlpActivated`
- `MlpOutput`
- `FinalHidden`

设计建议：

- 第一版优先采用明确命名的强类型槽位
- 仅保留少量扩展槽位用于调试或实验
- 避免将所有中间结果都做成字符串键值表

### 4.4 `KV Cache` 区

建议独立于普通中间张量区。

建议包含：

- 每层 `K Cache`
- 每层 `V Cache`
- 当前 cache 有效长度
- 本轮写入位置

用途：作为跨 token 持久存在的数据结构。

### 4.5 模型与层访问区

建议包含：

- 当前 `BitNetModel` 引用
- 当前层定义或当前层权重视图

用途：让步骤无需自行查找当前层对象。

### 4.6 工作缓冲区 / 内存池区

建议包含：

- 可复用临时向量
- scratch buffer
- tensor pool 或 allocator 入口

用途：为后续 `CPU` 优化保留结构位置，减少频繁分配。

---

## 5. Step 设计原则

### 5.1 统一协议

每个步骤都遵循同一原则：

- 输入来自 `BitNetSession`
- 输出写回 `BitNetSession`
- 步骤本身只关注自己的职责范围

### 5.2 步骤元信息

建议每个步骤至少在设计层面明确四件事：

- 步骤名称
- 读取哪些槽位
- 写入哪些槽位
- 是否覆盖写回已有结果

### 5.3 复合步骤与原子步骤

建议分两层：

#### 原子步骤

示例：

- `QKVProjectionStep`
- `AttentionSoftmaxStep`
- `MlpActivationStep`

#### 复合步骤

示例：

- `AttentionBlockStep`
- `MlpBlockStep`
- `LayerPipelineStep`
- `TransformerStackStep`

这样既能保持顶层结构清晰，也能在内部继续细化。

---

## 6. 顶层推理 Pipeline 设计

建议顶层使用 `GenerationPipeline` 思路组织，步骤树如下。

### 6.1 `SessionInitializeStep`

职责：

- 初始化本轮执行元信息
- 设置 `Prefill` 或 `Decode`
- 准备 token 范围和 position 信息
- 清理本轮短生命周期槽位

读取：

- `BitNetSession`
- `BitNetModel`
- generation 配置

写入：

- 执行模式
- 当前 token 范围
- 当前 position 信息

### 6.2 `EmbeddingStep`

职责：

- `TokenIds -> Hidden`

读取：

- 输入 token ids
- embedding 权重

写入：

- `Hidden`

### 6.3 `TransformerStackStep`

职责：

- 逐层执行 `LayerPipelineStep`

读取：

- `Hidden`
- 层数量
- 模型层定义

写入：

- 更新后的 `Hidden`

### 6.4 `FinalNormStep`

职责：

- 对最终 `Hidden` 执行输出前归一化

读取：

- `Hidden`

写入：

- `FinalHidden`

### 6.5 `LmHeadStep`

职责：

- `FinalHidden -> Logits`

读取：

- `FinalHidden`
- 输出层权重

写入：

- `Logits`

### 6.6 `SamplingStep`

职责：

- 从 `Logits` 中选择下一个 token

读取：

- `Logits`
- 温度、`TopK`、`TopP` 等参数

写入：

- `NextTokenId`
- 采样附加信息

### 6.7 `SessionCommitStep`

职责：

- 将本轮结果写回 `BitNetSession`
- 推进 sequence length 和 position
- 为下一轮 decode 准备状态

读取：

- `NextTokenId`
- 当前 position 信息

写入：

- 会话 token 历史
- 更新后的长度和位置

---

## 7. `TransformerStackStep` 设计

`TransformerStackStep` 本身不做复杂数值计算，仅负责按层调度。

建议结构：

- 对每一层创建或复用当前层视图
- 设置当前 layer index
- 调用 `LayerPipelineStep`

每一层由两个大块组成：

- `AttentionBlockStep`
- `MlpBlockStep`

这是第一版最合适的粒度。

---

## 8. `AttentionBlockStep` 设计

建议拆为以下子步骤。

### 8.1 `AttentionInputNormStep`

职责：

- 对 `Hidden` 做 `RMSNorm` 或对应归一化

读取：

- `Hidden`
- 当前层 attention norm 权重

写入：

- `AttentionInput`

### 8.2 `QKVProjectionStep`

职责：

- `AttentionInput -> Q / K / V`

读取：

- `AttentionInput`
- 当前层 `Wq / Wk / Wv`
- 对应量化参数

写入：

- `Q`
- `KCurrent`
- `VCurrent`

备注：

- 这是最重要的 `BitNet` 热点步骤之一
- `BitNet-B1.58` 的 `Q / K / V` 投影不是普通 `float matmul`
- 对当前 `GGUF` 模型，`Wq / Wk / Wv` 是 `I2_S` 打包权重，运行时路径等价于 `BitNetLinear(attn_norm, wq/wk/wv)`
- 对拍时不要先把整张权重反量化成 `float` 再做普通矩阵乘法，这会稳定偏离 runtime
- 若模型存在对应 bias，应在各自投影输出后追加 `bq / bk / bv`

### 8.3 `PositionApplyStep`

职责：

- 对 `Q` 和 `KCurrent` 应用位置编码

读取：

- `Q`
- `KCurrent`
- position 信息

写入：

- `QPositioned`
- `KPositioned`

备注：

- `RoPE` 只作用于 `Q` 和 `KCurrent`
- `VCurrent` 不做 `RoPE`
- 对当前调试 dump，`Qcur / Kcur` 是 `RoPE` 之前的运行时节点，不能直接拿去和 `RoPE` 之后的实现结果对拍

### 8.4 `KvCacheWriteStep`

职责：

- 将本轮 `K / V` 写入当前层 cache
- 更新当前层 cache 长度

读取：

- `KPositioned`
- `VCurrent`
- 当前层 `KV Cache`
- 当前写入位置

写入：

- 更新后的 `K Cache`
- 更新后的 `V Cache`
- cache 有效长度

备注：

- 当前 `BitNet-B1.58` 路径里，`V Cache` 的存储类型会影响后续 attention 数值
- 单 token调试时，runtime 看到的 `attn_ctx` 不是原始 `VCurrent(F32)` 的直接重复，而是写入 cache 后再读出的值
- 对当前模型可近似记为：`VCacheValue = fp16_to_fp32(fp32_to_fp16(VCurrent))`
- 如果实现里直接复用原始 `VCurrent(F32)`，`AttentionContext`、`attn_sub_norm` 和 `attn_o_out` 都会出现小但稳定的偏差

### 8.5 `AttentionScoreStep`

职责：

- 计算 query 与历史 keys 的相关性分数

读取：

- `QPositioned`
- 当前层 `K Cache`
- mask 信息

写入：

- `AttentionScores`

备注：

- `Prefill` 时偏向块状计算
- `Decode` 时偏向 query 对历史 cache 的点积

### 8.6 `AttentionSoftmaxStep`

职责：

- 对分数做缩放、mask 和 softmax

读取：

- `AttentionScores`

写入：

- `AttentionWeights`

### 8.7 `AttentionReduceStep`

职责：

- `AttentionWeights * VCache -> AttentionContext`

读取：

- `AttentionWeights`
- 当前层 `V Cache`

写入：

- `AttentionContext`

备注：

- 对当前模型，`AttentionContext` 的逻辑 shape 应理解为 `[n_head, head_dim]`
- `VCurrent` 的逻辑 shape 应理解为 `[n_head_kv, head_dim]`
- `GQA` 映射规则是连续分组，而不是取模
- 对当前模型：`n_head = 20`、`n_head_kv = 5`、`n_gqa = 4`
- 因此映射规则为 `kv_head = q_head / n_gqa`
- 即 `0..3 -> 0`、`4..7 -> 1`、`8..11 -> 2`、`12..15 -> 3`、`16..19 -> 4`
- flatten 为一维向量时应保持 `head-major` 顺序：`flat_index = q_head * head_dim + dim`
- 不要误用 `dim-major` 展平，也不要误用 `kv_head = q_head % n_head_kv`
- 在 `单 token + pos = 0 + clear KV cache` 的调试场景下，softmax 权重退化为 `1`
- 因此当前模型的单 token `AttentionContext` 应与 `repeat_gqa(fp16_roundtrip(VCurrent))` 对齐

### 8.8 `AttentionOutputProjectionStep`

职责：

- `AttentionContext -> AttentionProjected`

读取：

- `AttentionContext`
- 当前层 `Wo`

写入：

- `AttentionProjected`

备注：

- 这是另一个重要的 `BitNet` 热点步骤
- `AttentionContext` 在当前 `BitNet-B1.58` 模型里不会直接进入 `Wo`
- 正确顺序是：`AttentionContext -> attn_sub_norm -> Wo -> optional wo_scale -> optional bo`
- `attn_sub_norm` 是 attention 后的 `RMSNorm`，这一层是 `BitNet-B1.58` 与常见实现相比最容易漏掉的差异之一
- 对当前模型的第一层调试结果，`attn_sub_norm` 是对整个 `2560` 维向量做一次 `RMSNorm`，不是按 head 分段归一化
- 对当前 `ggml-model-i2_s.gguf`，`GGUF` 中不存在额外的 `attn_output.scale` 和 `attn_output.bias`
- 因此当前模型的 `attn_o_out` 仅包含 `BitNetLinear(attn_sub_norm, attn_output.weight)`
- `attn_output.weight` 尾部自带的那个 `float scale` 只是 `I2_S` 权重的打包量化 scale，不是额外的 `wo_scale`

### 8.9 `AttentionResidualStep`

职责：

- `Hidden + AttentionProjected -> Hidden`

读取：

- `Hidden`
- `AttentionProjected`

写入：

- `Hidden`

设计建议：

- 直接回写主 `Hidden` 槽位，减少中间结果残留

### 8.10 `BitNet-B1.58` Attention 实现说明与避坑

下面这条链路是当前仓库实现 `BitNet-B1.58` attention 时应优先对齐的 runtime 语义：

```text
inpL
  -> attn_norm
  -> Qcur / Kcur / Vcur
  -> RoPE(Qcur / Kcur)
  -> KV cache write
  -> attention core
  -> attn_ctx
  -> attn_sub_norm
  -> attn_output.weight
  -> attn_o_out
  -> residual add
```

关键约束：

- `Attention` 的直接输入是 `QKVProjection` 的输出，不是 `RMSNorm` 的输出
- `RMSNorm` 只是 `QKVProjection` 的输入准备步骤
- `Q / K / V / Wo` 都应走 `BitNet` 专用量化线性路径，而不是普通浮点矩阵乘法
- `Q / K` 会做 `RoPE`，`V` 不做 `RoPE`
- attention 后必须有 `attn_sub_norm`
- 之后才进入 `Wo`

对当前仓库最容易踩错的点：

- 把 dump 出来的 `Qcur / Kcur` 当成 `RoPE` 之后的值
- 在单 token 场景下误以为 `Q / K` 没参与计算就是实现错误
- 直接把原始 `VCurrent(F32)` 做 `GQA` 重复，而忘记 `KV cache` 的 `F16 round-trip`
- 把 `attn_sub_norm` 按 head 分段归一化，而不是按当前 runtime 的整条 `2560` 维向量归一化
- 把 `attn_output.weight` 尾部量化 scale 误认为额外的 `wo_scale`

单 token runtime 对拍建议：

- 仅在 `单 token + pos = 0 + clear KV cache` 条件下做第一步对拍
- 推荐按以下顺序定位偏差：
  - `attn_norm`
  - `Qcur / Kcur / Vcur`（注意这里是 `RoPE` 前）
  - `attn_ctx`
  - `attn_sub_norm`
  - `attn_o_out`
- 如果 `attn_sub_norm` 只有极小误差，但 `attn_o_out` 误差明显放大，优先检查进入 `Wo` 前的激活量化边界，而不要先怀疑 `Wo` 本身

当前仓库已经验证过的一条重要经验：

- `attn_sub_norm` 的微小偏差会在 `Wo` 前的激活量化阶段翻转少量量化码
- 一旦量化码翻转，`attn_o_out` 的误差会被明显放大
- 因此 attention 调试时，必须先把 `attn_ctx` 与 `attn_sub_norm` 压到足够接近 runtime，再看输出投影

---

## 9. `MlpBlockStep` 设计

建议拆为以下子步骤。

### 9.1 `MlpInputNormStep`

职责：

- 对 attention 后的 `Hidden` 做归一化

读取：

- `Hidden`
- 当前层 MLP norm 权重

写入：

- `MlpInput`

### 9.2 `MlpUpProjectionStep`

职责：

- `MlpInput -> MlpUp`

读取：

- `MlpInput`
- up projection 权重

写入：

- `MlpUp`

备注：

- 属于主要 `BitNet` 优化目标之一

### 9.3 `MlpGateProjectionStep`

职责：

- `MlpInput -> MlpGate`

读取：

- `MlpInput`
- gate projection 权重

写入：

- `MlpGate`

### 9.4 `MlpActivationStep`

职责：

- 执行激活与门控组合

读取：

- `MlpUp`
- `MlpGate`

写入：

- `MlpActivated`

说明：

- 若模型使用 gated MLP，则组合两路输入
- 若模型采用普通 FFN，可退化为单输入激活

### 9.5 `MlpDownProjectionStep`

职责：

- `MlpActivated -> MlpOutput`

读取：

- `MlpActivated`
- down projection 权重

写入：

- `MlpOutput`

### 9.6 `MlpResidualStep`

职责：

- `Hidden + MlpOutput -> Hidden`

读取：

- `Hidden`
- `MlpOutput`

写入：

- `Hidden`

---

## 10. 完整步骤树

建议的完整树状结构如下：

- `GenerationPipeline`
  - `SessionInitializeStep`
  - `EmbeddingStep`
  - `TransformerStackStep`
    - `LayerPipelineStep[0]`
      - `AttentionBlockStep`
        - `AttentionInputNormStep`
        - `QKVProjectionStep`
        - `PositionApplyStep`
        - `KvCacheWriteStep`
        - `AttentionScoreStep`
        - `AttentionSoftmaxStep`
        - `AttentionReduceStep`
        - `AttentionOutputProjectionStep`
        - `AttentionResidualStep`
      - `MlpBlockStep`
        - `MlpInputNormStep`
        - `MlpUpProjectionStep`
        - `MlpGateProjectionStep`
        - `MlpActivationStep`
        - `MlpDownProjectionStep`
        - `MlpResidualStep`
    - `LayerPipelineStep[1]`
    - `...`
  - `FinalNormStep`
  - `LmHeadStep`
  - `SamplingStep`
  - `SessionCommitStep`

---

## 11. `Prefill` 与 `Decode` 的处理方式

建议保持步骤树稳定，不为 `Prefill` 和 `Decode` 分别设计两套完全不同的调度结构。

建议做法：

- 步骤名称和职责保持一致
- 具体行为由 `BitNetSession` 上的模式字段决定
- 每个步骤内部根据模式选择不同策略

例如：

### `QKVProjectionStep`

- `Prefill`：处理一个 token 段
- `Decode`：仅处理最新 token

### `KvCacheWriteStep`

- `Prefill`：批量写入一段 `K / V`
- `Decode`：追加一个位置

### `AttentionScoreStep`

- `Prefill`：块状 score 计算
- `Decode`：单 query 对历史 cache 的点积

这样可以最大限度保持结构一致，便于测试和演进。

---

## 12. `BitNet` 特性应该落在哪一层

关键原则：`BitNet` 是算子实现特征，不是步骤调度特征。

因此建议：

- `QKVProjectionStep` 仍然只是投影步骤
- `MlpUpProjectionStep` 仍然只是投影步骤
- `AttentionOutputProjectionStep` 仍然只是投影步骤
- `MlpDownProjectionStep` 仍然只是投影步骤

具体使用：

- 普通线性 kernel
- `BitLinear` kernel
- ternary 或其他低比特 kernel

由步骤内部依赖的底层实现决定。

这样可以得到以下收益：

- 推理骨架稳定
- reference 版与优化版可替换
- 后续可独立优化热点步骤而不破坏流程结构

---

## 13. 第一版实现建议

建议分阶段推进。

### 阶段一：跑通 reference pipeline

目标：

- `BitNetSession` 能完整流转
- 所有步骤的读写关系清晰
- 先保证正确性

建议特点：

- 可先不追求 SIMD
- 可先不追求极致内存布局
- 可先用相对直接的张量表示

### 阶段二：替换热点线性步骤

优先优化：

- `QKVProjectionStep`
- `AttentionOutputProjectionStep`
- `MlpUpProjectionStep`
- `MlpGateProjectionStep`
- `MlpDownProjectionStep`
- 如有需要再优化 `LmHeadStep`

### 阶段三：引入 `CPU` 侧优化

包括：

- 权重打包布局
- block 化访问
- `SIMD` 加速
- scratch buffer 复用
- cache 友好访问
- 多线程策略

---

## 14. 风险与约束

### 14.1 `BitNetSession` 过度膨胀

风险：

- 字段过多
- 生命周期混乱
- 各步骤读写边界不清晰

规避建议：

- 将上下文按区域分组
- 区分长期状态与短期状态
- 对关键槽位做显式命名

### 14.2 步骤粒度过细

风险：

- 调度复杂度上升
- 代码阅读成本增加
- 实现时容易碎片化

规避建议：

- 第一版按数学子阶段拆分
- 不继续下钻到极细粒度操作

### 14.3 步骤读写不透明

风险：

- 难以定位错误来源
- 难以验证前置依赖

规避建议：

- 为每个步骤明确读写契约
- 在调试阶段可加步骤级 dump 或 profiling 标记

---

## 15. 推荐的最小可行骨架

若以尽快落地为目标，建议第一版先实现如下骨架：

### 顶层

- `SessionInitializeStep`
- `EmbeddingStep`
- `TransformerStackStep`
- `FinalNormStep`
- `LmHeadStep`
- `SamplingStep`
- `SessionCommitStep`

### 每层

- `AttentionBlockStep`
- `MlpBlockStep`

### 每个 block 内部

先按本文档中的建议顺序组织子步骤，但实现时可先用简单顺序调用，不必一开始就做复杂框架。

---

## 16. 结论

本设计的核心原则可归纳为一句话：

`BitNetSession` 负责持有状态，`Step` 负责消费状态并产出状态，`BitNetRuntime` 负责按顺序驱动 `Step`，`Kernel` 负责执行真正的数值计算。

这套结构适合 `BitNetSharp` 的早期演进路径：

- 先建立清晰、稳定的推理骨架
- 再逐步替换热点线性层为 `BitNet` 优化实现
- 最后在 `CPU` 上围绕内存布局、缓存命中和向量化持续优化
