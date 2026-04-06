# BitNetSharp 架构设计草案

## 1. 文档目标

本文档用于整理 `BitNetSharp` 的第一版推理架构设计，目标是支持在 `CPU` 上执行 `BitNet` 模型推理，并为后续的正确性实现、性能优化和 `prefill` / `decode` 分化提供稳定骨架。

当前设计原则是：

- 以 `SessionRuntimeContext` 作为统一执行上下文
- 每个推理步骤只接收一个 `SessionRuntimeContext`
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

- 创建和复用 `SessionRuntimeContext`
- 组织顶层推理流水线
- 控制 `prefill` / `decode` 执行模式
- 按顺序调度各步骤

建议由 `BitNetRuntime` 作为推理入口。

### 2.4 Step 层

职责：以统一协议组织推理步骤。

每个步骤：

- 只接收 `SessionRuntimeContext`
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
- 创建 `SessionRuntimeContext`
- 驱动推理步骤树
- 区分 `prefill` 与 `decode`
- 控制一轮或多轮 token 生成

### 3.4 `SessionRuntimeContext`

定位为单次执行期的统一数据交换中心。

这是整个设计的核心。

建议原则：

- 所有步骤共享同一个上下文实例
- 步骤之间不直接传递零散参数
- 上下文中显式区分长期状态与短期状态
- 关键槽位尽量强类型化，避免完全依赖字符串字典

---

## 4. `SessionRuntimeContext` 结构设计

建议将 `SessionRuntimeContext` 分为以下区域。

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

- 输入来自 `SessionRuntimeContext`
- 输出写回 `SessionRuntimeContext`
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

- `QkvProjectionStep`
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

### 8.2 `QkvProjectionStep`

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
- 内部可以调用普通线性实现或 `BitLinear` 实现

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
        - `QkvProjectionStep`
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
- 具体行为由 `SessionRuntimeContext.Mode` 决定
- 每个步骤内部根据模式选择不同策略

例如：

### `QkvProjectionStep`

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

- `QkvProjectionStep` 仍然只是投影步骤
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

- `SessionRuntimeContext` 能完整流转
- 所有步骤的读写关系清晰
- 先保证正确性

建议特点：

- 可先不追求 SIMD
- 可先不追求极致内存布局
- 可先用相对直接的张量表示

### 阶段二：替换热点线性步骤

优先优化：

- `QkvProjectionStep`
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

### 14.1 `SessionRuntimeContext` 过度膨胀

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

`SessionRuntimeContext` 负责持有状态，`Step` 负责消费状态并产出状态，`BitNetRuntime` 负责按顺序驱动 `Step`，`Kernel` 负责执行真正的数值计算。

这套结构适合 `BitNetSharp` 的早期演进路径：

- 先建立清晰、稳定的推理骨架
- 再逐步替换热点线性层为 `BitNet` 优化实现
- 最后在 `CPU` 上围绕内存布局、缓存命中和向量化持续优化
