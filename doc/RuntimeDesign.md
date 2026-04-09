# BitNetSharp Runtime 设计文档

## 1. 文档目标

本文档用于整理 `BitNetSharp` 下一阶段的运行时重构方向，目标是为单 `token` 推理建立一套职责清晰、易于演进的架构骨架，并为后续 `CPU` / `SIMD` / `Tensor` 策略扩展以及未来的 `CPU` / `GPU` offload 留出稳定接口。

当前文档关注的是**运行时组织方式**，不是一次性完成全部实现。

本阶段的核心目标如下：

- `Node` 只保留数据定位和外层编排逻辑
- 所有数值计算统一下沉到 `OpProvider`
- `IMemoryManager` 作为统一抽象，负责为多个推理 `session` 提供共享内存管理，并作为模型权重加载的目标位置
- `MemoryOwner` 未来支持可追踪访问，但当前只记录 trace 期间的读写请求，不实现版本管理
- `Runtime` 以 `Graph` 的方式串联 `Node`
- `Graph` 当前只承载 `Node`，不引入额外的运行时节点抽象接口
- `SamplingNode` 作为 `Node` 保留在 graph 中，使 `Runtime` 保持模型无关
- 跨 `MemoryManager` 的同步通过包装器实现，而不是直接写入管理器本体
- `BitNetRuntime` 永远保持简单干净，复杂优化能力放在外围组件中

---

## 2. 当前实现与目标方向

当前仓库已经具备以下基础：

- `BitNetSession` 作为会话状态入口
- `BitNetMemoryManager` 作为会话共享内存管理器
- 多个 `Node` 已实现 `Init` / `Forward`
- `MathHelper` 中已经承载了部分核心算法与 backend 分支
- `BitNetRuntime` 仍处于很轻量的占位状态，尚未串成完整单 `token` 推理链路

下一阶段的目标不是推翻现有实现，而是把运行时执行单元统一收敛为 `Node` 概念，并把计算职责进一步收敛到统一算子提供者中，使运行时的职责边界稳定下来。

---

## 3. 设计原则

### 术语区分

本文档中有两个容易混淆的“层”概念，需要显式区分：

- **模型层（model layer）**：神经网络语义中的层，例如 transformer block、norm、attention、feed-forward 等模型结构概念
- **运行时节点（runtime node）**：`Graph` 中的执行单元，负责数据定位、调用 `OpProvider`、参与序列化与初始化

仓库后续统一把 graph 中的执行单元称为 `Node`。因此：

- `BitNetLayerDefinition` 仍然表示模型层定义
- `EmbeddingNode`、`AttentionNode`、`SamplingNode` 等表示运行时 graph 节点

### 3.1 Node-only Graph

`RuntimeGraph` 当前只用于串联 `Node`。

本阶段假定：

- 一次单 `token` 推理的**主计算路径**只包含 `Node`
- 其他非 node 行为，如运行前检查、调试拦截、统计、未来同步逻辑，统一由 `Runtime` 在执行前后拦截处理
- 不在 graph 中建模 `IRuntimeNode`、`SyncNode`、`CopyNode` 等额外节点

这可以让当前设计先聚焦在主推理链路，而不是过早抽象运行时节点体系。

### 3.2 Node 负责 What，OpProvider 负责 How

每个 `Node` 负责：

- 定义本层输入输出地址
- 定义本层需要的权重地址
- 保留必要的外层逻辑和流程控制
- 通过 `MemoryManager` 解析本层所需数据
- 调用对应的 `OpProvider` 方法

每个 `Node` 不负责：

- 具体数值循环
- `CPU` / `SIMD` / `Tensor` 分支
- 各类热点算法实现

所有算法细节统一放入 `OpProvider` 体系。

补充约定：

- `OpProvider` 不直接引用 `MemoryManager`
- `OpProvider` 不直接引用 `BitNetSession` 或 `Runtime`
- `OpProvider` 只接收计算所需的数据与参数
- `Node` 或 `Runtime` 负责把数据准备好再交给 `OpProvider`

### 3.3 memory manager 只管理存储，不管理策略

`IMemoryManager` 只负责：

- 为多个推理 `session` 提供统一内存管理
- 作为模型权重加载的目标位置
- 用 `id + key` 定位数据
- 负责申请、获取、释放和清理

当前建议继续保持简单的管理器形态：

- 用 `IMemoryManager` 表达统一抽象
- 内置主内存实现命名为 `HostMemoryManager`
- 当前实现可以继续沿用 `GetMemory<T>` / `RequestMemory<T>` 这类方法级泛型访问方式
- 不在这一阶段把内存抽象提前拆分成过多接口层次

`IMemoryManager` 不负责：

- 算法实现
- layer 执行顺序
- backend 调度
- 跨 manager 的同步策略

### 3.4 同步是包装能力，不是底层存储职责

未来如需在多个 memory manager 之间同步同一份 `id + key` 数据，不将同步逻辑写进某一个 `IMemoryManager` 实现内部，而是通过包装器协调多个管理器共同工作。

这样可以保持基础内存管理接口简单，也便于未来引入 `CPU` / `GPU` offload。

### 3.5 显式初始化优先

仓库已经明确约定：`Node.Forward` 之前必须先执行显式 `Init`。

该约定在新设计中继续保留，用于承载：

- eager cache 预取
- 权重读取准备
- 未来 backend 预热

### 3.6 配置序列化与运行时状态分离

`Node` 需要支持从 graph 文件安全反序列化，因此建议继续保持无参构造，并通过属性级标记控制哪些配置参与序列化与反序列化。

建议原则如下：

- 只序列化显式标记的配置属性
- 不序列化运行时计算属性、缓存字段和临时状态
- graph 文件只恢复配置，不触发内部初始化
- `Init` 负责完成 node 的内部初始化和运行前准备

这样设计不仅便于 graph 反序列化，也有利于后续单元测试。测试可以在不依赖复杂构造参数或完整 runtime 组装的前提下，直接创建 `Node`、填充最小配置、显式调用 `Init`，再分别验证配置恢复、初始化行为和 `Forward` 逻辑。

推荐采用白名单式策略，而不是依赖默认“所有 public 属性都可序列化”的模式。这样可以避免把运行时状态错误写入 graph 定义。

### 3.7 Runtime 保持简单，优化放到外围

`BitNetRuntime` 的定位应长期保持稳定：

- 只负责 graph 执行
- 只负责必要的运行时拦截与装配
- 不内置复杂图优化逻辑
- 不承担 backend 专属图改写

未来如果需要按 `CPU` / `SIMD` / `GPU` 自动分析并优化 graph，应通过外围组件完成，例如未来的 `GraphOptimizer`。这类组件可以负责：

- 自动加载 graph
- 分析 node 依赖关系
- 识别可聚合区段
- 按 backend 生成优化后的执行图

但这些能力都不应进入 `BitNetRuntime` 本体。

### 3.8 逻辑 graph 与优化 graph 分离

当前 graph 应表达**逻辑层顺序**，而不是提前编码优化结构。

这意味着：

- 当前不实现聚合层或融合层
- 当前不让单个 node 隐式代表多个逻辑层
- 未来如果需要做 node 聚合，应由外围优化器重写 graph

这样可以保持：

- graph 文件语义稳定
- node 配置属性清晰可见
- 单元测试边界清晰
- backend 优化与逻辑建模解耦

### 3.9 访问追踪优先于版本管理

当前内存追踪只关注“访问请求可观测性”，不实现版本管理。

建议方向如下：

- `MemoryOwner` 作为可扩展的基础拥有者抽象
- 提供 `BeginTrace` 与 `EndTrace`
- 记录 trace 区间内的读写请求
- 通过只读访问与可写访问接口区分访问意图

当前阶段不做：

- 版本号推进
- 精确脏区检测
- 真实字节级修改比对

未来如果需要，可以在同一抽象上继续扩展版本和同步元数据。

---

## 4. 总体架构

建议的运行时依赖方向如下：

`BitNetRuntime -> RuntimeGraph -> Node`

对应的职责关系如下：

```text
BitNetRuntime
  -> RuntimeGraph
      -> Node
          -> IMemoryManager
          -> OpProvider
```

补充约定：

- `RuntimeGraph` 当前只包含 `Node`
- `Runtime` 自身可以在 node 执行前后插入检查、日志、调试或未来同步逻辑
- `Node` 负责数据定位，并协调 `IMemoryManager` 与 `OpProvider`
- `OpProvider` 统一处理 backend 差异，但不拥有 `IMemoryManager`
- `OpProvider` 不承担外部资源释放责任，不参与 `Dispose` 传递
- graph 中保留 `SamplingNode`，使 `Runtime` 不需要理解 `vocab`、采样策略或其他模型细节
- graph 的默认最终输出是字符串结果，但中间 node 仍可保留 token 和 logits 等逻辑输出

---

## 5. 核心组件职责

## 5.1 `BitNetRuntime`

`BitNetRuntime` 是推理执行协调器。

建议负责：

- 持有 `RuntimeGraph`
- 持有主 `BitNetSession`
- 持有主 `IMemoryManager`
- 按 graph 顺序执行各节点
- 在节点执行前后做必要的运行时拦截
- 选择或提供当前使用的 `OpProvider` 策略
- 作为默认的 node 包装装配点，为日志、性能统计和诊断附加包装层

不建议负责：

- 直接实现具体算法
- 承担 `Node` 内部数值循环
- 让 graph 承担非 node 节点抽象
- 承担复杂 graph 优化或 backend 专属图重写

### 运行时拦截的典型用途

运行时拦截可以覆盖但不限于：

- 执行前状态检查
- 调试观测
- tracing / profiling 钩子
- 未来的 memory sync
- 失败回滚或诊断上下文补充

这些行为属于 runtime 编排，不进入 graph。

### Node 包装扩展

`BitNetRuntime` 可以作为默认的 node 包装装配点，在不改变 graph 语义的前提下，为 node 附加运行时包装层，用于：

- 日志
- 性能统计
- tracing
- 异常诊断补充

同时，包装层机制本身应保留为公开扩展点，而不只服务于内置 runtime。这样第三方使用者可以在**不重写 `Runtime`** 的前提下，为 node 注入横切逻辑。

这里的关键边界是：

- wrapper 属于运行时增强，不属于 graph 序列化内容
- wrapper 默认不改变 graph 的逻辑结构
- wrapper 适合做横切能力，不要求承载全部 runtime 核心能力

---

## 5.2 `RuntimeGraph`

`RuntimeGraph` 当前阶段是一个**node 顺序容器**。

建议负责：

- 保存 node 的执行顺序
- 支持节点串联
- 为后续拓展更复杂图结构预留基础

当前不负责：

- 执行非 node 节点
- 表达 runtime side-effect 节点
- 引入 `IRuntimeNode`

当前也不负责：

- 直接表达聚合层或融合层
- 表达 backend 专属优化结构

第一阶段可以先实现为线性结构，例如：

- `EmbeddingNode`
- `RmsNormNode`
- `QKVProjectionNode`
- `AttentionNode`
- `ResidualNode`
- `FeedForwardNormNode`
- `FeedForwardNode`
- `FeedForwardResidualNode`
- `FinalNormNode`
- `LmHeadNode`
- `SamplingNode`

其中 `SamplingNode` 作为 graph 的最后一个 `Node`，负责将采样相关逻辑保留在模型执行链中，从而让 `Runtime` 继续保持模型无关。

未来如果 graph 末端逻辑变复杂，应优先按需拆分为多个逻辑 node，而不是现在就引入聚合层。后续如需性能优化，再由外围 `GraphOptimizer` 做聚合或改写。

---

## 5.3 `Node`

`Node` 的目标是变成“薄节点”。

建议每个 `Node` 只保留：

- 本层输入地址
- 本层输出地址
- 本层权重地址
- 必要的维度与流程校验
- 通过 `MemoryManager` 解析输入输出容器
- 调用对应 `OpProvider` 的入口

不建议 `Node` 保留：

- `RmsNorm` 数值计算
- `MatMul` 或投影循环
- backend 选择分支
- 大量底层 buffer 操作细节

### 当前 `Node` 约定保留

- `Init` 必须显式调用后才能 `Forward`
- `EnableCache` 这类与权重读取策略直接相关的配置可以继续保留在 node
- `InferenceConfig` 仍然可以保留，但 backend 的最终执行逻辑应逐步收敛到 `OpProvider`

### `Node` 配置属性约定

为了支持 graph 文件反序列化，建议将 `Node` 属性分为两类：

1. 配置属性
   - 参与序列化 / 反序列化
   - 由 graph 文件恢复
   - 仅包含基础类型、枚举和必要的 `struct` 配置

2. 运行时属性
   - 不参与序列化 / 反序列化
   - 用于缓存、内部状态、临时计算结果和初始化标记

建议通过 attribute 明确标记配置属性，例如使用项目内自定义配置特性或显式忽略特性，而不是依赖约定猜测。

推荐采用“只有被标记的属性才参与配置序列化”的白名单模式。

### `Node` 配置暴露

`Node` 的一部分配置属性不仅用于 graph 序列化，也可以作为对外可配置项暴露给 runtime 或上层宿主。

当前更推荐的方式是：

- 通过属性级 attribute 声明哪些配置可以被外部读取和配置
- 由 `Runtime` 或外围宿主读取这些元数据后，对外展示或转发配置

当前不建议先引入全局 runtime `property bag` 来承载 node 专属配置，因为这会削弱 node 的边界，并把类型安全退化为字符串键值管理。

### `Init` 的职责边界

graph 反序列化完成后，`Node` 仅处于“配置已恢复”的状态，并不保证已经具备可执行性。

`Init` 应负责完成以下内部准备：

- 配置合法性校验后的内部建模
- 权重预读或缓存建立
- backend 相关预热
- 只应在运行前执行一次的初始化操作

这种分层也让测试边界更清晰：

- 配置序列化测试只验证配置属性
- `Init` 测试只验证内部准备是否完成
- `Forward` 测试只验证计算行为和输出结果

这意味着：

- 反序列化负责恢复配置
- `Init` 负责恢复运行时内部状态
- `Forward` 只在 `Init` 成功后允许执行

---

## 5.4 `OpProvider`

当前 `MathHelper` 承担了较多计算职责。下一阶段建议逐步重构为 `OpProvider` 体系。

建议的目标命名：

- `IOpProvider<TContainer>`
- `OpProviderBase<TContainer>`
- `CpuOpProvider<TContainer>`
- `SimdOpProvider<TContainer>`
- `TensorOpProvider<TContainer>`

### 职责建议

`OpProviderBase<TContainer>` 负责：

- 提供统一算子入口
- 放置默认正确性实现
- 放置共享参数校验逻辑
- 放置公共辅助逻辑

`OpProvider` 的边界约束如下：

- 不持有 `MemoryManager`
- 不引用 `BitNetSession`
- 不引用 `Runtime`
- 不负责任何外部资源释放
- 尽量保持为无状态辅助计算组件

派生类负责：

- 按 backend 覆盖热点实现
- 在不改变语义的前提下做 `CPU` / `SIMD` / `Tensor` 优化

### 设计原则

采用“**基类保底实现 + 派生类按需优化覆盖**”模式。

这样可以避免：

- 三种 backend 各自维护三套完整算法副本
- node 内部散落 backend 分支
- 算法逻辑分散到多个调用点
- 因持有 `MemoryManager` 而引入生命周期和 `Dispose` 责任传递

### 数据输入方式

`OpProvider` 不应自己去 `MemoryManager` 取数据。

建议的数据流是：

- `Node` 先根据地址从 `IMemoryManager` 获取输入、输出和权重容器
- `Node` 再把这些容器或其视图传给 `OpProvider`
- `OpProvider` 只对传入数据执行计算

这样可以保证 `OpProvider` 后续如果需要改为静态类或静态方法集合，整体调用方式仍然成立。

### 典型算子

建议逐步将以下计算迁入 `OpProvider`：

- `RmsNorm`
- `Add`
- `Softmax`
- `QKV` 投影相关计算
- `FeedForward` 内部核心算子
- `FinalNorm`
- `LmHead` 投影相关计算

第一阶段建议先从 `RmsNorm` 迁移，作为完整样板。

---

## 5.5 `BitNetSession`

`BitNetSession` 继续作为一次推理会话的状态载体。

建议负责：

- 持有 `BitNetModel`
- 持有会话 `Id`
- 维护会话级输出状态
- 暴露运行时需要的主要缓冲区访问入口
- 绑定外部传入的 memory manager

当前仓库已有重要约定，后续设计继续遵循：

- 不在 `BitNetSession` 内部自行构造 `BitNetMemoryManager`
- 由外部传入共享内存管理器
- 简单标量，如 `CurrentToken`，仍作为普通属性保留
- `BitNetMemoryManager` 优先用于大块内存，而不是简单标量状态
- 输出缓冲区采用懒初始化的 get-only 风格，并在原地写入

---

## 5.6 `IMemoryManager` 与 `HostMemoryManager`

建议将当前 `BitNetMemoryManager` 逐步演进为稳定的 `IMemoryManager` 抽象，并提供一个内置的主内存实现：`HostMemoryManager`。

### 核心职责

- 为多个推理 `session` 提供共享内存管理
- 作为模型权重加载的目标位置
- 按 `id + key` 管理命名内存块
- 提供当前仓库所需的获取、申请、释放与查询能力
- 管理资源生命周期

### 设计原因

当前最重要的不是把 `MemoryManager` 抽象得足够复杂，而是先忠实表达仓库的真实需求。

当前这套设计主要服务于两件事：

1. 管理多个推理 `session` 的运行期内存
2. 为模型权重提供统一的加载目标位置

因此，这一层更适合先保持为“按作用域和 key 管理命名内存块”的统一抽象，而不是过早拆成复杂的 owner / view / container 体系。

这里的“抽象”优先使用接口，而不是先引入共享基类。只有在未来多个实现确实出现大量公共代码时，再考虑补充 `MemoryManagerBase` 一类的共享基类。

### 当前形态建议

- 使用 `IMemoryManager` 作为统一接口
- 继续允许使用 `GetMemory<T>` / `RequestMemory<T>` 这类方法级泛型 API
- 让 `HostMemoryManager` 表示当前基于计算机主内存的内置实现
- 在真正需要异构容器抽象之前，不强行引入更复杂的类型层次

### `MemoryOwner` 追踪扩展

考虑到所有 node 都通过内存容器读写数据，未来需要在内存容器层提供访问追踪能力，以便了解哪些数据被读取、哪些数据被申请写入。

当前建议方向：

- `MemoryOwner` 采用可扩展的基础抽象
- 通过 `BeginTrace` 和 `EndTrace` 记录 trace 区间内的访问请求
- 通过只读访问与可写访问接口区分访问意图

这里记录的是：

- 读请求
- 写请求
- 调用来源
- 请求发生顺序或时间

当前不要求：

- 记录真实字节变化
- 维护版本号
- 在第一阶段实现复杂一致性模型

这样可以先满足调试、诊断和未来优化分析需求，并为后续扩展保留空间。

---

## 5.7 `MemorySync` 与同步包装器

未来为了支持多个 memory manager 之间的数据同步，建议增加独立同步抽象，例如：

- `IMemorySync`
- `SynchronizedMemoryManager`

### 职责建议

`SynchronizedMemoryManager`：

- 对外仍表现为 `IMemoryManager` 的同步包装实现
- 内部持有多个实际 manager
- 维护主 manager 与次级 manager 间的同步关系
- 根据策略执行写主、副本同步、显式同步或延迟同步

### 设计约束

同步逻辑不进入基础 `MemoryManager` 实现内部。

这样可以让：

- `HostMemoryManager` 保持简单
- 单 backend 路径不承担额外复杂度
- 未来 `CPU` / `GPU` offload 有自然扩展点

---

## 6. 数据寻址模型

为了统一 node 与算子之间的数据访问，建议引入显式地址对象，而不是在各处直接散落字符串。

建议的最小地址模型：

```text
MemoryAddress = (Id, Key)
```

其中：

- `Id`：当前会话、层实例或其他作用域标识
- `Key`：具体的数据槽位名称

### 推荐使用方式

- `Node` 负责定义和选择地址
- `Node` 负责根据地址从 `IMemoryManager` 获取容器
- `OpProvider` 只接收已解析的数据容器或视图

对于 graph 最终输出，建议仍以 node 方式表达。换句话说，graph 的最后一个逻辑节点可以是 `SamplingNode` 或未来拆分出的相关 node，而不是把这些模型相关逻辑塞进 `Runtime`。

### 示例 key 语义

- `Embedding`
- `RmsNorm`
- `QKVQuery`
- `QKVKey`
- `QKVValue`
- `AttentionOutput`
- `FeedForwardOutput`
- `FinalNormOutput`
- `Logits`

这与当前 `BitNetSession` 中已有的 key 常量体系是一致的，可逐步平滑迁移。

---

## 7. 执行流程

单 `token` 推理建议采用以下主流程：

1. `Runtime` 准备会话与输入状态
2. `Runtime` 按 graph 顺序执行各 `Node`
3. 每个 `Node` 在 `Forward` 中解析本节点所需地址
4. `Node` 通过 `IMemoryManager` 获取输入、输出和权重容器
5. `Node` 调用 `OpProvider` 完成数值计算
6. graph 末端的 `SamplingNode` 或相关 node 生成最终结果
7. `Runtime` 返回 graph 的最终输出

### 一个简化示意

```text
Runtime
  -> Graph
      -> EmbeddingNode
      -> RmsNormNode
      -> QKVProjectionNode
      -> AttentionNode
      -> ResidualNode
      -> FeedForwardNormNode
      -> FeedForwardNode
      -> FeedForwardResidualNode
      -> FinalNormNode
      -> LmHeadNode
      -> SamplingNode
```

注意：`SamplingNode` 是 graph 中的 `Node`，而不是运行时外部行为。

---

## 8. 为什么当前不引入 `IRuntimeNode`

本阶段明确不引入 `IRuntimeNode`。

原因如下：

- 当前 graph 的唯一目标是串联 `Node`
- 单 `token` 主推理路径可以视为纯 node 序列
- 其他操作已经可以通过 runtime 拦截实现
- 过早引入统一 runtime 节点，会使设计提前承载还未验证的复杂度

同时，未来的复杂优化不应通过 `IRuntimeNode` 进入 `Runtime` 本体，而应通过外围 graph 重写或包装机制实现。

因此，本阶段采用更直接的约束：

- `Graph` 只管理 `Node`
- `Runtime` 管理所有非 node 行为

如果未来确实出现大量必须进入图执行体系的非 node 操作，再重新评估是否需要引入节点抽象。

---

## 9. 与现有类型的映射建议

当前仓库中的主要类型与目标方向可按下表理解：

| 当前类型 | 目标方向 | 说明 |
|---|---|---|
| `BitNetRuntime` | `BitNetRuntime` | 继续保留，补全 graph 执行与 runtime 拦截职责 |
| `BitNetSession` | `BitNetSession` | 继续保留，作为统一会话状态载体 |
| `BitNetMemoryManager` | `IMemoryManager` / `HostMemoryManager` | 先保留现有实现，并逐步收敛成 `IMemoryManager` 抽象与内置主内存实现 `HostMemoryManager` |
| `MemoryOwner` 体系 | 可追踪访问的容器抽象 | 未来加入 `BeginTrace` / `EndTrace`，当前不实现版本管理 |
| `MathHelper` | `OpProviderBase` 体系 | 逐步把算法和 backend 分支迁入算子提供者 |
| 各个 `*Node` | 各个 `*Node` | 作为 graph 中的运行时执行节点，保留现有节点类型并持续瘦身 |
| `SamplingNode` | graph 末端 `Node` | 保持采样与最终输出逻辑在 graph 内，而不是放入 runtime |
| `InferenceConfig` | `InferenceConfig` | 暂时保留，后续逐步与 `OpProvider` 协同 |

未来还可新增外围组件，例如：

- `GraphLoader`
- `GraphOptimizer`
- node wrapper 提供者

这些组件用于扩展 runtime 生态，但不改变 `BitNetRuntime` 本体保持简单的原则。

这里的关键不是立即重命名所有文件，而是先稳定职责边界，再逐步迁移实现。

---

## 10. 最小可行迁移顺序

建议按以下顺序推进，而不是一次性重写：

### 阶段 1：收敛算子职责

- 保留现有 `Node` 结构
- 抽出 `IOpProvider` 与 `OpProviderBase`
- 先迁移 `RmsNorm` 计算
- 让 `RmsNormNode` 只保留地址与外层流程

### 阶段 2：收敛内存抽象

- 从当前 `BitNetMemoryManager` 提炼 `IMemoryManager` 接口
- 提供 `HostMemoryManager` 作为内置主内存实现
- 继续复用当前 session key 体系

### 阶段 3：补全 runtime 主链路

- 为 `BitNetRuntime` 增加 node-only `RuntimeGraph`
- 把单 `token` 推理串成完整执行顺序
- 把 `SamplingNode` 作为 graph 末端 `Node` 接入

### 阶段 4：为异构内存预留扩展

- 增加 `IMemorySync`
- 增加同步包装器
- 先做 manager 间同步样板，不急于落地真实 `GPU` offload

### 阶段 5：为外围优化和观测预留扩展

- 保持 `BitNetRuntime` 简单
- 在外围引入 `GraphOptimizer` 等组件
- 允许 runtime 默认包装 node，也允许第三方通过公开扩展点附加 wrapper
- 在 `MemoryOwner` 层增加 trace 能力，为后续优化分析提供数据基础

---

## 11. 本阶段不做的事情

为了控制复杂度，以下内容明确不在当前重构范围内：

- 引入 `IRuntimeNode`
- 在 graph 中表达非 node 节点
- 完整的多 `token` runtime / `KV Cache` 调度重构
- 直接落地完整 `CPU` / `GPU` offload
- 为所有 node 一次性改造成最终形态
- 立即实现聚合层或融合层
- 在当前阶段引入 runtime 级全局 property bag 来承载 node 配置
- 当前就为内存追踪引入版本管理
- 因为文档设计而提前修改无关测试或层次结构

---

## 12. 总结

本阶段的目标可以概括为一句话：

> 让 `Runtime` 负责执行编排，让 `Node` 负责数据定位，让 `OpProvider` 负责全部计算，让 `IMemoryManager` 负责存储，并通过 node-only graph 串起单 `token` 推理主链路。

这套设计有几个直接收益：

- 运行时职责更清晰
- node 更薄，更容易维护
- backend 优化点集中到 `OpProvider`
- `OpProvider` 不承担外部资源生命周期，后续更容易保持为无状态 helper
- `IMemoryManager` 先围绕 session 内存和权重加载这两个核心职责保持简单，`HostMemoryManager` 作为当前内置主内存实现
- graph 可继续保持逻辑语义稳定，未来优化交给外围 `GraphOptimizer`
- `SamplingNode` 保持在 graph 中，使 `Runtime` 继续模型无关
- node 配置属性既可序列化也可对外暴露，仍保持在 node 边界内
- 内存访问 trace 可为调试、诊断和未来优化提供基础数据
- 为未来同步包装和异构后端留出明确扩展点

在当前仓库阶段，这是一条比引入更多运行时节点抽象更稳、更可落地的演进路线。
