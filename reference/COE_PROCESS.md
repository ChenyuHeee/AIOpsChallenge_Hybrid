# Chain-of-Event（COE）在项目中的实现说明

面向未读论文的读者，解释我们如何把 COE（事件链/调用链推理）落地到 Hybrid 流水线。

## 目标与思路
- 目标：从 metrics/logs/traces 中提取事件与调用关系，构建轻量事件图，辅助解释“异常如何传播、为什么是该组件”。
- 思路：先把遥测时间对齐，再从组件/调用/操作名提取有向边，形成事件图；图信息被 Graph/Trace 专家使用，并在共识/理由阶段提供路径证据。

## 数据流与接口
- 输入：按时间窗过滤后的 metrics/logs/traces（parquet）。
- 构建：`TelemetryLoader._build_event_graph(metrics, logs, traces)` 生成 `event_graph: Dict[str, List[str]]`。
- 使用：`GraphSpecialist`/`TraceSpecialist` 引用 `telemetry.event_graph` 产出图相关假设；共识与 LLM 理由可引用这些证据。

## 事件图构建（代码对应）
位置：`contest_solution/data/loader.py`
- 数据读取：`TelemetryLoader.load` 依次读 `metric-parquet` / `log-parquet` / `trace-parquet`，按时间窗过滤（`_filter_by_window` 带 padding）。
- 边生成 `_build_event_graph`：
  - metrics/logs：从组件列（component/service/pod/hostname/k8_pod 等）提取节点；如有操作列（operation/spanName 等），添加 component→operation 边。
  - traces：从 service 列提取节点；如有 peer.service，生成 service→peer 边；同样可加 service→operation 边。
  - 归一化：`_normalize` 统一大小写，去掉 hipstershop./service=/svc- 噪声，过滤空/unknown。
  - 去重：同一源的目标列表去重，形成轻量邻接表。
- 输出：`TelemetryFrames(event_graph=graph)`，与 metrics/logs/traces 一起返回。

## 专家如何利用事件图
- `GraphSpecialist`（`agents/specialists.py`）
  - 输入：`telemetry.event_graph`。
  - 逻辑：对每个源节点，按出度计算置信度（1.0 + 0.2*出度），有 hints 则非 hint 源降权；生成 `Hypothesis`（modality=graph，summary=传播指向）。
- `TraceSpecialist`
  - 虽主要基于 traces 的时延峰值，但可结合图结构理解跨度分布；同文件中使用 `telemetry.traces` + service 列，输出 latency 证据。

## 与 SOP/管线的衔接
- SOP 阶段：`ConstructGraph`（见 `contest_solution/sop.py`），目标是“图中有候选因果链”。
- 上游：`planner` 给出时间窗；`TelemetryLoader` 按窗口加载与对齐；`_build_event_graph` 构建图。
- 中游：Graph/Trace 专家用图或链路信息生成 `Hypothesis`（组件、confidence、evidence）。
- 下游：`consensus` 融合这些假设；`reasoning`/LLM 可引用图证据说明“异常路径/传播链”。

## 配置与扩展点
- 时间窗 padding：`TelemetryLoader(window_padding=...)`（默认 45 分钟）；可调以涵盖抖动。
- 列名适配：`_first` 选择列时已列出常见别名，如需新格式可扩展候选列。
- 边类型扩展：可加入错误码、异常关键词共现边，或 trace span 的父子关系，增强因果感。
- 权重/评分：GraphSpecialist 目前简单用出度评分，可改成基于异常密度/路径长度的打分。

## 文件索引
- 事件图构建：`contest_solution/data/loader.py` (`_build_event_graph`)
- 图/链路专家：`contest_solution/agents/specialists.py` (`GraphSpecialist`, `TraceSpecialist`)
- SOP 定义：`contest_solution/sop.py` 的 `ConstructGraph` 步骤

## 输出示例（结构）
- `event_graph`（邻接表）：`{"checkoutservice": ["paymentservice", "redis"], "frontend": ["productcatalogservice"], ...}`
- 图假设：`Hypothesis(component="checkoutservice", source="GraphSpecialist", confidence=1.4, evidence=["[graph] Propagation to paymentservice, redis"])`

COE 在项目中承担“把遥测转成可解释路径”的角色，帮助 downstream 在理由中给出“异常如何传播、为何怀疑该组件”。
