# Hybrid 端到端运作说明（从原始遥测到提交答案）

## 1. 最终产出是什么

对每个 case，系统最终产出一条结构化答案记录，核心字段是：

- `uuid`：题目唯一标识
- `component`：最终预测的根因组件（必须与评测要求的组件 token 严格一致）
- `reason`：简短原因（会刻意包含若干“证据关键词”以提高可匹配性）
- `reasoning_trace`：步骤化的分析过程（含证据摘录/摘要）

## 2. 输入从哪里来、长什么样

每个 case 至少包含：

- `uuid`
- `query`：自然语言描述（可能包含组件名、症状、时间信息）

系统还会在遥测侧读取四类数据：

- 指标（metrics）：时间序列 KPI（多数为 wide 表：时间 + 标识列 + 多个数值 KPI 列）
- 日志（logs）：文本日志（message/level/pod/node 等字段可能存在或缺失）
- 调用链（traces）：span 级别记录（duration、process tags、span tags 等）
- 事件图（graph）：由 metrics/logs/traces 抽取到的“传播/关联”结构（source → targets）

其中“事件图”不是单独的数据源，而是由加载阶段从已读入的数据中构建出来的中间产物。

## 3. 总体流程概览

对每个 case，管线按如下顺序运行（概念级）：

1) 规划（Planner）：从 `query` 抽取关键词、组件提示、时间窗，并生成 SOP 分析步骤；同时检索“专家知识片段”作为 reasoning 提示。

2) 读取与预处理（Telemetry Loading）：按时间窗读取三类遥测，并在可配置条件下读取“基线窗口（baseline windows）”用于对比；同时构建 event graph。

3) 四类专家并行产出候选（Specialists → Hypotheses）：
- Metric 专家：为 service/node/TiDB 组件打分并输出异常摘要
- Log 专家：按错误模板/消息为组件打分并输出摘要
- Trace 专家：按延迟尾部和（可选）对比基线 p95 为组件打分并输出摘要
- Graph 专家：按传播出度等简单启发式为组件打分并输出摘要

4) 融合与排序（Consensus）：把各专家的候选做加权投票/规则重排，得到一个有序的 `ranked_components`。

5) 推理文本生成（Reasoning / LLM）：给 LLM 提供候选榜单与证据 JSON，要求它用严格 JSON 输出 `component/reason/analysis_steps`；若 LLM 不可用或输出不合规，则回退到“共识 top1 + 证据拼接”。

6) 校验与增强（Validator）：统一格式、长度限制、必要证据 token 增强、并生成最终的 `reasoning_trace`，确保可被评测程序接收。

为了便于理解，下文会反复用到两个中间对象：

- Evidence：一条“可引用的证据摘要”，包含 modality、summary、score。
- Hypothesis：针对某个 component 的一组证据与总体置信度（confidence），并带来源信息（来自哪个专家）。

下面按“每一类数据如何被处理”展开。

## 4. Metrics：从 wide 表到组件异常证据

### 4.1 识别“要给谁打分”（service / node / TiDB 三路并行）
指标数据常见的问题是：同一行里可能同时包含服务标识和节点标识，如果只用一种标识会遮蔽另一种根因。

因此做法是：

- 选择一条“service-like”标识序列（优先使用 object_id/pod/component/service/instance 等列，并做归一化）
- 选择一条“node-like”标识序列（优先使用 node_name/hostname 等列，并做归一化）
- 对 TiDB，若存在 namespace/object_type，会把 object_type 映射成 replica 级的 TiDB 组件 token（例如 tikv/pd/tidb 分别映射到对应组件），避免出现系统性 token 不匹配

这里的“归一化”非常关键，本质上是在做 token 对齐与去噪：

- 去掉常见前缀/装饰（例如 service=、svc-、某些命名空间前缀），统一大小写与空白。
- 丢弃无意义 token（unknown/null/none 等）以及不会出现在评测答案里的 token（例如裸 IP:port）。
- 节点 token 需要精确保留（例如 aiops-k8s-XX），不能随意裁剪。
- replica token（如 xxxservice-0）默认保留后缀：因为很多 ground truth 是 pod 级别；把后缀裁掉会导致系统性 LA 损失。
- TiDB token（tidb-...）默认保留完整 replica 名称：同样是为了避免系统性 token mismatch。

### 4.2 选择“哪些列是 KPI”（数值列筛选）
从表里剔除时间列、标识列、以及一些明显非 KPI 的字段，只保留数值 KPI 列作为候选。

此外还有一个“防串扰”的约束：

- node 指标（例如 node_*）只用于给 node 组件打分。
- service 指标用于给 service/pod/TiDB 这类非 node 的组件打分。

目的：避免“node KPI 被服务聚合稀释”或“服务 KPI 误把 node 拉高”。

### 4.3 为每个（组件 × KPI）计算异常严重度（severity）
对每个组件、每个 KPI：

- 在故障窗口内取该组件的 KPI 值序列
- 计算异常严重度（核心是鲁棒 z-score；并对 usage_rate/cpu/memory 等比率类指标加入“相对跳变”修正，避免 padding 稀释或 std 过小造成误判）
- 若启用 baseline windows：再计算“故障 vs 基线”的对比严重度（以基线分布为参照，常用 base_mean/base_p95 或相对变化）
- KPI 名称会映射到权重（例如 node_cpu/node_memory/timeout/latency/error_ratio 等更高权重），最终分数约为 `severity × weight`

计算严重度时有两个常见坑会被特别处理：

- padding 稀释：如果时间窗被扩展，峰值可能被更多“正常点”稀释，导致 z-score 变小。
- std 过小：某些稳定高位的比率类 KPI 可能出现“std 很小但水平很高”，单纯 z-score 可能误判。

所以对 usage_rate/cpu/memory 等比率类 KPI，会额外引入“相对跳变”（peak 相对 median 的增幅）并映射到 z-score 量级，取两者较大值。

### 4.4 过滤阈值与证据摘要
- 对一般组件使用一个最小分数阈值；对 TiDB 组件使用更低的最小阈值（避免 TiDB under-score）
- 对 TiDB 组件还会乘一个温和的倍率（用于“有明确 TiDB 指标证据时”让其更有竞争力）
- 每条 evidence 会写成可读摘要：
  - 无基线：如 “某 KPI spike/avg”
  - 有基线：如 “fault_p95 vs base_mean/base_p95”

这里的“阈值”与“倍率”本质是在做两件事：

- 阈值：把噪声候选尽早砍掉，减少后面融合阶段的干扰。
- 倍率：在“证据真的存在”的前提下，让某些类别（例如 TiDB 的 metrics）不至于因为先验偏置而长期 under-score。

注意：倍率只对“有明确证据的组件”生效；它不是硬抬所有 TiDB 候选分数。

### 4.5 输出形式（Hypothesis）
对每个组件聚合多条 metrics evidence，输出：

- `component`
- `confidence`：
  - node：取最强 evidence（避免均值稀释强节点信号）
  - TiDB：取最强 evidence 并允许更高 cap（避免 TiDB 被过低封顶压制）
  - 其他：取 evidence 平均
- `evidence[]`：每条含 modality=metrics、summary、score

一个组件的 metrics 证据通常会包含多条 KPI 摘要；下游融合并不需要看到整段原始序列，而是依赖这些“可比较的摘要分数”。

## 5. Logs：从原始 message 到“模板增量”证据

日志侧要解决两个挑战：

- 大量噪声（非错误日志、重复日志）
- 故障窗口“哪些模板突然变多”比单条 error message 更有诊断价值

### 5.1 基础过滤（错误优先）
默认只保留包含 error/exception/fail/timeout/critical 等关键词的日志。

采样顺序也很关键：

- 先用关键词过滤（尽量保留稀有但关键的错误）
- 再做采样（控制上限，避免全量扫描过慢）

另外存在一个可选模式：不只看 error 日志，而是允许把“非错误模板频率的变化”也纳入对比（适合某些压力/流量型故障）。

### 5.2 模板化（template）
把日志 message 做归一化成“模板”，核心是把不稳定 token 替换掉：

- UUID、IP、十六进制串、长数字、空白等

模板长度会被截断以稳定分组。

模板化的目标不是“完美抽象语义”，而是让同类日志能稳定聚类，从而做频率对比。

### 5.3 （可选）与 baseline 做“模板频率对比”
当启用 baseline 对比时：

- 在故障窗口与基线窗口分别统计（component, template）的出现次数
- 先把日志行映射到 component：通常来自 pod 字段；在特定条件下也会把 node 字段当作 component（避免 node 线索被完全丢掉，但默认更保守，因为 node token 很容易在全量日志里“话太多”）
- 在故障窗口与基线窗口分别统计（component, template）的出现次数
- 用 `log1p(count_fault) - log1p(count_base)` 得到增量 delta
- 只保留 delta 显著为正的模板（表示“故障期突然变多”）
- 每个组件取 top 模板 delta 作为 evidence，并附加 1 条原始日志样本提高可读性

在不启用 baseline 时，则回退为“直接把错误日志 message 作为 evidence”。

这两种模式的差异可以理解为：

- baseline 对比：更适合“故障期出现了新的错误模式/频率显著变化”。
- 直接 message：更适合“故障期的关键错误很少但很明确”。

### 5.4 输出形式
对每个组件聚合 logs evidence，给一个上限封顶的置信度（奖励重复证据以便打破多组件平分）。

## 6. Traces：从 span duration 尾部到服务/节点/Pod/边的证据

Trace 侧的关键点是：

- 只看尾部（anomalous tail）以控规模
- 尽量输出与评测组件 token 兼容的标识（service、pod、node）

### 6.1 选取 duration 并聚焦尾部
- 选择 duration 类字段作为延迟
- 只保留延迟最大的 top-K（上限约几千条），避免窗口缺失时全量爆炸
- 如果 duration 的数量级像微秒，会自动换算到毫秒

只看尾部的效果是：

- 运行时间可控
- 证据更集中（更像“真正异常”的 span）
- 但也会引入偏置：如果故障表现为“大面积中等变慢”，尾部抽样可能低估它。这类情况通常需要依赖 metrics/logs 来补足。

### 6.2 （可选）基线对比：base p95
启用 baseline 对比时：

- 计算每个服务在基线窗口的 p95 延迟
- 同时计算每个节点在基线窗口的 p95 延迟（为 node fault 准备）
- 故障窗口内的 span 会用 `ms / base_p95` 的倍率来打分，分数形态为 `log1p(ratio) × scale`

引入 base_p95 的意义：把“本来就慢的服务”与“突然变慢的服务”区分开。

### 6.3 组件归因：service / pod / node
对每条高延迟 span：

- 提取 service（process.serviceName 或同义字段）
- 提取 pod（process tags 中的 pod 名称），若 pod 是 replica 形式（如 service-0），会比 service 证据略“更强”，以鼓励更细粒度定位
- 提取 node（process tags 中的 node_name；明确过滤掉评测不接受的 master 形式，避免稳定拉低 LA）

这里的策略倾向是：

- pod 证据略强于 service：因为评测往往在 pod 粒度给 GT。
- node 证据独立累积：为 node-fault 场景提供“跨服务一致”的支撑。

### 6.4 处理“边”（a→b）但不直接依赖边作为最终提交
从 client span 的 tags 提取 peer 信息，构造 `client→peer`。

重要细节：评测对边的处理更像“端点集合”，因此 trace 证据会把边的 evidence 归档到左右端点组件名下（left/right），从而让融合阶段仍能选出合法组件 token。

直观理解：

- 边信息更像“传播证据”，可以解释因果链路
- 最终提交仍需是单个合法 component token，所以边通常只作为“端点组件的加权证据”，而不是最终答案本身

### 6.5 输出形式
对每个组件输出 traces evidence（数量会截断），置信度通常是 evidence 的平均分。

## 7. Graph：从“传播结构”到轻量候选

事件图是“source → targets”的邻接表形式。

Graph 专家主要做两件事：

- 只对“看起来像组件 token”的节点打分（服务、节点、TiDB、部分常见依赖）
- 分数与出度相关：出度越大，越像传播源

并且会做一些简单的保守约束：

- 只接纳“看起来像组件”的 token（避免把操作名/span 名当作组件候选）
- 如果 query 里有明确组件提示，会对“不在提示集里”的节点做轻微降权（避免图结构把无关节点推太高）

可选地，它也会生成少量 edge 候选（source→target），但这通常只作为辅助证据。

## 8. Hypothesis Bank：把多模态证据装箱

Specialists 的输出会按组件名聚合成一个“假设库”，每个组件下可能有多个 hypothesis（来自不同专家），每个 hypothesis 内含多条 evidence。

可以把它理解为一个映射：

- key：component
- value：来自不同专家的 hypothesis 列表

每条 evidence 都会尽量写成“能被复述、能被对比、能被关键词匹配”的短摘要。

这个结构的价值在于：

- 共识阶段做融合投票时可以看到“哪个组件被哪些模态支持”
- LLM 推理阶段可以拿到结构化 evidence JSON，避免自由发挥

## 9. Consensus：从候选堆到有序榜单

共识融合的目标是把“topK 里有 GT”尽量推进到“top1 就是 GT”。

它的核心机制可以概括为：

1) 加权投票：不同专家/模态有不同权重，把 evidence 分数累积到组件总分。

2) 模态支持度跟踪：记录每个组件由哪些模态支持（metrics/logs/traces/graph）。

3) 规则层重排（可配置开关）：
- 抑制“热门服务”在证据弱时的坍缩
- 对 Planner 提示过的组件做加分
- 偏好 replica：当 evidence 是 pod 级或 query 明确指向某个 replica 时，倾向 replica 组件 token
- 过滤过弱的节点候选：避免榜单里出现太多相近 node，干扰 LLM 选择

这里可以用一句话总结：

> 先用“可加和的证据分数”把候选拉到一个大致合理的排序，再用少量规则把常见的系统性失分点（热门服务坍缩、replica 粒度错误、node 候选挤爆榜单）纠正回来。

最终输出 `ranked_components = [(component, score), ...]`，并保留 supporting_evidence 与模态支持信息给后续阶段使用。

## 10. Reasoning（LLM）：把“榜单+证据”变成可解释答案

LLM 并不是自由推理，而是一个强约束的结构化生成器：

- 输入包含：query、SOP 步骤、检索到的专家知识片段、共识榜单（会把 node/service 分组展示）、以及每个组件的 evidence JSON
- 要求输出严格 JSON：`component`、`reason`、`analysis_steps`
- `component` 必须从候选集合中选一个，禁止发明新名字
- `reason` 会被要求尽早包含 1-3 个“证据 token”（例如某个 KPI 名称或错误关键词），以提高评测侧的关键词匹配稳定性

LLM 在这里的角色边界是：

- 更像“把证据组织成可解释文本”的写作器
- component 的选择被强约束在候选集合内，避免它凭语义臆测出不存在的 token

如果 LLM 不可用或输出缺字段：

- 回退到“共识 top1 组件 + 证据拼接”的可解释文本

## 11. Validator：保证可提交、可匹配、可读

最终阶段会做三类事情：

- 合规性：字段存在、长度/词数限制、组件 token 规范化
- 稳定性：对可能导致评测失败/失分的格式做处理（例如避免把 edge 组件直接作为最终提交；必要时回落到端点组件）
- 可匹配性增强：把关键证据 token 放到 reason 的靠前位置；生成结构化 reasoning_trace（3-7 步）并注入证据摘要

你可以把它理解为“最后一道把关 + 文本工程”：

- 防止 token/格式问题导致直接 0 分
- 在不改变 component 的前提下，尽量把强证据的关键词暴露在 reason/trace 里

## 12. 为什么会出现“GT 在 topK 但 top1 不是 GT”

从机制上看，常见原因是：

- 不同模态证据量纲不同：例如 metrics 很强但 logs/traces 对某个服务更“话多”，导致融合分数偏移
- 跨类型冲突：node/service/TiDB 同时都有信号时，规则层如果偏向某一类，会让另一类从 top1 掉到 topK
- token 颗粒度不一致：service vs pod(replica) 的竞争，如果不做 replica 偏好，容易选到错误粒度

因此，改进 LA 的关键通常不是“再多召回一些候选”，而是：

> 让融合规则与证据结构更稳定地把“证据最强且 token 正确”的组件推到 top1。

## 13. 开关/策略对 top1 的影响图谱（跨 kind 冲突视角）

这一节不列任何工程细节，只从“机制→会把 top1 推向哪里→常见副作用”来理解。

### 13.1 Baseline 对比相关

- 开启 metrics baseline 对比：
  - 典型收益：把“本来就高/慢”的 KPI 与“故障期突变”区分开，减少误报。
  - 常见副作用：如果基线窗口本身不干净（含轻微异常），可能降低真实故障的对比强度。

- 开启 logs baseline 模板对比：
  - 典型收益：把“故障期模板突然增多”的组件推上来，尤其是错误模式切换明显时。
  - 常见副作用：当大量服务同步打印相似错误模板时，会出现“话多的服务”集体上浮，压制真正根因。

- 开启 traces baseline（base_p95）：
  - 典型收益：把“突然变慢”的服务/节点识别出来，减少对“天然慢服务”的偏置。
  - 常见副作用：如果 trace 样本量不足或采样偏差大，base_p95 会不稳定，导致排序抖动。

### 13.2 粒度（service vs pod/replica）相关

- 偏好 replica（pod 级 token）：
  - 典型收益：当 GT 是 replica（如 xxx-0）时，能把“正确粒度”从 topK 推到 top1。
  - 常见副作用：在证据只到 service 粒度时，过强的 replica 偏好可能把某个“无充分证据的 pod”抬到 top1。

建议理解方式：

> replica 偏好是用来修正“token 粒度错误”的，不是用来提升召回。

### 13.3 跨 kind 冲突（node vs service vs TiDB）相关

- node 侧信号强化（例如更偏向 node KPI、或对 node 候选做更激进的规则）：
  - 典型收益：在 node-fault 题型里，能把“跨服务一致的基础设施异常”推到 top1。
  - 常见副作用：在 service-fault 题型里，如果 node KPI 有轻微波动但并非根因，可能把 node 误推到 top1。

- TiDB metrics 竞争力增强（仅在 TiDB 指标证据存在时）：
  - 典型收益：解决 TiDB under-score，使 TiDB 在多模态竞争中不再长期吃亏。
  - 常见副作用：如果 TiDB 指标只是“被影响端”而非“根因端”，可能出现 TiDB 抢走 top1。

建议理解方式：

> 跨 kind 的关键不是“谁更重要”，而是“哪一类证据更像根因证据”。node/ TiDB 的加权应尽量绑定到清晰、可复述的证据摘要上。

### 13.4 抑制热门服务坍缩

- 对热门服务施加惩罚/需要多模态支持：
  - 典型收益：当 evidence 很弱但热门服务因为先验/频次被推到 top1 时，能把它压下去，让更有证据的候选上来。
  - 常见副作用：如果热门服务确实是根因，但证据刚好偏单模态，可能被错误压制。

### 13.5 LLM 参与 component 选择的边界

- 把 component 选择强约束在候选集合内：
  - 典型收益：防止 token 发明，减少直接失分。
  - 常见副作用：如果候选集合本身没把 GT 推到靠前，LLM 很难“写作式补救”。因此提升 LA 的主要杠杆仍在 consensus 排序。

