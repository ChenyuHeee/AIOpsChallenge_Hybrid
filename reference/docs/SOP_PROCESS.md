# SOP 流程搭建说明

## 设计目标
- 把 RCA 过程固定为可复用的流水线：限定时间窗/范围 → 抽取信号 → 生成/融合假设 → 验证与说明。
- 让各模块有清晰输入/输出，方便替换或增删步骤。
- 让 LLM 推理严格对齐 SOP，减少随意发挥和幻觉。

## 流程全貌
1) 接收用户描述（含时间窗/关键词）并生成计划。
2) 载入遥测数据，完成时间窗对齐与事件图构建。
3) 多类专家并行生成假设（组件 + 证据 + 可信度）。
4) 共识层按权重、先验、记忆融合，得到排序组件。
5) LLM 基于 SOP 步骤、检索知识、融合结果生成理由与推理链。
6) 校验格式/长度/步数后输出提交。

## SOP 步骤与落地
- `ScopeTelemetry`
  - 目的：确认数据是否齐备且时间窗一致。
  - 成功标准：metrics/logs/traces 已加载或有兜底策略。
  - 代码：`contest_solution/sop.py` 定义；执行时 `data/loader.py` 做对齐与准备。
- `SurfaceSignals`
  - 目的：快速找出高严重度的异常信号。
  - 成功标准：带组件指向的异常列表。
  - 代码：专家在 `agents/specialists.py` 的指标/日志/链路分析。
- `ConstructGraph`
  - 目的：生成事件/调用图，刻画传播路径。
  - 成功标准：图中有候选因果链。
  - 代码：`data/loader.py` 的 `_build_event_graph`；图类专家复用。
- `GenerateHypotheses`
  - 目的：专家产出候选根因假设（组件 + 证据 + narrative）。
  - 成功标准：每条假设含信号、组件、信心水平。
  - 代码：`agents/specialists.py` 四类专家 `run` 返回 `Hypothesis`。
- `RunConsensus`
  - 目的：按权重/先验/记忆融合假设，得到排序列表。
  - 成功标准：`[(component, score)...]` 排序结果。
  - 代码：`agents/consensus.py` 的 `_specialist_weight`、`vote`、`_memory_reward`。
- `ValidateAndExplain`
  - 目的：校验输出、生成推理链、控制长度。
  - 成功标准：符合评测字段/长度/步数要求。
  - 代码：`agents/reasoning.py` 用 SOP 步骤拼 Prompt；`agents/validator.py` 限制长度与步骤。

## 执行链路（从入口到输出）
- 入口：`contest_solution/main.py` 调用 orchestrator（见 `agents/planner.py`/`orchestrator.py`）。
- 计划：`PlannerAgent.build_plan` 解析时间窗/关键词，返回 `CasePlan`，其中 `sop_steps = [step.name for step in INCIDENT_SOP]`。
- 数据：`data/loader.py` 按计划对齐时间窗，生成 `event_graph` 供图/链路专家使用。
- 专家：`agents/specialists.py` 并行产出 `Hypothesis`（组件、confidence、evidence）。
- 共识：`agents/consensus.py` 融合专家结果，叠加先验与记忆（命中奖励、unknown 惩罚）。
- 推理：`agents/reasoning.py` 在 Prompt 中嵌入 SOP 步骤、检索到的 `insights`、共识排序与证据；LLM 输出 component + reason + analysis_steps。
- 验证：`agents/validator.py` 处理长度/步数，最终输出提交 JSONL。

## 配置与扩展
- 增删步骤：修改 `contest_solution/sop.py` 的 `INCIDENT_SOP` 列表，并在 `reasoning.py`/`planner.py` 保持同步。
- 调整权重与开关：`contest_solution/config.py` 控制专家启用、阈值、长度、步数；权重在共识层可扩展为动态更新。
- 提升检索/记忆：更新 `resources/paper_insights.json` 或改进 `InsightRepository`；调节共识记忆参数以减少空答。

## 输入/输出概览
- 输入：用户描述（含时间窗/组件关键词）、metrics/logs/traces 数据。
- 中间件：`CasePlan`（含 SOP 步骤、时间窗、关键词、检索摘要）、`Hypothesis` 列表、`event_graph`。
- 输出：排序的组件候选 + LLM 理由 + SOP 对齐的推理链，符合评测格式。
