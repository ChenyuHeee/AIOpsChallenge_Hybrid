# Hybrid 全流程（含论文要点标注）

## 输入与输出
- 输入：用户描述（含时间窗/关键词）、metrics/logs/traces（parquet）。
- 输出：按评测格式的 JSONL：`component`、`reason`（限长）、`reasoning_trace`（3-7 步，SOP 对齐）、证据片段。

## 端到端步骤
1) 读取配置、加载记忆
   - 代码：`contest_solution/config.py`，加载 env 与 `PipelineConfig`，读入 `memory_path`。
   - 论文标注：记忆用于 [FLASH]。

2) 规划案例（Plan Case）
   - 代码：`agents/planner.py` → `PlannerAgent.build_plan`。
   - 动作：抽取关键词/组件提示、解析时间窗、检索论文摘要（`insight_repo.relevant`），产出 `CasePlan`。
   - 论文标注：[Flow-of-Action]（先定流程），[RAG]（检索论文摘要入后续 Prompt）。

3) 定义 SOP 步骤
   - 代码：`sop.py` → `INCIDENT_SOP`（ScopeTelemetry → SurfaceSignals → ConstructGraph → GenerateHypotheses → RunConsensus → ValidateAndExplain）。
   - 论文标注：[Flow-of-Action]（固定 RCA 轨道）。

4) 加载并对齐遥测
   - 代码：`data/loader.py` → `TelemetryLoader.load`（按时间窗过滤 + padding），返回 metrics/logs/traces + `event_graph`。
   - 论文标注：[Flow-of-Action]（时间窗对齐），[COE]（构建事件图）。

5) 事件图构建
   - 代码：`TelemetryLoader._build_event_graph`：提取 component/peer/operation，规范化、去重，生成邻接表。
   - 论文标注：[COE]（事件/调用链）。

6) 专家并行生成假设
   - 代码：`agents/specialists.py` → Metric/Log/Trace/Graph `run` → 产出 `Hypothesis`（component + confidence + evidence + source）。
   - 论文标注：[TAMO]（多信号并行），[COE]（Graph/Trace 专家利用事件图）。

7) 共识融合（加权投票）
   - 代码：`agents/consensus.py` → `ConsensusOrchestrator.vote`。
   - 打分：`final_score = confidence * weight * reinforcement * prior`。
     - 权重 `_specialist_weight` → [mABC]。
     - 记忆 `_memory_reward` → [FLASH]。
     - 先验 `_component_prior` → 高频组件偏好。
   - 论文标注：[mABC]（bandit-style 加权），[FLASH]（命中奖励/unknown 惩罚）。

8) 排序与兜底
   - 若无分或最高分<=0，回退最高先验组件；写入 `supporting_evidence`。
   - 论文标注：[mABC]（回退结合先验）。

9) 构建 LLM Prompt 并推理
   - 代码：`agents/reasoning.py` → `_build_prompt`。
   - 内容：SOP 步骤、检索的论文摘要、共识排序前 N、证据 JSON。
   - 论文标注：[Flow-of-Action]（步骤约束推理链），[RAG]（论文摘要入 Prompt）。

10) LLM 解析与兜底
    - 解析 JSON（component/reason/analysis_steps），若空则用共识 Top1 + 证据兜底。
    - 论文标注：无新增（延续上一步）。

11) 验证与格式化
    - 代码：`agents/validator.py` → `enforce_limits`/`build_trace`，控制长度、步数、字段完整性。
    - 论文标注：对齐评测（非论文，但硬性要求）。

12) 输出提交
    - 代码：`main.py`/`orchestrator.py` 汇总结果，写入提交 JSONL。
    - 论文标注：无。

## 模块与论文映射速查
- [Flow-of-Action]：`sop.py`，`planner.py`，`reasoning.py`（SOP 嵌入 Prompt）。
- [TAMO]：`agents/specialists.py`（多模态专家输出统一 Hypothesis）。
- [COE]：`data/loader.py`（事件图）、`specialists.py`（Graph/Trace）。
- [mABC]：`agents/consensus.py`（权重 + 先验 + 兜底）。
- [FLASH]：`agents/consensus.py`（记忆奖励/更新）、`config.py`（memory_path）。
- [RAG]：`knowledge/insights.py` + `resources/paper_insights.json`，在 `planner.py`/`reasoning.py` 注入 Prompt。

## 运行入口
- `contest_solution/main.py`：单一入口；读取配置→规划→加载→专家→共识→LLM→校验→输出。

如需把论文段落合并到其他文档，可直接引用本文件的“模块与论文映射速查”。

## 相关说明
- 组件（component）是否由 LLM 决定：`../../docs/组件来源说明.md`
- 组件来源对比试验记录：`../../docs/组件来源对比试验.md`
