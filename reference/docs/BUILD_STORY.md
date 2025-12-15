# Hybrid 方案构建脉络（基于参考论文）

## 1) Flow-of-Action：先把流程定死
- 论文要点：把 RCA 变成可复制的流水线——先限定时间窗和范围，再按步骤收集信号、列假设、验证结论，避免即兴发挥带来的不稳定。
- 代码落地：
	- `contest_solution/agents/planner.py`: `CasePlan.sop_steps = [step.name for step in INCIDENT_SOP]`，`build_plan` 解析时间窗/关键词。
	- `contest_solution/sop.py`: 定义 SOP 阶段（ScopeTelemetry → SurfaceSignals → Analyze → Confirm）。
	- `contest_solution/agents/reasoning.py`: `_build_prompt` 把 SOP 步骤写入 Prompt，约束 reasoning_trace。

## 2) TAMO：多信号、多工具并行出“假设”
- 论文要点：同时看指标、日志、调用链、拓扑，而不是盯单一信号；每条线都要给出“怀疑的组件 + 为什么 + 可信度”，方便后续融合。
- 代码落地：
	- `contest_solution/agents/specialists.py`: Metric/Log/Trace/Graph 四类专家的 `run` 产出 `Hypothesis`（组件+confidence+evidence）。
	- `contest_solution/utils/hypothesis.py`: 统一 `Hypothesis` / `EvidenceItem` 结构。
	- `contest_solution/config.py`: 专家开关/阈值在配置可调（权重固定在共识侧）。

## 3) mABC：让强专家更有话语权
- 论文要点：用多臂 bandit 思路给专家动态加权，持续放大可靠专家、压低常犯错的专家，减少“多数但不可靠”的投票噪声。
- 代码落地：
	- `contest_solution/agents/consensus.py`: `_specialist_weight` 设定专家权重；`vote` 中 `final_score = confidence * weight * reinforcement * prior` 并排序。
	- 目前权重为静态表，可扩展为在线更新以逼近真实 bandit。

## 4) Chain-of-Event：补上“路径”证据
- 论文要点：从时间序列、日志、调用链拼出事件图，追踪异常如何在组件间传播，用“哪条链路先出问题”来解释责任归因。
- 代码落地：
	- `contest_solution/data/loader.py`: `_build_event_graph` 从 metrics/logs/traces 提取边，生成轻量 `event_graph`。
	- `contest_solution/agents/specialists.py`: `TraceSpecialist`/`GraphSpecialist` 用时延峰值、拓扑邻域、调用链异常给组件相关性分。

## 5) 检索增强 LLM：把已知知识塞进 Prompt
- 论文要点：先检索行业/论文里的最佳实践，再让 LLM 结合上下文回答，减少幻觉；理由更像“对照手册后的结论”，而不是凭空猜。
- 代码落地：
	- `contest_solution/knowledge/insights.py`: `InsightRepository.relevant` 检索摘要。
	- `contest_solution/resources/paper_insights.json`: 可检索的论文摘要库。
	- `contest_solution/agents/planner.py`: `matched_insights = insight_repo.relevant(...)`。
	- `contest_solution/agents/reasoning.py`: `_build_prompt` 将 `insight_snippets` 拼入 Prompt。

## 6) FLASH 记忆：复用命中过的好线索
- 论文要点：把历史命中当“记忆”，命中过的组件下次得分更高，常错或 unknown 会被惩罚，帮助收敛、降低空答或乱答。
- 代码落地：
	- `contest_solution/agents/consensus.py`: `_memory_reward` / `_update_memory` 对命中组件加分，对 unknown/失败降权；记忆持久化到 `memory_path`。

## 7) 评测对齐：格式和指标是硬门槛
- 论文要点：评测脚本对字段、长度、步骤有硬性要求，格式错直接归零；必须在“组件、理由、路径、证据”各维度都合规。
- 代码落地：
	- `contest_solution/agents/validator.py`: `enforce_limits` 截断理由/步数；`build_trace` 生成 `reasoning_trace` 并追加证据片段。
	- `contest_solution/config.py`: 暴露 `max_reason_words`、`target_trace_steps` 等限制以适配评测。

## 8) 工程化与可复现：让别人能跑、能改
- 论文要点：提供清晰入口、集中配置、可跑样例，保证别人拉下代码就能跑、能改、能对比，不需要重走环境/流程坑。
- 代码落地：
	- `contest_solution/main.py`: 单一入口驱动全流程。
	- `contest_solution/config.py`: 集中配置（开关、权重、长度、超时）。
	- `submissions_*.jsonl`: 基线示例；`git_push.sh`: 一键提交/推送脚本。

## 9) 串起来看
- 先定流程（Flow-of-Action），再铺专家（TAMO），用 mABC 让融合更稳，用事件链补路径证据。
- 检索增强让理由不飘，FLASH 让命中率更高，评测对齐保证能得分。
- 工程化包装让方案可提交、可复现、可交付。
