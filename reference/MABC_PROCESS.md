# mABC 在项目中的实现说明

面向不了解论文的读者，说明我们如何把 mABC（bandit 风格的专家加权共识）落地到 Hybrid 流水线。

## 目标与思路
- 目标：融合多路专家的假设时，放大可靠专家、压低不可靠专家，减少“多数但弱”的噪声。
- 核心思路：对每条假设引入专家权重、组件先验、记忆奖励，形成加权得分并排序。

## 数据流与接口
- 输入：`Hypothesis` 序列（组件、confidence、evidence、source）。来源：`agents/specialists.py` 中的 Metric/Log/Trace/Graph 专家，及可能的 LLM reasoning。
- 处理：在 `ConsensusOrchestrator.vote(case_id, hypotheses)` 中做逐条累加。
- 输出：`ConsensusResult`，包含排序后的组件列表和汇总证据，供 LLM 与最终提交使用。

## 打分公式（代码对应）
在 `agents/consensus.py` 中，逐条假设计算：

```
final_score = hypothesis.confidence * weight * reinforcement * prior
```

- `hypothesis.confidence`：专家内部给出的基础置信度。
- `weight`：专家静态权重 `_specialist_weight(hypothesis.source)`。
- `reinforcement`：记忆奖励/惩罚 `_memory_reward(hypothesis.component)`。
- `prior`：组件先验 `_component_prior(hypothesis.component)`。

累加后按分排序，如全部空则回退到最高先验组件 `_top_prior_component()`。

## 关键机制拆解
- 专家权重（放大强专家）
  - 位置：`_specialist_weight`
  - 当前默认：Metric 1.2, Log 1.1, Trace 1.1, Graph 1.0, ReasoningAgent 1.3，其它 1.0。
  - 作用：在同等置信度下，历史表现好的专家影响力更大。
- 组件记忆（Hindsight/FLASH）
  - 位置：`_memory_reward` & `_update_memory`
  - 逻辑：命中过的组件提升 success_rate，reinforcement = 1.0 + 0.6 * success_rate；unknown 直接降到 0.7。
  - 更新：`_update_memory` 在每次 vote 后记录 champion（case_id, component, score），持久化到 `memory_path`（`config.py` 中默认 `.aiops_memory.json`）。
- 组件先验（偏好常见故障点）
  - 位置：`_load_component_priors` / `_component_prior`
  - 默认值：adservice 1.4, checkoutservice 1.3, recommendationservice 1.25, productcatalogservice 1.2, frontend 1.15, cartservice 1.1；可被已存储的 priors 覆盖。
  - 作用：在数据不足时偏向更常出故障或历史高频的组件。
- 证据汇总
  - 位置：`vote` 内 `supporting_evidence` 聚合 `[modality] summary`，供后续 LLM 理由引用。
- 回退策略
  - 若没有得分，或最高分 ≤ 0，则使用最高先验组件兜底，避免空答。

## 与 SOP/管线的衔接
- 上游：`specialists` 并行产出 `Hypothesis`，带来源（source）与置信度。
- 本层：`ConsensusOrchestrator.vote` 完成加权融合，输出排序组件与证据。
- 下游：`reasoning.py` 在 Prompt 中嵌入排序组件与证据，`validator.py` 负责最终格式与长度控制。

## 可配置与扩展点
- 记忆开关与路径：`config.py` 中 `PipelineConfig.enable_hindsight_memory`（默认启用）与 `memory_path`。
- 先验覆盖：`memory_path` 中的 `component_priors` 可持久化覆盖默认表。
- 动态 bandit：可将 `_specialist_weight` 与 `_update_memory` 扩展为在线更新（如按专家 ID 维护 success_rate），更贴近论文中的实时调权。
- 负反馈/惩罚：目前主要奖励命中，可加入对错误预测的惩罚项，进一步加速收敛。

## 文件索引
- 主体逻辑：`contest_solution/agents/consensus.py`
- 记忆持久化：`contest_solution/config.py` 的 `load_memory` / `store_memory`、`PipelineConfig.memory_path`
- 假设结构：`contest_solution/utils/hypothesis.py`（组件、confidence、evidence、source）

## 输出示例（结构）
- 排序列表：`[('checkoutservice', 2.85), ('adservice', 1.90), ...]`
- 证据：`{'checkoutservice': ['[metric] error_rate spike', '[log] timeout in checkout'], ...}`

这样，mABC 在项目中承担“把多路专家的信号融合为可靠排序”的职责，并通过权重、先验、记忆来实现“强者恒强、常错降权、缺证兜底”。
