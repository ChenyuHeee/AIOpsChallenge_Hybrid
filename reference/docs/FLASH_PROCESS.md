# FLASH 记忆在项目中的实现说明

## 目标与思路
- 目标：利用历史反馈放大已命中的组件、压低“unknown”或常错组件，减少空答和随机性。
- 思路：在共识打分时引入记忆系数（reinforcement），并在每次决策后更新记忆；记忆持久化到磁盘，下次运行继续生效。

## 数据流与接口
- 输入：`Hypothesis` 列表（组件、confidence、source、evidence）。
- 使用位置：`contest_solution/agents/consensus.py` 的 `ConsensusOrchestrator.vote`。
- 输出：`ConsensusResult`（排序组件 + 证据），其中分数已乘上记忆奖励/惩罚。

## 打分公式中的记忆项
代码：`consensus.py` → `_memory_reward`

```
final_score = confidence * weight * reinforcement * prior
reinforcement = 1.0 + 0.6 * success_rate
```

- 对 unknown：直接降到 0.7（不随 success_rate 增长）。
- 对普通组件：从 `component_success` 取 success_rate（默认 0.4），带入上式。

## 记忆的更新与持久化
代码：`consensus.py` → `_update_memory`
- 何时更新：每次 `vote` 后，对冠军组件（ranked[0]）更新；unknown 不更新。
- 更新公式：`updated = min(1.0, baseline*0.9 + (1 if score>0 else 0)*0.1)`，baseline 默认 0.5。
- 历史记录：追加 `{case: case_id, component: winner, score: score}` 到 `history`。
- 存储：`store_memory` 写入 `PipelineConfig.memory_path`（默认 `.aiops_memory.json`）。

## 配置与开关
- 文件路径：`contest_solution/config.py` → `PipelineConfig.memory_path`，默认 `.aiops_memory.json`。
- 开关：`PipelineConfig.enable_hindsight_memory` 提供启用/停用位（如需硬关闭，可在调用时绕过或清空记忆文件）。

## 与 SOP/管线的衔接
- 上游：Metric/Log/Trace/Graph 专家产出 `Hypothesis`（含 confidence、source）。
- 本层：`ConsensusOrchestrator.vote` 计算 weight×prior×reinforcement，总分排序组件。
- 下游：排序结果传给 `ReasoningAgent` 生成理由；`Validator` 负责最终格式。

## 扩展与改进建议
- 负反馈：当前只对冠军加正向奖励，可增加对误判组件的惩罚，或对亚军按名次衰减更新。
- 专家粒度记忆：现按组件聚合，可扩展为“组件 × 专家”二维 success_rate，精细化不同专家的记忆。
- 冷启动：默认 success_rate=0.4，可按赛题/数据重新设定，或结合先验频率自适应。
- 清理策略：定期衰减或截断 history，防止老案例绑死新分布。

## 文件索引
- 记忆与共识：`contest_solution/agents/consensus.py` (`_memory_reward`, `_update_memory`, `vote`)
- 持久化与配置：`contest_solution/config.py` (`PipelineConfig.memory_path`, `load_memory`, `store_memory`)
- 输入结构：`contest_solution/utils/hypothesis.py`（Hypothesis 定义）

FLASH 记忆在项目中的角色：为打分注入“历史经验”，让常命中的组件更易被选中，unknown/常错更难得分，从而减少空答、提升稳定性。
