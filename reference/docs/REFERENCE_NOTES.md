# Hybrid 方案论文映射说明

下列条目说明 Hybrid 流水线中各模块对应参考论文的借鉴点（结合当前代码实现）。

## flow_of_action_www2025.pdf（Flow-of-Action）
[翻译](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/FlowOfAction_SOP%E5%A2%9E%E5%BC%BA%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F_%E7%BF%BB%E8%AF%91/)
[笔记](https://chenyuheee.github.io/posts/AIOPS论文翻译与学习/FlowOfAction_学习笔记/)
- 借鉴点：将 RCA 拆为可复用的标准步骤（ScopeTelemetry → SurfaceSignals → Analyze → Confirm），确保每个案例都有一致的计划与检查点。
- 在代码中的落地：`agents/planner.py` 生成 SOP 风格的 plan，`sop.py` 描述步骤语义；`reasoning.py` 用 plan 片段构造 LLM 提示，保证推理路径可解释。

## tamo_tool_assisted_rca.pdf（TAMO / Tool-Assisted Multi-Objective）
[翻译](https://chenyuheee.github.io/posts/AIOPS论文翻译与学习/TAMO_多模态工具辅助智能体_翻译/)
[笔记](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/TAMO_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- 借鉴点：多工具/多信号的协同优化，避免单一信号偏差。
- 在代码中的落地：`agents/specialists.py` 并行启用指标/日志/链路/图专家，各自输出带分数的 `Hypothesis`；`config.py` 可调整专家开关与权重。

## mabc_emnlp2024.pdf（mABC 加权投票）
[翻译](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/mABC_%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%8C%BA%E5%9D%97%E9%93%BE%E5%8D%8F%E4%BD%9C%E6%A1%86%E6%9E%B6_%E7%BF%BB%E8%AF%91/)
[笔记](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/mABC_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- 借鉴点：基于 bandit 式自适应投票，提升高质量专家的权重。
- 在代码中的落地：`agents/consensus.py` 对专家分数做加权汇总，并结合先验与记忆强化，形成排序后的组件列表。

## chain_of_event_fse24.pdf（Chain-of-Event / 事件链）
[翻译](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/ChainOfEvent_%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%A0%B9%E5%9B%A0%E5%88%86%E6%9E%90_%E7%BF%BB%E8%AF%91/)
[笔记](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/ChainOfEvent_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- 借鉴点：从原始遥测构建事件图，追踪因果传播链。
- 在代码中的落地：`data/loader.py` 生成轻量事件图（节点/边）；`specialists.py` 的 trace/graph 专家利用调用链异常和拓扑邻域给出相关性打分。

## exploring_llm_agents_fse2024.pdf（检索增强 LLM Agents）
[翻译](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/LLM_Agents_for_RCA_%E7%BF%BB%E8%AF%91/)
[笔记](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/LLM_Agents_for_RCA_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- 借鉴点：将领域知识注入 LLM 提示，减少幻觉并提升理由一致性。
- 在代码中的落地：`knowledge/insights.py` 与 `resources/paper_insights.json` 提供论文摘要；`planner.py`/`reasoning.py` 把检索结果拼入 Prompt，辅助生成理由与推理链。

## flash_workflow_agent.pdf（FLASH 记忆机制）
[翻译](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/FLASH_%E5%B7%A5%E4%BD%9C%E6%B5%81%E8%87%AA%E5%8A%A8%E5%8C%96%E6%99%BA%E8%83%BD%E4%BD%93_%E7%BF%BB%E8%AF%91/)
[笔记](https://chenyuheee.github.io/posts/AIOPS%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E4%B8%8E%E5%AD%A6%E4%B9%A0/FLASH_%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- 借鉴点：利用历史反馈做记忆强化，鼓励重复命中有价值组件。
- 在代码中的落地：`consensus.py` 中的记忆奖励/惩罚机制，对近期成功组件加分、失败样本减分，降低反复输出 unknown 的概率。

## rcaeval_benchmark.pdf（评测对齐）
- 借鉴点：对齐评测格式与指标（组件、理由、路径、解释性）。
- 在代码中的落地：`validator.py` 控制字段完整性、步数与长度；输出 JSONL 满足评测脚本要求；`config.py` 暴露步数/长度限制以适配不同评测器。

## aiopslab_2025.pdf（实践/工程指南）
- 借鉴点：工程化落地与数据适配经验，用于稳定性与可复现性设计。
- 在代码中的落地：统一入口 `main.py`，明确的 `config.py` 配置面板，示例提交文件（`submissions_*.jsonl`）作为调试基准。
