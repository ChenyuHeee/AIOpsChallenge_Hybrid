# 根因分析方案

该程序结合 AIOPSPolaris、OpenRCA 以及多篇最新论文的思想，构建了一条可直接用于比赛提交的根因分析（RCA）流水线。

## 核心设计理念

- **SOP 驱动的规划（Flow-of-Action）**：每个案例都遵循标准操作流程，保证工具调用可控且易于复现。
- **多专家协同（TAMO / mABC）**：指标、日志、链路与事件图专家分别产出假设，再由加权投票层融合。
- **事件链推理（Chain-of-Event）**：将遥测转换为轻量
级传播图，追踪延迟突增或异常影响的上下游关系。
- **检索增强推理（Exploring LLM Agents）**：将论文洞见与历史记忆注入 LLM Prompt，提升事实依据。
- **事后记忆（FLASH）**：共识反馈会更新轻量记忆文件，后续案例可复用成功组件线索。
- **评测对齐（AIOpsLab / RCAEval）**：规划器与校验器确保输出满足比赛指标约束（组件准确率、理由长度、推理格式）。

## 仓库结构

```
contest_solution/
  agents/           # 规划、专家、共识、推理、校验模块
  data/             # 遥测读取与事件图构建
  knowledge/        # 论文洞见知识库
  llm/              # LLM 客户端（DeepSeek/OpenAI）与降级策略
  resources/        # 由论文抽取的摘要信息
  config.py         # 环境变量与流水线配置
  orchestrator.py   # 主调度逻辑
  sop.py            # 标准操作流程定义
  main.py           # 命令行入口
```

## 算法结构（Hybrid）

### 数据流概览
1) **入口**：`main.py` 读取 `metadata_xxx.csv`；按行构造案例上下文（uuid、时间窗、故障类别）。
2) **遥测加载**：`data/loader.py` 以时间窗截取指标、日志、链路（若缺失则返回空信号），并构建轻量事件图节点/边。
3) **SOP 规划**：`agents/planner.py` 按 Flow-of-Action 生成任务计划（ScopeTelemetry → SurfaceSignals → Analyze → Confirm），并注入 `resources/paper_insights.json` 中的论文摘要作为检索增强提示。
4) **专家并行打分**：`agents/specialists.py` 产出多路假设 `Hypothesis`：
  - Metrics 专家：基于异常指标与拓扑邻域的相关性得分。
  - Logs 专家：检索错误/超时关键词，定位关联组件。
  - Traces/Graph 专家：利用调用链异常与事件图边权，追踪上游/下游影响。
5) **共识融合**：`agents/consensus.py` 将专家候选做加权投票（mABC 思路），叠加两类先验：
  - **组件先验**：常见高风险组件（如 adservice、checkoutservice）有微弱提升，降低 unknown。
  - **记忆强化**：对近期命中的组件给予奖励，失败样本惩罚，形成轻量“记忆分数”。
  最终输出排序后的组件列表及分值。
6) **LLM 推理与理由生成**：`agents/reasoning.py` 结合 SOP 阶段、共识候选、证据 JSON 和论文洞见构建 Prompt，产出 component + reason + reasoning_trace；如 LLM 失败，使用启发式兜底（例如选最高分组件并给定固定解释模板）。
7) **校验与落盘**：`agents/validator.py` 检查字段完整性、步数/长度上限，填充缺失字段并输出符合评测器要求的 JSONL。

### 关键配置与可调参数
- `.env` / 环境变量：LLM Key、Base URL、模型名、并发度。
- `config.py`：流水线开关（启用哪些专家/先验/记忆）、最大步数、理由字数限制、超时与重试次数。
- `resources/paper_insights.json`：检索增强知识库，可追加新论文摘要。
- `agents/consensus.py`：先验强度、记忆奖励/惩罚、投票权重。
- `agents/specialists.py`：各专家的打分权重与启发式阈值。

### 运行基线
- Phase1（全量）在新评测脚本下得分：Component 14.69%、Reason 59.24%、Efficiency 81.87%、Explainability 14.67%、Final 39.23。
- Phase2（全量）在新评测脚本下得分：Component 2.08%、Reason 43.23%、Efficiency 81.87%、Explainability 17.23%、Final 28.04。

## 快速开始

```bash
python -m Beta.contest_solution.main \
  --telemetry-root /path/to/telemetry \
  --metadata metadata_phase12.csv \
  --output submissions.jsonl
```

可选参数：

- `--limit N`：仅处理前 `N` 个案例，便于调试。
- `--dry-run`：只打印 JSON 对象，不落盘。

## 扩展方向

- 在 `resources/` 中追加新的论文摘要，并通过 `RCA_PAPER_INSIGHTS` 指向相应文件。
- 在 `agents/specialists.py` 中调整专家权重或启发式，以适配不同数据集。
- 通过环境变量修改 `PipelineConfig`，灵活控制理由长度、轨迹步数或记忆文件位置。

该方案可在接入新遥测数据后快速迭代，同时保持根因说明的准确性与比赛格式的一致性。
