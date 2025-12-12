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

## 算法概要
- **遥测融合**：`TelemetryLoader` 读取指标/日志/链路，并生成轻量事件图供后续传播分析。
- **SOP 规划**：`PlannerAgent` 按 Flow-of-Action 模板提取事件时间窗与关键词，检索 `paper_insights.json` 中的最佳实践作为 LLM 提示。
- **专家集成**：Metrics/Logs/Traces/Graph 专家独立评估组件严重度，生成带证据的 `Hypothesis`，降低单一信号偏差。
- **带先验和记忆的共识**：`ConsensusOrchestrator` 融合 mABC 加权投票、FLASH 记忆，并对 adservice、checkoutservice 等高频组件设置先验，减少回退 `unknown`。
- **LLM 推理**：`ReasoningAgent` 结合 SOP 阶段、组件候选、证据 JSON 与论文洞见构建 Prompt；解析失败时采用启发式兜底。
- **轨迹格式化与校验**：`SubmissionValidator` 控制理由/推理步数上限，补齐参考证据，输出符合评测器要求的 JSONL。
- **迭代评分**：在 2025-06-07 全量数据上通过 `mtr/judge/evaluate.py` 获得 Component Accuracy 8.33%、Reason Accuracy 58.33%、Final Score 35.51，为后续拓展提供基线。

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
