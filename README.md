# AIOpsChallenge_Hybrid

基于 Beta 方案的混合式根因分析流水线，包含可直接提交的代码（`contest_solution`）以及各阶段示例输出。

## 仓库内容
- `contest_solution/`：全量流水线（规划、专家、共识、推理、校验）
- `submissions_phase1.jsonl`、`submissions_phase2.jsonl`、`submissions_2025-06-07.jsonl`：示例输出
- `ground_truth_sample.jsonl`、`submissions_sample.jsonl`：小样本校验文件
- `.env`：LLM 密钥与端点（不纳入版本控制）
- 📑 参考映射：[`reference/REFERENCE_NOTES.md`](reference/REFERENCE_NOTES.md)

## 快速开始
```bash
# 建议 Python 3.10+
pip install -r contest_solution/requirements.txt   # 或复用已有 venv

# 运行全流程
python -m contest_solution.main \
  --telemetry-root /path/to/telemetry \
  --metadata /path/to/metadata_phase1.csv \
  --output submissions.jsonl

# 可选参数
# --limit N   仅处理前 N 个案例
# --dry-run   只打印结果，不写文件
```

## 算法概览
1) 案例级加载时间窗遥测，若有信号则构建轻量事件图。
2) 规划器（Flow-of-Action）设定范围，并检索论文洞见作为提示。
3) 专家（指标/日志/链路与图）生成打分假设。
4) 共识层用 mABC 式投票叠加先验与记忆，排序组件。
5) 组件（component）默认取共识层 Top-1；推理 LLM 仅用于生成 reason + reasoning_trace（失败则启发式兜底）。
6) 校验器控制格式与长度，输出评测所需 JSONL。

## 配置入口
- `.env`：`DEEPSEEK_API_KEY`/`OPENAI_API_KEY`、Base URL、模型名、并发/超时等。
- `contest_solution/config.py`：启用专家/先验/记忆，步数与长度上限。
- `contest_solution/resources/paper_insights.json`：RAG 知识库，可自行扩充。
- `contest_solution/agents/consensus.py`：先验权重、记忆奖励/惩罚可调。

常用环境变量：
- `RCA_LLM_PROVIDER`：`deepseek` / `openai` / `dummy`
- `RCA_LLM_TIMEOUT`：LLM 请求超时秒数（DeepSeek 未设置时默认 60s）
- `RCA_WINDOW_PADDING_MIN`：遥测时间窗 padding（分钟），用于控制跨事件污染

## 基线成绩（新版评测）
- Phase1：Component 14.69%，Reason 59.24%，Efficiency 81.87%，Explainability 14.67%，Final 39.23
- Phase2：Component 2.08%，Reason 43.23%，Efficiency 81.87%，Explainability 17.23%，Final 28.04

## 说明
- `.env` 请勿入库，运行前自行导出环境变量。
- 若某些 telemetry 日期缺失，流水线会在空信号下进行兜底推理。
