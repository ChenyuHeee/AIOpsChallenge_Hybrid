# 组件（component）是否由 LLM 决定？

## 结论（当前默认）
- 默认：**component 不由 LLM 选择**，而是由“共识层（Consensus）”基于多模态证据（metric/log/trace/graph）投票得到的 Top-1 组件决定。
- LLM 的职责：主要用于生成 **reason** 和 **reasoning_trace**，以提升 TA/Explainability；LLM 失败时会自动兜底。

## 为什么要这么做（与 LA 直接相关）
- 评测对 component 的匹配非常“硬”：本质是 token 级精确匹配。
- LLM 在“命名”和“端点选择”上更容易漂移（同义词、大小写、把 edge 当 component 等），一旦 token 不一致就会直接记错，从而拉低 LA。
- 共识层的输出来自可追溯证据，稳定性更强；因此把“component 选择”放在共识层通常更利于 LA。

## 如何做消融对比（新增开关）
在 [contest_solution/orchestrator.py](../../contest_solution/orchestrator.py) 中加入了环境变量开关：
- `RCA_COMPONENT_SOURCE=consensus`（默认）：component = 共识层 Top-1
- `RCA_COMPONENT_SOURCE=llm`：component = LLM 输出（若为空则回退到共识 Top-1）

示例：
```bash
# 默认（推荐 baseline）
RCA_COMPONENT_SOURCE=consensus \
python -m contest_solution.main --telemetry-root ../data --metadata ../metadata_phase1.csv --output outputs/submissions/sub.jsonl --limit 10

# 消融：让 LLM 决定 component
RCA_COMPONENT_SOURCE=llm \
python -m contest_solution.main --telemetry-root ../data --metadata ../metadata_phase1.csv --output outputs/submissions/sub_llm.jsonl --limit 10
```

> 注意：即使切到 `llm`，仍会做长度/格式校验；并对 component 与候选集合做“大小写不敏感”对齐，以降低 token 漂移。

## 对比试验记录
- 试验报告：`docs/对比试验/week1/01_组件来源对比/报告.md`
