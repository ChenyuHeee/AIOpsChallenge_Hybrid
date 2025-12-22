# 基线纠偏方案对比

目标：在固定数据口径下，比较多种方案对 Final（以及 LA/TA/Explainability/Efficiency）的影响。

## 试验设置
- metadata: `../metadata_phase1.csv`
- ground truth: `../AIOpsChallengeJudge/ground_truth_phase1.jsonl`
- telemetry root: `../data`
- dates（按天全量）: 2025-06-07, 2025-06-08
- suite slug: `bias_fix_v1_hint_grid`（对应 outputs/experiments/bias_fix_v1_hint_grid/）

## 候选方案
- 方案A：hints 关闭（对照组）（key=hint_000）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_HINT_BONUS": "0"}`
- 方案：hints 加成（bonus=0.05）（key=hint_005）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_HINT_BONUS": "1", "RCA_HINT_BONUS": "0.05"}`
- 方案：hints 加成（bonus=0.10）（key=hint_010）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_HINT_BONUS": "1", "RCA_HINT_BONUS": "0.10"}`
- 方案：hints 加成（bonus=0.15）（key=hint_015）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_HINT_BONUS": "1", "RCA_HINT_BONUS": "0.15"}`
- 方案：hints 加成（bonus=0.20）（key=hint_020）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_HINT_BONUS": "1", "RCA_HINT_BONUS": "0.20"}`

## 结果汇总（按日期评测；并给出跨日期平均）

| 方案 | 平均LA | 平均TA | 平均Explain | 平均Eff | 平均Final |
|---|---:|---:|---:|---:|---:|
| 方案A：hints 关闭（对照组） | 16.67% | 95.83% | 32.65% | 81.87% | 56.45 |
| 方案：hints 加成（bonus=0.05） | 16.67% | 93.75% | 27.09% | 81.87% | 55.06 |
| 方案：hints 加成（bonus=0.10） | 16.67% | 97.92% | 28.18% | 81.87% | 56.84 |
| 方案：hints 加成（bonus=0.15） | 16.67% | 95.83% | 31.88% | 81.87% | 56.38 |
| 方案：hints 加成（bonus=0.20） | 16.67% | 97.92% | 30.03% | 81.87% | 57.02 |

## 分日期明细

| date | N | 方案key | LA | TA | Explain | Eff | Final |
|---|---:|---|---:|---:|---:|---:|---:|
| 2025-06-07 | 24.0 | hint_000 | 16.67% | 100.00% | 28.26% | 81.87% | 57.68 |
| 2025-06-07 | 24.0 | hint_005 | 16.67% | 95.83% | 28.26% | 81.87% | 56.01 |
| 2025-06-07 | 24.0 | hint_010 | 16.67% | 100.00% | 30.43% | 81.87% | 57.90 |
| 2025-06-07 | 24.0 | hint_015 | 16.67% | 95.83% | 30.43% | 81.87% | 56.23 |
| 2025-06-07 | 24.0 | hint_020 | 16.67% | 100.00% | 30.43% | 81.87% | 57.90 |
| 2025-06-08 | 24.0 | hint_000 | 16.67% | 91.67% | 37.04% | 81.87% | 55.22 |
| 2025-06-08 | 24.0 | hint_005 | 16.67% | 91.67% | 25.93% | 81.87% | 54.11 |
| 2025-06-08 | 24.0 | hint_010 | 16.67% | 95.83% | 25.93% | 81.87% | 55.78 |
| 2025-06-08 | 24.0 | hint_015 | 16.67% | 95.83% | 33.33% | 81.87% | 56.52 |
| 2025-06-08 | 24.0 | hint_020 | 16.67% | 95.83% | 29.63% | 81.87% | 56.15 |

## 产物路径
- submissions: `outputs/experiments/bias_fix_v1_hint_grid/`
- filtered gt: `tmp/filtered/` (gt_YYYY-MM-DD_*.jsonl)
