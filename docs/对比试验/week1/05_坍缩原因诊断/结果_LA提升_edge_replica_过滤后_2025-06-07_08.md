# 复跑：edge过滤后（2025-06-07/08）

目标：在固定数据口径下，比较多种方案对 Final（以及 LA/TA/Explainability/Efficiency）的影响。

## 试验设置
- metadata: `../metadata_phase1.csv`
- ground truth: `../AIOpsChallengeJudge/ground_truth_phase1.jsonl`
- telemetry root: `../data`
- dates（按天全量）: 2025-06-07, 2025-06-08
- suite slug: `la_boost_v2_edge_replica_rerun`（对应 outputs/experiments/la_boost_v2_edge_replica_rerun/）

## 候选方案
- 方案A：对照（legacy off, strip replica=1, edge off）（key=base_safe）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_ENABLE_HINT_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_USE_LEGACY_MEMORY": "0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_EDGE_COMPONENTS": "0"}`
- 方案B：保留 replica 后缀（edge off）（key=keep_replica）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_ENABLE_HINT_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_USE_LEGACY_MEMORY": "0", "RCA_STRIP_REPLICA_SUFFIX": "0", "RCA_ENABLE_EDGE_COMPONENTS": "0"}`
- 方案C：开启 edge 组件候选（strip replica=1）（key=edge_on）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_ENABLE_HINT_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_USE_LEGACY_MEMORY": "0", "RCA_STRIP_REPLICA_SUFFIX": "1", "RCA_ENABLE_EDGE_COMPONENTS": "1", "RCA_EDGE_TOPK": "3"}`
- 方案D：开启 edge 候选 + 保留 replica 后缀（key=edge_on_keep_replica）
  - env: `{"RCA_COMPONENT_SOURCE": "consensus", "RCA_WINDOW_PADDING_MIN": "20", "RCA_ENABLE_MODALITY_BONUS": "0", "RCA_ENABLE_HINT_BONUS": "0", "RCA_COMPONENT_PRIOR_SCALE": "1.0", "RCA_USE_LEGACY_MEMORY": "0", "RCA_STRIP_REPLICA_SUFFIX": "0", "RCA_ENABLE_EDGE_COMPONENTS": "1", "RCA_EDGE_TOPK": "3"}`

## 结果汇总（按日期评测；并给出跨日期平均）

| 方案 | 平均LA | 平均TA | 平均Explain | 平均Eff | 平均Final |
|---|---:|---:|---:|---:|---:|
| 方案A：对照（legacy off, strip replica=1, edge off） | 14.58% | 72.92% | 12.40% | 81.87% | 44.43 |
| 方案B：保留 replica 后缀（edge off） | 10.42% | 62.50% | 17.95% | 81.87% | 39.15 |
| 方案C：开启 edge 组件候选（strip replica=1） | 14.58% | 72.92% | 12.40% | 81.87% | 44.43 |
| 方案D：开启 edge 候选 + 保留 replica 后缀 | 10.42% | 62.50% | 17.95% | 81.87% | 39.15 |

## 分日期明细

| date | N | 方案key | LA | TA | Explain | Eff | Final |
|---|---:|---|---:|---:|---:|---:|---:|
| 2025-06-07 | 24.0 | base_safe | 25.00% | 70.83% | 17.39% | 81.87% | 48.26 |
| 2025-06-07 | 24.0 | keep_replica | 12.50% | 62.50% | 17.39% | 81.87% | 39.93 |
| 2025-06-07 | 24.0 | edge_on | 25.00% | 70.83% | 17.39% | 81.87% | 48.26 |
| 2025-06-07 | 24.0 | edge_on_keep_replica | 12.50% | 62.50% | 17.39% | 81.87% | 39.93 |
| 2025-06-08 | 24.0 | base_safe | 4.17% | 75.00% | 7.41% | 81.87% | 40.59 |
| 2025-06-08 | 24.0 | keep_replica | 8.33% | 62.50% | 18.52% | 81.87% | 38.37 |
| 2025-06-08 | 24.0 | edge_on | 4.17% | 75.00% | 7.41% | 81.87% | 40.59 |
| 2025-06-08 | 24.0 | edge_on_keep_replica | 8.33% | 62.50% | 18.52% | 81.87% | 38.37 |

## 预测 component 分布（Top-5）

说明：该表用于快速观察是否出现 ‘adservice 坍缩’（例如 Top-1 占比异常高）。

| date | 方案key | Top-5 components（component:count） |
|---|---|---|
| 2025-06-07 | base_safe | checkoutservice:17; adservice:2; cartservice:2; aiops-k8s-07:1; emailservice:1 |
| 2025-06-07 | keep_replica | checkoutservice:10; cartservice:7; adservice:3; aiops-k8s-07:1; emailservice:1 |
| 2025-06-07 | edge_on | checkoutservice:17; adservice:2; cartservice:2; aiops-k8s-07:1; emailservice:1 |
| 2025-06-07 | edge_on_keep_replica | checkoutservice:10; cartservice:7; adservice:3; aiops-k8s-07:1; emailservice:1 |
| 2025-06-08 | base_safe | checkoutservice:20; adservice:3; aiops-k8s-07:1 |
| 2025-06-08 | keep_replica | cartservice:7; checkoutservice:7; adservice:5; emailservice:2; aiops-k8s-07:1 |
| 2025-06-08 | edge_on | checkoutservice:20; adservice:3; aiops-k8s-07:1 |
| 2025-06-08 | edge_on_keep_replica | cartservice:7; checkoutservice:7; adservice:5; emailservice:2; aiops-k8s-07:1 |

## 产物路径
- submissions: `outputs/experiments/la_boost_v2_edge_replica_rerun/`
- filtered gt: `tmp/filtered/` (gt_YYYY-MM-DD_*.jsonl)
