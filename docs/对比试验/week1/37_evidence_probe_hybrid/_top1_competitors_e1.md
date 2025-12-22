## Top1 竞争对手分析（candidate=e1_fusion_no_infra_rules；topK=80）

- N=120, LA=4.17%, GT-in-topK=70.00%
- 重点样本：GT 在 topK 但未命中（miss_but_gt_in_topk）= 79

### Top1 最常见竞争对手（Top 15）

| 排名 | top1 组件 | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | cartservice | 21 | 26.58% |
| 2 | emailservice | 14 | 17.72% |
| 3 | frontend-1 | 7 | 8.86% |
| 4 | frontend-2 | 6 | 7.59% |
| 5 | aiops-k8s-04 | 6 | 7.59% |
| 6 | frontend-0 | 3 | 3.80% |
| 7 | checkoutservice-0 | 3 | 3.80% |
| 8 | paymentservice | 2 | 2.53% |
| 9 | aiops-k8s-06 | 2 | 2.53% |
| 10 | aiops-k8s-07 | 2 | 2.53% |
| 11 | checkoutservice-2 | 2 | 2.53% |
| 12 | adservice-0 | 1 | 1.27% |
| 13 | frontend | 1 | 1.27% |
| 14 | tidb-tidb-0 | 1 | 1.27% |
| 15 | aiops-k8s-03 | 1 | 1.27% |

### GT 类型 vs top1 类型（计数）

| GT_kind | top1_kind | 次数 | 占比 |
|---|---|---:|---:|
| service | service | 40 | 50.63% |
| tidb | service | 14 | 17.72% |
| node | service | 11 | 13.92% |
| service | node | 9 | 11.39% |
| service | tidb | 3 | 3.80% |
| tidb | node | 2 | 2.53% |

### top1 的模态组合（Top 10）

| 排名 | top1_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 43 | 54.43% |
| 2 | logs+metrics+traces | 27 | 34.18% |
| 3 | metrics | 6 | 7.59% |
| 4 | logs+metrics | 3 | 3.80% |

### GT（best-rank 命中项）的模态组合（Top 10）

| 排名 | gt_best_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 44 | 55.70% |
| 2 | metrics | 28 | 35.44% |
| 3 | traces | 3 | 3.80% |
| 4 | logs+metrics | 3 | 3.80% |
| 5 | logs+metrics+traces | 1 | 1.27% |

### 代表性样本（前 12 条，按 gt_best_rank 由小到大）

| date | uuid | GT_parts | GT_kind | top1 | top1_kind | top1_mod | gt_best_rank | gt_best_mod |
|---|---|---|---|---|---|---|---:|---|
| 2025-06-27 | 238afe01-573 | checkoutservice-0 | service | cartservice | service | metrics+traces | 2 | metrics+traces |
| 2025-06-27 | 4b98f078-560 | checkoutservice+emailservice | service | checkoutservice-2 | service | logs+metrics+traces | 2 | traces |
| 2025-06-24 | 6d1ad486-497 | aiops-k8s-08 | node | cartservice | service | metrics+traces | 3 | metrics+traces |
| 2025-06-28 | bb585575-589 | aiops-k8s-08 | node | emailservice | service | metrics+traces | 3 | metrics+traces |
| 2025-06-07 | 0fccdcad-125 | recommendationservice | service | cartservice | service | metrics+traces | 4 | metrics+traces |
| 2025-06-07 | abb62970-110 | adservice | service | adservice-0 | service | logs+metrics | 4 | metrics+traces |
| 2025-06-14 | a63a0c8b-290 | aiops-k8s-05 | node | cartservice | service | metrics+traces | 4 | metrics |
| 2025-06-24 | 2ff9d8a6-501 | aiops-k8s-06 | node | frontend-1 | service | logs+metrics+traces | 4 | logs+metrics+traces |
| 2025-06-07 | f18b68cd-119 | aiops-k8s-07 | node | emailservice | service | metrics+traces | 5 | metrics+traces |
| 2025-06-24 | a0d5329b-489 | checkoutservice | service | emailservice | service | metrics+traces | 5 | metrics+traces |
| 2025-06-27 | aceeda23-570 | checkoutservice+paymentservice | service | frontend-2 | service | logs+metrics+traces | 5 | metrics+traces |
| 2025-06-13 | 49e9814e-266 | frontend+cartservice | service | emailservice | service | metrics | 6 | metrics |
