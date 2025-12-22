## Top1 竞争对手分析（candidate=e0_fusion_baseline；topK=80）

- N=120, LA=13.33%, GT-in-topK=70.00%
- 重点样本：GT 在 topK 但未命中（miss_but_gt_in_topk）= 68

### Top1 最常见竞争对手（Top 15）

| 排名 | top1 组件 | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | aiops-k8s-05 | 10 | 14.71% |
| 2 | aiops-k8s-03 | 7 | 10.29% |
| 3 | aiops-k8s-04 | 6 | 8.82% |
| 4 | tidb-tidb-0 | 6 | 8.82% |
| 5 | aiops-k8s-07 | 6 | 8.82% |
| 6 | aiops-k8s-06 | 5 | 7.35% |
| 7 | tidb-tikv-0 | 5 | 7.35% |
| 8 | aiops-k8s-01 | 4 | 5.88% |
| 9 | cartservice | 4 | 5.88% |
| 10 | aiops-k8s-08 | 3 | 4.41% |
| 11 | frontend-1 | 3 | 4.41% |
| 12 | frontend-2 | 2 | 2.94% |
| 13 | emailservice | 2 | 2.94% |
| 14 | adservice-0 | 1 | 1.47% |
| 15 | frontend-0 | 1 | 1.47% |

### GT 类型 vs top1 类型（计数）

| GT_kind | top1_kind | 次数 | 占比 |
|---|---|---:|---:|
| service | node | 33 | 48.53% |
| service | service | 13 | 19.12% |
| tidb | node | 8 | 11.76% |
| service | tidb | 7 | 10.29% |
| node | service | 2 | 2.94% |
| tidb | tidb | 2 | 2.94% |
| node | tidb | 2 | 2.94% |
| node | node | 1 | 1.47% |

### top1 的模态组合（Top 10）

| 排名 | top1_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 26 | 38.24% |
| 2 | metrics | 23 | 33.82% |
| 3 | logs+metrics+traces | 15 | 22.06% |
| 4 | logs+metrics | 4 | 5.88% |

### GT（best-rank 命中项）的模态组合（Top 10）

| 排名 | gt_best_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 40 | 58.82% |
| 2 | metrics | 21 | 30.88% |
| 3 | traces | 3 | 4.41% |
| 4 | logs+metrics | 3 | 4.41% |
| 5 | logs+metrics+traces | 1 | 1.47% |

### 代表性样本（前 12 条，按 gt_best_rank 由小到大）

| date | uuid | GT_parts | GT_kind | top1 | top1_kind | top1_mod | gt_best_rank | gt_best_mod |
|---|---|---|---|---|---|---|---:|---|
| 2025-06-24 | 5fb39f4b-500 | tidb-tidb-0 | tidb | tidb-tikv-0 | tidb | metrics | 2 | metrics |
| 2025-06-24 | 7db04792-496 | tidb-tikv-0 | tidb | tidb-tidb-0 | tidb | metrics | 2 | metrics |
| 2025-06-27 | 238afe01-573 | checkoutservice-0 | service | cartservice | service | metrics+traces | 2 | metrics+traces |
| 2025-06-27 | 4b98f078-560 | checkoutservice+emailservice | service | checkoutservice-2 | service | logs+metrics+traces | 2 | traces |
| 2025-06-07 | abb62970-110 | adservice | service | adservice-0 | service | logs+metrics | 4 | metrics+traces |
| 2025-06-24 | 2ff9d8a6-501 | aiops-k8s-06 | node | frontend-1 | service | logs+metrics+traces | 4 | logs+metrics+traces |
| 2025-06-28 | e52c3079-598 | cartservice | service | tidb-tidb-0 | tidb | metrics | 5 | metrics+traces |
| 2025-06-07 | 0fccdcad-125 | recommendationservice | service | aiops-k8s-05 | node | metrics | 6 | metrics+traces |
| 2025-06-13 | 49e9814e-266 | frontend+cartservice | service | aiops-k8s-07 | node | metrics | 6 | metrics |
| 2025-06-24 | a0d5329b-489 | checkoutservice | service | aiops-k8s-05 | node | metrics+traces | 6 | metrics+traces |
| 2025-06-07 | a363d826-117 | frontend+cartservice | service | aiops-k8s-05 | node | metrics | 7 | metrics+traces |
| 2025-06-27 | aceeda23-570 | checkoutservice+paymentservice | service | aiops-k8s-03 | node | metrics | 7 | metrics+traces |
