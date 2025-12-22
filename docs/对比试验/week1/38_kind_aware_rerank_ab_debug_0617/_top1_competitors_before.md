## Top1 竞争对手分析（candidate=before_baseline；topK=80）

- N=24, LA=25.00%, GT-in-topK=58.33%
- 重点样本：GT 在 topK 但未命中（miss_but_gt_in_topk）= 8

### Top1 最常见竞争对手（Top 20）

| 排名 | top1 组件 | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | aiops-k8s-05 | 2 | 25.00% |
| 2 | aiops-k8s-08 | 1 | 12.50% |
| 3 | currencyservice-0 | 1 | 12.50% |
| 4 | aiops-k8s-06 | 1 | 12.50% |
| 5 | checkoutservice-1 | 1 | 12.50% |
| 6 | currencyservice | 1 | 12.50% |
| 7 | aiops-k8s-07 | 1 | 12.50% |

### GT 类型 vs top1 类型（计数）

| GT_kind | top1_kind | 次数 | 占比 |
|---|---|---:|---:|
| service | node | 4 | 50.00% |
| service | service | 3 | 37.50% |
| tidb | node | 1 | 12.50% |

### top1 的模态组合（Top 10）

| 排名 | top1_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 3 | 37.50% |
| 2 | metrics | 3 | 37.50% |
| 3 | logs+metrics+traces | 1 | 12.50% |
| 4 | logs+metrics | 1 | 12.50% |

### GT（best-rank 命中项）的模态组合（Top 10）

| 排名 | gt_best_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 4 | 50.00% |
| 2 | metrics | 4 | 50.00% |

### 代表性样本（前 12 条，按 gt_best_rank 由小到大）

| date | uuid | GT_parts | GT_kind | top1 | top1_kind | top1_mod | gt_best_rank | gt_best_mod |
|---|---|---|---|---|---|---|---:|---|
| 2025-06-17 | c3f4fde9-330 | checkoutservice+emailservice | service | checkoutservice-1 | service | metrics+traces | 2 | metrics+traces |
| 2025-06-17 | 83524cef-323 | tidb-tikv-0 | tidb | aiops-k8s-06 | node | metrics+traces | 3 | metrics |
| 2025-06-17 | 76222cb2-331 | currencyservice | service | currencyservice-0 | service | logs+metrics | 4 | metrics+traces |
| 2025-06-17 | 4fd42adb-327 | adservice | service | aiops-k8s-08 | node | logs+metrics+traces | 6 | metrics+traces |
| 2025-06-17 | fc3737af-320 | productcatalogservice | service | aiops-k8s-07 | node | metrics | 10 | metrics |
| 2025-06-17 | a42d2eb4-319 | emailservice | service | aiops-k8s-05 | node | metrics | 19 | metrics |
| 2025-06-17 | e539dea4-322 | checkoutservice | service | aiops-k8s-05 | node | metrics | 23 | metrics+traces |
| 2025-06-17 | e1562aa2-335 | checkoutservice-2 | service | currencyservice | service | metrics+traces | 45 | metrics |
