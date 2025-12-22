## Top1 竞争对手分析（candidate=after_kind_aware_rerank；topK=80）

- N=24, LA=20.83%, GT-in-topK=58.33%
- 重点样本：GT 在 topK 但未命中（miss_but_gt_in_topk）= 9

### Top1 最常见竞争对手（Top 20）

| 排名 | top1 组件 | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | tidb-tidb-0 | 3 | 33.33% |
| 2 | tidb-tikv-0 | 3 | 33.33% |
| 3 | currencyservice-0 | 1 | 11.11% |
| 4 | checkoutservice-1 | 1 | 11.11% |
| 5 | currencyservice | 1 | 11.11% |

### GT 类型 vs top1 类型（计数）

| GT_kind | top1_kind | 次数 | 占比 |
|---|---|---:|---:|
| service | tidb | 4 | 44.44% |
| service | service | 3 | 33.33% |
| node | tidb | 2 | 22.22% |

### top1 的模态组合（Top 10）

| 排名 | top1_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics | 6 | 66.67% |
| 2 | metrics+traces | 2 | 22.22% |
| 3 | logs+metrics | 1 | 11.11% |

### GT（best-rank 命中项）的模态组合（Top 10）

| 排名 | gt_best_modalities | 次数 | 占比 |
|---:|---|---:|---:|
| 1 | metrics+traces | 6 | 66.67% |
| 2 | metrics | 3 | 33.33% |

### 代表性样本（前 12 条，按 gt_best_rank 由小到大）

| date | uuid | GT_parts | GT_kind | top1 | top1_kind | top1_mod | gt_best_rank | gt_best_mod |
|---|---|---|---|---|---|---|---:|---|
| 2025-06-17 | c3f4fde9-330 | checkoutservice+emailservice | service | checkoutservice-1 | service | metrics+traces | 2 | metrics+traces |
| 2025-06-17 | 3d284cf0-333 | aiops-k8s-06 | node | tidb-tidb-0 | tidb | metrics | 3 | metrics+traces |
| 2025-06-17 | 76222cb2-331 | currencyservice | service | currencyservice-0 | service | logs+metrics | 4 | metrics+traces |
| 2025-06-17 | bbfefe10-321 | aiops-k8s-06 | node | tidb-tidb-0 | tidb | metrics | 4 | metrics+traces |
| 2025-06-17 | 4fd42adb-327 | adservice | service | tidb-tikv-0 | tidb | metrics | 7 | metrics+traces |
| 2025-06-17 | fc3737af-320 | productcatalogservice | service | tidb-tikv-0 | tidb | metrics | 12 | metrics |
| 2025-06-17 | a42d2eb4-319 | emailservice | service | tidb-tidb-0 | tidb | metrics | 19 | metrics |
| 2025-06-17 | e539dea4-322 | checkoutservice | service | tidb-tikv-0 | tidb | metrics | 25 | metrics+traces |
| 2025-06-17 | e1562aa2-335 | checkoutservice-2 | service | currencyservice | service | metrics+traces | 45 | metrics |
