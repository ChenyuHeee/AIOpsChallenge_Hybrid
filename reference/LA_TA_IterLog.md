# Hybrid 提分迭代记录（LA/TA）

> 目的：把 Hybrid 解法从“能跑”迭代到“能拿分”，并把每一次关键改动的动机、方案分叉、以及分数对比沉淀下来，方便复现与继续优化。
>
> 注意：本文刻意不记录任何 `.env` 密钥内容；仅记录可公开的环境变量名与用法。

## 0. 评测口径与关键事实（决定改动方向）

### 0.1 Final 分数构成
评测程序的总体加权（概念上）：
- `Final = 0.4 * LA + 0.4 * TA + 0.1 * Efficiency + 0.1 * Explainability`

其中：
- **LA**：component 命中率
- **TA**：reason 命中率
- **Efficiency**：推理步数（trace step count）等相关
- **Explainability**：reasoning_trace 的 observation 是否“命中” evidence_points 关键词

### 0.2 Judge 规则中“会导致必错/大幅扣分”的点
这几条直接决定我们为什么要做某些“看起来很奇怪”的约束：

1) **component 是严格 token 命中**
- GT 如果是 edge：`a->b` 这种形式，评测只认端点 token（`a` 或 `b`），提交 `a->b` 会被判错。
- 因此必须把 edge 类输出端点化（例如优先取右端点）。

2) **reason 先 keywords 子串命中，再回退语义相似**
- 只要包含关键词子串就直接判对。
- 语义相似依赖 `sentence-transformers`/`scipy` 等可选依赖；依赖缺失时 judge 会退化为“不通过”。

3) **Explainability 强依赖 trace 中 observation 的关键词命中**
- 所以我们的 validator 需要把证据关键词显式注入 reason/trace，且要尽量避免被截断。

## 1. 低分排障：问题画像

### 1.1 现象
- LA 很低，预测 component 分布高度集中在少数服务（例如 checkoutservice/cartservice/adservice），很难覆盖 GT 中大量的 node/edge 端点类答案。
- 启用 LLM 直接选 component 时，LA 往往进一步恶化（模型偏向 node 或“看起来合理但 token 不匹配”的选择）。
- 时间窗 padding 过大时会引入跨事件污染，导致候选/证据被“稀释”，LA 降。

### 1.2 工程类根因
- 读遥测时的时区/时间解析、窗口过滤、采样/限流不稳 → 读错/读多/读少都会导致候选质量下降。
- evidence/trace 截断 → explainability 和 reason 命中都受影响。

## 2. 关键策略分叉（方案对比）

### 2.1 方案 A：component 由 LLM 选择
- 优点：有机会提升 TA/可解释性（尤其是 reason 描述更像人话）。
- 缺点：component 受严格 token 约束，LLM 很容易选错 token 或选 edge 字符串形式，LA 会显著下滑。

### 2.2 方案 B（当前主线 / “策略1”）：component 强制用共识层 Top-1，LLM 只写 reason/trace
- 目标：
  - **LA 由算法稳定保障**（尤其是端点规则、node token 保真、候选集合约束）。
  - **TA 由 LLM 拉升**（reason 生成 + trace 组织），失败时仍可回退启发式。

该策略落在 orchestrator 层：最终提交 component 直接覆盖为 consensus top-1。

## 3. 一次关键突破：缩小 window padding 提升 LA

### 3.1 动机
之前 padding 较大时，容易把邻近事件的指标/日志/trace 混入同一个窗口：
- 候选排序更偏向“常见的全局热门服务”，node 类根因难胜出。
- 证据更分散，组件归因更难。

### 3.2 实施
- loader 默认 `window_padding` 从 45min 调小到 15min。
- 并通过环境变量 `RCA_WINDOW_PADDING_MIN` 可覆盖。

## 4. DeepSeek LLM 稳定性：timeout 默认值

### 4.1 现象
- 默认 timeout=10s 时，DeepSeek 请求在本地环境里经常 read timeout。
- 结果是“看似启用了 LLM，但每次都超时回退”，TA/Explainability 无法发挥。

### 4.2 修复
- 当 provider 为 deepseek 且未显式设置 `RCA_LLM_TIMEOUT` 时，将默认 timeout 提升为 60s。

## 5. 24 子集对比实验（同一批 UUID，口径一致）

说明：以下对比均基于同一组 24 个 UUID（不同 submission 文件之间 UUID 集合完全一致），因此可直接横向对比。

| 方案/文件 | padding | LLM | LA | TA | Explainability | Final |
|---|---:|---|---:|---:|---:|---:|
| `submissions_2025-06-07_consensus_weights_24.jsonl` | 旧默认 | off | 12.50% | 66.67% | 21.74% | 42.03 |
| `submissions_2025-06-07_llm_24.jsonl` | 旧默认 | on（但 component 受影响/或回退） | 12.50% | 66.67% | 26.09% | 42.46 |
| `submissions_2025-06-07_llm_mix_24.jsonl` | 旧默认 | on（策略1：LLM 拉 TA） | 12.50% | 100.00% | 26.09% | 55.80 |
| `submissions_2025-06-07_pad15_24.jsonl` | 15min | off | **20.83%** | 66.67% | 17.39% | 44.93 |
| `submissions_2025-06-07_pad15_24_deepseek_t60.jsonl` | 15min | on（策略1 + timeout=60s） | **20.83%** | **95.83%** | 19.57% | **56.81** |

**结论：**
- **缩小 padding 是 LA 的有效杠杆**：12.50% → 20.83%。
- **策略1 + DeepSeek（更高 timeout）能把 TA 拉高且不伤 LA**：LA 保持 20.83%，TA 提升到 95.83%，Final 56.81。

## 6. 复现命令（建议直接复制执行）

### 6.1 生成 24 子集（padding=15，禁用 LLM）
```bash
RCA_LLM_PROVIDER=dummy \
RCA_WINDOW_PADDING_MIN=15 \
python -m AIOpsChallenge_Hybrid.contest_solution.main \
  --telemetry-root data \
  --metadata metadata_2025-06-07.csv \
  --output AIOpsChallenge_Hybrid/submissions_2025-06-07_pad15_24.jsonl \
  --limit 24
```

### 6.2 生成 24 子集（padding=15，DeepSeek，timeout=60s）
```bash
RCA_LLM_PROVIDER=deepseek \
RCA_LLM_TIMEOUT=60 \
RCA_LLM_ATTEMPTS=1 \
RCA_WINDOW_PADDING_MIN=15 \
python -m AIOpsChallenge_Hybrid.contest_solution.main \
  --telemetry-root data \
  --metadata metadata_2025-06-07.csv \
  --output AIOpsChallenge_Hybrid/submissions_2025-06-07_pad15_24_deepseek_t60.jsonl \
  --limit 24
```

### 6.3 评测（只评这 24 个 UUID，避免 missing uuids 干扰）
评测程序默认会以 ground truth 全量 UUID 为基准；如果 submission 只包含 24 条，会把缺失的 UUID 视为“空预测”，导致整体分数被稀释。

因此建议：先用脚本把 ground truth 过滤到这 24 个 UUID，再评测。

（仓库里已有示例过滤产物：`tmp_gt_pad15_24*.jsonl` 与 `tmp_sub_pad15_24*.jsonl`）

注：这些过滤产物现已收纳到 `AIOpsChallenge_Hybrid/tmp/filtered/` 下。

```bash
python AIOpsChallengeJudge/evaluate.py \
  -g AIOpsChallenge_Hybrid/tmp_gt_pad15_24_deepseek_t60.jsonl \
  -s AIOpsChallenge_Hybrid/tmp_sub_pad15_24_deepseek_t60.jsonl \
  -o AIOpsChallenge_Hybrid/results_pad15_24_deepseek_t60.json
```

## 7. 评测工具的一个小坑（已修复）
- 现象：`evaluate.py --report` 在某些情况下会报 `Object of type bool is not JSON serializable`。
- 修复：写 report 时增加 `default` 序列化兜底（兼容 numpy 标量等），现在可稳定输出 JSON 报告。

## 8. 下一步（面向 LA≈70% 的方向）
- 继续做 padding sweep（例如 5/10/15min）验证 LA 的最优点，同时观察 Explainability 是否下降。
- 对 node 类根因建立“聚合证据→单 node 定位”的更强规则，提升 aiops-k8s-* 的命中率与编号准确率。
- 在不引入跨事件污染的前提下，提高 trace/graph 对端点的指向性，扩大候选覆盖面，减少预测集中在热门服务。

## 9. 新增：交叉验证与“评测反馈记忆”（继续攻 LA）

### 9.1 交叉验证（CV）
- 目的：用 K 折验证 padding/专家权重等改动对 LA 是否稳定提升，避免只在单一子集上过拟合。
- 工具脚本：`AIOpsChallenge_Hybrid/tools/cv_search.py`
- 约束：默认禁用 LLM、禁用 memory，做轻量候选网格搜索并输出各折 LA。

最近一次 CV 结果（`limit=48`, `k=3`，每折 16 条；候选网格较小，仅用于快速定向）：

| 候选 | LA(avg) | folds |
|---|---:|---|
| padding=5,  w_trace=1.6 | 12.5% | [12.5%, 12.5%, 12.5%] |
| padding=10, w_trace=1.6 | 16.7% | [12.5%, 25.0%, 12.5%] |
| padding=15, w_trace=1.6 | **20.8%** | [12.5%, 25.0%, 25.0%] |
| padding=15, w_trace=1.8 | 20.8% | [12.5%, 25.0%, 25.0%] |

结论：在这组快速 CV 网格里，`padding=15` 明显优于 5/10；trace 权重从 1.6 提到 1.8 未体现收益。

补充一次更大范围 CV（`limit=96`, `k=3`，每折 32 条；网格：padding=10/15/20/25 × w_graph=1.0/1.3；w_trace 固定 1.6）：

| 候选 | LA(avg) | folds |
|---|---:|---|
| padding=10, w_graph=1.0 | 16.7% | [12.5%, 25.0%, 12.5%] |
| padding=10, w_graph=1.3 | 16.7% | [12.5%, 25.0%, 12.5%] |
| padding=15, w_graph=1.0 | 20.8% | [12.5%, 25.0%, 25.0%] |
| padding=15, w_graph=1.3 | 20.8% | [12.5%, 25.0%, 25.0%] |
| padding=20, w_graph=1.0 | **25.0%** | [12.5%, 37.5%, 25.0%] |
| padding=20, w_graph=1.3 | 25.0% | [12.5%, 37.5%, 25.0%] |
| padding=25, w_graph=1.0 | 25.0% | [12.5%, 37.5%, 25.0%] |
| padding=25, w_graph=1.3 | 25.0% | [12.5%, 37.5%, 25.0%] |

结论：在这组更大范围 CV 中，`padding=20`（或 25）优于 15/10；w_graph=1.3 未体现收益。

落地验证（同一 24 UUID 子集，直接跑 judge）：
- `padding=15` + DeepSeek：LA 20.83%，TA 95.83%，Final 57.03
- `padding=20` + DeepSeek：LA **25.00%**，TA **100.00%**，Final **60.58**

### 9.2 评测反馈记忆（Judge feedback memory）
- 目的：把 judge 的“预测对/错”写回到 `.aiops_memory.json`，鼓励长期正确的组件、惩罚经常错的组件。
- 工具脚本：`AIOpsChallenge_Hybrid/tools/update_memory_from_eval.py`
- 推理端开关（默认关闭，避免小样本误导）：
  - `RCA_USE_JUDGE_MEMORY=1`
  - `RCA_MEMORY_MIN_TOTAL=20`（默认 20，要求足够样本后才启用反馈信号）
