# Week1 学习汇报（智能运维探索：RCA 排序与证据融合）

## 1. 我这一周在学什么

我把智能运维（尤其是 RCA）拆成两个更具体的问题来学习：

1) **候选生成（Recall）**：能不能把 GT 放进候选池。
2) **排序/融合（Rank）**：当 GT 已经进候选池时，能不能把它推到 Top1。

这一周的真实收获来自第 2 点：我反复看到“GT 在 topK，但不在 Top1”，而这会**直接损失 LA**（Top1 命中才计分）。

## 2. 我理解到的关键机制：service / node / TiDB 的跨 kind 矛盾

- 很多失败不是“找不到”，而是“找到但没选上”。
- 造成“没选上”的主因之一是：**service、node、TiDB 三类候选的证据强度不在一个量纲里**，导致排序阶段被“看起来更强”的另一类候选抢走 Top1。

我把这种现象总结为：**跨 kind 冲突（kind conflict）**。

它对 LA 的影响非常直接：

- 只要 Top1 kind 选错，即使 TopK 里有 GT，LA 还是掉。

## 3. 我学到的“证据优先级”方法（按 GT kind 决定先看什么）

这周我逐渐形成一个朴素但很有用的学习框架：

> 先判断 GT 可能属于哪一类（service / node / TiDB），再按该类最可信的证据去验证。

### 3.1 GT 是 service（如 cartservice）时：优先看 traces/logs/graph

- **traces**：该 service 的 span 错误率/延迟是否显著激增；异常是否沿依赖链聚集到它（区分“受害者/传播者”）。
- **logs**：是否出现 service 相关错误模板集中爆发（如 500、timeout、conn reset 等）。
- **graph**：在依赖传播路径上是否处于关键节点（上游集中失败或下游连锁异常的枢纽）。

我学到的点：service 类 GT 的“强证据”往往不在节点资源曲线上，而在调用链与错误模式里。

### 3.2 GT 是 node / k8s node（如 aiops-k8s-xx）时：优先看 metrics + node/system logs

- **metrics**：CPU iowait、磁盘、网络丢包、load、OOM 等是否在故障窗口出现尖峰且对齐。
- **node/system logs**：kernel / kubelet / containerd 等系统层异常是否出现。

我学到的点：node 类 GT 的证据更偏“数值/系统信号”，并且非常容易在融合阶段压过 service 证据，因此更需要“确认证据闭环”。

### 3.3 GT 是 TiDB/TiKV/PD 时：优先看 TiDB 指标 + TiDB 日志

- **tidb/tikv 指标**：store down、raft、region、scheduler、coprocessor、latency、QPS 等是否异常。
- **tidb 日志**：slow query / error logs（txn conflict、lock wait、grpc error 等）是否集中。

我学到的点：TiDB 既可能“证据很硬”（组件级指标很集中），也可能因为融合权重/先验偏置而 under-score；这解释了为什么 TiDB 方向非常需要“温和可控”的策略。