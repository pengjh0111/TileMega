# Queue-driven L2 end-to-end result

## T1 intervention (2026-09-08)

新的 seq=4/128 分片探索扫描见 [T1 report](../L2_ATTRIB/t1_result.md)，
含每格 25 轮配对 L2/L1 与 bootstrap 区间。该扫描不代替冻结源码的最终四臂
归因门槛，也尚未覆盖 seq=512。下文保留的是 T1 改动前结果，不是本轮新数据。

❌ 冻结源码最终四臂发现 notify 显著增加，按停止规则不扩展到 seq=512。
以下是该最终数据的进程内 L2/L1 中位数及 95% bootstrap CI（每格 25 轮），
完整配对差/Wilcoxon 位于 `../L2_ATTRIB/t1_final/paired.tsv`：

| model / seq | base L2/L1 [CI] | load+split+S128 L2/L1 [CI] |
|---|---|---|
| gqa2 / 4 | 1.071096 [1.067080, 1.072403] | 1.106965 [1.106436, 1.109168] |
| gqa2 / 128 | 1.071429 [1.071041, 1.072114] | 1.096233 [1.094551, 1.098136] |
| mha4 / 4 | 1.085350 [1.082968, 1.086995] | 1.115138 [1.111622, 1.117543] |
| mha4 / 128 | 1.090238 [1.089263, 1.091842] | 1.118637 [1.117069, 1.119201] |

Evidence status: ✅ measured on RTX 4090 (`sm_89`), BF16, 2026-09-07, after
logical-task events, kAll aggregation, and selective event publication. Every
number from the stage loop and the two intermediate queue implementations is
excluded.

## L2 versus L1

`run.sh` uses `κ=1`, the winner of the independent final κ sweep. Each of 25
fresh processes times L1 and L2 in the same run. The statistic is the median
within-process ratio, with a 20,000-resample bootstrap CI and a two-sided
Wilcoxon signed-rank test.

| model | seq | L1 median (ms) | L2 median (ms) | paired L2/L1 | 95% CI | p | L2 faster |
|---|---:|---:|---:|---:|---|---:|---:|
| gqa2 | 4 | 0.409600 | 0.443392 | 1.082707 | [1.081788, 1.085213] | 1.293e-5 | 0/25 |
| gqa2 | 128 | 0.568544 | 0.616448 | 1.080851 | [1.076168, 1.082734] | 1.302e-5 | 0/25 |
| gqa2 | 512 | 1.781984 | 1.938432 | 1.087256 | [1.086682, 1.087949] | 1.306e-5 | 0/25 |
| mha4 | 4 | 0.804000 | 0.883648 | 1.096613 | [1.095202, 1.100458] | 1.304e-5 | 0/25 |
| mha4 | 128 | 1.139712 | 1.255424 | 1.101495 | [1.099196, 1.102959] | 1.304e-5 | 0/25 |
| mha4 | 512 | 3.563520 | 3.967776 | 1.113357 | [1.112638, 1.114281] | 1.305e-5 | 0/25 |

✅ All **150/150** fresh processes pass the fixed PyTorch comparison and
L0.5/L1/L2 hash check. ❌ L2 is not faster: its residual event/queue cost is
8.1–11.3%, and every paired observation favors L1. Logical-task publication
therefore buys observable overlap, not an end-to-end win on these models.

## Exact windows versus forced `kAll`

`run_windows.sh` uses one `κ=1` executable and changes only host materialization
through `TILEMEGA_FORCE_ALL_DEPENDENCIES`. Exact mode allocates fine events for
narrow edges; all mode represents every incoming edge by one producer-stage
aggregate. Both arms pass 25/25 in every cell (**300/300** total).

| model | seq | exact median (ms) | all median (ms) | paired all/exact | 95% CI | p |
|---|---:|---:|---:|---:|---|---:|
| gqa2 | 4 | 0.443392 | 0.447488 | 1.006944 | [1.006897, 1.009310] | 1.307e-5 |
| gqa2 | 128 | 0.617312 | 0.615424 | 0.998180 | [0.996477, 0.998600] | 7.476e-4 |
| gqa2 | 512 | 1.939456 | 1.916800 | **0.988291** | [0.987756, 0.988906] | 1.307e-5 |
| mha4 | 4 | 0.882688 | 0.884736 | 1.003218 | [1.001277, 1.004407] | 1.569e-3 |
| mha4 | 128 | 1.254592 | 1.251328 | 0.996745 | [0.994304, 0.999821] | 1.569e-3 |
| mha4 | 512 | 3.966784 | 3.921920 | **0.988372** | [0.987622, 0.990447] | 1.307e-5 |

✅ The window is now a measurable execution choice: every confidence interval
separates from one. ⚠️ Its value changes sign. Exact readiness wins at seq=4
by 0.694% / 0.322%, but aggregate `kAll` wins at seq=512 by 1.171% / 1.163%.
The number of fine events grows with sequence length, and their publication
cost overtakes the overlap saved on these two DAGs. Thus the old “windows are
constant across placement” result is invalid, but “narrower is always faster”
is also false.

Direct overlap evidence is in `../OVERLAP/`; the event-cost decomposition is
in `../L2_ATTRIB/`.
