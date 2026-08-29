# 实验索引

> **为什么需要这张表**：结论会被后续实验推翻。直接去读各个 `result.md`
> 很容易读到已过期的结论。**先查这里的「是否被推翻」列，再去读原文。**
>
> 本表只记录**实验之间**的相互推翻关系。骨架 `Tilemega_skeleton.md`
> 由维护者更新；本表中「与骨架 §x.y 冲突」只是实验侧的观察记录，
> 不代表骨架已经改动。

工具链版本对结论至关重要（同一份 MLIR 在 13.1 与 13.3 上结论相反）：
- **R1 / R2**：cuda-tile `8a775693`（2026-01）+ tileiras **13.1** (V13.1.80)
- **V0 / V1**：cuda-tile `af241704`（Tile IR 13.3.3）+ tileiras **13.3** (V13.3.36)

---

## Round 1（E0–E7）

| 编号 | 主题 | 一句话结论 | 是否被推翻 | 结果文件 |
|---|---|---|---|---|
| E0 | 基线工具链走通性 | verifier→translate→tileiras→实机全链路正常 | 否 | `R1R2/experiments/E0_baseline/result.md` |
| E1 | 原语盘点 | 无 device 端函数调用；tile 形状须编译期常量且为 2 的幂；无独立 fence op；token 是纯编译期排序约束 | 否 | `R1R2/experiments/E1_primitives/notes.md` |
| E2 | 异构 tile 分支 dispatch | 单 entry 内分支的资源开销取 **max** 而非 sum | 否（R2-G 推广到 N=8） | `R1R2/experiments/E2_heterogeneous_tiles/result.md` |
| E3 | runtime 上界的 persistent grid-stride loop | 三个循环边界均可为运行时值，全链路通过 | 否 | `R1R2/experiments/E3_persistent_loop/result.md` |
| E4 ⭐ | 跨 block 生产者/消费者自旋 | (a) 不串 token 则自旋循环被**静默消除**；(b) 大 grid 必现死锁 | **(a) 否，(b) 被 V1 推翻** | `R1R2/experiments/E4_spin_wait/result.md` |
| E5 | occupancy/CGA hint 能否修复死锁 | hint 无效；「grid=80 安全」不是确定性阈值而是时序竞态 | **被 V1 推翻**（挂起本身不复现） | `R1R2/experiments/E5_coexistence/result.md` |
| E6 | 最小 3 角色 megakernel | 小 grid（=3）下三角色并发正确，200/200 | 否 | `R1R2/experiments/E6_minimal_megakernel/result.md` |
| E7 | warp-specialization 可控性 | 旧版 cuda-tile 不支持 `num_worker_warps_per_cta` | **部分过期**：当前版本支持，仅限 [4, 8] | `R1R2/experiments/E7_warp_specialization/result.md` |

## Round 2（R2-A–H）

| 编号 | 主题 | 一句话结论 | 是否被推翻 | 结果文件 |
|---|---|---|---|---|
| R2-A | 活锁 vs 死锁判别 | 是局部死锁（PC 20s 不前进）；卡死点在 post-spin reduce 收尾码 | **被 V1 推翻**：那段代码本身是 Bug C 的 UB 现场，且挂起在 13.3 不复现 | `R1R2/experiments/R2_A_livelock_vs_deadlock/result.md` |
| R2-B | test-and-test-and-set 修复 | `CCTL.IVALL` 成功移出循环，但挂起率无改善 | **前半有效**（test-and-test-and-set 写法本身正确）；**后半被 V1 推翻** | `R1R2/experiments/R2_B_relaxed_plus_acquire/result.md` |
| R2-C | 自旋退避 | 小 grid 有量级缓解，grid=170 完全无效 | **被 V1 推翻**（无挂起可退避） | `R1R2/experiments/R2_C_backoff/result.md` |
| R2-D | REG=228 vs occupancy 矛盾溯源 | 是 harness 传错 blockSize=1，不是 driver bug | 否。**V1 加固**：必须读 `EIATTR_REQNTID`，连 `MAX_THREADS_PER_BLOCK` 都不能替代 | `R1R2/experiments/R2_D_occupancy_audit/result.md` |
| R2-E | harness 卫生检查 | 未开始 | pending | — |
| R2-F | 软件流水/双缓冲判定 | 观察到跨迭代批量预取，是否构成双缓冲未定点；**自旋循环体内含 `BAR.SYNC.DEFER_BLOCKING`** | 否（V0 确认 13.3 仍如此） | `R1R2/experiments/R2_F_pipelining/result.md` |
| R2-G | max-not-sum 推广 | 推广到 N=8 成立；loop-carried 跨分支未测 | 否 | `R1R2/experiments/R2_G_max_vs_sum_scaling/result.md` |
| R2-H | 编译+查询墙钟耗时 | 未开始 | pending | — |

## V0：工具链重基线（2026-08-28）

| 编号 | 主题 | 一句话结论 | 是否被推翻 | 结果文件 |
|---|---|---|---|---|
| V0 | tileiras 13.1 vs 13.3 静态对比 | REQNTID 256→128（驻留容量 170→340）；`CCTL.IVALL` 225→28；**但自旋循环内的 `BAR.SYNC.DEFER_BLOCKING` 仍在** | 否。与骨架 §2.2「每 block 256 线程」的常量表述冲突（待维护者裁决） | `V0_toolchain_rebaseline/result.md` |
| V0-GPU | 13.1 侧挂起率扫描 | grid=30 挂 22/50、grid=80 挂 37/50 后因 GPU 污染中断 | 不完整；13.3 侧结论由 V1 取得 | 同上 |

## V1：修正后的跨 block 同步测试（2026-08-29）

| 编号 | 主题 | 一句话结论 | 是否被推翻 | 结果文件 |
|---|---|---|---|---|
| V1-ctrl | 原样未改的测试在 13.3 上 | **300 次零挂起**（grid 3→340） | 否。推翻 E4/E5/R2-A/B/C 的挂起结论；与骨架 §2.4 冲突（待维护者裁决） | `V1_sync_corrected/SUMMARY.md` |
| V1-a | 修 Bug A+C，逐元素消费者 | 全 grid 零挂起 + 全槽位校验通过；修掉 Bug B 让 cooperative 上限 340→2040 | 否 | 同上 |
| V1-b | 修 A+B+C，保留 reduce | 不挂但**静默算错**，失败率随 grid 上升 | 否 | 同上 |
| V1-min | 最小复现（写 1 tile / 读 1 tile） | **阈值尖锐**：grid≤80 全过，grid≥120 全错；丢失整数条 `STG.E` 份额 | 否 | 同上 |
| V1-cuda | CUDA C++ 结构等价对照 | **全 grid 0 失败** → 缺陷隔离到 Tile IR 工具链，非硬件 | 否 | 同上 |
| V1-d | 分散事件（per-block flag） | grid≥128 同样失败 → **per-tile 事件模式不能规避** | 否 | 同上 |
| V1-c | `num_worker_warps_per_cta` 4 vs 8 | 未做 | pending | — |

---

## 实验侧当前的结论（供维护者裁决，尚未回写骨架）

1. 原 R1 / §2.4「大 grid 挂起」在 tileiras 13.3 上**不复现**（V1-ctrl，300 次零挂起）。
2. 取而代之的是一个**新现象**：修掉三处测试 bug 后不再挂起，但 grid ≥ 120 时
   跨 block 数据**静默损坏**，丢失整数条 `STG.E` 份额。已由 CUDA C++ 结构等价
   对照隔离到 **Tile IR 工具链**（非硬件、非内存模型），目前无 workaround。
   V1-d 表明 per-block 事件标志也不能规避。
3. 因此「§2.4 是否可以关闭」与「是否新增一条阻塞项 / 风险条目」都需要维护者决定，
   本表不代行。
