# TileMega 待办

> **文档地图（v2.1）**：设计与契约 → `TileMega_skeleton.md`；实现状态 → `docs/STATUS.md`；待办 → `docs/TODO.md`；实测发现 → `docs/FINDINGS.md`；v2.0 待办原文 → `docs/archive/TODO_v2.0.md`；开工前验证计划 → `docs/VERIFICATION_PLAN.md`。每类信息只有一个权威位置，其他位置只放指针。
>
> 本文件是待办的唯一权威来源（v2.1 起由 skeleton §7 迁入）。v2.0 §7 原文见 `docs/archive/TODO_v2.0.md`；本文件 §2 为其中未关闭条目的原文副本，§3 为已完成条目的索引。

## 0. 约定

> `[ ]` 待办 `[~]` 进行中 `[x]` 完成 `[!]` 阻塞 `[-]` 已放弃（保留并注明原因）
>
> **前置**：`docs/VERIFICATION_PLAN.md` 的项目先完成。
>
> **对所有"分析层"验收项的一条常设附加条件（本轮补入，代价见下）**：
> 一项分析工作只有当它的输出**进入生产路径**时才算通过——不是"算法有单测"、
> 也不是"探针能复现表格"，而是生成的 `.cu` / 求解器的决策里能指出哪一处用了
> 它，并且把它拿掉会看得见地变化。
> 这条不是事后总结的漂亮话，它有价签：P3.3 的验收（§2.7 的 13 条边全部自动
> 推出、isl 整表交叉验证）与 P3.5 的验收（global 事件张量、release/acquire
> 顺序、ABA 场景、与 L1 逐位一致 50/50）**各自都通过了**，两条合起来仍然漏掉
> 了"推导出的 `C` 必须进生成器"：`Frontend.cpp` 从不调用
> `CouplingDerivation`，IR 里的 `CouplingMapAttr` 是 `{ [0] -> [0] }` 占位，
> codegen 只能发 `kAll`，于是一条依赖退化成一次 barrier。这一个缺口同时解释
> 了三个负结果——L2 比 L1 慢、κ 的收益侧恒为零、编排层量到 −1.4%——而每一个
> 都曾被当作独立的结论记在案。验收项之间的**接缝**没有主人，就是这样漏的。
>
> 本轮又发现同一模式的第二种形态：§5.4 明写“worker schedule + 逐 task wait”，
> 实现却是“无 schedule + stage 外循环 + stage wait”。它仍通过了“生成的
> L0.5/L1 与手写版逐位一致”，因为手写版与生成版共享同一个 stage-loop 偏差。
> 因此新增常设条件：**参照实现若与被测实现共享控制流，逐位一致只能证明二者
> 一致，不能证明执行模型符合设计。** 对调度/同步的验收必须同时包含结构检查
> （生成表确实被 kernel 读取）和一个旧模型不可能产生的可观测量。本轮的可观测量
> 是跨 stage 提前启动，旧实现按构造为 0，新队列 50 个全新进程
> 的均值为 34.36% / 40.18%。
>
> 同一缺陷还出现了第三层：第一版队列虽然逐 `TaskRef` 调度，却把事件键写成
> `(producer stage, owner CTA / κ)`，并只在一个 CTA 完成该 stage 的最后一个
> task 时发布。这保证正确但没有实现逻辑 task 粒度；当 task 数大于 worker 数时，
> 消费者仍被迫等待同一 CTA 拥有的无关工作。现改为按
> 每 stage 一个聚合行加 `(stage, logical_task / κ)` 精细行的前缀表索引事件，
> 并由每个 `TaskRef` 完成时同时贡献精细与聚合 arrival。
> 第一版队列的 κ / Place / L2 数字再次全部作废并重测。由此再加一条结构验收：
> **不仅要检查 kernel 是否读取 task schedule，还要检查 readiness 的事件键与发布点
> 是否真由 logical task 定义。**

> **v2.1 新增常设条件**
>
> 1. 每条 L2 性能结论都必须注明执行器配置：π 模式、σ 模式、W、同步协议版本、κ、发射策略。不同执行器配置之间的结论不得复用。
> 2. L2 性能验收必须同时报告放置吞吐上界与关键路径上的每跳同步时间，否则"同步贵"与"没有并行余量"无法区分（F-126）。
> 3. 验收门在实现前写定；未过门的项不打勾，负结果保留并写入 FINDINGS。
> 4. 每个 EX 条目对应一次独立的执行 prompt，§1.3 为初版规格，由对应 prompt 细化。验收门一旦写入执行 prompt，实现后不得修改。执行中在 §1.2 台账更新状态与证据列。

## 1. 主线：执行模型与执行感知求解（EX，v2.1）

> 依据：`docs/STATUS.md` §1.5.2（差距 G1–G14）、`TileMega_skeleton.md` §5.7（Plan 契约）、F-126–F-130。
>
> 术语：
> - **每跳延迟**：从生产者 task 的事件发布完成，到消费者观测到该事件（最后一次 poll 成功）的时间差；同 worker 依赖不计。
> - **放置吞吐上界**：去掉 wait 与 notify 后的 L2 时长（`neither` 臂），或由 EX-D1 重建的"各 worker 的 task 时长之和"的最大值。
> - **研究门**：在同会话配对测量下，求解器选出的 Plan 使实测 L2 < L1，且 CI 不含 1。

### 1.1 依赖与里程碑

```
EX-D1 ──► EX-D2 ──► {EX-E1, EX-E3, EX-E4}（先后由 EX-D2 的分叉规则决定）
EX-E1 ──► EX-E2 ──► EX-S2
EX-D1 ──► EX-S1 ──► EX-S2 ──► EX-S3
EX-E1 ──► EX-S5
EX-E1..E4 ──► EX-E5 ──► EX-S4
EX-V1 与所有性能项并行；EX-C1 任意时间（slot 部分随 EX-E1）
```

| 里程碑 | 条目 | 退出条件 |
|---|---|---|
| M1 诊断 | EX-D1、EX-D2 | 分叉结论写入 FINDINGS |
| M2 执行器能力 | EX-E1、EX-E2、EX-E3 步骤 1–3、EX-E4 第 1 步 | 任意合法 Plan 可执行且正确；每跳延迟有测量 |
| M3 执行感知求解 | EX-S1、EX-S2 | 研究门 |
| M4 扩展 | EX-S3、EX-S5、EX-E3 步骤 4–6、EX-E4 第 2 步、EX-E5、EX-S4 | 各自的验收门 |

（⚠️ v2.1 第二轮：M3 的退出条件「研究门」本轮未达成，M3 未退出。上面的依赖图不变，但建议的推进顺序追加一条：EX-S2 在 `W = 1` 下已经到顶（纯 EFT 与无代价模型的闭式 wavefront 相差不到 3%，F-150），所以 M3 的下一次尝试应排在 EX-E3 与 EX-E2 之后，而不是继续在放置轴上迭代。原顺序保留不删。）

### 1.2 条目台账

状态词沿用 ROUND5_LEDGER：未开始／进行中／已验证／触发停止门槛／待外部条件／经用户批准取消。实现与验证分列；代码存在不等于完成；历史数据不得冒充本轮运行。

| ID | 范围与验收（不可删减，详见 §1.3） | 依赖 | 实现状态 | 验证状态/剩余项 | 证据、commit |
|---|---|---|---|---|---|
| EX-D1 | trace v2：逐 slot 时间线、每跳延迟、HOL、关键路径重建 | — | 已验证（含 R4 task DAG 归因维护） | D1-a/b/c/e 通过；**D1-d 触发停止门槛**：按 §3.6 节点定义重建误差 12.10%–14.51%（门槛 5%），原因已测定为节点权重未计生产者自身 publish（补上后 2.08%–3.17%），未放宽定义；⚠️ sm_120 复算后该停止门槛加剧：重建误差 14.33%–20.07%，publish 修正只收到 4.73%–6.51%（四格中两格仍超 5%），说明该修正不是架构无关的解释（F-141） | `docs/experiments/TRACE_V2/`（`resolution.md`、`analysis.md`、`raw/perturbation.txt`、`sass_identity/`）；F-131、F-132、F-133、F-134；commit 477f5432、11c0199f、37c4a1a6；sm_120 复算见 `docs/experiments/sm120_round_one_20260912.md` 与 `TRACE_V2/raw_sm120/`，F-137、F-138、F-141 |
| EX-D2 | 余量诊断 + 跨 stage 连续轮询放置 | EX-D1 | 已验证 | D2-a..D2-e 全部通过；mode 5 两模型 × seq∈{4,128} 各 50/50；四臂 8 组合 × 4 cell × 25 轮无缺样；`FORK rule=2` 由脚本产出；✅ sm_120 复算通过：400/400 正确、mode 5 比值 0.6514/0.7403/0.6313/0.7491、脚本再次判定 `FORK rule=2`（F-139、F-140）；⚠️ real-width 只覆盖 seq=4，seq=128 因磁盘耗尽 FAIL，headroom 未在 sm_120 复现（F-142） | `docs/experiments/PLACE_ROTATE/`（`raw/summary.tsv`、`raw/fork.txt`、`raw/place_stats.txt`、`headroom.md`）；F-135、F-136；commit f2f5b558、5c04690e；sm_120 复算见 `PLACE_ROTATE/raw_sm120/`（`summary.tsv`、`fork.txt`、`insitu/`），F-139、F-140、F-142 |
| EX-D3 | TaskBody 内 setup / load_wait / mainloop / epilogue 分相 | EX-D1 | 已实现（独立默认关闭） | 四格各 50/50；参考扰动最大中位 1.045860；135568 节点闭合误差 0；最终默认位同见 H2 | F-180–F-182；`PHASE/`、`JOINT/summary.md` |
| EX-E1 | Plan 契约：求解器输出 (π, σ, W, …)，host 按 Plan 物化 | EX-D2 | 已验证 | E1-a..E1-e 全部通过：legacy 位同 diff 为空（两模型 sha256 逐字节相同，正对照为每模型恰好一行差异）；模式 0/4/5 经新契约物化后 36/36 dump 文件与 `TILEMEGA_PLACEMENT` 逐字节相同（24/24 cell PASS，`mode_identity` 仅 E2E_TIME/E2E_ITER 行不同）；四组单元测试含两个负对照全部通过；CTest 全绿且 SEQSCAN 子集 12/12 各 50/50；`lib/Codegen/Codegen.cpp` 内仅剩一处消费调用 `solver::BuildVariantStageSchedule`，非调度决策。本轮 `window` 恒为 1，执行器语义未改 | `docs/experiments/PLAN_CONTRACT/`（`legacy_identity/`、`mode_identity/`、`sigma/`、`seqscan/`、`h4_grep.txt`、`ctest.txt`）；F-143、F-144；commit bbe813e5、157b5960、e7b4c5cc、e3b8934c、acbad55a |
| EX-E2 | 窗口执行器（W）+ 窗口感知的提升与本地依赖 | EX-E1 | 已验证（机制成立，增益为负）| E2-a：W ∈ {1,2,4} 各 30 格 1500/1500 全通过；E2-b：W=1 与现状无显著差异，四格 1.0000/1.0000/0.9990/1.0007、pooled 1.0000 [0.9994, 1.0003]、p 0.2158–0.7760；**E2-c 负对照按 H4 成立**：W=1 的提升/省略规则跑在 W=2 执行器下，四格全败、0/50 通过、失败率 1.0000，共 200 新鲜进程——按 R3 §10，负对照**不**失败才是停止条件，且绝不可读作「规则多余」（F-160）。**E2-d 预期落空**：窗口没有回收 HOL，反而增加——`hol_workers_nonzero` 在 W=4 与 W=1 相同（8/16/256），`hol_reclaimable_ns` 随 W 上升，四个参考格 `measured_l2_ms` 全部变慢（0.428→0.470、0.593→0.646、0.850→0.935、1.278→1.324）；代价侧 `wait_total_ns` +12.4%、`publish_total_ns` +13.5%、`hop_p90_ns` 2048→25600，而 `idle_fraction` 几乎不动（0.7505/0.7562/0.7587），即窗口买到的是更多轮询而不是重叠。⚠️ `cp_lb_nosync` 随 W 膨胀，原因已按精确计数定位：H4 取消同 worker producer poll 的省略后这些 poll 真的执行并进入 trace，`TRACE_V2/analyze.py` 的 `preds` 由记录到的 wait 构造、`longest` 又按「同 worker」判队列边，于是每条多出的 wait 记成一条跨 worker 边（seq 4 精确对上：+80 / +208，同 worker 边不变），因此 **`measured_over_ceiling` 跨 W 不可比**；按 H8 每个配置各自报自己的天花板，研究门仍锚在 §7.3 固定参考。下一步（天花板改由 Plan 的依赖结构重建、W 暂留 1）见 `docs/experiments/PLACE_EFT2/summary.md` §11 与 F-159 | `docs/experiments/WINDOW/`（`raw/summary.tsv`、`raw/matrix.tsv`、`raw/negative.tsv`、`raw/identity.tsv`、`raw/analysis/`）；F-159、F-160；commit 97a2f1f0 |
| EX-E3 | 同步协议 v2（六步，逐步开关、逐步验收） | EX-D1 | 六步已实现；第 1–3 步沿用 R3，C1/C2/C3(a) 已验证，cluster 待 sm_120 | R3 §2 的四步全部实现并逐步单独实测，每步一个开关、默认关（H2）。正确性：四步各自两参考模型 × seq∈{4,128} 共 4 格 200/200 新鲜进程，SEQSCAN 30 格 1500/1500，全部通过。配对 `full` 臂比值（gqa2 s4、gqa2 s128、mha4 s4、mha4 s128，25 轮轮换臂序）：**E3-0 标定等待策略** 0.9954/1.0050/0.9958/1.0053——⚠️ seq=128 两格是回归且 CI 不含 1，隔离跳成本降了 86%（1206.5 → 165.3 ns，168 点曲线）却没有传导到 kernel，原因已定位在四臂分解：策略只动 `wait` 项（0.023–0.041 ms），而 `notify` 项是它的 2.6–3.5 倍，见 F-151；**E3-1 屏障缩减** 0.9997/0.9971/1.0000/0.9983（SASS `bar_sync` 11→8，按 §5 口径每 task ≤ 2），见 F-152；**E3-2 单成员事件直接发布** 0.9951/0.9950/0.9954/0.9954，见 F-153；**E3-3 release 归约发布** 0.9656/0.9749/0.9709/0.9732（p ≤ 2.5e-4，CI 上界均 < 1），是唯一能动 `notify` 的一步（降 29.6%–34.1%），同时 `wait` 升 17.3%–53.6%——按 R3 §5 发布侧与轮询侧分开报告、不合并；`membar` 4→2、计数 `atomics` 9→7，见 F-156。**E3-4 litmus 只出结论、不改规则（H3）**：四个臂 × 18 格 × 50 新鲜进程 = 3600 次，`per_writer` 与 `thread0_fence` 各 900/900；但探测器只在 6 格醒着（acquire=0 且 tile≤4096），结论的边界是 6 而不是 900，且负对照 `no_barrier` 在其中 1 格（grid 128、tile 4096）没有触发，按 CLAUDE.md 如实记录不上取整，见 F-157。⚠️ 屏障条数三处不一致：R3 §1.2 称每 task 6 次、skeleton §5.5.1 称最多 5 次、SASS 计得 11 次——以 SASS 为准，差异记录不调和。✅ 本轮补充：三个数字不是同一量纲——前两个是每 task 的动态执行次数，11 是 `tilemega_l2_kernel` 全核的静态 `bar_sync` 位点数（含 task 循环以外的位点）。且 `verify.py:204` 的 `per_task = BARRIERS_PER_TASK_BASE - drop` 是用 skeleton 声明的 5 减静态差值 3，所以 `E3-1-barriers` PASS 行上的 `per_task=2` 是「声明值减实测值」而非实测，本轮两臂都没有测过动态每-task 屏障数；`TILEMEGA_BARRIER_V2` 的第四处（`ModelHarness.cuh:583`）并非删除，而是把等待屏障由条件执行改为无条件执行，减法模型表达不了这一点，见 F-165。⚠️ 第 5 步（异步发布）与第 6 步（cluster 同步）本轮明确不做（R3 §2），仍未开始。✅ 本轮补测了累积阶梯（`raw_stair1..4`，每格每步 25 轮配对）：E3-0/E3-1/E3-2 三步累积后仅 −1.01/−0.40/−0.70/−0.43%，加上 E3-3 才到 −5.16/−3.48/−4.79/−3.74%；累积值比 `e3_steps.tsv` 四个隔离比值的乘积再多 0.69–1.02 pp（轻度超可加）。⚠️ 阶梯跑在 `legacy_grid_stride` 上（`run_barrier.sh` 不传 `-DTILEMEGA_PLACEMENT`，取头文件默认 0），与 F-161 在 rotate 上读到的 −2.80/−2.73/−2.70/−3.09% 不是同一放置；同一放置下两者一致到 0.35 pp 以内。协议的收益按候选而非按基线变化：去掉 `balanced` 后百分比收益与基线 L2 的相关系数为 −0.003，而 rotate 是唯一四格都低于 3.2% 的候选，并因此在三格丢掉它在配置 A 下的第一名——这正是 F-161 中最优候选由 rotate（A）变为 chain（B）的机制，见 F-163。⚠️ E3-3 的轮询侧（`TILEMEGA_EVENT_LOAD_POLL`，旧负结果按 R3 §5 必须重测而不是引用）已经跑完一轮，但结果是**空数据**，不是复现的负结果：`raw_poll` 四格配对比值 1.001584/0.999687/1.000000/0.999719，CI 全部跨 1、p 0.073–0.81，其中 mha4 s4 的比值与差值恰好是 1.000000 与 0.000000。原因是两臂编译出的是同一个 device image——SASS 逐字节相同，sha256 前 16 位 `e31a9311ca09a94c` 与 `raw_barrier`、`raw_solo` 的基线一致，census 也完全相同（l2 11/4/1/9、l1 11/3/1/4、stage 9/0/0/0）。开关本身有效（单独探针下 `ATOMG.E.ADD.64.STRONG.GPU` → `LD.E.64.STRONG.GPU`，140 行差异），但本轮默认开关下 `EventPoll` 的三处调用点全部在编译期死掉：生成源在自己第 9 行就用 `atomicAdd` 展开 `TILEMEGA_GENERATED_WAIT_global`（早于第 18 行 include），harness 的覆盖只在 `#if TILEMEGA_WAIT_POLICY` 下生效；`ProbeTaskDependencies` 及其唯一调用点在 `#if TILEMEGA_SLOT_WINDOW > 1`（:599–:653）内，默认 W=1；`ClusterSync::StageBarrier` 只从 `#elif TILEMEGA_GENERATED_CLUSTER_DIM > 1` 分支到达，而该宏的 harness 默认值是 1。所以轮询侧的代价至今未测，见 F-162。必须重跑：`OFF_FLAGS=<标定策略>`、`ON_FLAGS=<标定策略> -DTILEMEGA_EVENT_LOAD_POLL=1`（复用 E3-0 已有的 ON_FLAGS/OFF_FLAGS 通路），并先用 SASS diff 确认两臂确实不同，再花 GPU 时间 | `docs/experiments/SYNC_V2/`（`backoff.tsv`、`backoff_policy.tsv`、`backoff_fit.txt`、`spin_interference.tsv`、`e3_steps.tsv`、`raw_wait/`、`raw_barrier/`、`raw_solo/`、`raw_red/`、`raw_poll/`、`raw_notify/`、`raw_litmus/`、`sass_identity/`）；F-151、F-152、F-153、F-156、F-157、F-162、F-163、F-165；commit 9c60161f、cb7ff628、c2eccfe0、55dc7e5c、83846a12、c5233bb2、41d85013、12faeef8 |
| EX-E4 | 分相 TaskBody：CG 推导的无入边操作数预取 | EX-D1、EX-D3 | 未开始（R5 条件未触发） | FORK5 rule=3；本轮仅完成分相 ABI，未实现预取；首次取数暴露中位占比 0.006，主循环内等待另需测量 | F-181、F-182；`PHASE/` |
| EX-E5 | 混合/动态发射策略 | EX-E1–E4 | 未开始 | — | 待填 |
| EX-S1 | 执行模拟器作为 L2 代价模型 | EX-D1 | 已实现（R5 分层求值与发布需求定价） | R5 S1c-a/b 未达：完整求值 140.260/138.348 ms，粗排 rho=0.56124；降级支持 S3；原记录： S1-a/b/d/e 通过，**S1-c 触发停止门槛**：单 Plan 求值参考模型最坏 276.8 ms（门槛 1 ms，超 276.8×）、real-width 99.8 ms（门槛 10 ms，超 10.0×），按 §9 第四条「先报告」记录，未改门；S1-a 逐 task 开始时刻误差 18 格全部报数（\|p50\| 11.6%–40.5%、\|p90\| 19.7%–65.2%、\|max\| 23.0%–81.5% of span），18/18 有符号 p50 为负（系统性早预测）；S1-b Spearman 0.8803、配置内 0.973、argmin 6/6、模式 0 与模式 5 的相对次序 6/6 正确；S1-d `hop_ns(N,R)` 覆盖 N≤256、R≤64 共 96 格 × 4 臂；S1-e 标定集（gqa2 seq{4,128} 模式 0/5 + 争用微基准）与评测集分离且未逐 cell 拟合 | `docs/experiments/SIMULATOR/`（`contention.tsv`、`hop_ns.tsv`、`hop_fit.txt`、`raw/predicted.tsv`、`raw/dump/`、`raw/time/l2.tsv`）；F-145、F-146、F-147；commit f3781430、a177995b；⚠️ sm_120 复测（2026-09-13）：扫描通过，`hop_ns = 448.277066 + 1.134462·log2(1+N/R) − 0.436421·log2(R)`。争用结论复现——`c1 = +1.13 ± 1.96` 与 sm_89 的 `−0.40 ± 2.41` 同样在一个标准误内为零，R2 §0 第四条在两个架构上都是否定；不同的是常数 448 ns 而非 1235 ns、其中退避只占约 32 ns 而非 910 ns，即 F-145 指出的可优化项在 sm_120 上小一个量级。F-145 原文按 `CLAUDE.md` 不改，差异记在其交叉引用与 `docs/experiments/sm120_simulator_place_eft_20260913.md`；S1-c 的求值预算未在 sm_120 上单独计时；⚠️ v2.1 第三轮：S1-c 本轮让路（R3 §2 明确将 EX-S1c 模拟器提速排除在本轮范围之外），仍未解决，**EX-S3 前必须解决**——EX-S3 的外层搜索要在内层反复调用 EX-S1 求值，276.8×/10.0× 的超预算会直接乘进搜索代价 |
| EX-S2 | EFT 放置与排序 + 闭式模板候选（研究门） | EX-S1、EX-E1、EX-E2 | 已验证 | S2-a 通过：34/34 arm-cell 各 50/50 新鲜进程，SEQSCAN 子集 12/12 各 50/50；**S2-b 研究门未达成（负结果）**：eft/模式 5 中位比 1.0096/1.0141/1.0022/1.0273，四格 95% CI 全部完全大于 1，0/4 通过；按 H7 不改门，也不退回「快于 L1」（eft/L1 为 0.7320/0.7276/0.8091/0.8537）；S2-c real-width 已报数：seq4 1.0273 [1.0269,1.0281]、seq128 1.0487 [1.0483,1.0493]，同样不过；S2-d 归因已报数（四臂分解显示模式 5 的无同步下界好 2.4–4.1×，eft 的同步项便宜 3.6–4.2×，两者抵消）；S2-e 逐格 Spearman +0.824/+0.794/+0.812/+0.928 | `docs/experiments/PLACE_EFT/`（`README.md`、`raw/summary.tsv`、`raw/samples.tsv`、`raw/place_stats.txt`、`raw/predicted.tsv`、`raw/correctness.tsv`、`verify.py`）；F-148、F-149、F-150；commit aa68bc19、a1516c62；⚠️ v2.1 第三轮：按 R3 §7 重测六候选 × 四配置（A 无标志 / B 协议 / C 协议+chain / D 全开 W=2），96 个 cell-arm-config 各 25 轮、配置与候选一起轮换。**研究门 S2r-b 0/4**：四参考格最优中位数 280.6/440.3/557.8/833.6 µs，目标 ≤218/328/362/528 µs，只收掉第二轮测量值到天花板距离的 8%/7%/5%/3%（门要求 50%），按 H7 不改门、不改口径。配置序 **B < A < D < W** 在 24 个（格，候选）对中成立 23 个，唯一例外是 gqa2 s128 的 chain 上 A 与 D 并列 519.2 µs；逐对看 B<A、B<D、D<W 各 24/24。四格最优全部是 B（协议单开），没有一格由 C 或 D 拿下——三个杠杆不叠加，窗口吐回的比协议赢到的多。chain 候选的符号随序列长度翻转：seq 4 两格领先（280.6 vs rotate 284.7、557.8 vs 563.1），seq 128 两格分别落后 11.3% 与 36.0%，原因按 F-154/F-155 定位在 fill 阶段而非容量上限。⚠️ 研究门未达成的原因本轮已定位为两类，需要不同的解法（F-164）：其一是算术性的——按 H8 各配置自算天花板，chain 自身的 `queue_lb`（241.7/342.0/485.4/603.1 µs）在四格全部高于 §7.3 目标（218/328/362/528 µs），rotate 在 gqa2 s4 的路径下界 242.7 µs 同样高于目标，即同步代价归零也够不到；按 F-131 的 trace 扰动上界 1.0167 折算后四格全部仍然成立，其中 gqa2 s128 只余 8.4 µs。门要求四格中过三格，而 gqa2 s4 对两个被 trace 的候选都不可达。其二是开销性的——在下界允许目标的三格里，rotate 的 B 配置距自身绑定下界 2.82×/2.29×/2.74×，而协议杠杆只收掉约 3%。⚠️ 仅 rotate 与 chain 被 trace（`TRACE_ARMS`），其余四个候选的下界未测。⚠️ 另记一处口径缺陷：`cp_lb_nosync` 对 chain 系统性偏低，因为 `TRACE_V2/analyze.py` 把同 worker 前驱判为队列边并在 `include_queue=False` 时整条跳过，而 chain 的整条脊正是被有意放在同一 worker 上——gqa2 s4 上 rotate 与 chain 的重建路径同为 20 节点、同为 242688 ns 任务时间，`cp_lb_nosync` 却是 242688 对 122880，被跳过的那段以 `queue_lb` 241664 重新出现（rotate 为 41984）。这与 F-159 记录的「随 W 膨胀」同源而方向相反，修法同为 `summary.md` §11 EX-E2 的第 1 步，即 `preds` 改由 Plan 的依赖结构与物化 σ 重建。⚠️ CHAIN 的 S2c-d 实测臂已补齐（六格 × 25 轮配对，正确性 18 个 arm-cell 各 50/50、共 900/900）：chain/rotate 中位比 1.0070/1.0114/1.1368/1.2581（gqa2 s4、mha4 s4、gqa2 s128、mha4 s128）、real-width 1.0235/1.1329，六格 95% CI 全部完全大于 1，**chain 在任何一格都没有跑赢 rotate**；legacy 对照同轮为 1.1932–1.5345，说明该比较足以分辨比 chain 目标效应大一个数量级的放置差异。预测与实测在四个 seq-128 类格一致（1.213→1.2581、1.135→1.1329、1.067→1.1368），在两个 seq 4 参考格符号相反——求解器预测领先 1.2%/1.1%，实测落后 0.70%/1.14%，偏差集中在这两个 Plan 放下的 2 条与 3 条同 worker 关键路径边上；成本模型把每条记为零代价的省去 hop，实际是一次串行化，这与 F-164 的下界侧记录同源。F-161、F-164、F-166；⚠️ sm_120 复测（2026-09-13，`raw_sm120_retry_20260913/`，F-149 交叉引用）：研究门再次 0/4 且幅度更大，eft/模式 5 为 1.0409/1.0528/1.0542/1.0468（pooled 1.0491 [1.0473,1.0510]），正确性 1200/1200、配对 600/600、S2-e 四格均 +0.812、预测方向 4/4 叫反；首次运行在门之前失败于 plan/grid 不匹配（grid 256 表 vs grid 340 设备），守卫正确、准备步骤有错，已由 `prepare_sm120.py` 改为在目标机器上重新求解（commit 2918f55b）；real-width 与完整 `verify.py` 未在 sm_120 复测 |
| EX-S3 | g / split-K / κ / W / residency 与 Plan 联合搜索 | EX-S2、EX-D3 | 已实现（缩小候选集的降级搜索，W=1） | R5 参考四格研究门 4/4；所选配置四格各 50/50；全候选集 top-3% 尚无实测证明；SEQSCAN 12 格各 50/50（600/600）；real-width 两格各 50/50，配对比 0.873807 / 1.009736（后者 CI 跨 1） | F-184、F-185；`JOINT/summary.md` |
| EX-S4 | 发射策略作为 variant 级决策 | EX-E5 | 未开始 | — | 待填 |
| EX-S5 | 参数化 Place：由 ISL 在 seq 区间上证明合法性 | EX-E1 | 未开始 | — | 待填 |
| EX-V1 | real-width 作为 L2 主基准 + 逐机制消融 | — | 未开始 | ⚠️ 外部条件：sm_120 上 real-width seq=128 在 PyTorch 导出阶段 FAIL，运行后文件系统 100% 占满（推断为磁盘耗尽，未进一步隔离）；seq=4 PASS（F-142） | `docs/experiments/sm120_round_one_20260912.md`；F-142 |
| EX-C1 | 清理死字段与遗留头文件 | — | 已验证 | `kLastTaskOfStage` 与 `GeneratedLlamaRuntime.cuh` 已删除，两参考模型生成的 `.cu` 逐字节不变；⚠️ v2.1 第二轮：`TaskPlacement::slot` 经 EX-E1 后仍是只写字段（被消费的 σ 是 `MaterializedPlan::slot`），已删除并保留 `lengths[chosen]++` 的计数副作用，CTest 全绿 | `docs/experiments/PLACE_ROTATE/raw/c1/`；commit 48554a81 |

（⚠️ v2.1 第四轮，2026-09-15：EX-D1 工具修复通过 R4 A-a/A-b/A-c，
32 个放置 dump 加 12 个窗口 dump 共 44/44；原口径保留为 `_legacy`。
EX-E2 的 HOL 结论修订：完整 DAG 下 W=4 在四格都回收了部分 HOL，
但原始配对计时的回退并未被分析器修复消除，W 默认仍为 1（F-167）。
EX-E3 未完成第 4–6 步：复核 litmus 共 3600 个新进程，candidate
900/900，但 grid=128、tile=4096、acquire=0 的无屏障负对照为
50/50 PASS，触发 R4 §11；§8.5 不解封（F-169）。另一本轮编译确认
R3 E3-3 的原子到达实际为 relaxed，加上前置 writer fence 保持序关系，
不能把已有收益归因于 `red.release` 指令（F-168）。EX-S2c 代价感知
修改与重测尚未执行。原里程碑与历史台账保留；恢复顺序是修复敏感性见证、
补齐候选自身 A trace 与有效冻结表、完成 B 五臂，再推进 C/D。
停止报告：`docs/experiments/SYNC_V3/summary.md`。）


<!-- R4_FINAL_BEGIN -->
（⚠️ v2.1 第四轮恢复完成，2026-09-16：此前 2026-09-15 的停止状态保留为历史。
旧停止报告归档于 SYNC_V3/summary_pre_resume.md；当前完整结果见 SYNC_V3/summary.md。
C1/C2 各四参考格 50/50、SEQSCAN 子集各 300/300；新的双套件敏感 litmus
1800 个进程全部符合正/负对照预期，§8.5 已在复核后追加解封注记。C2 的 κ=2
相邻 slot 见证为 68/68/108/140 条，四格 50/50。C3(a) 在 W=2/4 各四格
50/50，窗口对照也全过；RED/shard 组合四格 50/50。cluster 作用域已实现、
sm_89 退化 SASS 与 sm_120 编译/脚本自检有证据，sm_120 硬件验证仍待运行。
G4/EX-E3 的研究门保持默认放置 wait+notify≤barrier，代表配置
local2 四格倍数为 1.3625/1.8065/1.2784/1.7961，达成 0/4（要求 3/4）；
差距缩减与全部五臂见 SYNC_V3/summary.md、F-177/F-178，不能由实现完成推定达门。
G5/EX-E2：修正 HOL 表明 W=4 回收了等待，但分析器修复消除原始计时回退的比例为
0%，窗口默认仍为 1；本轮 shared/control 净差异另由新配对测量给出（F-173/F-177）。
G7/EX-D1：32 个放置 dump 加 12 个窗口 dump 的界与 split 共 44/44 通过，
旧口径以 _legacy 保留，24 个候选自身 target 在新 A trace 后单独冻结且不再改动。
EX-S2c 的价格测试与队列放置修正已测，mha4 s128 保持 35 跳，参考正确性四格
50/50；六格 chain/rotate、匹配反馈对照和原链对照全部 25 轮（F-176）。
窗口 no-wait 探针的漏覆盖在本轮修复，未完成旧矩阵保留、最终矩阵全新重跑（F-175）。
后续优先 EX-S1c+EX-S3，再 EX-S5，最后 EX-E4；本轮未实现这些排除项。）
<!-- R4_FINAL_END -->

### 1.3 条目详述

#### EX-D1 trace v2

- **目标**：重建每个 task 的时间线，量化每跳延迟、HOL 与关键路径构成（G7）。
- **设计要点**：
  - 新增编译开关（如 `TILEMEGA_TRACE_V2`，默认关）。关闭时，默认构建的 SASS 必须与改动前逐字节相同。
  - 每个 slot 由 thread0 写入独立数组，不使用全局原子。记录的时刻为：wait 开始、最后一次 poll 成功（ready）、RunTask 开始、RunTask 结束、发布完成；同时记录 `%smid`、worker、stage、logical_task。每个事件行另记录 epoch 的发布时刻。
  - 先实测目标 GPU 上 `%globaltimer` 的分辨率（连续读取的最小非零增量）并记录。若大于 100 ns，同时记录 `clock64`，并给出每 SM 偏移的估计方法与误差。
  - 离线分析脚本输出：
    - 每跳延迟分布（p50 / p90 / max）；
    - HOL 可回收时间：worker 等待队首期间，其队列中已有就绪后续 task 的时间总和；
    - 每个 worker 的忙与闲；
    - 关键路径构成（task 时间 / 同步 / 空闲）；
    - work 下界与关键路径下界。
- **验收**：
  - 两参考模型的 trace 构建在 50 个全新进程中 50/50 正确；
  - trace 开/关的 L2 配对 25 轮，中位比 ≤ 1.02；
  - 重建出的关键路径长度与实测 kernel 时长之差 ≤ 5%；超出则报告原因，不放宽。
- **停止条件**：若 trace 扰动 > 2% 且无法降低，暂停并报告，不以扰动后的数据下结论。
- **证据目录**：`docs/experiments/TRACE_V2/`。

#### EX-D2 余量诊断与跨 stage 连续轮询

- **目标**：在投入执行器与求解器之前，量出"并行余量"与"每跳成本"哪个是主瓶颈（F-126）。
- **设计要点**：
  - **(a) 离线**：用 EX-D1 实测的 task 时长与每跳延迟，计算 work 下界、关键路径下界与 EFT list scheduling 的模拟 makespan。覆盖两参考模型 seq∈{4,128}；real-width 4×4096（REALMODEL）seq∈{4,128} 能跑则跑。
  - **(b) 在线**：新增一个 host 放置模式，即新的 `TILEMEGA_PLACEMENT` 取值：
    - 映射为 `π(s,t) = (t + base_s) mod grid`，其中 `base_s = (Σ_{stage_order 中 s 之前的 s'} count(s')) mod grid`。效果是把轮询从"每个 stage 都从 worker 0 重新开始"改为跨 stage 连续进行。
    - 队列仍为 stage-major，W=1；只改 host 端 `task_owner` 的初始化。
  - 四臂（full / nowait / neither / l1nosync）在同会话内配对 25 轮；每格用 50 进程验证正确性。
- **预先声明的分叉规则**（写入实现 prompt 后不得修改）：
  1. `neither` 比默认放置下降 ≥ 5%，而 full L2 不降 → 每跳成本为主，先做 EX-E3、EX-E4；
  2. 两者都下降 → 先做 EX-E1、EX-S2；
  3. `neither` 不降 → 该规模下 DAG 缺乏并行余量，先做 EX-E4、融合与 EX-V1。

  同时报告跨 worker 边比例的变化。
- **验收**：每格正确性 50/50；分叉结论与数据写入 FINDINGS。
- **证据目录**：`docs/experiments/PLACE_ROTATE/`。

#### EX-D3 TaskBody 内分相（R5 新增）

- 默认关闭的 `TILEMEGA_TRACE_PHASE` 与 trace v2 独立；只由 thread0 写 slot 私有存储，无新增原子、屏障或轮询写入。
- GEMM 在描述符/指针设置、首块操作数可用、主循环结束、写回完成处记时；SIMT 按结构划分，无法分离的取数留在主循环。
- 两参考模型 × seq {4,128} × 放置 {0,5} 加 real-width 两序列两放置；关键路径与全部节点并列，保留 globaltimer 与 clock64。
- 冻结分叉：`FORK5 rule=3 load_share=0.006 fixed_share=0.232 math_share=0.763 cells=4`。提交早于 S3，实现后所选几何的补充相位不改此行。
- 四段之和含单列的 executor 收尾区间，等于 run_end-run_begin；关闭时默认 SASS 位同依最终 source/artifact 父子提交验收。

#### EX-E1 Plan 契约与物化

- **目标**：让任意合法的 (π, σ) 都能落地（G1、G2，以及 G14 中的 slot 部分），实现 skeleton §5.7.4。
- **设计要点**：
  - **CG**：`tilemega.placement` 承载求解输出，包括：
    - `mode ∈ {legacy_grid_stride, rotate, balanced_legacy, template, host_list_schedule}`；
    - 模板参数或物化策略参数；
    - `window`（W）与 `policy`。

    `map=[0]` 的写法仅作为 legacy 模式保留。
  - **Codegen**：发出 `RuntimeVariantDesc` 中的 SchedulePlan 描述；`BuildVariantSchedule` 迁入 `lib/Solver`，Codegen 只消费；`ScheduleStageDesc` 仅作为 legacy 输入保留。
  - **Host**：在绑定 θ、完成 split-K 改写与驻留 grid 之后，按 (π, σ) 物化每个 worker 的队列；按 §5.7.3 L-d 计算 TaskWait；按 L-a、L-b 校验。`TaskPlacement::slot` 要么被消费，要么删除。
- **验收**：
  - legacy 模式 + W=1：两参考模型在 seq∈{4,128,512} 下物化出的 TaskRef、TaskWait 与事件表与改动前逐字节相同（dump 后 diff）；生成的 `.cu` 除新增字段外逐字节相同（列出 diff）。
  - balanced 模式经新契约得到的队列，与现 `TILEMEGA_PLACEMENT=4` 下的队列相同。
  - 新增单元测试：任意合法的 σ 被严格遵守；含环的 σ 被拒绝。
  - 全部 CTest 通过；SEQSCAN 子集 seq∈{4,128,2048} × past∈{0,512} 各 50/50。
- **证据目录**：`docs/experiments/PLAN_CONTRACT/`。
- （⚠️ v2.1 第二轮：已验证。`TaskPlacement::slot` 这一条本轮查明仍未被消费——被物化路径读取的 σ 是 `MaterializedPlan::slot`——已随 EX-C1 删除，见 commit 48554a81。`window` 字段已进入 Plan 与 `RuntimeVariantDesc`，但本轮恒为 1，执行器语义未改，harness 对非 1 的 W 直接 `exit(2)`。）

#### EX-E2 窗口执行器

- **目标**：消除 HOL，使静态计划能容忍代价模型的误差（G3、G5）。
- **设计要点**：
  - 设备端窗口 W：编译期给上限，运行期取值。
  - 队首未就绪时，对窗口内的后续 slot 做非阻塞探测（每个等待只做一次加载）。
  - 用 shared memory 位图记录本 worker 已完成的 slot。
  - host 按 §5.7.3 L-d 生成提升、省略与本地依赖。
  - W=1 时退化为现行实现。
- **验收**：
  - W∈{1,2,4} 各跑完整 SEQSCAN 矩阵（1500/1500 口径）。
  - W=1 的性能与现行实现无显著差：配对 25 轮，CI 含 1 或 |Δ| < 0.5%。
  - 负对照：构造一个能触发乱序的最小图，证明"W>1 执行器 + W=1 提升规则"会失败（≥ 50 进程，报告失败率）。若在参考模型上触发不了，如实记录，并保留构造图与失败证据。
  - W>1 的性能收益在 EX-S2 之后评估，不作为本项的门。
- **证据目录**：`docs/experiments/WINDOW/`。

#### EX-E3 同步协议 v2

- **目标**：把每跳成本降到不高于 barrier（G4）。
- **步骤**：每步有独立的编译开关与独立的验收。若某一步使 notify 上升，即回退并记录，先例见 F-89。
  1. **去掉冗余的 CTA 屏障**：trace 关闭时去掉 RunTask 之前的屏障；把 RunTask 之后与 NotifyTask 内的屏障合并。目标是每个 task ≤ 2 次。
  2. **单成员事件**（members = 1）：省去 arrivals 原子，直接发布 epoch。
  3. **aggregate 行**：生产者做无返回值的 release 归约，消费者以 acquire 加载轮询计数目标，取代"最后到达者发布 epoch"。此前对"直接轮询计数"的否定结论，得自 thread0 串行 + RMW 轮询的 stage-loop 执行器，必须重测。
  4. **litmus**：比较"`__syncthreads()` 之后仅由 thread0 执行一次 release fence 再发布"与现行的"每个 writer 各自 fence"。
     - 按 F-1/F-3/F-10 的要求：地址复用、小 tile（含 ≤ 4096 元素）、CTA 协作写、grid 64/128/256、每格 ≥ 50 个全新进程，并包含"无屏障"负对照。
     - 通过之前，§8.5 的规则不变。
  5. **发布异步化**（依赖第 4 步）：由一个 warp 完成 fence 与发布，其余 warp 进入下一个 task 的 Prefetch（与 EX-E4 联动）。
  6. **本地与 cluster 同步**（依赖 EX-E2）：同 CTA 依赖改用 shared memory 标志；在具备 `caps.cluster` 的目标上评估 cluster 级同步（承接 P3.5、P4.7）。
- **每步验收**：
  - 两模型 50/50，并通过 SEQSCAN 子集；
  - 四臂配对 25 轮，报告 notify、wait 与每跳延迟（EX-D1）的变化；
  - 目标（不是正确性门）：默认放置下 wait + notify ≤ barrier。
- **不变量**：§8.2 的单调 epoch 不变；§8.5 在第 4 步通过之前不改。
- **证据目录**：`docs/experiments/SYNC_V2/`。

#### EX-E4 分相 TaskBody 与跨 task 预取

- **目标**：把同步延迟与加载延迟移出关键路径（G6）。
- **设计要点**：
  - 分析层按操作数输出"无入边操作数"掩码（例如 GEMM 的权重），并随生成表发出。掩码由每个操作数的读关系 R 与 CG 入边推出，不接受手写标注。
  - TaskBody ABI 增加可选的 Prefetch（skeleton §5.3.1）。
  - **第 1 步**：在等待期间或上一个 task 收尾时，对下一个 slot 的无入边操作数发出 L2 预取。不占 shared memory，不改变 §8.6。
  - **第 2 步**（依第 1 步的结果决定是否进行）：做 shared memory 级的跨 task 流水。这需要修改 §8.6 的生命周期约定，并用 F-40 的闭式与实测核对 occupancy。
- **验收**：
  - 掩码与人工核对表一致（两参考模型的全部 stage）；
  - 正确性 50/50，并通过 SEQSCAN 子集；
  - decode seq∈{1,4} 上 L2 配对 25 轮，报告每跳延迟的变化。
  - 注意：参考模型的权重可能整体驻留在 L2 中，收益应以 real-width 为主评估。
- **证据目录**：`docs/experiments/PREFETCH/`。

#### EX-E5 混合/动态发射策略

- **目标**：为时长依赖数据的 stage（如 ragged attention）提供负载均衡。
- **设计要点**：
  - 每个 producer stage 带 jit 标记，默认由 Tier 或时长方差决定；
  - 就绪队列；
  - 生产者派发即推送（early push）；
  - worker 先执行已就绪的 jit task，再检查 aot 队首。
- **验收**：通过正确性矩阵；在 attention 占比高或 L_s 不均的配置上与 static 对比。依赖 EX-E1–E4。

#### EX-S1 执行模拟器（L2 代价模型）

- **目标**：取代 `CostModel::EventNs`（计数 × 速率），作为 L2 的目标函数（G9）。
- **设计要点**：以离散事件模拟 skeleton §5.7.2 的执行语义。输入包括：
  - 精确的 runtime task DAG（`RuntimeProjection` / `ExactRuntimeTaskGraph`）；
  - `CostModel::TaskInstanceNs` 在实际坐标上给出的 task 时长；
  - 同 SM 共驻 task 的资源向量合并（各道需求相加后取 max）；
  - EX-D1 标定的每跳延迟与 notify/poll 开销。
- **验收**：
  - 与 EX-D1 的 trace 对照（≥ 2 模型 × 3 seq × 3 种放置），报告逐 task 开始时间的误差分布。
  - 在"放置 × 配置"扫描上沿用排序口径：Spearman；模型 top-3 中至少一个落入实测 top-3%。
  - 单个 Plan 的求值时间：参考模型 < 1 ms，real-width < 10 ms。
  - 绝对误差不作为门，但必须报告。
- **证据目录**：`docs/experiments/SIMULATOR/`。
- （⚠️ v2.1 第二轮：排序门（S1-b）通过，求值时间门（S1-c）未通过且差一个数量级以上——参考模型最坏 276.8 ms 对 1 ms、real-width 99.8 ms 对 10 ms。按 R2 §9 第四条这是「先报告」的停止条件，本轮如实记录、未改门、未改算法。现实现是逐事件推进的单线程模拟，复杂度随 task 数与边数增长；若要进入联合搜索（EX-S3），必须先给它一个增量或分层的求值路径。）
- （⚠️ v2.1 第二轮：标定集与评测集的分离按 H8 执行并在 `SIMULATOR/README.md` 中列出；未做逐 cell 系数拟合。S1-a 的误差在 18/18 格上有符号 p50 为负，是系统性早预测而非随机误差，见 F-146。）

#### EX-S2 放置与排序（研究门）

- **目标**：由求解器输出 (π, σ)，使实测 L2 < L1（G2、G10）。
- **设计要点**：
  - 在精确的 task DAG 上做 EFT 式 list scheduling：
    - 优先级为以 ns 计的向上秩，含跨 worker 的每跳延迟；
    - 为每个 task 选择最早完成时间最小的 worker：同 worker 前驱不计同步代价，跨 worker 前驱加上每跳延迟，同 SM 上的两个 worker 按资源向量计干扰；
    - 只使用驻留 grid（resident-only）。
  - 候选还包括 legacy、EX-D2 的连续轮询、AFFINE_PROBE 中的 band/wavefront（F-93、F-97）以及 balanced，由 EX-S1 选优。
  - Label 作为子决策：只有相关 task 同簇时才可选 cluster 同步。
- **验收**：
  - 两参考模型与 real-width 在 seq∈{4,128} 上，EX-S1 选出的 Plan 实测 L2 < L1（配对 25 轮，CI 不含 1）；
  - 每格正确性 50/50；
  - 报告跨 worker 边的比例，以及关键路径上的同步时间。
- **证据目录**：`docs/experiments/PLACE_EFT/`。
- （⚠️ v2.1 第二轮：研究门由「实测 L2 < L1」提高为「实测快于模式 5」，原句保留不改。理由是模式 5 已经通过了 L1 门（F-135，L2/L1 = 0.807/0.724/0.828/0.716），继续用 L1 作分母无法分辨求解器是否带来新的东西。本轮结果为负：eft 对模式 5 的四格中位比 1.0096/1.0141/1.0022/1.0273，95% CI 全部大于 1；对 L1 的比值 0.7320/0.7276/0.8091/0.8537 则远低于 1，即旧门通过而新门未过（F-149）。按 H7 门不回退。）
- （⚠️ v2.1 第二轮：「Label 作为子决策」本轮未实现，仍未开始。）
- （⚠️ 2026-09-13 sm_120 复测：研究门在第二个架构上同样为负，且幅度由 0.2–2.7% 扩大到 4.1–5.4%，四格 95% CI 全部大于 1。门仍未改、未退回「快于 L1」。四臂分解显示机制相同而系数不同：模式 5 的无同步下界好 2.6–9.4×、eft 的同步项便宜 3.1–3.7×，两者不再抵消。另记一条跨设备约束：物化的 EFT 表把 `(worker, slot)` 绑在求解时的驻留 grid 上，跨设备准备必须重新求解而不能搬运——首次运行即被 host 的计划守卫按此拒绝。见 `docs/experiments/sm120_place_eft_retry_20260913.md`。）
- （⚠️ v2.1 第三轮：研究门的定义**第二次**改动，原句与第二轮的改定义说明都保留不改。由「实测快于模式 5」改为「把到天花板的距离走掉一半」：以 `cp_lb_nosync`（关键路径上各 task 实测时间之和）为天花板，`target_cell = ceiling_A + 0.5 × (measured_A − ceiling_A)`，四个参考格的目标为 ≤ 218 / 328 / 362 / 528 µs（gqa2 s4、gqa2 s128、mha4 s4、mha4 s128），要求至少 3/4 格上最优配置的实测中位数 ≤ target 且 95% CI 上界 < target。改动的理由是第二轮的门以另一个候选作分母，无法分辨「求解器更好」与「两个候选一起离天花板很远」；新门的分母是该配置自身 trace 重算出的天花板（H8），不再依赖任何候选。按 H7，本轮门在实现之前固定，未加宽、未缩窄覆盖、未改指标定义。）

#### EX-S3 联合搜索

- **目标**：把 g 与 Plan 联合优化（G8、G11）。
- **设计要点**：
  - 外层搜索 (g, split-K, κ, W, residency)，用 max(work 下界, 关键路径下界) 剪枝；内层用 EX-S2 + EX-S1 评估 top-k；最终编译并实测 top-3。
  - ChainDP（L1 目标）保留，作为 L1 路径与初始候选的来源。
  - 若证实有必要，把 κ 从全局编译宏改为每个 producer stage 的运行期字段。
- **验收**：
  - 在预先定义的候选集上，预测最优落入实测 top-3%；
  - 报告所选 split-K 与 L1 DP 所选的差异——这个差异本身作为结果记录。

#### EX-S4 发射策略决策

- 把 static / static+W / hybrid 作为 variant 级决策，默认由 Tier 决定。在 EX-E5 之后进行。

#### EX-S5 参数化 Place

- **目标**：把不变量 I1 从依赖推广到放置。
- **设计要点**：
  - 把 π/σ 模板写成 task 坐标与 θ 的拟仿射函数；
  - 在每个 seq 区间上，用 ISL 证明 §5.7.3 的 L-a（无环）、L-b（驻留）以及依赖跨度上界；
  - 每个 binary ≤ 2 个变体（F-66）。
- **验收**：区间内的证明通过；在区间端点与 3 个内点处，host 物化结果与模板求值一致。

#### EX-V1 基准迁移

- **目标**：避免只在纯延迟区得出结论（G12）。
- **设计要点**：
  - 以 real-width 4×4096 / intermediate 14336（REALMODEL）和一个真实的 1B 级配置作为 L2 主基准；
  - decode seq∈{1,4,16,64}，另含 128/512；
  - 按机制逐项消融：放置、W、同步 v2、预取。
  - 数值判据不变；条件 7/9 的判据产物按 F-102 处理。
- **验收**：每个 EX 性能结论同时给出参考模型与 real-width 的数据，或说明缺失原因。

#### EX-C1 清理

- **内容**：
  - `kLastTaskOfStage`：删除或启用；
  - `GeneratedLlamaRuntime.cuh`：若没有包含者则删除；
  - `TaskPlacement::slot`：随 EX-E1 处理。
- **验收**：构建与全部 CTest 通过；生成的 `.cu` 逐字节不变。

## 2. v2.0 延续项（原文迁入）

> 本节为 v2.0 §7 中状态为 `[ ]`、`[~]`、`[!]` 的条目原文副本（含 Phase 5 进入条件与 P6.1 说明），按原 Phase 与小节分组；以"→ 承接 / → 协同"开头的行与 P4.8 的注记为 v2.1 追加。完整原文见 `docs/archive/TODO_v2.0.md`。

### Phase 3：分析层（CG 的填充）

#### P3.2 访问关系构造（W / R）

- [ ] Tier 0 对齐静态情形：走纯 CuTe 路径，验证与 ISL 路径结果一致

#### P3.4 Tier 分类与松弛

- [!] **Tier 本身无法由 `C` 推出，这一条按"做不到就说做不到"记录**：Tier 是
      来源属性而不是几何属性——Tier 1 取决于 layout id、Tier 2 取决于任务空间
      是否有运行时 extent、Tier 3 取决于下标是否来自张量，这三样都**不在** `C`
      里。曾经实现过一版按 `isl_map_is_single_valued` 分类的
      `TierClassifier::Classify`（一对多即判 Tier ≥ 2），实测对 §2.7 的 21 条
      派生边**误判 4 条**，其中 `attn_combine→wo`、`add1→rmsnorm2`、
      `silu→wdown` 三条是表里明确标为 Tier 0 的：它们一对多只是因为
      `wait > 1`，这对精确仿射边完全正常，与可解析性无关。该分类器已删除，
      而不是留着一个会误判的实现；Tier 仍由 `CouplingDerivation` 从访问映射
      的上下文（layout id / runtime extent / data-dependent 标志）赋值。

- [!] `docs/experiments/P3/derived-*.md` 自 `b1dbc38` 起未再生成过，`C` /
      `wait` / `fanout` 三列仍是 isl 迁移前的语法；本次一并重新生成。
      `test/Dialect/CouplingGraph/*.mlir` 三个文件引用已删除的
      `#tilemega.closed_form`，全部解析失败，且未接入 ctest（本构建无
      lit/FileCheck），**尚未修复**。

#### P3.5 L2 落地

- [~] cluster 路径：`ClusterSync<Arch>` 原语层已实现（DSMEM `map_shared_rank`
      + 单调 epoch + 两级 barrier），能力开关是 `Caps<Arch>::kCluster`，
      sm_89/sm_90/sm_120 三个目标交叉编译通过，PTX 层面确认 `mapa` /
      `barrier.cluster` 只出现在 kCluster 目标上。**生成器尚未接线**：
      `Codegen.cpp` 仍只接受 `sync_kind = "global"`，megakernel 也仍以
      `<<<grid, threads>>>` 启动。需 sm_90+ 硬件，本机未运行。
      见 `docs/experiments/CLUSTER/`（`run_on_cluster_gpu.sh` 在非 cluster 机器上
      硬失败，退出码 3，拒绝把单 CTA 回退路径当成簇结果测量）
      发现：`if constexpr (Caps<Arch>::kCluster)` **不足以**关掉 cluster 代码——
      `cooperative_groups::this_cluster` 是非依赖名，sm_89 上根本没有声明，
      被丢弃的分支照样做名字查找而编译失败；`.cluster` 作用域的 fence 同样
      要求 `.target sm_90+`。可用性只能由 `#if defined(_CG_HAS_CLUSTER_GROUP)`
      判定，策略仍由 `Caps<Arch>::kCluster` 决定，两者由 `static_assert` 绑定
      **本轮已接线**：`Codegen.cpp` 现在接受 `sync_kind = "cluster"`，并要求
      它与 placement 的簇宽**完全一致**——混用被拒绝而不是被调和（一个 grid
      只有一种 stage barrier），发出 `TILEMEGA_GENERATED_CLUSTER_DIM`；
      `ModelHarness.cuh` 的 `LaunchL1`/`LaunchL2` 相应有 `cudaLaunchKernelEx`
      变体。仍然欠一台 sm_90+ 机器来跑端到端，`run_on_cluster_gpu.sh` 的
      自检改成读 `TargetSpec::Probe().caps.cluster` 而不是比较架构号。
      ⚠️ **能力边界**：sm_120 有 Thread Block Cluster / DSMEM / TMA，但**没有**
      tcgen05，也没有 B200 的 L1.5/LRC 层——一台 5090 上跑通不等于九条资源道
      都被验证过
  → 承接：EX-E3（第 6 步）、EX-S2（cluster 共置）

#### P3.6 生成器一般化（去掉 Llama 结构写死）

- [!] 一般化的范围是 decoder-layer 家族（RMSNorm→QKV→RoPE→KVAppend→
      Attention→O→残差→RMSNorm→SwiGLU→残差），不是任意 ATen 图；层数/
      宽度/GQA-MHA 比例从权重形状结构化推导。换一个不匹配这个数据流形状
      的模型（例如纯 MLP 堆叠）需要新增一条**模式**（`GraphPattern.h` 上的
      声明式数据），而不是新增一条分支：匹配规则里已经没有参数名与 target
      字面量，不匹配的算子降级为一个算子一个 task space 而不是报错。
      详见 §1.5.1 与 `docs/experiments/SEMANTIC/result.md`。

#### P3.7 求解权威迁移到 isl/barvinok（原则三的落地）

- [!] 事件张量 extent 的 verifier 交叉检查退回为"能否求值"：从 `C` 反推
      `image(C_κ)` 需要逐维回答"生产者坐标是否真的依赖这个消费者坐标"，而
      isl 唯一可用的查询（`isl_map_involves_dims`）是语法性的，会把"只是给
      域定界"的坐标也算作相关，从而高估 image。推导侧改为在构造时记录
      哪条约束引用了哪个坐标（`CouplingDetail::occurring`），verifier 拿不到
      这个上下文，故不做该项交叉推导——这是有意不上线一个不可靠的强检查。

### Phase 4：求解层（CG 上的优化）

#### P4.1 代价查询接口

- [ ] `[!]` 缓存尚未进库：目前只有实验脚本按文件缓存
      （`docs/experiments/ORACLE/run.sh`），查询接口本身每次重算。闭式 3µs
      的量级下这不是瓶颈，但 tier 3 的结果值得落盘。
      缓存键必须含源码、目标架构、CUDA 与 CUTLASS 版本（§1.2 原则二）。

#### P4.2 层1 合法性剪枝

- [!] 目标「每算子 8~20 候选」**未达到**：一个 GEMM 算子在 RTX 4090 上的
      合法候选是 **224** 个（300 → 264 形状合法 → 224 装得下 smem → 216 真
      能编译）。不改口径去凑这个数字；8~20 是层2 对齐传播（P4.3）之后才
      可能出现的量级，层1 单独做不到。

- [!] **split-K 这根轴层1 剪不掉**：它是 host 侧的粒度选择而不是 collective
      的 trait，合法性是运行时的 `split_k ≤ k_tiles`。所以 224 个形状要乘满
      5 个 split 因子 = 1120 个配置进 tier 3，而层1 只把 1500 削到 1120
      （25.3%）。P6.2 的 oracle 显示 split-K 单独值 2.0~2.1×、且出现在
      **每一个** top-10% 配置里，把它留给固定默认值就是剪错了轴。

- [!] tier 2 排序的诚实结果：实测最优 `32x32x32s3` 在解析排序里排 **42/224**，
      而排序前 8 名全部**编译失败**（`cute/int_tuple.hpp(890)`，80 例同一处）。
      collective 级 traits 合法 ≠ megakernel 可编译。缺的是访存项，
      因为 `TargetSpec::Calib` 未标定（见 P4.1 上游）。

#### P4.4 代价模型

- [ ] `T_bubble`：软件流水气泡 + Label 带来的 smem 占用延长。**未纳入**。
      §2.2(b) 的填充深度 `d = stages·resident_tiles_per_SM − 1` 在 12 个标定
      点上**不可辨识**——带填充深度的包络与那条直线是同一条线的重参数化，
      恢复出来的 per-CTA setup 常数会继承占用率缩放——所以默认关闭；
      Label 的 smem 项要等簇真跑起来才有可测的东西。
  → 承接：EX-S1（在模拟器中表达流水气泡与共驻）

- [ ] regime 判别：从 (batch, seq_len, prefill/decode 比例) 判 A/B/C。**未实现**。
  → 承接：EX-S4

#### P4.7 层4 Label（簇划分）

- [ ] `[!]` 消融：Label 开 / 关的端到端对比 **未执行**——手上是 sm_89，
      `Caps<Sm89>::kCluster` 为 false，簇 kernel 在这台机器上编不出来
      （这是特意的：`static_assert` 让簇形状的 kernel 无法静默退回平坦
      barrier 再继续报时间）。已经建好并交叉验证过的部分：生成器发
      `TILEMEGA_GENERATED_CLUSTER_DIM`，`GridBarrier` 在 dim>1 时走
      `ClusterSync::StageBarrier`，启动走 `cudaLaunchKernelEx`，grid 被裁成
      簇的整数倍。sm_90 与 sm_120 交叉编译：dim 1 有 0 条 `UCGABAR`、
      dim 2 与 dim 8 各 8 条（PTX `barrier.cluster` 0/4/4）；默认构建的 SASS
      与加簇之前逐字节相同。`docs/experiments/CLUSTER/run_on_cluster_gpu.sh` 把
      megakernel 臂也备好了，缺的只是一台有簇的机器。
  → 承接：EX-E3（第 6 步）、EX-S2；需 caps.cluster 目标

#### P4.8 层5 Place

> v2.1：Round 5 的 balanced 映射（B2）负结果由 EX-S2 承接；P4.8 的 Round 5 更新段落与"已作废的 pre-queue Place 实验"原文见 archive。

- [ ] 分层 DAG 上的 list scheduling（关键路径优先）——**未做，且测量说不要在
      这个目标函数上做**。留空而不是打勾：真正该建的是什么还没确定。
  → 承接：EX-S2

- [ ] 掩盖同步延迟：队列顺序让等待被独立 task 填充——未做。
  → 承接：EX-E2、EX-E4、EX-S2

- [!] 解析半边与硬件半边测的不是同一个尺寸：`place_probe` 绑 `S = 512`，
      E2E fixture 是 `seq = 4`。后者在 `tile_m = 16` 下**只有一个 M 分片**，
      于是 `w(c₁,c₂)` 是常数、目标函数在整个置换群上恒等——五个臂的目标值
      完全相同，而硬件上它们相差 24 个百分点（L1 上 pair +17.0 到 reverse −7.1）。这条先于测量写下：预测对了
      "目标函数是平的"，错在由此推出"置换不值钱"。目标函数是平的只说明它是
      瞎的。两个陈述都不外推到对方的尺寸上。

#### P4.9 实现契约与一致性校验

- [!] IR 侧的访问校验只在 task space 带 `index_map` 时生效，而前端 importer
      目前不发这个属性（它直接从 FX 节点建 task space，没有 `OperatorNode`）。
      没有 `index_map` 时访问模式是不可证伪的——正是 F-17 被发现时的状态。
      不用一个静默通过的检查掩盖它。

### Phase 5：符号化与运行时

#### Phase 5 进入条件（v2.0 原文）

**进入条件（Phase 5 不得绕过）**：

1. ✅ 生成器可在一个 `ModelSpec` 中携带同一算子的多粒度实例，并为每个
   `seq` 区间携带独立、精确的依赖表；host 已是 O(1) 区间查表。
2. ✅ smem/register/occupancy 曲线已量化：1/2/4/8/16 变体对应
   5/5/4/2/1 CTA/SM，因此第一版区间划分以 **2 个变体/binary** 为无损上限；
   不能只根据 C++ `sizeof(union)` 判断，4 变体的首个拐点来自寄存器。
3. ⚠️ **部分满足。** BF16 Tensor Core 已贯通并有独立 sm_89 标定（✅ 两模型
   50/50，SASS 各 96 条 BF16 HMMA）。修正 128-thread occupancy 与 per-SM smem
   预算后 ρ = **0.8984 / 0.8871**，仍低于 FP32 0.9432 / 0.9421，且
   top-1/3/10 全为 0，**区间边界只能当作初值而不是定论**。模型同时高估
   窄-N split=1 与激进 split=8/16；不是可用全局 scale 修掉的误差。
   **c8be09e 后状态修正**：上述 BF16 排名数字来自旧 partial 路径的有偏 PASS
   子集（mha4 的 308 个 RUNFAIL 实为数值判据失败），仅作历史记录，不能用作
   本轮达标判据；修复 partial 后的完整稳态排序尚未重测。
4. ✅ **Round5 A9.3 事件价格功能门通过（不是 A6/P5.1 全部完成）。**
   两模型十二格 κ0/1 预测差非零且方向符合实际等待数；1200/1200 队列字段对账，
   36/36 价格代入位一致。sm_89 使用十二格稳态结构标定，最长队列系数为零，
   poll 的 LOO 最大相对误差仍约95%；不将残差包装成高精度。
   `EVENT_COST/round5_structured.md` 记录完整证据。A6统一任务价格已通过：
   两个GEMM入口各904680位比较相等、FP32排名不降、四份DP计划不变；
   B入口已开放。L2候选级DP未完成；真实task符号价格5120 GEMM/768 scalar点
   已对照，完整CG-interface DP因KVAppend→attention的二次重复计数乘一次
   cache曲线产生seq³，触发B3.2停止门，(b)未退役。这不是以采样替代符号求解
   的理由。证据：`PARAMETRIC/task_prices/result.md`。以下为旧输入阶段的历史状态：
   `wait`/`fanout`/`volume`/`count` 是 `S`/`past`/`L_s` 的
   拟多项式而不是 `S_min` 上的整数（420/420 逐点复核）。这是 P5.1「代价函数
   以 θ 为参数 → DP 输出分段拟多项式」的输入前提，此前不成立。
   `div/scale/offset/count` 的三点拟合也已换成 isl 端点发现 + 全参数域集合等价
   证明，参考模型与 16 层均 0 fallback。⚠️ 本轮新增直接 CG 输入与 θ 绑定，
   输入侧 2154/2154 FP32 double 位模式相等；四个指标尚未完整进入价格计算，
   分段 DP 未完成，不能据此勾选 P5.1（`PARAMETRIC/result.md`）。
   c8be09e 后 BF16 输入 gate 为 1540/1540（含全部 308 个旧数值失败配置），
   FP32 2154/2154 保留。新功能审计在 seq=4 的首边发现 wait_sum=512、
   fanout_sum=16：前者含 nominal tile 的 producer 范围，后者限 actual domain。
   这两个 QP 不能直接当作同一物理图定价；event_ns 尚未接入，T1.5 未通过。
   详见 `EVENT_COST/result.md`。方案 (a) 是终态目标，旧有限枚举 (b) 不作完成标记。
5. ⚠️ L2 已改为 worker task queue 并在 50 进程观察到平均 34.36%/40.18%
   跨 stage 提前启动；
   κ=1 的六格 L2/L1 为 **1.081–1.113**，仍未胜过 L1。新的四臂归因见
   `docs/experiments/L2_ATTRIB/`，旧 stage-loop 分解已作废。
6. ✅ `seq×past` 矩阵 1500/1500，且直接删除 `TaskWait` 的新负测试 0/50；
   修改已退役 stage wait 的旧 clamp 仍为 50/50，两者共同证明守门覆盖活跃路径。
7. ✅ **Round5 定性关闭：判据产物，已归因。** 依据同输入的深度扫描、
   common-FP32 噪声地板及 golden 线程敏感性，不再作为本轮数值修复前置；
   未修改逐元素容差、未把历史失败改写为 PASS。历史记录：
   4×4096 / intermediate=14336 为 50/50；完整
   16×2048 的 973M 参数 decoder 虽然可生成、编译和运行，但固定 BF16 判据为
   0/50。✅ c8be09e 后固定输入/权重前缀深度扫描为 6 深度×50=300 进程：
   2/4/6 层各 50/50，8/12/16 层各 0/50；max_abs 随深度为
   .015625/.03125/.03125/.046875/.0625/.078125，未见定性跳变。
   16 层相对共同 FP32 golden 的 L2 误差，TileMega/PyTorch BF16 比为
   .9913284393，支持累积/噪声地板归因，但没有采用新的 k 判据。
   CPU golden 线程数 8→56 单独复现 mismatch 162→198，TileMega 哈希不变；
   这不是运行时精度提升。`REALMODEL/depth_result.md` 保留全部证据与局限。
   不能仅由层间 bit-identical 排除所有共同实现误差。论文端到端主张不能把这项写成完整
   1B 已验证。
8. ⚠️ 32 次 host 迭代的正反构建都 50/50，说明当前 harness 结构上触发不了
   ABA；增长 KV cache 与 kernel 内迭代必须在 Phase 6 的 serving loop 一起验证。
9. ✅ **Round5 A0 定性关闭：判据产物，与条件7同因。** 原始失败的
   4×4096 split8 相对共同 FP32 的 k_L2=.9961308506568204，落在用户指定的
   [0.989,1.003] 归因区间；56线程CPU BF16 golden 逐位复现。
   原二进制经用户授权一次采集，hash/diff 复现原始失败；不修改容差。
   `REALMODEL/condition9_result.md` 留存方法、三比较和边界。T2.d 数值可行域线取消。
   以下旧结论保留为历史，不再代表本轮待办：历史代价模型在 4×4096 上排名最高的 split-K 16/8 配置都违反未放宽的
   BF16 判据；最佳 split=1 候选则 50/50，并在 25 轮配对中使 L1 快
   28.05%。因此 P5 的 DP 在比较代价之前必须有数值可行性约束；不能让
   不可验收的 split-K 方案成为分段最优。本轮 FP32 partials 修复在两参考模型
   seq=128/past=3、split=1/2/4/8/16 各 50/50（500/500）；该修复消除了这组
   配置的失败，未重跑的 4×4096 最优配置不能直接宣布通过。流量模型同步增加
   partial 字节数；973M 条件 7 与数值可行域的全域证明仍未解决。
   c8be09e 已提供 sm_120 同组 500/500 的跨架构原始证据。✅ 本轮原始
   4×4096 最优 split8 在 FP32 partials 下仍失败：1 个超差元素，
   max_abs=.046875，0/1 后触发停止，split16 未运行；不是 50 进程验收。
   **条件 9 仍未关闭**，数值可行域设计与原始日志见
   `REALMODEL/condition9_result.md`，未禁用大 split 或放宽容差。
   FP32 dtype 的 partial 开关 GPU 回归另为 400/400；combine 新速率未实测。

⚠️ 这些是 Phase 5 前置的**实测状态**，不是全绿的启动条件。尚未完成的硬项是：
代价模型消费拟多项式、BF16 达到 FP32 排名/命中目标、真实规模 per-operator DP
验证，以及能触发 ABA 的增长 KV-cache 迭代。下面的参数化 DP、交点求解或 serving
本身也尚未实现。


#### P5.1 参数化解

- [ ] 代价函数以 `θ` 为参数 → DP 输出分段拟多项式
  → 协同：EX-S5（放置的参数化）

- [ ] 求分段交点 → 最优解的区间划分

- [ ] 每区间生成一套模板实例化（受 smem union 与编译时间约束）

- [ ] 区间数上限：P6.2 的平台期（top-34 在 ±10% 内）说明相邻区间的最优解
      多半互相可用，所以「求精确交点」应当让位于「合并差异 <10% 的相邻
      区间」——这直接决定要编多少个变体，而变体数是 P5.2 的硬约束

#### P5.2 运行时选择

- [ ] Host launcher：代入实际 `θ` → `O(1)` 区间查表 → 选 kernel

- [ ] 变体数量上限控制

#### P5.3 Tier 2 运行时支持

- [ ] indptr 前缀和（kernel 内或 host 预计算）

- [ ] 事件张量 extent 与 `wait` 的运行时填充

#### P5.4 尾 wave

- [ ] Stream-K 式归约维切分：`C` 从 `m ↦ {m}` 变 `m ↦ {(m,0..K)}`

- [ ] 复用 CUTLASS 的 Stream-K tile scheduler 作为参照实现

### Phase 6：Serving 集成与评估

#### P6.1 L5 Serving harness


按 §4.6 实现：paged KV（Tier 1）、continuous batching（kernel 内调度 task）、
chunked prefill（Tier 2，regime C 的载体）、MoE 路由（Tier 3 的 indptr 一般化）。

#### P6.2 评估

- [ ] **bucketing 损失曲线**：扫 batch = 1..128，「逐形状最优」vs
      「power-of-two 向上取整复用」。不依赖代价模型精度，纯结构性损失

- [ ] Label（簇）消融

- [ ] 混合 batch regime 对比
  → 承接：EX-V1

- [ ] Warmup 时间（目标：0 次 CUDA graph capture）

- [ ] 端到端对比
  → 承接：EX-V1

#### P6.3 消融

- [ ] L2 → L3（Reparam + Coarsen + Label 的贡献）
  → 承接：EX-V1（逐机制消融）

- [ ] 符号化的贡献（vs bucketing）

### Round 5 未关闭条目（指针；状态以 docs/experiments/ROUND5_LEDGER.md 为准）

| ID | 实现状态 | 验证状态/剩余项 | 关联 |
|---|---|---|---|
| A12.1 | 进行中 | A1 ComputeMetrics及A2无效grid实走错误分支before0/after0；工具remaining0；未覆盖全部新增错误出口，不作全量关闭 | 独立（isl 引用审计） |
| B1.2 | 固定实现域已验证，联合搜索未完成 | 两模型×seq4/128各4/16模式；基线位同；residency外层；尚欠tile/split/chunk联合域与规模优化 | 独立（融合） |
| B1.4 | 正确性通过，RMS方向门失败 | GEMM两链2seq×2状态×50=400/400；add预测/实测一致；RMS预测快、实测慢6.25%/35.21%，局部停止，未调参；链不是完整decoder | 独立（融合） |
| B1.5 | runner已实现 | CPU校验通过；真实fusion构建/预测manifest未生成，sm120未运行 | 待 sm_120 |
| B2.2 | resident-only已实现 | IR往返/拒绝、lowering与CUDA编译通过；400进程约束通过，无超驻留证明声明 | EX-S5 |
| B2.5 | 正确性通过、性能负 | 两臂400/400，新映射wait200/200对账；25轮配对四格显著变慢，价格预测下降，sign门不通过 | EX-S2 |
| B2.6 | runner已实现 | CPU校验通过，未生成sm120实际构建manifest，未运行 | 待 sm_120 |
| B3.1 | 负门已触发 | BF16历史770/462子集ρ分别−.00022857/−.00003006；FP32全1077×2预测字节不变。曲线保留OFF，不默认替换CDF；不是新BF16全oracle | 独立（符号 DP） |
| B3.2 | 高次门局部停止 | 真实GEMM5120、scalar768对照过；6→8接口二次计数×一次cache产生三次项，两模型复现exit2且零残留。完整选择门未过，(b)未退役 | 独立（符号 DP） |
| B4.1 | 未开始 | 与A12.1同一任务，不重复计数 | 独立（isl 引用审计） |

## 3. 已完成条目索引（v2.0，原文见 archive）

- P0.1 仓库与依赖 — [x] 3 项 — 证据：见 archive 原文
- P0.2 工具链 — [x] 3 项 — 证据：见 archive 原文
- P0.3 测试基础设施（优先级高于任何功能代码） — [x] 7 项 — 证据：见 archive 原文
- P1.1 torch.export 接入 — [x] 3 项 — 证据：见 archive 原文
- P1.2 符号形状桥（θ） — [x] 3 项 — 证据：见 archive 原文
- P1.3 CG dialect — [x] 3 项 — 证据：见 archive 原文
- P1.4 FX graph → CG 骨架 — [x] 4 项 — 证据：见 archive 原文
- P2.1 TaskBody 模板库 — [x] 5 项 — 证据：见 archive 原文
- P2.2 L0 参考实现 — [x] 1 项 — 证据：见 archive 原文
- P2.3 L0.5 host 端 stage 循环 — [x] 3 项 — 证据：见 archive 原文
- P2.4 L1 单 kernel megakernel — [x] 5 项 — 证据：`docs/experiments/E2E_GEN/`
- P3.1 CuTe ↔ ISL 桥 — [x] 6 项 — 证据：`docs/experiments/P3_ISL/`
- P3.2 访问关系构造（W / R） — [x] 2 项 — 证据：见 archive 原文
- P3.3 耦合推导（C）与派生量 — [x] 5 项 — 证据：`docs/experiments/P3_ISL/result.md`、`docs/experiments/P3/table27.md`、`docs/experiments/WIRING/`
- P3.4 Tier 分类与松弛 — [x] 7 项 — 证据：`docs/experiments/P3/attributes.md`、`docs/experiments/P3/derived-*.md`
- P3.5 L2 落地 — [x] 7 项 — 证据：`docs/experiments/CLUSTER/`、`docs/experiments/E2E_L2/`
- P3.6 生成器一般化（去掉 Llama 结构写死） — [x] 5 项 — 证据：`docs/experiments/P3_GENERALIZATION/run.sh`、`docs/experiments/E2E_GEN/`、`docs/experiments/P3_GENERALIZATION/`、`docs/experiments/SEMANTIC/result.md`
- P3.7 求解权威迁移到 isl/barvinok（原则三的落地） — [x] 9 项 — 证据：见 archive 原文
- P4.1 代价查询接口 — [x] 3 项 — 证据：`docs/experiments/ORACLE/run.sh`、`docs/experiments/BACKEND/result.md`
- P4.2 层1 合法性剪枝 — [x] 1 项 — 证据：`docs/experiments/BACKEND/result.md`
- P4.3 层2 对齐传播 — [x] 4 项 — 证据：`docs/experiments/SOLVER/alignment.md`
- P4.4 代价模型 — [x] 6 项 — 证据：`docs/experiments/ORACLE/`、`docs/experiments/CALIB/result.md`、`docs/experiments/COST_MODEL/result.md`
- P4.5 层3 链上 DP（Reparam + Coarsen） — [x] 6 项 — 证据：`docs/experiments/SOLVER/result.md`
- P4.6 Coarsen（事件粒度 κ） — [x] 5 项 — 证据：`docs/experiments/P3_ISL/`、`docs/experiments/COARSEN/result.md`
- P4.7 层4 Label（簇划分） — [x] 5 项 — 证据：`docs/experiments/CLUSTER/run_on_cluster_gpu.sh`、`docs/experiments/CLUSTER/result.md`
- P4.8 层5 Place — [x] 6 项 — 证据：`docs/experiments/PLACE/round5_balanced_result.md`、`docs/experiments/OWNERSHIP/`、`docs/experiments/PLACE/result.md`、`docs/experiments/OWNERSHIP/result.md`
- P4.9 实现契约与一致性校验 — [x] 3 项 — 证据：`docs/experiments/CONTRACT/result.md`、`docs/experiments/L2_ATTRIB/`
- P6.2 评估 — [x] 2 项 — 证据：`docs/experiments/ORACLE/result.md`
- P6.3 消融 — [x] 1 项 — 证据：见 archive 原文

已放弃（`[-]`）条目：

- P0.1 仓库与依赖：mirage / MPK 代码未引入；

（⚠️ v2.1 第五轮：EX-E3 六步实现已完成；R4 的 cluster 硬件验收仍待 sm_120。
协议/barrier 几何平均降 36% 并未改善端到端，R4 端到端最快仍是 baseline。
R5 不再改同步、窗口或链化；M4 的配置重选先于新增执行器机制。
EX-S1c 完整预算与粗排排序同时未达时，以明确缩小的候选集推进 EX-S3，
不是把未达门改成通过。EX-E4 因新诊断 rule3 本轮不进入。
κ 仍为全局编译选项；目前没有按 stage 定价/实测证据要求引入运行期字段。
最终数值、停摆与降级清单、自检完整输出见 `JOINT/summary.md`。）
