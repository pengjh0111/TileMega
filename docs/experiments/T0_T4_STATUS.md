# T0–T4 可续接状态（基线 e305a9f）

## Round 5 A/B（基线 c1281b9，当前执行轮）

逐项真值表：[ROUND5_LEDGER.md](ROUND5_LEDGER.md)，含全部 A/B 编号、依赖、
独立实现/验证状态、易遗漏清单及交付物。以下旧轮记录保留，不覆盖。

✅ A0 common-FP32 三比较完成，k_L2=.9961308506568204；按新 prompt，条件9
关闭为判据产物，条件7同样定性关闭，T2.d 取消。不修改容差，不声称原失败变 PASS。
一次原二进制GPU输出采集经用户单独授权，其余CPU计算。
⚠️ A1正在整理恒等式/历史审计证据；A2–A9未通过，B尚未开工。

本文件是进度记录，**不是整轮完成报告**。不覆盖历史负结果，不降低验收阈值。

| 项目 | 已完成与证据 | 尚未完成 |
|---|---|---|
| T0 | warmup/repeat、逐臂资源、min-blocks 开关；4090 400/400，四臂 25 轮；OCCUPANCY/result.md | 5090 仅人工脚本，未运行 |
| T1 | 直接 CG 输入、角色/别名绑定；2154 输入侧位模式；有限 S=1..16 DP 32/32 点对照 | 四指标完整进入价格计算、一般符号转移/交点、完整 SOLVER alignment 对照；输入 gate 不等于完整 gate |
| T2.a | FP32 partials 默认启用、旧路径开关；两模型 × 五 split × 50=500/500；流量增量及 FP32 CPU 2154 位模式 | FP32 GPU 新 50 进程回归；全部 shape 数值域；combine 新速率实测 |
| T2.b | golden 来源与三条路线评估，判据不变 | 独立同序逐层诊断、FP32 中转实现与真实成本、973M 通过 |
| T2.c | 原二进制 308/308 分类为数值失败，逐项日志与哈希 | 不能将一次分类重放冒充 50 进程同步验证 |
| T2.d | 参考模型大 split 失败消失 | 4×4096 最优配置重跑、全域可行性结论；不宣布条件 9 全面关闭 |
| T3 | skeleton 先加入第六操作 Fuse；按 T0 修正 residency 预算 | 区间 DP、两条真实融合、50 进程、FUSION/run_sm120.sh |
| T4 | band/wavefront 离线六例合法且 locality 正结果，负载失衡同时报告 | T4.2 真实 CG→runtime task 投影与接入、I3/GPU 验证、PLACE/run_sm120.sh |

## 继续时的关键约束

- 当前 BF16 实例是 128 threads / 212 registers / 24576 B shared / 2 CTA/SM，
  不套用题述的 256-thread 实例。T0 编译清单固定在 partial 修复前，不混用源版本。
- 两级原子扇入与“TC 永不获胜”两条方向仍被否定，不重启。
- FP32 partials 同时保留 Linear→BF16→residual 舍入边界；默认运行时宏与
  CostModelOptions 同步为 true。旧流量模型用 --bf16-partials-baseline。
- T1 选择 (b) 是由于 SDCM 的 sqrt/exp 和 IEEE 重复加法不属于现有 QP 表示。
  有限整数枚举不冒充一般符号交点；四个 coupling 指标当前仍未完整定价。
- PARAMETRIC 工具有 isl context 引用残留警告，尚待定位，不因 exit=0 忽略。
- 生产 placement 不能把 18 算子探针的固定 ReferenceModel 硬塞进 30/60 stage
  模型；必须处理 attention 等逻辑 task 到 runtime stage 与 split rewrite 的投影。
- 已写的 sm_120 脚本只限 OCCUPANCY、BF16，均未运行。缺少的两份不能用空脚本
  或写死结果占位。
- `third_party/barvinok` 原有未跟踪内容保留，未作清理，不纳入本轮提交。
# T1–T5 current status (baseline c8be09e)

This section supersedes the status of the old round above; it does not erase
its experiments or negative results. **The new round is incomplete.**

| Task | Verified in this continuation | Open work / blocking limitation |
|---|---|---|
| T1.1 | No production cache change | Measured-curve switch, BF16/FP32 ranking comparison |
| T1.2/1.3 | Two-architecture steady-state four-arm reanalysis; explicit rate units/missing reasons; actual CG metric audit | wait_sum=512 versus fanout_sum=16 on first partial-tile edge; physical-domain metrics and runtime count projection unresolved. CV underestimates short-seq costs by 94–98%. No valid event price / no functional gate |
| T1.4/1.5 | BF16 input bits 1540/1540 including 308 classified failures; FP32 2154/2154 unchanged; concrete/finite DP regressions agree | (a) not implemented; (b) remains historical diagnostic only, not terminal design or functional acceptance |
| T2 | No new fusion implementation | Exact L-task rewrite, interval DP, live resource/wave budgets, real two-edge GPU tests, sm_120 script; depends on valid T1 event pricing |
| T3.1 | 6/6 finite instances have strictly dominating balanced mappings; 78/78 graphs acyclic, explicit zero isl refs; Pareto plot and raw data | Finite/offline only, not overresident I3 or GPU evidence |
| T3.2–3.5 | Not implemented | Logical→runtime stage / split projection, L-sched writeback and lowering, BF16 50-process tests, price/timing comparison, sm_120 script |
| T4.3 | Primary-paper evaluation lookup recorded in BF16/related_work.md | Explicit numerical reference/tolerance/depth protocol not found; no inferred community criterion |
| T4.1/4.2 | 300 fresh processes; depth2/4/6 each50/50, depth8/12/16 each0/50; all inter-level hashes agree. At16 k_L2=.9913284393 supports noise-floor attribution. CPU golden threads8→56 alone reproduce162→198 mismatches | Fixed criterion still fails at depth; no new k threshold adopted, no proof excluding all shared bugs |
| T4.4 | Original 4×4096 split8 with FP32 partials fails0/1, one mismatched element, max_abs=.046875; stopped immediately | Condition9 remains open; split16 compiled/not run, no50-process claim. Numerical-domain feasibility design recorded, not implemented |
| T4.5 | FP32 GPU partial-storage regression400/400; both models, split1/16, both states,50fresh processes each; identical paired hashes | Actual FP32-partial combine-rate calibration still missing; analytic traffic increment is not a measured coefficient |
| T5.1 | Points() leak localized by before/after ref guard; caller-owned contexts; zero-ref tool evidence | No claim about unexecuted error branches; audit deliberately rejects new leaks |
| T5 enumeration follow-up | Exact finite-basic-component union avoids pathological disjointization; independent set tests and generated-CUDA byte equality | Wide split8 >1089s unfinished→17.1849s;16-layer321.8086→322.9925s has no observed improvement |
| T5.2 | Four-arm comparison now interleaves build states; logs execution_index | New future fusion/placement runners must preserve both state and arm rotation |
| T5.3 | Nested polylib generated autotools files locally ignored | Local git info/exclude, not an upstream submodule change or portable ignore policy |
| T5.4/5 | BF16 input gate migrated, all old failures included; status/findings updated | Full L2 symbolic CostBreakdown gate awaits T1 implementation |

The event-price prototype was removed before committing because it mixed
ns/runtime-task units with event-image cardinalities. CostModel/ChainDP keep
their original L1 prices, rather than exposing a plausible but invalid L2
answer. Relation/QP audits and calibration metadata are preparatory work,
**not consumption acceptance**. The unresolved interfaces are the physical
incidence graph and its runtime projection, not an implicit fallback to .cu.

No numerical tolerance, occupancy goal, atomic fan-in hypothesis or TC-lane
attribution was revived. The new GPU evidence is limited to the explicit
T4 diagnostics/regressions above, not new comparative performance results.
sm_120 raw evidence was supplied at c8be09e, not executed locally.

The condition9 stop is recorded in `REALMODEL/condition9_result.md` with the
sole failed log, resources, unrun split16 and a numerical-feasibility design.
`REALMODEL/depth_result.md` records the independently completed depth/noise
experiment and its golden-thread detour. These do not close T1, fusion,
production placement, BF16 ranking or the measured combine-rate debt.

Verification: full build, check-policy, five-target audit zero failures,
24/24 CTest. FP32 predictions/finite-DP TSVs remain byte-identical to HEAD
before the new input-gate code. Commits are split by implementation category
and experiment, with single-line repository-convention messages.

---
