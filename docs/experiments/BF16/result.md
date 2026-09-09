# BF16 end-to-end and calibration

## 本轮 T2.a：FP32 partials（基线 e305a9f）

✅ RTX 4090，seq=128、past=3，两个模型各 split={1,2,4,8,16}，每格
50 个全新进程。修复路径 500/500 通过；修复前后总计 1000 次运行，
原始记录 `raw_splitk/correctness.tsv`，完整 ptxas 与运行日志分别在
`raw_splitk/ptxas/`、`raw_splitk/logs/`。BF16 容差没有修改。

| 模型 | split | BF16 partials 通过 | FP32 partials 通过 | max_abs 前 → 后 | partial bytes 前 → 后 |
|---|---:|---:|---:|---|---|
| gqa2 | 1 | 50/50 | 50/50 | .03125 → .03125 | 0 → 0 |
| gqa2 | 2 | 0/50 | 50/50 | .09375 → .03125 | 4194304 → 8388608 |
| gqa2 | 4 | 0/50 | 50/50 | .0625 → .03125 | 8388608 → 16777216 |
| gqa2 | 8 | 50/50 | 50/50 | .0625 → .03125 | 16777216 → 33554432 |
| gqa2 | 16 | 0/50 | 50/50 | .09375 → .03125 | 33554432 → 67108864 |
| mha4 | 1 | 50/50 | 50/50 | .03125 → .03125 | 0 → 0 |
| mha4 | 2 | 0/50 | 50/50 | .078125 → .046875 | 9437184 → 18874368 |
| mha4 | 4 | 0/50 | 50/50 | .078125 → .046875 | 18874368 → 37748736 |
| mha4 | 8 | 0/50 | 50/50 | .09375 → .046875 | 37748736 → 75497472 |
| mha4 | 16 | 0/50 | 50/50 | .09375 → .046875 | 75497472 → 150994944 |

max_abs 是每格 50 次中的最大值，不是逐元素相对判据；因此同样 .0625 可能通过，
也可能失败。不能单凭它反推 pass。扫描没有显示误差随 split 单调增长。

实现：`ModelRuntime.h:22` 的 `TILEMEGA_FP32_PARTIALS`，默认 1，=0 为原基线；
`GemmStageTaskBody.h:391` 用 FP32 output epilogue 写 split partial，
`ModelHarness.cuh:863` 按实际元素类型分配，`GemmCombineTaskBody.h:22` 合并。
还必须保持图中的舍入边界：完整 GEMM 归约之后先转 BF16，再加 residual，
不能让第一路 partial 提前承担 residual，也不能把 Linear→BF16→add 改成单次末尾舍入。
这属于同一 partial-storage 实现所需的语义处理，不把收益全归因于少一种舍入。

流量：`CostModel.cpp:342` 的 EpilogueBytes 随 split partial 的 4 B 存储增长；
`:372` 的 combine 路径计入额外 2 B/partial 读取，并用新工作集评估 L2/DRAM。
该增量是解析流量项，**不是重新标定的 combine 系数**。需 DRAM 系数而未标定时
显式报 not_calibrated，不取零或退回旧路径。CostModelOptions 与 runtime 默认同步，
`tilemega-costmodel --bf16-partials-baseline` 可重放旧流量模型。

✅ FP32 代价模型开关前后 2154/2154 全 CostBreakdown double 位模式相等，
见 `../PARAMETRIC/input_gate.log`。⚠️ 不将 CPU 位模式闸门冒充新的 FP32 GPU 50 进程回归。
条件 9 的“大 split 在该参考配置上必然失败”已消失；尚未对所有 tile、所有参数域
证明数值可行性，更未宣称 973M 条件 7 或 BF16 排名条件已经达标。
sm_120 仅写 `run_splitk_sm120.sh`，未运行。

以下为历史记录，旧 ρ 数字不作为本轮达标证据；RUNFAIL 分类见文末 T2.c。

Evidence status: ✅ measured on RTX 4090 (`sm_89`) on 2026-09-05.

The dtype is read from every `ExportedProgram` FakeTensor and stored on the
L-sem operator and generated `ModelSpec`; it is not inferred from granularity.
The implementation registry enumerates the SM80+ BF16 Tensor Core family only
for BF16, including its distinct legality (`tile_m % 32`, `tile_n % 16`,
`tile_k % 16`, 128 threads and 8-element alignment).  Dispatch remains behind
`ArchDispatch::Caps::kBf16TensorCore`.

All bodies store BF16. GEMMs, RMSNorm and attention dot/softmax/value reductions
accumulate in FP32; explicit BF16 materialization boundaries preserve the
exported graph's `linear -> residual`, SiLU and RoPE rounding semantics.  The
two reference models passed PyTorch L0 → L0.5 → L1 → L2 in 50 fresh processes
each (see [`correctness.tsv`](correctness.tsv)). `cuobjdump` finds 96 static
`HMMA.16816.F32.BF16` instructions in each generated executable, so this is
not a BF16-storage/SIMT computation.

The BF16 tolerance is `1.6e-2 + 1.6e-2*abs(reference)`.  FP32 retains its
`3e-5` rule.  At `seq=2048`, `1e-2` left four values on legitimate BF16
quantization boundaries among millions; `1.5e-2` left zero, so the selected
bound includes a small explicit margin rather than reusing an FP32 threshold.

`tilemega-calibrate --dtype bf16 --base configs/targets/sm_89.json` now writes
`calibration_by_dtype.bf16` and preserves the original `calibration` object
byte-for-byte in meaning.  The accepted third run achieved 97.37% of the
reported DRAM pin rate and only 0.00073% start/end drift; two earlier runs were
correctly rejected (9.30% and 46.12% drift) and were not committed.  The final
profile measures the BF16 MMA instruction and the real BF16 CUTLASS Stream-K
collective; compact values are in [`calibration.tsv`](calibration.tsv) and
[`streamk.tsv`](streamk.tsv).

The cost model is now dtype-aware too: BF16 mainloop work populates the `tc`
lane rather than the CUDA-core lane, storage traffic uses two-byte elements,
and per-lane controlled ablations are emitted by `tilemega-costmodel`.  The
final Spearman/top-3% and SMEM/L2-identifiability conclusions are generated
from the BF16 oracle in `../ORACLE/raw_bf16/cost`; they must not be substituted
with the old FP32 validation set.

## The oracle has now run, and Part 2.4's acceptance fails

✅ measured, 1540 generated/compiled/measured points, `../ORACLE/result.md`
§6.7.  Part 2.4 asked that BF16's ρ and top-3% hit rate be **no worse** than
FP32's ρ 0.9450 / 0.9435 with top-1 and top-3 inside the measured top 3%.

| | gqa2 | mha4 |
|---|---:|---:|
| full model ρ | **0.5605** | **0.6239** |
| MAPE % | 39.41 | 37.59 |
| top1 / top3 / top10 | 0 / 0 / 0 | 0 / 0 / 0 |
| rank the model gives the true optimum | 104 | 51 |
| uncalibrated analytic `tier2-baseline` ρ | **0.8778** | **0.8738** |

The acceptance is not met and no threshold was moved to meet it.  In BF16 the
calibrated model ranks *worse than the analytic baseline it replaces*, having
beaten it 2:1 in FP32.

The attribution is in §6.7 and is not a defect of §2.2's structure: the
`+splitk` layer is the one that inverts, because `combine_fixed_ns` is **0** in
the BF16 profile (FP32: 108.1 ns).  The calibrator stores `max(0, fit)` for an
intercept it honestly reports as unresolved (`|value| < 300 ns`, 150% spread,
either sign); in FP32 the fit landed positive and the clamp never bound.  With
a reduction stage that costs the model ~0.12 µs across the whole graph, split-K
is nearly free and the model's eight best configurations are all split-K 16,
predicted 2–3× faster than they measure.  Three of the six BF16 Stream-K points
also fit a **negative** per-CTA setup (`a_ns` −119.2 / −438.6 / −400.7,
`fit_r2` 0.922 against FP32's 0.974), which cannot be a setup time.

⚠️ Part 2.1's premise is also not confirmed on this target.  The `tc` lane —
the reason nine lanes were restored — changes ρ by **0.001** when removed
(0.5605 → 0.5595).  Six of the nine lanes are exactly inert.  BF16 makes the
mainloop fast enough that the bottleneck moves *away* from the compute lanes,
rather than into `tc`.

Fixing this is a measurement problem (resolve the reduction stage's fixed cost
instead of clamping it) and is deliberately left to the next round rather than
attempted at the end of this one.

## The repair, and a correction to the attribution above

⚠️ **The attribution in the section above is wrong, and the correction is the
useful part.** `combine_fixed_ns` was a real defect and it was fixed; it turned
out not to be what broke the ranking. Three changes were made, each measured
separately against the same 1540-point validation set.

### (1) The launch baseline was a different kernel

`NullKernel` — an empty `__global__ void f(){}` — was launched at the same grid
and its duration subtracted from every calibration point. It has neither the
shared-memory footprint nor the `__launch_bounds__` of the kernel it stands in
for, so it is not the same launch. ✅ The evidence is arithmetic: the
launch-subtracted duration of the reduction at **one** output came out at
**−928 ns**, which is impossible.

That bias is the size of `a` itself on the small tiles, and it is why three
BF16 Stream-K shapes fitted a negative per-CTA setup. The baseline is now the
*same kernel doing nothing* — `CalibCombineKernel(..., count = 0)` for the
reduction, and `CalibGemmKernel(table, tiles_m = 0, ...)` for the GEMM, which
returns before its first division. `a` is then measured directly at zero
mainloop iterations rather than extrapolated, and `c` is fitted through it:

| profile | `a_ns` before | `a_ns` after | `ac_r2` before | after |
|---|---|---|---:|---:|
| BF16 | −119.2 / −438.6 / −400.7 among six | none negative beyond the ±150 ns floor | 0.922 | **0.984 – 0.9997** |
| FP32 | 993 – 4501 | 0 – 4096 (smaller: the empty kernel under-charged this kernel's dispatch) | 0.9999 | **0.9992 – 1.0** |

### (2) The clamp is gone

`combine_fixed_ns = max(0, fit)` is replaced by a direct measurement at the
narrow end of the width sweep. On this target every small-width reading lands
inside one ~100 ns timer tick, so the honest answer is that the quantity is
**below the measurement floor**: the profile now carries `0.0` together with
`combine_fixed_resolved = false`, and a zero can no longer be mistaken for a
measured "free". `tilemega-target-audit` gained the field and caught its
absence in the four uncalibrated target files on the first run.

### (3) What actually broke the ranking: one scalar for the per-CTA setup

✅ Neither (1) nor (2) moved the ranking: ρ went 0.5605 → 0.5560 (gqa2) and
0.6239 → 0.6235 (mha4). The attribution to `combine_fixed_ns` is **falsified**.

The cost model never reads the per-shape `a` directly. It fits one scalar
`setup_ns` across the calibrated points and prices all 154 shapes through it.
In FP32 the calibrated `a` spans 993 – 4501 ns (4.5×) and one scalar is a fair
approximation; in BF16 it spans 0 – 10112 ns, and the fit reported
`setup = 1187.94 ns, rms 3247 ns` — a residual **2.7× its own value**. A per-CTA
setup is dominated by the tile the CTA must materialize, so it is now fitted as
`setup = alpha + beta · tile_m · tile_n` over the same points — two numbers
instead of one, nothing newly measured.

| | gqa2 ρ | mha4 ρ |
|---|---:|---:|
| before | 0.5560 | 0.6235 |
| **after** | **0.8246** | **0.8247** |
| FP32 control (unchanged by this) | 0.9432 | 0.9421 |

### (4) The SMEM lane prices a path BF16 does not use

With the setup term fixed, the residual was a clean monotone bias in `split_k`:
median predicted/measured **0.93 at split 1 falling to 0.55 at split 16**. The
model believed splitting K helps ~1.7× more than it does.

The SMEM lane is the cause. Its rate comes from a scalar `ld.shared`
microbenchmark and its work term is the SIMT mainloop's shared-memory traffic,
which scales with iterations — so splitting K divides a term that the BF16
kernel does not pay. A Tensor Core collective moves operands `cp.async` →
shared → `ldmatrix` → MMA registers, and nothing in the BF16 profile measures
that path. The lane is therefore `kNotCalibrated` for BF16 and drops out of the
`max`, which is the mechanism the nine-lane vector exists for.

✅ The split-K bias disappears — predicted/measured becomes **flat at 0.46–0.48
across every split factor** — confirming the diagnosis rather than merely
improving the score:

| split_k | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| before | 0.93 | 0.67 | 0.62 | 0.57 | 0.55 |
| after | 0.48 | 0.46 | 0.48 | 0.48 | 0.48 |

⚠️ Recorded honestly: the lane was found harmful **by ablation first** and
explained afterwards. The explanation is a micro-architectural argument, not a
fit, and it must be re-checked on a target where the operand feed can be
measured apart from the byte traffic.

## Part 2.4, re-measured

| | gqa2 | mha4 |
|---|---:|---:|
| BF16 full model ρ | **0.8942** | **0.8834** |
| uncalibrated analytic `tier2-baseline` ρ | 0.8778 | 0.8738 |
| rank the model gives the true optimum | 48 | 46 |
| MAPE % | 51.96 | 49.94 |
| FP32 control, same code | 0.9432 | 0.9421 |

✅ **The minimum bar is met**: the calibrated model is no longer worse than the
analytic baseline it replaces. ❌ **The target is not**: FP32's ρ 0.9450 /
0.9435 with top-1 and top-3 inside the measured top 3% is not reached, and
`top1 = top3 = top10 = 0` still. No threshold was moved.

The remaining gap is now a *scale* error rather than a shape error: the model
under-predicts by a near-constant 2.1× (which is what MAPE 52% is), because
removing the SMEM lane removed most of the predicted mainloop time and the
`tc` and `l2` lanes alone do not replace it. The named next step is a BF16
operand-feed term — measured on the `cp.async` / `ldmatrix` path rather than on
scalar `ld.shared` — not another scalar refit.

## Which lane carries BF16 (Part 2.3(a))

One-variable ablations on the repaired model:

| removed lane | gqa2 ρ | mha4 ρ |
|---|---:|---:|
| — (full) | 0.8942 | 0.8834 |
| `l2` | 0.8851 | **0.8658** |
| `tc` | 0.8882 | 0.8780 |
| `cuda`, `sfu`, `tmem`, `l1_5`, `ddr`, `net` | 0.8942 | 0.8834 |

The answer to "which lane, if not `tc`" is **`l2` first and `tc` second**, and
for the first time both are non-trivial (F-71 had `tc` worth 0.001 against the
broken model). Six lanes remain exactly inert. The nine-lane structure is kept:
what changed is that a lane whose rate does not describe the path in use is now
marked `kNotCalibrated` instead of being charged anyway.

## SMEM / L2 identifiability (Part 2.3(b))

The collinearity is unchanged as a *fit* property — both lanes are built from
`occupancy · 2 · Tk · (Tm + Tn)`, ratio 3.4715200776, Pearson 1 over the 20
calibrated points. What the repair adds is that the question is no longer
open in BF16: the SMEM lane is not carried by the profile at all, so there is
one byte lane (`l2`) and no unidentifiable second one. ⚠️ This is a decision
about what is *calibrated*, not a demonstration that the two pipes are the
same; on a target where TMA moves the L2 side without the SMEM side they must
be re-measured and the lane restored.


The requested SMEM/L2 retest has one important negative result already:
they are **still exactly collinear in the implemented BF16 model**.  Across all
20 calibrated `(shape, occupancy)` points, both lanes use the same
`occupancy * 2 * tile_k * (tile_m + tile_n)` feature; their ratio is a constant
3.4715200776 and Pearson is 1.0 (`identifiability.tsv`).  Consequently this
fit cannot independently attribute a BF16 timing change to SMEM versus L2.
Tensor Core work is a different feature and can break the old one-lane
ranking, but changing dtype alone does not make the two byte lanes
identifiable.  This is a limitation of the present feature construction, not
evidence that the hardware pipelines themselves are identical.

## Current final diagnosis (task-queue round)

The oracle was re-evaluated with CUDA's recorded `ctas_per_sm` as the
authoritative validation feature. The previous analytical feature assumed the
FP32 kernel's 256 threads for BF16 and used a per-CTA shared-memory ceiling as
if it were the per-SM budget. Correcting those defects changes the full-model
rank correlation only modestly:

| | gqa2 | mha4 |
|---|---:|---:|
| BF16 full model ρ | **0.8984** | **0.8871** |
| MAPE | 51.88% | 49.86% |
| top-1 / top-3 / top-10 in measured top 3% | **0 / 0 / 0** | **0 / 0 / 0** |
| rank assigned to measured optimum | 48 | 46 |
| FP32 control ρ | 0.9432 | 0.9421 |

The queue-round lane ablation is consistent with the earlier diagnosis but is
recorded from the corrected validation path: removing `l2` gives
**0.8886 / 0.8693**, removing `tc` gives **0.8923 / 0.8817**, and each of the
other six calibrated/available lanes leaves the full score unchanged. Thus
`l2` is still the strongest identifiable BF16 lane and `tc` is second.

❌ The required `ρ ≥ 0.94` and top-1/top-3 acceptance is not met. No threshold
is changed. The complete predicted top ten and their measured ranks are in
[`topk_diagnosis.tsv`](topk_diagnosis.tsv): gqa2's ten land at ranks
46–232 and mha4's at 29–145, while the top-3% cutoffs are ranks 23 and 14.

The false leaders are bimodal. One family uses narrow output tiles
(`tile_n=16/32`) with `split_k=1`; the other uses aggressive `split_k=8/16`.
Their predicted/measured ratios are only 0.36–0.47. Conversely, both measured
top tens overwhelmingly use `tile_n=16, split_k=8`, but the model ranks the
true optima 48th/46th. Thus this is not merely a global scale error: the
current setup/operand-feed construction prices the interaction between narrow
tiles and split-K incorrectly.

Four new calibration shapes cover exactly the missing axes (`32x16x16s2`,
`32x128x32s3`, `128x32x32s2`, `64x64x32s5`) and calibration now measures every
resident occupancy rather than stopping at four CTAs/SM. ⚠️ A trial expanded
profile made validation worse, so it is not installed in `sm_89.json`; adding
points does not repair a misspecified feature.

No GEMM tile shape can decouple the current SMEM and L2 lanes: both are the
same feature `occupancy·2·Tk·(Tm+Tn)` for every possible input, not merely for
the original 20 samples. A valid decoupling experiment needs a different
kernel that reuses a fixed shared-memory tile while sweeping a larger L2
working set, or holds the global working set fixed while repeating shared
loads. Until that measurement exists, BF16 marks the scalar-`ld.shared` SMEM
lane `not_calibrated` and carries one identifiable byte lane (`l2`). Keeping
two fitted coefficients would create an unidentifiable degree of freedom.
# T0–T4 本轮进度（基线 e305a9f）

## T2.c：308 个 RUNFAIL 的分类

✅ 已对原始 308 个失败配置的**原有二进制**逐一重放，308/308 均为数值判据失败，
不是编译失败、资源不足或 CUDA 执行异常。split=2 与 split=16 各 154 个。
新分类在 `runfail_audit/classification.tsv`，每项包含退出码、原二进制 SHA256 和
完整日志；`classify_runfail.py` 可复现。不把一次重放称作 50 进程并发正确性验证。

原筛选脚本 `docs/experiments/ORACLE/run_bf16.sh:117` 先检查退出码，任何非零就
写 RUNFAIL 并 break；harness 的数值 MISMATCH 也返回非零，因此从未走到后面的
MISMATCH 分支。它还丢弃 stderr、未保存这一轮 stdout，所以这里严格称为
“原二进制重放分类”，不伪称恢复了丢失的历史日志。

代表：`runfail_audit/logs/mha4_32x16x16s2k2.txt`，L0.5 对 golden 有 1 个元素失败，
max_abs=0.046875；L1 对 L0.5、L2 对 L1 均 0 mismatch，三层 hash 一致。
这解释的是原筛选标签与样本排除，不独自证明 partial 精度是全部误差根因。
在完成数值可行性复核前，不以旧 BF16 ρ/top-k 判定本轮是否达标。

## T2.b：golden 来源（先查清，再决定）

✅ `python/tilemega/export_bridge.py` 输出图/shape/dtype 元数据，不计算数值 golden。
973M 的实际入口是 `REALMODEL/export_real.py:114`：模型和输入在 CPU 上构造为
torch.bfloat16，`:137` 用 exported program.module() 执行，`:145` 按 BF16 写参考。
模型中 linear/matmul 是 PyTorch CPU 后端运算，没有指定 CUTLASS 的 tile 或
split-K 顺序。具体 CPU GEMM 归约树不在 exported graph 中，尚不能从图恢复。

`V_H/export_probe.py:21` RMSNorm 将平方/均值/rsqrt 放在 FP32，随后 cast 回输入
dtype 再乘权重；attention softmax 同样先转 FP32 再转回 BF16。残差、线性层输出
和 KV cache 是 BF16。它不是“整层全 FP32、最后统一转 BF16”的 golden。

三条路线的当前评估（均不能以层间 hash 一致替代 PyTorch 对照）：

1. **相同归约顺序 golden**：图有 BF16 materialization 边界，却未编码 CPU/GPU
   GEMM 的 reduction tree。可独立实现给定 tile/split 的 FP32 运算序列解释器，
   逐层检查后生成同序参考；直接把 TileMega L0.5 输出改名 golden 是循环验证，
   不可采用。当前没有这样的独立解释器，故尚不能声称此路已经解决 973M。
2. **重设深层判据**：相对范数/余弦只约束整体方向或平均误差，可掩盖少量大离群值，
   尤其最终 hidden 还不是 logits。需要事先固定 FP32/BF16 共用指标、独立的扰动
   负控制及 4×4096 对照，才能证明没有放水。当前证据不足，本轮不选此路，
   不修改逐元素容差，也不把历史 0/50 重新标成通过。
3. **FP32 层间激活**：只扩大存储而下一层立即转 BF16 不会消除舍入；要有效就要
   修改消费端的混合 dtype 运算和 residual/materialization 语义。每个跨层激活
   增加 `2*seq*hidden` 字节存储，每次完整写后再读增加 `4*seq*hidden` 字节流量。
   它不等价于扩大所有权重到 FP32。TaskSmem 是否增长取决于实际混合 dtype
   collective，不能由上述 global 字节数推断 occupancy 或时间。尚未实现该路径，
   因而 smem/时间的实测代价仍缺失，不填推测数字。

选择：保留固定判据，优先独立同序逐层诊断，而不是更换验收指标。
⚠️ 这是路线评估，T2.b 的独立诊断与第三条路线实测尚未完成；973M 仍未通过。
