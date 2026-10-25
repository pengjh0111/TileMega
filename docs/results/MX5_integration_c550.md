# MX-5 dense integration checkpoint

Verified checkpoint, 2026-10-25. Authoritative prompt SHA256:
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
G-A passed; G-B and G-PROGRAM remain open. Manual-control correctness and diagnostic timings are recorded below; no
solver-selected model acceptance, MoE or DNN result is claimed.

## Calibration and evidence

The two source-consistent 512-thread campaign manifests cover exactly 4545
keys: GEMM 3395, head 750, attention 400, with zero missing/extra keys.
Those earlier-source timing campaigns are drained; domain coverage is not
completed timing, and their partial measurements do not price the new source.
The default target still uses the MX-4 table. A qualified partial table is
never labeled a complete table or substituted for 576-thread measurements.

All 63 derived bench libraries were rebuilt with final MIR capture. Their
4545 selected coordinates map to 251 kernels and 521 nonempty MMA rings:
zero selected spills, generic loads, private accesses, calls or ring4160;
maximum MTreg253, frame0. All 1178 compiled kernels have 2595 passing rings.
Two unused Residual+Norm+StreamK TN128 depth2/3 symbols have frames56/84;
they remain recorded and do not represent covered model stages.

All 63 MIR rebuilds have executable MACA device ELF images identical to
the original candidate libraries referenced by the campaigns.
One whole-library hash differs only in bundled LLVM bitcode; repeated
print/no-print rebuilds establish identical executable images. The bitcode
difference is more than padding, and its compiler cause remains unresolved.
Raw proofs: `runs/mx5/calibration/derived_mir_61/`.

Fresh 576 queries retain 820 GEMM, 100 head and 400 attention P0 resource
failures. Of P0-fitting keys, 455 GEMM and 150 head primary-prefetch failures
all fit after the first prescribed halving. No geometry was pruned, R_next
invented, or 512 plan exception granted. Two parameter-lifetime diagnostics
still leave the D64/Q4/576 attention bench spilling (216→244/156 bytes).

## Dense code and integration failures

StageProgram emits actual compile-time body/shape aliases, operand expressions,
layer-loop bounds, DownLast/head edges and the prefetch chain. The new dense
program removes embed and uses 5L+1 stage counters. Its new runtime retains
the serving L1 ABI, by-value bindings and exact occupancy check. Watchdog and
diagnostics compile separately from timing builds. Explicit `mx_decode_v2`
pins are manual/resource probes; the existing default has not been switched.

The body cost table is being measured while this dense compile integration
proceeds. This overlaps the prompt's B/C schedule to expose integration
resource defects early. Neither gate is waived; MoE/DNN remain later stages.
The user authorized recording differences and following the closest compliant
direction. This does not authorize a legacy fallback or a resource bypass.

An initial one-layer fixture spilled284 bytes at576. D7 found scratch-field
loads hoisted across stages. Per-stage opaque kernarg reloads repair that
fixture to MT167/frame0, preserving AS4 loads, P-B wait placement and zero
generic/private loads. This fix is integrated; it proves no model result.

Actual minimum-prefetch resource probes currently fail:

| Model/B | Threads | MTreg | Frame B |
|---|---:|---:|---:|
| Qwen3-0.6B B1 | 576 | 168 | 660 |
| Qwen3-0.6B B16 | 576 | 168 | 660 |
| Qwen3-0.6B B1/B16 copied-source controls | 512 | 256 | 252 |
| Llama-3.2-1B B1 | 576 | 168 | 292 |
| Qwen3-0.6B B1, head WN8 control | 512 | 256 | 252 |
| Qwen3-0.6B B1, wider legal groups control | 512 | 256 | 316 |

The resource check rejects these binaries at
`include/tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh:150`; they are not run.
D7 global/control reload and launch-control-record experiments reduce the
512 frame to236/196 but do not meet the gate. Their changes were not applied.
The 576 attention rings reload spilled Q fragments; the 512 rings have zero
private accesses but scalar/control packs still spill. A wrapper wave-bound
assumption does not propagate through the bodies' distinct opaque thread IDs.
Per-PEU allocation is an inference; no mathematical lower bound is established.
Raw sources, commands, MIR, failed trials and hashes remain under
`runs/mx5/actual_model_d7_*/` and `runs/mx5/program_probe/`.

## Prefill KV write path

The optional fragment writer reuses the existing prefill shared arrays and
arithmetic. One query tile owns each complete32-token K/V block; vector stores
publish whole fragments with no cache read-modify-write. The covered contract
is seq64/past0. Original MX-4 geometry pins acquire only an explicit
`mx_kv_fragments_v2` layout marker; macro/manifest identity checks enforce it.
Default prefill layout remains unchanged until native decode is accepted.

Verified: 192 old/new unit kernels, all frame0, max MT222; actual-helper CPU
ownership proof1728 configurations; five fresh guarded processes ×576 cases.
Context, unpacked K/V, untouched cache tails and repeated results are bitwise
equal to the MX-4 reference. This is component correctness, not C-1/C-2.
Raw artifacts: `runs/mx5/prefill_fragments_v2/`.

## Later invariant repair and revalidation

Verified later compiler-only trials restore the actual launch's thread/wave
bounds inside the bodies, pass launch controls by value for stage-local reload,
and restore past bounds already checked by the runtime. Together these give
actual Qwen B1/B16 copied-source controls MT254/frame0 at512;576 still spills592B.
No GPU model check or timing has run. Raw proof is
`runs/mx5/actual_model_d7_past_bounds1/`. The source change is now under validation;
old campaigns are drained and their rows remain tied to their original code.
New R_next and three-process measurements are mandatory for the new source.

The intermediate unit build exposed three TN128/WN1/D2/Norm/StreamK GEMM instances
with frames24/60/24, all MT256. Their failed rows were appended to the resource
CSV. Legal-shape assumptions alone reduce frames to16/40/16 and still fail;
fixed104-CTA assumptions do not repair them. Neither diagnostic was adopted.
The accepted annotation repair retains stream-K's original wave guard while
restoring the thread bound in both partitions. All30 TN128/D2 instances then
fit (max MT250/frame0); three non-normalized stream-K instances gain6 registers,
which remains recorded. Qwen B1/B16 copied-source512 executable device ELFs
remain byte-identical to the preceding MT254/frame0 controls. Raw proof:
`runs/mx5/gemm_streamk_wave_ablation1/`.

Verified final unit builds: GEMM434/maxMT254, head30/maxMT222,
attention72/maxMT243, chain512/maxMT171 and chain576/maxMT133; all frame0 and
within their respective limits. Full fresh MIR covers538 kernels/1179 rings:
zero resource/private/call/generic/ring4160 failures. All29 executable device
ELFs equal the frozen unit binaries; two whole-library differences are confined
to non-ELF offload bitcode. Raw proof: `runs/mx5/final_unit_mir_v5_61/`.
Five fresh guarded processes each pass2074 GEMM,300 head and1008 attention
cases. Both512/576 chain paths pass5×4000 iterations,104 CTAs, zero violations.
Fresh P0 builds cover68 libraries and9090 resource queries. Of512 keys,
4495 fit;50 TN128/WN1/D2/Norm/TileRange store/SwiGLU keys have MT256/frame0,
exceeding255, and remain rejected. Their bench-only D7 investigation is
isolated. The576 queries retain820 GEMM,100 head and400 attention failures.
Raw query: `runs/mx5/calibration/dense_rnext_v5.jsonl`; the complete table is open.
These compiler-only results do not establish G-B or G-PROGRAM.

The later bench-only operand-lifetime repair changes no computation body:
current operands are reconstructed each iteration; following operands live only
inside the next-prefetch lambda. Full fresh GEMM P0 builds and6790 guarded
queries now pass3395/3395 at512 and2625/3395 at576 (770 failures retained).
No formerly fitting inventory row regresses. Several already-spilling576
frames grow; raw comparison retains them. New GEMM source:
`19e6e3676512512a0a480ec2144c59c8307d283e363efd6bfb4d266bf3ee5088`.
Head/attention source is unchanged; all1150 selected512 keys fit. The150
head576 primary failures fit after the prescribed first halving, with all
failed/successful attempts retained. Three-process timings are underway.

The actual Qwen B1 fragment-prefill plan rebuilt successfully from unchanged
MX-4 geometry with watchdog0 and verified identity. Its legacy prefill resources
are identical to the drift reference (L1 MT248/frame264); the prompt leaves
this prefill structure unchanged. This does not validate native decode.

## Actual native build and engine contract

Verified normal compiler-entry builds under `runs/mx5/native_dense_v5/`:
Qwen B1 timing/watchdog and B16 timing all MT254/frame0 at512, with verified
artifact identities. The B1 timing program has14 clean steady rings and no
whole-kernel generic/private/call instructions. Both576 minimum-prefetch
controls still spill592B and are rejected. B16 watchdog has MT256/frame0,
exceeding255; it is rejected and never executed. Its D7 control-lifetime
investigation runs only in an ignored copy; no gate or limit is relaxed.
Three control-lifetime variants do not reduce MT256; one introduces7 generic
loads and is rejected. No trial is adopted. Raw D7 proof:
`runs/mx5/native_control_lifetime_v5_61/`.

Verified CPU source audit found that equal kv_layout=vt labels allowed old
linear prefill with new fragment decode to bind incompatible bytes. The engine
now additionally requires equal kv_storage (old manifests default linear),
before weights/state binding. Nine focused compatibility cases and existing
identity/feature-contract suites pass27/27 with GPU visibility disabled.
This small engine change records a difference from the prompt's assumption
that the existing layout check alone suffices. Current full pipeline audit:
`MX5_pipeline_c550.md`.

## First manual model check and CPU solver

Verified actual Qwen B1 correctness at512: compatible watchdog1 prefill/decode,
512 steps, C-2 separate/loop bitwise equal; HF C-1 pass, max_gap0.03125,
gap_le_0_5_ratio1.0, gap_zero_ratio0.98828125. It is a manual minimum-prefetch
resource control, not a solver-selected final cell. G-PROGRAM remains open.
Raw evidence: `runs/mx5/native_dense_v5/qwen3_0_6b/B1/check_manual3/`;
all three attempts are recorded in `MX5_correctness_c550.csv`.

Attempt1 stopped before kernel launch because memory accounting lacked native
pair/u32/u64 types. Accounting now includes their byte widths and explicit
minimum allocations;22 focused CPU cases pass. Attempt2 executed the kernels
but requested diagnostic-only step clocks, so C-2 did not complete. The checker
now disables optional step events; no kernel instrumentation was added.

Verified CPU solver implements exact six-list top4, full legal-domain/resource
evidence checks, attention interpolation over L65..1087 and the1% end-to-end
selection threshold. CTest, sanitizers and8290 real GEMM/head domain-key
comparisons pass. It is not yet wired to final model selection/defaults.
Developing this CPU solver overlaps table timing; no acceptance gate is waived.

## Fixed-grid repair, five development checks and first D7 ledger

Verified later builds preserve the evaluated fixed104-CTA launch invariant in
both GEMM benchmark and native kernel. Assuming directly on the gridDim getter
was ignored by Clang because it has side effects; materializing its value first
makes the existing launch contract visible. No algorithm or searched coordinate
changed. Fresh five-process benchmark validation covers8 keys per process,
P0/Pw2, B1/B16, store/SwiGLU: each2000 iterations ×104 CTAs ×two launches.
Torch last-rotation oracles and within/across-process exact output checks pass5/5.
Raw proof: `runs/mx5/grid_invariant_v7_validation_61/`.

Actual normal-entry builds in `runs/mx5/native_dense_v7/` give these resources:

| Model/B |512 timing/watchdog MTreg; frame|576 minimum-prefetch frame B|
|---|---|---:|
|Qwen0.6B B1|250/250;0|568|
|Qwen0.6B B16|248/248;0|564|
|Llama1B B1|171/171;0|0 (MT168)|
|Llama8B B1|229/231;0|392|
|Llama8B B16|227/229;0|384|

All five512 manual-control cells pass C-1 and C-2 at512 steps. Max HF gaps
in table order:0.03125/0.046875/0.0625/0.375/0.375; all gap<=0.5 ratios1.0,
all separate/loop mismatches0. Correctness uses compatible watchdog1 prefill
and decode. All ten development prefill timing/watchdog plans now exist, using
unchanged MX-4 geometry plus the explicit fragment marker. Source/binary IDs
are retained in the append-only correctness/resource CSVs. Runtime HEAD was
not captured for these later rows and is left empty, rather than inferred.
These manual512 controls grant no blanket DT-W exception or G-PROGRAM pass.

Verified uninstrumented **old-grid** Qwen B1 manual minimum-prefetch TPOT:
2.377973717/2.375923825 ms in two fresh processes, warm1+timed2, step-events0;
median2.376948771 ms, TTFT9.094190 ms. MX-4 is1.983551895 ms (+19.832951%);
fastest baseline SGLang2.4579 ms (ratio0.967065). This is a regression requiring
D7, not a final result or an acceptable default. The first timing attempt
incorrectly requested diagnostic-only step events and remains in raw failures.

`MX5_stage_ledger_c550.csv` records the same old-grid manual control using
non-trace diag-skip differences and three-process untraced stage benchmarks.
Per-layer inferred contributions/bench, us: qkv12.790/17.880,
attention23.393/22.270, O6.090/9.180, gate_up12.908/17.010,
down8.501/11.960; head271.890/309.560 us per step.
Attention bench averages interpolated knot medians over L65..1087. Common skip
control cost262.180 us/step and nonadditive residual45.617 us/step are separate;
no unobserved absolute stage times were fabricated. Diagnostic masks include
head skip and zero token IDs to keep embedding reads valid, so their numerical
sequence is not a correctness/performance acceptance workload. Trace is off;
diagnostic-only step clocks are present and only skip differences are used.
Raw sources,19 successful guarded jobs and corrected protocol commands:
`runs/mx5/manual_d7_stage_ledger_v5_61/`.

Source inspection (verified) finds the attention LA merger computes per-partial
Exp2 serially in only Q threads and does not explicitly preserve a batch of
partial-O loads before consumption. This departs from §4.7's parallel merge;
its causal timing contribution is inferred pending MIR and repaired A/B.
A copied-header D7 investigation is isolated from current GEMM/head calibration.
The latter use immutable source closures, not mixed-source recycled timings.

Fresh GEMM v7 queries:3395/3395 P0-fit at512;2720/3395 fit at576,675 failed
rows retained. Of576 derived candidates,2380 fit primary,295 fit first halving,
45 remain unresolved; the675 P0 failures are also retained. At512 all3395
primary candidates fit. Head-only calibration resumes strict completed batches;
old attention timings remain unqualified for its changed benchmark source.
Attention v8 P0 CPU builds now fit400/400 at512 and100 D64Q4 rows at576;
300 D128576 rows still spill. Resource queries/timings of this closure are held
while the attention merge defect is repaired; no missing cost is fabricated.

Verified protocol-script fixes (`ba573643a`): candidate timing guards900s;
prefill watchdog defaults0 and accepts WATCHDOG=1; checks prefer prefill_w and
decode_w; uninstrumented decode measurements explicitly disable step events;
expert jobs reserve40960MiB, including solver device queries. Bash syntax and
whitespace checks pass. Native solver routing/MoE prompt and automatic prefill
watchdog-twin construction are still pending; the script is not a final MX-5
campaign driver yet. Existing manual development twins were built separately.

Verified full attention-merger MIR audit refines the D7 attribution: m/l has
two scalar LDG_B32→ARRIVE64→STS chains; its Q-thread leader has34 Exp2 and34
LDS waits, independent of actual producers. O already issues31 loads together,
then mixes waits/consumption. Thus O is not wholly serialized; the copied
repair targets proven m/l and alpha departures while enforcing bounded O
issue-before-consume. No performance causality is claimed before an A/B.

## Reviewed top-four route and actual-program attribution

Verified `c532da9b5` wires an explicit --mx-decode-v2-solve CPU preparation
route and immutable four-candidate orchestration. Main host CLI and selection
unit build;10 Python CPU tests and native CTest pass. Candidates remain marked
manual/diagnostic, final=false and default_enabled=false. Resource failures stop
with exact reports;512 execution is rejected until per-plan576 reductions/D7
policy exists. Raw process/guard/artifact evidence is independently revalidated
before the native1% selection consumer accepts measured pairs. Actual complete
table/model execution, automatic reductions, production promotion and script
routing remain pending. Synthetic CPU tests are not measured stage rows.
Raw review/build/tests: `runs/mx5/v2_orchestration_v1/`.

Verified actual Qwen B1 v7 program provenance (`15314b1f2` tooling): a copied
debug build reproduces every allocated device ELF section,153736 text bytes,
MT250/ST108/frame0 and19335 normalized MIR records. DWARF changes nonallocated
debug sections. Compiler-source body buckets(MIR/bytes):qkv949/8496,
attention10453/91376,O717/6560,gate_up1424/13616,down1386/12832,
head876/8792. Setup, prefetch edges, entry prefetch and control remain separate;
residuals are2111 unattributed MIR records,176 shared and24 unattributed bytes,
plus256 text bytes outside the kernel. These metrics are independent, static
compiler provenance; CSE/debug-line ownership has explicit limits. Every other
plan needs its own identity proof. Appended attribution row preserves the old
build row. Main-tool positive reproduction and a changed-MIR-hash negative test
pass; no counts transfer on failed proof. Raw: `runs/mx5/program_attribution_v7_61/`.

## Isolated attention repairs and rejected prefix candidate

Verified copied LA repair: one aligned AS1 m/l pair, parallel per-slot alpha,
original fixed lambda/O accumulation order,32 O loads before their wait/use.
All4 actual copied Qwen timing/watchdog functions and12 bench functions have
frame0, no generic/private/calls and clean MMA rings; Qwen native MT250/248 is
unchanged. The full72-kernel unit matrix has144 clean rings,maxMT243/frame0.
Five fresh guarded processes each pass1008 cases, context/K/V/tail/determinism
checks. Continuous2000-iteration paired baseline/candidate checks are running;
no new end-to-end performance or synchronization gate is claimed yet.
Raw: `runs/mx5/attention_la_units_v1_61/` and `attention_la_parallel_d7_v1_61/`.

Verified separate prefix MIR audit: current source issues owner key norm/RoPE,
then Q/norm, then V_new and initial history K/V. This differs from §4.7's
segment-start issue schedule (`DecodeAttentionV2.h:124-195`). A copied repair
issues the full batch first and retires it before transforms, preserving rings
and the LA suffix. All18 compiler invocations succeed, but six D128512
candidate functions fail: B1 MT256/frame132;B16 MT256/frame120. D64/576
fitsMT129/frame0. No failed candidate executes or enters main, and no resource
limit/threshold is relaxed. Failure/source/MIR evidence:
`runs/mx5/attention_prefix_d7_v1_61/FAILURE_MANIFEST.json`. This item remains
unfinished; a read-only D7 audit is separating required payload and avoidable
compiler liveness from allocator artifacts, without a second implementation.

Verified script follow-up `23a53044b`: explicit WATCHDOG=1 prefill solves now
write prefill_w rather than overwriting timing prefill. Both plans still need
explicit builds. No final CELLS/default/native MoE route is enabled.

## LA model checks and the segment-prefix repair

Verified main LA repair `78795d384`: five fresh continuous paired processes
each cover60 coordinates,>=2000 iterations,104 CTAs and two launches. Context,
new K, repeated-launch and cross-process output bits match the baseline.
Five actual512 manual development plans then pass512-step C-1/C-2; maximum
gaps are0.03125/0.046875/0.0625/0.375/0.375 and both execution modes match.
Ten timing/watchdog512 builds fit, four minimum-prefetch576 builds still spill,
and Llama1B B1 at576 fits168/frame0. Resource failures remain in the ledger.
Raw: `runs/mx5/native_dense_la_v1/`; these are not solver-selected final plans.

Verified two fresh uninstrumented rounds per Qwen manual control, each warm1
and timed2, ISL/OSL64/1024, step events0. B1 TPOT median2.309565870ms,
TTFT9.014596ms; B16 TPOT4.061410098ms, TTFT52.452863ms. B1 is16.4359% slower
than MX-4; G-PROGRAM remains open. B16 is about10.5% faster than MX-4 and
about1.0% slower than the fastest stored baseline. Old-grid/card7 and new
fixed-grid/card6 controls are not a causal LA P-PAIR. Four raw process rows
were appended unchanged to MX5_integration_measurements_c550.csv.

Verified copied prefix repair after the initial rejected build: retaining BF16
metadata packed until the norm inverse and materializing the true CTA bound
avoids premature expansions and signed range controls. All four actual copied
Qwen timing/watchdog functions fit253/255/253/253 MTreg, frame0. Full72-kernel
matrix has144 clean MMA rings, maxMT247/frame0;5/5 fresh processes each pass
1008 attention cases and15 exact MX-4 transform comparisons. Another5/5 fresh
continuous60-pair processes pass2000x104x2 and exact output comparisons.
Raw: `attention_prefix_cta_metadata_d7_v1_61/`, `attention_prefix_units_v1_61/`
and `calibration/attention_prefix_p0_staged_v1_61/stage_validation/gpu_runs_NEW/`
under runs/mx5. This candidate remains outside main: its extra GVM0 before
the prefix transform loses overlap, and diagnostic B1 L65..1088 rises from
21.16..22.69us to23.55..27.30us. Component results do not set defaults.

Verified CPU-only next repair uses exact younger-history wait counts, retaining
all prefix issue sites, arithmetic, layout, LA and synchronization. Actual
Qwen resources still fit253/255/253/253, and a new full72-kernel/144-ring audit
passes maxMT247/frame0. Five-process GPU checks are in progress under
`attention_younger_wait_units_v1_61/`. Fifty steady max/alpha exchange intervals
contain LDS waits only. Later segment-prefix norm code still acquires compiler
GVM0 waits; this is a measured MIR overlap limitation, not the spec's steady
DT-X exchange failure. The earlier stricter local goal is explicitly corrected
in QUEUE_AUDIT_QUALIFIED.json; no user gate is waived or performance claimed.

## Bounded actual-plan resource policy, CPU interface

Verified39 CPU tests pass after adding mx5_plan_resources.py:29 new policy
tests and10 existing selection tests. The immutable origin retains the exact
original four candidate IDs/predictions and candidate0 anchor. Actual attempt
receipts bind pins, emitted aliases/shapes, source/compiler/MIR/ELF/resources;
resource retries halve the largest reducible shared-class numeric Pw, ties
qkv/O/GU/down/head, floorD, at most three whole-plan reductions. Changed widths
require their own three-process source-qualified costs; missing costs stop with
exact keys. No baseline rank is silently repriced or claimed globally exact
after a repair. These are recorded routine interpretations of §4.5, not new
search dimensions. CPU fixtures are synthetic; no measured row is fabricated.
Native receipt production, occupancy capture and orchestration remain pending.
All-D excess MTreg requests D7; stack-only failure or exhausted reductions
stays blocked. This slice grants no512 permission or default promotion.
Review/CPU evidence: `runs/mx5/plan_retry_v1/`.

## Calibration stop: BF16 boundary discrepancy

Verified2026-10-26: GEMM campaign stops at batch129/round1/key3 with the
unchanged output assertion in stage_bench.py:142. Exact coordinates:
N16384,K3072,B8,TN32,WN1,D2,Norm1,SwiGLU,part1=TileRange,Pw8,threads512.
The first129 batches completed successfully; the failed partial batch remains
immutable and cannot qualify as three rounds. Earlier informal root text
incorrectly called part1 StreamK; DecodeShapes.h:28 establishes the enum.
The diagnostic directory/labels retain that historical naming error.

Verified same original seed2020, all27 diagnostic outputs have one identical
over-tolerance coordinate(row4,col4277), no NaN/count/ticket anomaly. Actual
part1/Pw8, part0/Pw2 and part1/Pw2 all produce-9.1875 there versus original
GPU FP32 reference-9.375, error0.1875>0.1700. Independent CPU FP64 dot/norm
gives gate2.4453114273989174, below BF16 midpoint2.4453125; rounded gate2.4375
produces-9.1875. The GPU FP32 reference instead rounds gate2.453125.
This identifies a boundary discrepancy, not yet an accepted oracle correction;
the failed candidate/table gate stays open and tolerances are unchanged.
Raw: `runs/mx5/gemm_streamk_failure_v1/`; additional independent audit pending.

## Verified oracle correction and literal-prefix adoption

Verified independent full-shape CPU FP32 matmul gives gate2.4453113079071045;
FP64 and70-digit Decimal agree on the BF16 low side. Exact input-only rational
comparison dot²-midpoint²*(ss/3072+float32(epsilon)) is negative. Neither input
proof nor new expected values use native output. Adopted db7647c34 changes the
single unconditional GEMM reference backend to CPU PyTorch FP32, retaining all
original BF16 casts and the exact .02+.016*abs(reference) predicate. Six portable
CPU tests cover the contract and an independent two-element boundary example.
New rows record torch_cpu/float32/gemm_cpu_fp32_v1, Torch version and thread
count; old GPU-oracle rows and the failed batch are immutable. New24-key retry
uses identical ordering/seeds and passes all three fresh processes. This fixes
a reference calculation discrepancy, without changing the threshold or body.
Raw review: `runs/mx5/gemm_cpu_fp32_oracle_v1_61/`; new measurements:
`runs/mx5/calibration/gemm_cpu_oracle_retry129_v1/`.

Verified6b346c997 removes an added precondition on the *nonexecuting* Pw0
reference: §4.5 defines R_next as its observed MTreg, not its eligibility to run.
Only a successful genuine zero-iteration attribute query with complete observed
attributes, binary/build/source-archive identities may retain diagnostic
nonfit data. Old unmarked nonfit records cannot be relabeled or backfilled.
Actual primary/timed rows and model caps, frame0, static0 and occupancy1 remain
unchanged. This narrows the stack0 gate to executing kernels; the interpretation
and difference from the previous stricter producer are explicit here. Root's
combined61 Python CPU tests and five native CTests pass, including malformed
attributes after a nonfit field. Raw: `runs/mx5/main_p0_prefix_cpu_v1/`.

Verified fe0829b8a adopts the exact previously copied full-prefix/packed-lifetime/
CTA-bound/younger-wait headers. Five fresh full units each pass1008 attention
and15 legacy transforms; five fresh60-pair continuous runs pass2000x104x2,
numerical, context/new-K/repeat/cross-process bits and scratch/occupancy checks.
The source/binary evidence stays separate from earlier variants. Diagnostic
Qwen B1 PF0/bmin1 atL65/1088 changes extra-GVM0's23.58/27.29us to23.02/26.54us;
B16 atL512/1088 changes71.94/92.60 to72.13/93.38us. This is not a uniform
speedup or default decision. Later-prefix automatic waits remain explicit.
Raw: `runs/mx5/attention_younger_wait_units_v1_61/` and
`attention_prefix_younger_wait_d7_v1_61/stage_validation/gpu_runs_NEW/`.
Main host rebuild and the source identity check pass;15 actual manual model
builds are running under `runs/mx5/native_dense_prefix_v1/`. Acceptance is open.

## Actual prefix resources and the 576 allocation budget

Verified all15 prefix builds complete: ten manual512 timing/watchdog builds
pass; at576 Llama1B fits139/frame0, while Qwen B1/B16 and Llama8B B1/B16
remain168/frame596/608/484/480. Full strict failures, including missing success
identities on rejected builds, remain failures. Source snapshots, ELF and MIR
are retained. Appended15 resource rows do not manufacture success identities.

Verified installed mxcc disassembly: getWarpsPerPEUForBlock computes
ceil(ceil(threads/64)/4); register granule8; getMaxNumMTRegs uses the resulting
per-PEU wave count. The observed xcore1002 budget512 gives512-thread256 and
576-thread168. This establishes the toolchain model; AP-total227 alone does
not establish launchability. A diagnostic max227 override of the earlier
full-prefix stage source yields eightMT227 functions, seven with frame/private.
Root's genuine zero-launch queries return attributes/occupancy status0 and
occupancy0 for all eight, including the Norm0/bmin1/frame0 function. No override
kernel executes. Raw: `runs/mx5/attention_576_register_budget_audit_v1_61/`.

Verified separate same-source max255 diagnostics for the four *actual final
prefix* minimum-prefetch576 programs give255/251/235/233 MTreg, frame0 and
no private/generic/calls/static shared memory. All56 MMA rings pass. These are
current compiler unspilled allocations, not mathematical lower bounds. The
postRA estimated peak for LlamaB16 is226 and cannot prove233 necessary; no new
avoidable allocation error has been demonstrated. Original168-budget failures
remain unchanged. Actual-eight-ELF occupancy queries are prepared, not yet run;
dynamic memory must come from each tm_plan_shared_bytes uint64 host query.
Raw: `runs/mx5/native_576_max255_d7_v1_61/`. D7/512 policy remains open; none of
these diagnostics grants a blanket exception or relaxes a resource/runtime gate.

Verified subsequent root-only actual-eight-ELF query: all attribute/occupancy
calls return status0. The four original168 allocations have occupancy1 but
nonzero frames; max255 allocations255/251/235/233 have frame0 but occupancy0.
Dynamic smem comes from each ELF's uint64 host query:32816 atB1 and32864 atB16.
No plan is created or launched. Root preserved the prepared int-return driver
and executed a separately hashed copy with the exact uint64 ABI. Raw:
`native_576_max255_d7_v1_61/query_result_uint64_v1.json` under runs/mx5.

Verified final-prefix actual manual controls: all five pass512-step C-1/C-2,
with the same maximum gaps as the earlier LA controls. Two fresh uninstrumented
rounds each give Qwen B1 TPOT2.393908293ms/TTFT9.077861ms and B16
TPOT4.214849114ms/TTFT52.530732ms. Versus stored MX-4: +20.687959% and
-7.102210%. Versus the earlier LA controls both regress; defaults stay off.
Four new timing rows and five correctness rows are appended, never substituted
for earlier values. Raw: `runs/mx5/native_dense_prefix_v1/development/gpu_runs_v1/`.

Verified b67500aaa adds actual pre-resource-judgment compile receipts without
altering CUDA bodies or ABI. Six native CTests pass, including87 Python tests
and a real CPU subprocess writer test. Root then invokes actual cucc twice:
512 produces253/frame0 and succeeds;576 compiles168/frame596, retains a valid
receipt, and is rejected by the original resource gate. Both generated CUDA
files exactly match their previously checked counterparts. No GPU launch.
Raw: `runs/mx5/receipt_writer_main_v1/`. The independent scoped receipt helper
does not supply occupancy or authorize512. Automatic retry/selection remains
unfinished. CTest now includes the portable policy/receipt/P0/reference suites.

Verified final attention source e3fbfaae has800 genuine P0 zero-iteration
queries and800 prescribed-primary queries. At512 all400 pass; at576 D64's100
pass and D128's300 remain resource rejections. Prescribed widths: D128/PF0 at512,
D128/PF1 at576, D64/PF2 at both. All failures remain; no forced-register binary
enters timing. The400-key512 three-process campaign is running. Selected rows
record actual active KV working sets>=256MiB, not only allocated capacity.
Head750 keys complete three rounds. GEMM final batches are still running.

Verified qualification manifest views preserve parent manifests/check rows and
all raw jobs byte-for-byte. Failed GEMM129 is excluded from the success view
and explicitly linked to the fresh same-order/seed24-key CPU-oracle retry.
An initial view fails on unused576 P0 rows missing prefetch; a new view uses
same-value checked subsets with parent hashes/membership checks. Check-only
coverage is exact4545:3395 GEMM+750 head+400 attention,0 missing/extra.
This is not measured-domain completion until every remaining batch passes.
Raw: `runs/mx5/calibration/final512_views_v2/`; the failed v1 remains preserved.

Under the user's recorded-difference/judgment authorization, root is reviewing
an explicit schema3 complete-disposition extension: each requested candidate
has either three real costs or a source-bound primary resource rejection with
no ns. Schema2 and executing resource gates remain unchanged. Rejections do
not become costs, cannot bridge attention's five-L grid, and cannot approve512
or prove a complete plan impossible. No schema3/default is enabled yet; this
is explicitly different from literal whole-domain timing, whose failed
resource candidates cannot safely execute. Ignored CPU review is ongoing.


## Qualified costs and actual top-4 resource checkpoint (verified)

The final512 table now qualifies all4545 keys (3395 GEMM,750 head,400 attention),
each from three fresh processes,0 missing/extra. SHA256:
`51aa665260b360f191c75ceccd5dfc140a5e1f535a76d43383444af6d65575a0`.
Raw: `runs/mx5/calibration/final512_qualified_v1.json`. This remains512-scoped,
full_table=false; it does not approve512 or change the default target table.
The completed attention campaign supersedes the preceding running checkpoint.

Two host integration defects were repaired without changing measured rows.
Parsed JSON numbers previously lost lexical precision when dumped, breaking
strict row membership;275be22ad preserves parsed lexemes. The actual4545-row
canonical roundtrip now matches; the old writer changed73 rows. Root's first
solve request also misread the inclusive past range:64..1087 is invalid for
capacity1088;64..1086 gives1023 decode iterations and attention L65..1087.
All20 initial attempts stopped before CUDA generation, with no compiler call
or resource observation.2a7235f4b rejects this interval before preparation.
Raw: `json_number_roundtrip_v1_61/`, `solve_interval_validation_v1_61/`, and
`solver_development_resource_v1/` under runs/mx5; all failures remain immutable.

The corrected five previews pass strict source/cost membership validation.
Their20 actual native resource attempts all have successful device compilation
and verified pre-judgment receipts, but only Llama1B B1 original IDs0/1/2 pass:
MT247/243/247,frame0. The other17 have MT256/frame380..600. All original IDs,
predictions, pins and failed artifacts remain; none has GPU execution approval.
Raw: `runs/mx5/solver_development_resource_v2/CPU_RESOURCE_REVIEW.json`.

D7 debug attribution reproduces Qwen originalID0's executable sections and MIR.
Of420 frame bytes,384 are24 vector slots associated with the terminal head
prefetch packet's control-flow merge. This is allocation evidence, not proof
that the packet is live throughout every body. Clearing consumed packets makes
the frame worse(544) and is rejected. Keeping the final down/head/tail barrier
in the terminal branch reduces it to156, still rejected. Those remaining slots
hold spilled scalar controls/pointers and masks, not head packet vectors.
A second copied variant reloads top-level Binding fields at their stage using
the existing uniform-opaque AS4 mechanism: QwenID0=251/frame0; Llama1BID3=
223/frame0, both private0/generic0/calls0. Bodies, pin/Pw and stage/next order
are unchanged. This is CPU-only evidence pending independent review, occupancy
and fresh model correctness; the patch is not enabled by this checkpoint.
Raw: `runs/mx5/solver_spill_diagnosis_v2_61/`; rejected variants remain.

The schema3 ignored CPU patch's independent review found three reproducible
qualification bugs: forced-register P0 acceptance, tail-position MIR observer
rejection, and Python cost-width validation omission. All remain outside main;
a revised patch must pass these negative cases before adoption. Raw:
`runs/mx5/outcome_domain_review_v1_61/`. No fabricated cost or512 approval.

The main guarded occupancy helper669f107a0 calls only inspected host shared-byte
and runtime attribute/residency queries. Two actual artifacts return status0,
resident1:512 MT253/frame0 and576 MT168/frame596. The latter still fails the
unchanged frame gate; residency alone cannot approve execution. No plan create
or launch occurs. Raw: `runs/mx5/actual_plan_occupancy_v1/root_actual_pair_v1/`.
After receipt/occupancy/JSON/interval integration,6 native CTests and102 portable
Python tests pass. Guard wall upper bound is14.0125 card-hours/427 closed jobs,
including waits; no active GPU job at this checkpoint. Budget remains60.
