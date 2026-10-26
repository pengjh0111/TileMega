# MX-5 dense integration checkpoint

Verified checkpoint, 2026-10-26 (latest snapshot below). Authoritative prompt SHA256:
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
G-A passed; G-B and G-PROGRAM remain open. Manual-control correctness and diagnostic timings are recorded below; no
solver-selected model acceptance, MoE or DNN result is claimed. Earlier sections
are source-specific historical checkpoints; the final section gives current status.

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

Historical source inspection (verified) found the then-current attention LA
merger computed per-partial Exp2 serially in only Q threads. Main78795d384 later
repaired m/l issue, parallel alpha and bounded O issue-before-consume; the
current merger does not retain that serial departure. The old causal timing
contribution was inferred, not measured in isolation. The copied repair was
isolated from the then-current GEMM/head calibration.
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

Verified historical attention-merger MIR audit refines that D7 attribution: m/l has
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

Verified separate prefix MIR audit of the then-current LA-only source: it issued owner key norm/RoPE,
then Q/norm, then V_new and initial history K/V. This differs from §4.7's
segment-start issue schedule (`DecodeAttentionV2.h:124-195`). A copied repair
issues the full batch first and retires it before transforms, preserving rings
and the LA suffix. All18 compiler invocations succeed, but six D128512
candidate functions fail: B1 MT256/frame132;B16 MT256/frame120. D64/576
fitsMT129/frame0. No failed candidate executes or enters main, and no resource
limit/threshold is relaxed. Failure/source/MIR evidence:
`runs/mx5/attention_prefix_d7_v1_61/FAILURE_MANIFEST.json`. That rejected attempt
remains archived. The later packed-lifetime/CTA-bound/younger-wait repair was
adopted separately in fe0829b8a, as recorded below.

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
are unchanged. At that checkpoint this was CPU-only evidence pending independent review,
occupancy and fresh correctness; later9aaf3bbf2 adoption is recorded below.
Raw: `runs/mx5/solver_spill_diagnosis_v2_61/`; rejected variants remain.

The schema3 ignored CPU patch's independent review found three reproducible
qualification bugs: forced-register P0 acceptance, tail-position MIR observer
rejection, and Python cost-width validation omission. That rejected revision
remains outside main; the corrected revision passed those negative cases before
its later f7180fca0 adoption. Raw:
`runs/mx5/outcome_domain_review_v1_61/`. No fabricated cost or512 approval.

The main guarded occupancy helper669f107a0 calls only inspected host shared-byte
and runtime attribute/residency queries. Two actual artifacts return status0,
resident1:512 MT253/frame0 and576 MT168/frame596. The latter still fails the
unchanged frame gate; residency alone cannot approve execution. No plan create
or launch occurs. Raw: `runs/mx5/actual_plan_occupancy_v1/root_actual_pair_v1/`.
After receipt/occupancy/JSON/interval integration,6 native CTests and102 portable
Python tests pass. Guard wall upper bound is14.0125 card-hours/427 closed jobs,
including waits; no active GPU job at this checkpoint. Budget remains60.


## Production emitter resource repair and bounded descendants (verified)

9aaf3bbf2 integrates terminal packet scope and stage-local top Binding AS4
reloads after independent source/MIR review. Six related CTests and105 host
Python tests pass. The18 generic emitter cases execute two steps each with
unchanged Run/Next/barrier/final-broadcast order; this preserves semantics,
not the literal lexical pf_h placement of prompt4.3. Actual GPU model checks
below supply separate correctness evidence. G/H/A body closures stay unchanged.
Raw: `solver_spill_review_v1_61/`, `integration_scope_main_v1/` under runs/mx5.

Five new native previews strictly preserve all20 original IDs/ranks/pins/Pw/
predictions/evidence, with new producer/preview identity and explicit v2 parent
lineage. The base45 logical jobs produce44 actual successful device compiles
and44 verified receipts;8BB1's conditional WD1 is skipped because WD0 fails.
Original512:14 fit/6 rejected. Minimum576:1 fit/19 rejected. WD1:4 fit.
The one minimum576 fit is Llama1B originalID1,MT167/frame0 and actual resident1;
this vetoes any blanket512 policy. Other minimum plans retain MT168/nonzero
frames. All44 actual artifacts have successful zero-launch runtime queries.
Raw: `solver_development_resource_v3/`, `solver_plan_queries_v3/` under runs/mx5.

The register rule applies to512 as well as576; the old576-only ledger is a tool
limitation. Nine extra actual builds follow only prescribed largest-reducible
Pw halvings with parent receipt/ELF/pin hashes.8BB1's four origins each fit after
GU8->4 alone(MT225/frame0); origin0 WD1=223/frame0.8BB16 IDs1/3 try down4->2->1
and remain256/frame120,112 with allPw=D; those items stop as D7 defects at
DecodeProgramV2.cuh:174, without pruning IDs or relaxing resource gates.
Raw: `solver_resource_retry512_v3/`. Original predictions are never overwritten.

The only fit changed-width class has three fresh guarded stage processes:
GEMM N28672,K4096,B1,TN32,WN8,D4,SwiGLU/Norm/TileRange,Pw4. Median153110ns,
p90 154150,spread0.0007,MT116/frame0/resident1,Rnext104. The standard aggregator
and variant-row validator pass. It is separate plan-retry evidence, not a
fabricated stage-primary failure/selection. Raw: `variant_costs_gu4_v1/`.

All five new origin0 development controls pass512-step C-1/C-2 (8BB1 uses its
explicit GU4 descendant). Max gaps: QwenB1/B16 .015625/.03125,Llama1B .09375,
8BB1/B16 .3125/.34375; every C2 mismatch count0. This is manual development
correctness, not solver/default/thread-exception approval.53 resource rows,
five correctness rows and four timing rows append without changing old rows.
Raw: `solver_development_gpu_v3/`. Correctness CSV runtime HEAD is9aaf3bbf2.

Uninstrumented origin0 two-fresh medians: QwenB1 2.024607199ms,QwenB16
3.733467490ms;8BB1 GU4 descendant10.665876486ms. QwenB1's complete manual top4
medians are2.024607199/2.014710134/2.029546297/2.029032014ms. The fastest gains
only0.48884% versus original0, so the1% rule retains0 and its MX-4 regression.
QwenB16 remaining-top4 diagnostics are running. No final pins/default change.
The first timing controller reads an incorrect guard/result.json path after
successful first-round measurements; a new controller validates the existing
actual guard_result.json and preserves those measurements, then runs only the
second fresh round. An earlier B16 query preparation misses a Python import,
causes an empty-query guard failure, and is retried in a new directory. Both
failed orchestration logs remain; no plan launch or measurements are invented.

D7 readonly pairing shows HeadD3 failures spill the same26/28 SGPR words to
private memory that HeadD2 fits keep in a single VGPR carrier. No effective
Head96-weight payload is loaded during attention. Standalone HeadD3 fits186,
so no intrinsic>255 lower bound is claimed. Three isolated CPU variants are
rejected and unadopted: next-step formation delays leave104B (8BB1), explicit
uniform terminal predicate leaves120B, opaque Head geometry leaves128B
(8BB16). The last changes Head source identity and never borrows f070 costs.
Raw: `llama8b_control_spill_v3_61/`, `llama8b_b16_uniform_terminal_v3_61/`,
`llama8b_b16_head_geometry_v3_61/`. No extra protocol/fallback is introduced.

Llama1B B1's complete576 domain189 keys now has396 genuine zero-execution
queries:189 P0,189 primary,18 legal halving steps;0 errors. Source archives and
normal flags are checked, including four newly compiled exact-source libraries.
Primary136 fit/53 rejected;18 fresh halving chains recover, giving154 selected
fit and35 remaining rejections. Those cost campaigns were prepared at that checkpoint; the completed old-source
154-cost/35-rejection control is recorded below. Schema3 revision5aef2ee1's
three earlier defects passed independent Python/native repros before f7180fca0
adoption. Candidate-bound749f77ae2 auditing still grants no execution/default policy.
Raw: `llama1b576_domain_v1_61/`, `outcome_domain_review_v2_root/`.


## Latest diagnostic records and source boundary (verified, 2026-10-26)

The append-only measurement CSV now retains QwenB16 original IDs1–3, two fresh
rounds each, plus both8BB16 original0 rounds; round/executor/implementation and
actual decode/prefill IDs remain explicit. All are manual/diagnostic/final=false,
WD0/TRACE0/DIAG0, step-events0, ISL64/OSL1024, warm1+timed2 per N arm.
QwenB16 medians by original0..3 are3.733467490/3.731914691/3.735340963/
3.739188528ms. The fastest gains0.04159%, so the diagnostic1% comparison retains
original0; the four selection CSV rows do not promote a final/default plan.
8BB16 rounds13.238265565/13.244280723ms give median13.241273144ms. This does not
rescue the separate all-PwD resource failures of its original IDs1/3.
Raw: `solver_development_gpu_v3/qwen3_0_6b_B16_top4_manual/` and
`solver_development_gpu_v3/llama3_1_8b_B16_timing/` under runs/mx5.

The new v3 ledger appends six stages×two rounds plus two common-floor rows to
MX5_stage_ledger_c550.csv. Per-layer net intervention/selected-bench medians,
us: Qkv6.994/9.860, A25.390/24.913, O4.904/8.500, GU9.718/13.700,
down6.908/10.130. Head186.818/193.890us per step is specifically confounded:
mask0 vs Head-skip32 changes generated token history. Each other reference uses
Head-skip32; intervention masks33/34/36/40/48 add the corresponding body skip.
These differences include dataflow, overlap, packet retirement and synchronization;
they are not independent body costs and must not be summed. The345.173us/step
common skip63 floor retains Next loads, skipped-consumer retirement, barriers,
broadcasts and DIAG clocks, and remains separate/unattributed. No absolute
per-body trace time is used. DIAG1/TRACE0 is an intervention workload, not a
numerical acceptance or uninstrumented TPOT workload. Raw:
`runs/mx5/v3_stage_ledger_61/ledger_summary_v1.json`; all14 appended rows state
round, formulas, confounding/nonadditivity, coordinates and source identities.

Historical/current spec boundary:78795d384 repaired the old §4.7 serial merge;
fe0829b8a adopted segment-start issue/packed-lifetime/CTA/younger-history waits.
Later-prefix automatic global waits still limit overlap; this is distinct from
DT-X's per-block softmax exchange/ring gates.9aaf's terminal packet scope and
stage-local AS4 reload preserve actual Run/Next/barrier order while differing
from prompt4.3's literal outer pf_h lexical scope. The engine's kv_storage check
also records the physical-layout compatibility requirement beyond kv_layout.
None of these differences waives end-to-end gates or grants blanket512 execution.

Llama1B B1's old-source576 control is now qualified schema3:189 legal keys=
154 costs from3fresh processes +35 witnessed primary resource rejections,
0 missing. Rejections contain no ns; Attention keeps all five L knots. Sources
remain G9705/Hf070/Ae3. Full table SHA256:
`2e38227936dbec0dfdd0420f61292dfb734c08e5f821f77da4ddce5f7782ce33`.
Raw: `runs/mx5/llama1b576_qualified_outcomes_v1_61/`. This scoped complete
cost-or-rejection control is not literal all-key timing or a current-family
execution certificate. Main f7180fca0 schema3 and749f77ae2 thread-exception
audit add CPU evidence contracts only; schema2/defaults/performing resource
limits are unchanged, and audit APIs keep execution_authorized=false.

Candidate-bound QwenB1 D7 retains normal576168/frame556 versus controlled
max255255/frame20; actual root API0/occupancy0/shared32816/zero launches matches
the diagnostic ELF/MIR. This is an allocation observation, not a mathematical
required-register lower bound or global576 impossibility proof. Historical
old603/producer19c45 READY became correctly blocked when four source inputs and
the live host changed during schema3 adoption; old refs/archives remain unchanged.
It cannot approve current source execution. Raw:
`runs/mx5/qwenB1_min576_budget_v1_61/REVIEW_QUERIED.md`.

Verified update:3843f6477 adopts the §4.8#1 idle-group Prefetch fix after
40/40 guarded fresh-process jobs pass (eight admitted families×five;40 distinct
test PIDs). Rejected576 complete families were never executed. The four new
null-idle test files are byte-identical to the tested frozen sources; the build
helper's `--null-idle` accepts only GEMM/head unit sources and rejects benches.
Eligible kLoads, per-compute-wave next() and P-B source issue order stay fixed;
actual allocated ring waits change. No speed gain follows from omitted requests.
New G/H/A closures cannot borrow old-source calibration. Raw:
`runs/mx5/prefetch_guard_root_validation_v1/verified_summary.json`.

Real native QwenB1 fixed-Pw=D controls omit historical R_next metadata and use an
immutable archived producer. WD0 MT251/WD1 MT255,both frame0/static0,15 clean
MIR rings/no generic/private/calls; actual occupancy queries API0/resident1,
zero launch. WD1 passes512-step C-1/C-2 (max gap0.015625,mismatch0), with the
entire token record exactly equal to the old control. These are manual controls.
Same-card P-PAIR order old1/guard1/old2/guard2,each fresh,warm1+timed2,step-events0:
old2.026855987/2.026719919ms,guard2.035135174/2.034372341ms. Medians
2.026787953/2.034753758ms,+0.393026%,below1% and indistinguishable. Relative to
stored MX4 1.983551895ms,the guard median is still about2.58% slower: G-PROGRAM
remains open. This is not a solver-selected result or a default change. All four
rounds,one check and two resource records append to the CSVs without editing
old rows. Raw:`runs/mx5/prefetch_guard_e2e_v1/`.

A separate attention candidate shares per-(q,w) M/Exp2 factors in wave zero's
dead prefix (9Q floats≤64+D/2), preserving fixed wave accumulation order and
adding one warp+one existing CTA sync. Normal QwenB1 copied-device compile:
MT251/frame0,15 clean rings,EXP_F32 52→38,but text201784→202904B and MIR
25566→25712; no code-size/speed benefit is inferred. Its complete512 family
passes5/5 fresh numerical processes. Four576 D128 libraries remain resource
rejected. Both chain libraries are whole-ELF,source,allocated-MIR and resource
identical to the already five-fresh checked guard chains; those GPU checks are
reused with exact identity proof. Candidate remains off main pending isolated
real-plan C-1/C-2 and E2E. Raw:`runs/mx5/attention_cta_scalar_share_d7_v1_61/`.

Budget checkpoint 2026-10-26T06:40:28.308091+08:00: union `runs/mx5/gpu_jobs.log` and
`runs/mx5/llama1b576_domain_v1_61/modern_dispositions_v2/timing_gpu_jobs.log`,
canonical distinct out: 499 closed jobs, active_count=2; closed wrapper wall
15.872500 card-hours plus active elapsed0.016560 to checkpoint,
upper bound15.889060/60 card-hours, including waits/cooldowns. At that snapshot
13 copied-validation jobs had closed successfully,2 were active; no assertion
that all40 were complete. Current campaigns use7/6, at most two cards. Full
source snapshots and append/hash/dedup proof: `runs/mx5/docs_latest_evidence_v1_61/`.
G-A passed; G-B/G-PROGRAM, final25 dense, native MoE, LLM30 and DNN25 remain open.

Verified 2026-10-26 continuation: the attention CTA scalar-sharing experiment
passes real512-step C-1/C-2 with identical control tokens, but same-card P-PAIR
GU/scalar medians2.034739340/2.042545932ms (+0.383665%, indistinguishable).
It was not adopted. Its isolated WT/producer/patch, two resources, one check and
four timing rounds remain source-specific evidence, not final cells. Raw:
`runs/mx5/attention_scalar_e2e_v1_61/ROOT_DECISION.json`.

D7 then identified live BF16-to-FP32 statistics expansion at GEMM/head Consume.
5c032c501 schedules that expansion after all MMA atoms and before ring refill,
using compiler dependency barriers; Square arithmetic and accumulation order
are unchanged. No runtime fence, extra load, axis or rounding change was added.
The exact tested G header SHA256 is
`2be242390f5fe4a1764a0ff637b0b1ba94d61f15d80f99113d17f444ea436152`.
Earlier late-V and barrier-localization diagnostics still spilled and remain
unadopted. Their observations are not required-register lower bounds.

The new GU+Consume source has50 normal unit builds:42 libraries fit,8 rejected
(all576); all failures retained. Complete512 G/H/chain/null-idle families pass
25/25 fresh guarded processes, five per family and25 distinct test PIDs;
chain tests use104CTAs and2000 iterations. This does not admit arbitrary512
programs or the rejected576 families. Root seal:
`runs/mx5/consume_guard_root_validation_v1/verified_summary.json`.

Actual native diagnostic plans bind a real detached8bba+Consume WT and an
immutable mixed-host producer622ae426...; historical host-object dependency
limits remain explicit in its delivery manifest. QwenB1 WD0/1 MT241/245;
8BB16 original1 allPw=D MT241/243, all frame/static0,15 clean MIR rings,
no whole private/generic/calls. Whole MIR still has waits4160 outside the
certified rings. All four actual queries return API0/resident1/zero launches.
Both model controls pass512-step C-1/C-2 (max gaps0.015625/0.34375,mismatch0).
Qwen same-card GU/Consume P-PAIR medians2.034798772/2.053543765ms (+0.921221%,
indistinguishable by1%/spread); no speedup is claimed and MX-4 acceptance still
fails. Native raw:`runs/mx5/gu_stats_native_v1_61/`. Main ordinary rebuild at
5c032c501 passes with matching fingerprinta868fa47...; its archived producer
SHA23fde9d14daf0c52703d25e29870f1dbc3faca1d4176e48dbdd785dcef9b6000.
Old source costs/query outcomes cannot qualify this new body epoch.

Eight further native builds (CPU resources only at this checkpoint) all fit:
QwenB16 MT243/241;8BB1 originalGU8 MT245/249 without reduction; real1B576
MT152/153;8BB16 original3 allPw=D MT241/243. Each has15 clean MIR rings.
All eight actual queries also return API0/resident1/zero launches;512-step
checks and two-fresh timings are running separately. No original candidate is deleted. Raw:`runs/mx5/gu_stats_remaining_controls_v1_61/`.

Fresh GU+Consume G/H/A calibration is prepared separately. The preceding GU-only
P0 epoch has8290 genuine zero-launch queries, including resource failures;
none is relabeled as a cost or as new-source evidence. Its primary/halving
recipes were prepared but not compiled. The current QwenB1 actually uses
attention PrefetchKind0, so source-level PF1/2 idle-load behavior cannot explain
its measured latency. Static path counts are attribution, not measured time.

Verified budget at2026-10-26T09:44:05+08:00: union three raw job logs by canonical
out gives579 closed/1 active,17.420278 wrapper-wall card-hours through checkpoint
of60, including active elapsed/waits/cooldowns. The third log is
`runs/mx5/next_body_epoch_calibration_v1_61/gh_p0_query_gpu_jobs.log`.
Append-only CSV prefix/hash proof:`runs/mx5/docs_consume_evidence_v1/`.
G-B/G-PROGRAM and final25 dense/native MoE/LLM30/DNN25 remain open.

Verified8BB16 original1 two-fresh TPOT13.346345006/13.335000643ms,
median13.340672824ms (manual diagnostic,not final). This is below stored MX-4;
it is not an isolated performance comparison with original0 or a solver result.

Verified five-development-cell GU+Consume checkpoint: all pass512-step C-1/C-2;
new remaining4 WD pairs have8/8 actual API0/resident1/zero-launch queries.
Two-fresh medians Qwen0.6B B16=3.745389509ms;1B B1 genuine576=2.157856207ms;
8B B1 originalGU8=10.622665064ms;8B B16 original3=13.321567510ms.
Together with original1=13.340672824ms and QwenB1 P-PAIR2.053543765ms,
these are manual controls, not final solver-selected cells. QwenB1 is+3.5286%
versus stored MX-4. All10 new timing rounds,4 checks and8 resource records append.
The complete GU+Consume unit resource matrix adds1142 rows, including88 failures;
old CSV bytes remain unchanged. Proof:`runs/mx5/docs_consume_resource_matrix_v1/`.

The four-control driver's process metadata accidentally hardcoded functional
HEAD5c032c501:11 later processes actually started after docs-only332a72a1a.
Original files are preserved; `root_check_time_four/RUNTIME_HEAD_METADATA_CORRECTION.json`
binds each correction to the root reflog snapshot. Correctness CSV runtime HEADs
use per-process actual values. No measurements or implementation bytes changed.

The first attention O-load-guard proposal is strictly rejected:256MT/frame332,
private196,text+7904B,MIR+987. Actual conditional guards work, but this is not
execution eligibility. No GPU or second compilation was used for that proposal.
A separate implementation uses8-value O batches, within §4.7#5's ≤32 bound;
contribution order stays0..producers−1 and M/lambda code is unchanged. Its single
prototype is241/frame0,15 clean rings,whole private/generic/calls0;text−10304B,
MIR−1287. Full512 family passes5/5 fresh numerical processes;576 full family
retains all48 failing D128 kernels. Current Stats chain is exactly reusable;
its earlier five-fresh proof is bound explicitly, not described as newly run.

Real A8 WD0/1 plans bind a detached332+onlyA8 WT, source604/c9a3812d...,
immutable producer8c779798... with explicitly retained mixed-host limitations.
Both are241/245MT,frame0,15 clean rings; actual API0/resident1/zero launches.
512-step C-1/C-2 passes (gap0.015625,mismatch0),both token records exactly equal
parent Stats. Same-card P-PAIR order Stats1/A8-1/Stats2/A8-2:
Stats2.051458109/2.052354270ms,A8 2.007340806/2.007769378ms.
Medians2.051906190/2.007555092ms,ratio.978385416 (−2.16146%),distinguishable.
ca678cbc8 adopts the exact tested A header1a8de92f... . There is still+1.21011%
versus stored MX-4; G-PROGRAM is not passed. Other4 A8 development controls are
being rebuilt. Raw:`runs/mx5/attention_O_batch8_e2e_v1_61/ROOT_DECISION.json`.

ResolvedTop4 first slice1f16c6b2b passes31 root-main CPU tests (earlier194pass/
1skip regression). It binds immutable exact four origins, complete domains,
actual artifacts/MIR/query and exact-width cost lineage. Unknown earlier576
reductions remain null/needs_576_retry_history; no reset-to-zero, execution or
default permission is inferred. Canonical576 history and result consumers remain
work in progress. New G/H actual P0 queries cover8290 keys;120 primary libraries
are source/formula-bound and authorized, not yet three-fresh costs. G/H closures
remain exact after A8; only16 fresh A P0 libraries are queued. Old A data are
retained separately. No source/provenance/cost row is relabeled.

Verified full A8 development refresh: all8 remaining WD plans pass actual
API0/resident1/zero-launch queries,512-step C-1/C-2 (four additional checks),
and all8 fresh timing processes complete. Current medians:QwenB16 3.692649128,
1B B1 true576 2.130356674,8B B1 originalGU8 10.646278181,
8B B16 original3 13.271694603ms. Current QwenB1 remains2.007555092ms,
+1.21% versus MX-4; no G-PROGRAM pass or final selection is claimed.
New query driver v1 failed on a manifest-schema key before any plan query;
v2 uses a new guard/output and completes all8. Failed setup is not a register
retry or model defect. Old logs are intact in `QUERY_SETUP_FAILURE_NOTE.json`.
New driver runtime HEADs all match actual per-process reflog values; an
intermediate annotation compared full SHA with a short prefix and falsely
flagged12. `RUNTIME_HEAD_METADATA_VALIDATION_v2.json` explicitly supersedes it;
no process raw or CSV value changed. Append proofs under docs_batch8_development_v1.

Canonical576 historyv2 ed737aa1b passes26 root CPU tests (220pass/1skip prior
regression). It binds actual positive576 R_next,computed PRIMARY576 widths,
unchanged canonical attentionKind and actual failure chains across the shared
three-reduction budget. Manual PF0/unset-Rnext floors cannot establish prior0.
MT/frame-only P0 failures remain nonexecuting probes; static-smem or occupancy
anomalies are explicitly rejected. Substantive v2 D7/result consumption remains
unfinished; audit outputs never authorize execution/defaults.

A provenance-consumer constraint found110 G576 next-width keys with real
primaryHEAD1f16 and same-body existing childHEADca678. No metadata or validator
was changed. Two needed child configurations were genuinely compiled in a clean
detached1f16 worktree; actual ordinary build metadata reports1f16 automatically,
and device text matches the preserved ca counterparts exactly. A new190-key
query plan uses those110 children and the other80 ca children with unchanged
real parent references. Fixed real worktrees prevent future head drift.
Raw:`runs/mx5/calibration_stable_heads_v1_61/`.

G512 timing jobs0/1 both pass actual3fresh protocols (48keys),guard wall
113.443679/99.005964s. These are current wrapper durations,not substituted
stage costs or a final table. Remaining142-batch coverage is running on7/6;
source/round/PID/counter/oracle/rotation records stay raw until qualification.

## Canonical576 and qualified D7 continuation (verified)

`ed737aa1b` binds canonical positive576 R_next, PK and the shared at-most-three
reductions; `858e0ef6b` binds substantive D7/source/MIR and all four exact IDs.
90 related root CPU tests pass; no real candidate certificate or final selection
is claimed. Old baseline I1 and current static footprint remain distinct.
Actual new resource queries: G190 has185fit/5failed;H150 has150fit. The remaining
G5 are true D2 floors and retain their failures without another compilation.
A8 P0 actual queries:512400fit;576100fit/300failed.
Raw:`runs/mx5/checkpoint_gamma_validation_v1/`; wrapper-wall union at12:44:22
Shanghai is20.798983/60 card-hours (685 closed/2 active),including waits/cooldowns.
Isolated gamma-weight barrier scheduling has only CPU admission so far; full
512 A and fresh old-Rope bitwise helpers will run before native/E2E review.

## Gamma packed dependency scheduling (verified)

`2dbf763ab` adopts exact RowTransform SHA4e52ae2d7460ccb1ad8e295e6bcdf3fdbb40dfdcae02fe61bf2af5e1ff55b1cb.
Inverse/RotateEight/raw copies/tree/rounding and all pin coordinates remain exact.
Moving packed gamma dependencies once per chunk permits real CSE; current K/Q
prefix BF16 rounds each192→160 and MUL160→128 (counts,not measured timing).
All5 root fresh processes pass1008 A cases+two15-case old-Rope helpers each.
Both WD plans241/frame0,resident1,15clean rings,whole private/generic/calls0;
WD0 text191736→183128B and MIR24422→23360. Actual source606 SHAdaa6f6746b15972c729b6bacdab8e529e28012c719f5f4612f20abc2d0d688aa,
real dirty ed737 WT; immutable mixed producer2227fc3914dacd5f40a095ac81b26263d9c5e8ab52641db835b018799539779e.
Historical mixed-host limitations remain explicit,not final producer qualification.
C1/C2512 pass,maxgap.015625,parent token files byte-equal. Same-card P-PAIR:
A8 rounds2.007759914/2.006651832;gamma1.960924107/1.955569767ms;
medians2.007205873/1.958246937,ratio.975608414,range.005354340ms.
Gain2.439% is distinguishable;vsMX4−1.276%. Other4 controls/fresh A P0
rebuilds are pending; G-PROGRAM and final selection remain open.
Raw:`runs/mx5/attention_gamma_barrier_e2e_v1_61/ROOT_DECISION.json`.
Unit CPU seal accidentally included one running root log; old failed full-seal
check is retained. DELIVERY_v2 excludes only that mutable log;1511 immutable
files recheck exactly. Native1473 seal also passes. No binary/source was altered.
Historical cost child PIDs are unrecorded; archived runner88–94 independently
subprocess.run()s three actual rounds with receipts/raw+guard0. Future passive
PID observations supplement records without changing the measurement protocol.

## Explicit measured-selection consumer (verified CPU; no final cell)

2852cad5b adds mx5_resolved_runner.py and an explicit MX5_RESOLVED_V2
script branch. It revalidates genuine exact-origin four candidates, builds and
queries their exact WD1 twins, checks all4, then times2 fresh processes each.
Selection retains original candidate0 as the1% anchor; selected WD0 binary/pin
are the actual measured artifacts. Complete sidecars are linked without overwrite;
central GPU logging and cwd are explicit. Default target/pins are unchanged.
Root25integration tests pass; worker272pass/1optional prebuilt-harness skip.
All new test execution is synthetic; no real candidate certificate or selected
final cell is invented. Earlier fixture-import failures remain under runs.
Actual gamma A800 P0 and800 primary queries each return512400fit and
576100fit/300rejected. All queries are iterations0/query-only/status0.
Five ordinary primary builds plus3sameepoch PK0 reuses preserve prescribed PK;
D64 PK2 fits136MT@512 and143MT@576. No PK is lowered to fit rejected D128576.

## All five gamma development controls (verified; manual)

The4 follow-up pairs pass all8 actual resident1/API0/zero-launch queries,
C1/C2512 and8fresh uninstrumented timings onGPU6. Tokens in both modes
are exactly equal to their A8 parents. MT WD0/1:QwenB16 237/239,1B576
151/152,8BB1GU8 242/246,8BB16orig3 240/243; all frame/static/private/generic
/calls0 and15clean rings. Runtime HEAD is captured per process,not hardcoded.
Current TPOT medians:QwenB1 1.958246937,B16 3.571500050,1B B1
2.126210101,8B B1 10.625549654,B16 13.242555381ms. Versus MX4:
−1.276/−21.282/−3.838/−0.840/−8.931%;8BB1 is indistinguishable below1%.
These are development controls,not final solver-selected cells or default approval.
Raw:`runs/mx5/checkpoint_after_gamma_development_v1/CURRENT_DEVELOPMENT_CONTROLS.json`.
Append-only8resource/4check/8timing rows preserve every previous CSV byte;
proof:`runs/mx5/docs_gamma_development_v1/APPEND_PROOF.json`.
Root6 continues A512/A576 3fresh costs then remaining G51248..71;7
continues G512101..141. Global cost/reject domain is9090=8025 pending/current
measured costs+1065 actual rejections; partial coverage is not final qualification.

## Catalog and D7 attribution validation (verified; no final selection)

12b5ab839 retains byte-exact scoped tables under a50-member catalog and
binds actual model/request/root/leaf selection. Global diagnostic LS uses unique
current GH rows only; it does not alter candidate costs/ranks or average scope
R². Normal tilemega+mx_decode_solver_v2 build and C++unit pass;50 root Python
checks pass (worker297pass/1optional skip). Actual complete-table native preview
still awaits genuine full data. CurrentG/H/A source archives14/15/19files match.
32 guarded API-only queries of two historicald5 binaries:168MT all8 stubs
512/576resident1;227MT all8 512resident1/576resident0,including local0.
Allstatuses0,iterations0,launches0. imcOccupancy disassembly uses actual
per-PEU register/wave model; live ISA constants and physical topology are not
independently observed. No compiler-bug or physical>227lower-bound claim.
Gamma f8 norm-six3freshPIDs2277152/2277481/2277721 pass original oracles,
2000iter/cold>=256MiB. N1/N0 mediansus:L65 19.20/15.19,L51221.97/18.03,
L1088 23.69/19.26. Different standard ordinal seeds and changed code/loads,
registers/rounding/schedule prevent calling this pure normalization cost.
All3 lengths have no head-crossing CTA and one block per active wave;has_next
is false,so unused steady K/V overlap cannot explain this measured narrow case.
Current QwenB1 completed16 compact-mask jobs give netus/layer:QKV6.651,
A22.915,O5.033,GU9.978,down6.895. A five-point3fresh mean overL65..1087
is21.84655us;ratio1.04893. Head190.355us is history-confounded;346.015us
commonfloor is not synchronization-only and not additive. Diagnosticabsolute
step medians1.982265/1.984230ms are not the accepted normal1.958247ms.
Raw:`runs/mx5/checkpoint_diag_attribution_v1/`;other4 ledgers remain in progress.
Append50 new diagnostic probe rows (18timing+32nonexecuting attrs), preserving
all earlier bytes; no diagnostics enter final cost tables or model candidates.
