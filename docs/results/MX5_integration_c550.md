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
