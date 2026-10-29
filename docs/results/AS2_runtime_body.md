# AS-2 native runtime/body evidence

Device receipts retain their date/physical/PCI identity. Most earlier runs use
physical1→logical0; the2026-10-26 mapping refresh is recorded below. Software:
Ascend950PR_9579, CANN 9.1.0,
BiSheng clang15.0.5, pinned CATLASS 2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52.
No device lock/exclusive setting, reset, other-process operation or CUDA test.
Native full-layer-count phase kernels now execute; complete generation and
performance admission remain unfinished. Earlier probes retain their own scope.

## Runtime and resource scope

- `tools/probes/ascend950/as2_timer.cpp` compares warmed 4M/16M system-counter
  loops with ACL events: 4.00795/16.0083 ms, differential12.0003 ms. The initial
  1M-tick first-launch experiment failed its admission limit and is retained.
  Frequency 1 GHz is verified here, not inferred from the AI Core clock.
- `Target/Ascend/ProgressPrimitives.h::WaitProgress` uses a supplied counter
  deadline and abort. AS-2 progress probes use 2e9 ticks per wait. This is not
  the legacy 100000-poll limit re-labelled as seconds.
- `as2_progress.cpp`: G2/G4/G8 **5/5 fresh processes each**, each16 epochs and
  four directions. AIC producers execute CATLASS identity MMAD/FIX; both AIV
  lanes produce MTE2/MTE3 payloads. Consumers pre-read cached old data, publish
  preload, wait for payload completion/publish, invalidate/acquire and check
  exact values. Ready, preload and reverse acknowledgements have single owners.
- The old DSB and entire-DCache acquire remain the correctness baseline.
  This is not a minimal-cache-maintenance result. G14/28, mode4, BF16
  direct and UB→L1 remain unverified. New GM MIX lifecycle evidence is below.
- G2 vector tests compute separate rows on all four AIV workers. This is task
  assignment/result evidence, not a hardware busy-percentage measurement.
  `runs/as2/probes/resource_summary.json` indexes every independent process.

## Body contracts and numerical scope

`Backend/Ascend/GemmTile.h` is coordinate-driven CATLASS BlockMmad, not the
BasicMatmul/DeviceGemm scheduler. New variants use BF16 A and logical NK weights
as column-major KN, FP32 accumulation and BF16 output. Geometries are
32×64×64 and16×128×128, stages2; AIC Resource reserves full L1/L0 banks.
The direct NK profile entry admits N,K multiples16; unaligned K needs a separate
padding contract and is not silently accepted as verified. No split-K is enabled.

`LlmVectorBody.h` declares106496 B of explicit UB for BF16 vectors and FP32
work/reduction buffers; the compiler's additional reserved resources are not
reinterpreted as occupancy. TPipe is initialized once per worker in the probes.
RMSNorm consumes the entire row, with FP32 variance/epsilon and explicit BF16
rounding before gamma multiplication. SwiGLU rounds SiLU to BF16 before the up
product; residual add uses FP32 opmath. RoPE consumes supplied BF16 cos/sin and
preserves BF16 product/add cast points. The standalone body consumes supplied coefficients; the new native packer
uses pinned model-specific coefficient generation, including Llama scaled RoPE.

AttentionChunk transfers up to32 contiguous K/V rows in each DMA. Broadcast and
AR/RA vector reductions compute QK and weighted PV; it emits FP32(m,l,o).
AttentionMerge rescales all unnormalized contributions before division. K/V
head mapping is integer division; ratio3 is exercised. This all-valid-key body
has a separate diagnostic ABI. The integrated serving body adds KV append,
QK norm/RoPE, causal prefill and a normalized-partial/log2-LSE route; general
all-masked attention is not admitted.

| Numerical cases | FP32-reference max abs | BF16-reference max abs | Frozen-policy failures |
|---|---:|---:|---:|
| GEMM M1,N1024,K3072, both variants | 0.00769520 | 0 | 0 |
| GEMM M16,N2048,K8192, both variants | 0.0156002 | 0.015625 | 0 |
| RMSNorm rows16/H3072, rows4/H4095 | ≤0.0141602 | 0 | 0 |
| SwiGLU rows16/F14336, rows4/F129 | ≤0.0273438 | 0 | 0 |
| RoPE rows16/D64 and D128 | 0.0078125 | 0 | 0 |
| Attention GQA3/K65/D128 | 0.00139278 | 0.0000610352 | 0 |
| Attention GQA4/K129/D64 | 0.000973850 | 0 | 0 |

Attention FP32-reference NRMSE is0.0016588/0.00162023 (limit0.01).
All cases report nonfinite=0. GEMM relative maxima near zero are retained in
numeric.json; the pre-frozen absolute-plus-relative policy is not relaxed.
The default AS-1 row-major GEMM64×128×256 also passes exact BF16 protection;
its inputs are read from AS-1 and all new output/build files stay under AS-2.
This is not the pending encoder diagnostic/fast comparison.

Reproduce one case after sourcing `scripts/as2_env.sh`: build only as2_gemm,
as2_vector or as2_attention with parallel1/nice10. `as2_body prepare-*` creates
one input/reference case; run the matching CLI through `as2_run`; then use
`as2_body compare-*`. Every output directory must be new. Exact invocations,
binary hashes and comparison results are in `runs/as2/jobs/` and `probes/`.
Selected vector/attention binaries, source/header snapshots and compiler logs
are retained in `tilemega-as2/build/retained/`; AS-1 files are never cleaned.

## Native LLM integration (verified-code/run; experimental)

`LlmServingBody.h` implements real Embedding/RMSNorm, GQA QK normalization,
RoPE/KV append, SwiGLU/residual/argmax epilogues and full vocabulary reduction.
`AscendLlmCodegen.cpp` emits per-stage implementations/resources/layout/flag
contracts. GEMM uses AIC-only or AIC+AIV0; both AIV lanes also own independent
vector tasks. AIV1 is never forced to acknowledge a nonparticipating task.

MIX owner: wait semantic frontiers, MMAD/FIX completion, publish ready, wait
AIV done, publish external completion once. AIV0: wait dependencies/ready,
acquire, epilogue, done, wait owner completion, publish its own FIFO prefix.
The separate acknowledgement prevents early buffer reuse. No hardware
cross-core flag namespace is borrowed from CATLASS. Five sequential processes
with four forwards and346 equal payload/control files verify the G2 protocol;
this is not a proof that Qwen numerical correctness passes.

Unified attention uses32-key chunks, BF16 QK/scaled-score/probability storage
rounding and FP32 reductions. Causal prefill transforms private K/V from the
entire ready QKV output; each public cache row has a unique query-block writer.
It does not consume cache rows another query task has not yet published.
These private transforms are repeated, not a claimed optimized final design.
Decode uses the actual runtime past and never reads invalid KV padding.

Local UB remains106496 B. GM worker scratch is32/128 KiB, separate from UB;
raw GEMM epilogue workspaces are real Plan allocations. The optional old
128-key FP32-partial/log2-LSE path is numerically unadmitted. BF16 direct,
UB-to-L1, dual-AIV MIX, split-K and G14/28 remain disabled/unverified.

The 200-byte Stage contains64-bit operand/workspace offsets. The device reader
copies object bytes legally; uint32-pointer type-punning previously corrupted
QKV at optimization. PV accumulation includes a V-to-MTE2 dependency before
reloading its reused local source. Both integration failures and binaries are
retained. Task/FIFO/epoch/guard/ABI receipts and exact source/binary hashes live
under `runs/as2/` and `tilemega-as2/build/retained/`.

Llama1B B1 past64 zero-KV decode passes114 output/KV comparisons (logits
NRMSE0.006897, exact token62324). Qwen prefill logits/token pass but26
intermediates fail; Qwen zero-KV decode fails the frozen model tolerance.
These are phase results. They do not establish64/1024 generation,512-step
teacher-forcing, every model/B cell or speedup against eager.

Verified-run request extension: `as2_request.py` binds actual solved phase
allocations by names/packing/config identity, rewrites only addresses, and
hard-links immutable inputs. `LlmRequestHost.h` shares one arena and submits
one persistent launch per forward; prefill token64 feeds decode at past64.
The last formal output would be token1087 from past1086, not an extra decode.
The runner checks prefill abort before submitting dependent decode, preserves
error controls, resets KV/tokens/phase frontiers only at completed boundaries,
and moves dumps/guards outside request timing. CPU rejects OSL1025 before ACL.

Short natural requests execute all layers for Qwen0.6B and Llama1B, four tokens
each, exact against independent NPU eager. Qwen logits pass but five K-cache
rows contain failing elements. Llama logits have33 failing elements plus14
failing KV rows. Frozen tolerances remain unchanged. Same-input selected-layer
Norm/GEMM/RoPE/Attention/SwiGLU checks pass; this does not waive composed errors.
Qwen request reset passes5/5 fresh processes, two requests/four forwards each;
401 payload/KV/control files are bitwise identical, including diagnostic/fast.
Proof: `correctness/native_request_reset_c3_summary.json`. Full-length requests,
512-step teacher forcing and final model-level optimization remain unverified.

## Continuation: precision, profiling and sourced plan execution

Verified-run: one real first-layer eager SDPA trace casts Q/K/V toFP32,
pre-scales Q/K, and runsFP32 matmul/softmax/PV. Hardware metrics were disabled;
only one attention call was profiled, with one offline analysis worker.
`as2_sdpa_dispatch.py` records tensors/operator metadata privately.

ABI v3 declares four attention implementations: BF16 score/P, FP32 chunks,
FP32 score/BF16P, and math-SDPA pre-scale/FP32P/capacity-softmax. No invalid KV
padding is read. Local UB remains106496B; per-worker GM scratch131072B.
First prefill context differs in48/131072 BF16 elements against eager rather
than74593 with BF16P. Full Qwen four-token logits/tokens pass, but11 final KV
slices fail the original policy. No tolerance change or correctness claim.

Diagnostic2 timestamps Body service outside semantic/ready/done waits and
metadata/publication;512B per queue slot separates writers. Worker fractions
are profiled intervals, not hardware-utilization counters. Five fresh finite
`as2_native_cost` processes pass with8192 copy samples each and a512MiB working
set. Median dispatch AIC/AIV886.25/879.469ns, symmetric notification estimate
272.484ns, effective logical GM/L2 copy rate1137.54B/ns; all contended/contextual.

The priced Qwen plan changes actual queues/owners, executes one launch per
forward, and preserves399 payload/KV files bitwise. Controls/trace are audited
against the new plan rather than expected to match old ordinals. Numeric
failure versus independent eager remains separate. Measured decode prices are
limited to past<=66 and guarded before ACL; this is not long-request admission.

Earlier math-SDPA candidate, verified-run: full28-layer Llama3B executes four forwards with
6,735,441,920B shared arena (>4GiB addresses), valid task/launch receipts and
exact greedy tokens. Against independent SDPA/fused eager, logits NRMSE0.036681
with11020 failures and49 failing final KV slices. That earlier candidate fails; the split2 repair is recorded below.

## ABI4 cooperative FP32 attention (experimental, verified-run)

`LlmCubeBody` borrows one CATLASS resource bank across BF16 tiles and full-FP32
QK/PV, draining each MMAD before reuse. G8/head128/GQA2 is the initial experimental
domain; it is an explicit candidate, not a performance default. Per-group GM
workspace is2MiB; invalid public KV is never read. `ExecuteLlm` reserves four
ready/done values per ordinal for three phases; only AIC publishes logical task
completion, with reverse ack before AIV0 reuse. Empty AIV1 does not wait.

Body/protocol probes use deliberately constructed diagnostic queues, explicitly
not a model pipeline. Prefill/decode each pass repeatability in5/5 fresh
processes, including multiple query coordinates, BF16 K14336 interleaving and
four decode epochs. Full KV/control and large-K outputs are bitwise exact.
Decode context matches independent eager bitwise. Prefill has13/131072 differing
BF16 values, maximum .0009765625: strict bitwise comparison fails, unchanged
Body tolerance passes. `mix_fp32_c3_v4b_policy.json` records both separately.
All13 differences originate at position37/head6; same-input RMSNorm isolates
one BF16 query difference at element76. Ln/exp alone does not remove it.

Actual Qwen CTG/Solver ABI4 model execution is separate: prefill logits/token
pass but55/198 intermediate checks fail. Four natural outputs match eager,
final-logit NRMSE .012178 and7 KV slices fail; the model Gate is not passed.
New ABI4 prices must be remeasured: an attention owner's diagnostic service span
includes nested internal coordination, whose participant span must not be added
again. Old ABI3 profiles/prices are rejected by identity checks.11 CPU request
binding tests pass. Details/reproduction: `tools/probes/ascend950/AS2_mix_attention.md`.

## Pinned normalization and long-context diagnosis (verified-run)

`LlmVectorBody::RmsNorm` variants2/3 use only the installed CANN9.1.0
`norm_common` device reduction/Newton primitives, not an operator launcher.
The optional header hash is enforced by CMake; missing primitives are rejected
before ACL. Variant2 is bounded to H128/H1024. Variant3 pairs opposite half-rows
before the H1024 register reduction; model binding is H1024-only. Both retain
actual BF16 gamma multiplication and unchanged output cast points. Defaults
remain unchanged; no end-to-end performance improvement is yet admitted.

Independent captured Qwen operands expose two distinct rounding cases. H128
Newton removes the original QK difference. Full H1024 Newton removes prefill
drift:198 saved outputs and a four-token request are bitwise exact, with401
fast/diagnostic payload/control files matching in two fresh processes (C-2,
not a new C-3 claim). One AS-1 Encoder protection also passes bitwise:
four epochs,156tasks/epoch,8192 output bytes, original BF16 reference unchanged.

The prior past160/layer9 Norm1 failure has fused rstd .699561357498169 versus
CPU .6995614171028137. Variant3 repairs all1024 BF16 outputs and preserves the
independent64-row prefill case. Its98-output request is bitwise exact; the
preceding long candidate's first fork988/14 token failures remains preserved.
`as2_last_forward.py` compares independent hooks and native named/actual buffers;
captures cannot enter a formal performance process. B1 diagnostic histories are
bounded by capacity1088; B16 retains the short128-position history limit.

`as2_rms_reference --capture SOURCE --out CASE` prepares bounded actual Norm
operands; `--input CASE --out REFERENCE` observes independent fused Rstd/output.
`as2_vector 6 rows width CASE OUTPUT epsilon` records square-sum and Rstd in
`norm_statistics.bin` (64B/row, floats0/32). Model calls omit these stores.
Frozen tolerances and lowest-max token tie rule are unchanged.

The earlier cached drift at past1025/layer11 is isolated to FP32 sum order.
Observed Q/K/V, scores, max and exponentials match; the pinned register-fold
softmax sum repairs probabilities/PV/context at actual past1025 and181. Explicit
variant6 then passes Qwen0.6 B1/B16 natural64/1024 and teacher512 against
independent SDPA/fused Norm, bitwise for final logits/KV/tokens. B1 fast/diagnostic C-2 matches401 files; no new C-3/performance claim is implied.
Teacher uses Host token copies/checkpoints and is excluded from greedy timing.
H2048 probes use Llama epsilon1e-5/Qwen epsilon1e-6;33 rows (four-AIV tail)
and one-row Llama match fused Norm bitwise. Full/folded candidates pass; larger H
stays gated. A real Llama1B head64/GQA4 replay matches Q/K/V/scores/probabilities/PV
and context bitwise. That family and the probed H3072/GQA3 family have explicit
experimental bindings; other families stay gated. The old H2048 Qwen run failed; actual
eager tiling exposes K2048×3 Stream-K. Opt-in CTG partial/combine now makes full
prefill198/198,1024 greedy and teacher512 bitwise exact against SDPA/fused Norm.
Llama1B's earlier13-KV failure is repaired by actual CTG K2731/2731/2730
partial/combine tasks: full64/1024,teacher512,16-layer KV match SDPA bitwise;
five fresh reset processes pass281 files/request; fast/diagnostic217 files match.
B1 diagnostic cap4MiB captures FFN outside timing; B16 retains1MiB except logits.
Independent eager [FIA-v2](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md) (native GQA/BF16/static KV) executes in the private
process without TileMega code. It changes Qwen's long greedy path at output6;
977 tokens differ in historical diagnostics. The later user override waives
baseline numerical comparisons; this is no longer a baseline collection blocker.
Split3 request-reset C3 passes5/5 fresh processes; same-Plan fast/diagnostic373
files are bitwise equal. Llama3B H3072/GQA3 operand probes pass; its integrated
non-split candidate fails from output1. Prefill first drifts at`l1.norm1`
(8 elements); same-act down differs56. Actual2×K4096 CTG partial/combine then
passes1024 tokens/logits/KV and teacher512 bitwise. Five fresh reset processes
pass485 files/request; fast/diagnostic373 common files match. Diagnostic-only
allocations above1MiB are explicitly enumerated; tokens/logits/KV stay mandatory.
Positive measured stage prices select structural prefill and EFT decode;
both selected queues also pass1024/teacher512. Final eager/TPOT remains open.

H4096/fold2048 and head128/GQA4 observed Body probes pass. The old Llama8B
candidate fails first at packed QKV despite exact input Norm/Q. Opt-in separate
Q/K/V, seven-part KV/store combine and stage25 packing now pass complete4-token
intermediates,1024 outputs/logits/KV and teacher512 bitwise. Packing reads three
BF16 buffers and writes KV-major `[row,group,(Q+K+V)]`; one AIV owns each row.
Native packing is bounded to8 groups/head128/GQA4, reusing106496B vector UB and
existing DMA/publication protocol. Seven-part combine reads FP32 slices in
increasing K order and writes BF16 once; no residual operation is added.
Invalid descriptors/other seven-part shapes are rejected. GQA3/4 CPU exact-read
contracts and32-layer structural import pass; CANN binary/source identities
and4/4 actual runtime launches are retained. Reset C3 passes5/5 fresh processes
with681 files/request; fast/diagnostic489 common captures match,192 large
diagnostic extras are explicit. Positive empirical prices select two actual EFT
queues; selected plans also pass1024 greedy and teacher512 bitwise.

The observed Llama1B decode `[16,1,8192]` down projection matches32768 BF16
outputs after three disjoint K2731/2731/2730 partials and CPU ordered sum.
This component does not verify device combine or a model. An explicit opt-in
B16 decode decomposition now regenerates CTG; Host/Codegen/Device admit rows16
only for N2048/K8192/split3. Other B16 decompositions fail closed. Host tests
check the valid contract and reject K6144/K4096/rows1024 before ACL.
The initial no-split prefill/split3 decode request fails10/64 tokens, including
B16 row13 at the first output; that failure is retained. Selected first-two-layer
observations isolate the earliest drift to down, despite exact Norm/context.
The installed Blaze scheduler and actual M1024 tiling explain full-K for the
first28 MN tiles and seven K1171 slices only for the final4 N-tail tiles.
A uniform-seven replay fails13/32768 values; the guarded local replay is exact.
The opt-in local MIX Body keeps one original CTG output tile and one external
event. It computes seven disjoint FP32 intervals only at columns1792:2048, then
AIV0 combines, rounds BF16 and adds the residual. Other columns use full-K.
It has57344B private GM scratch/group/stage, visible in Plan/contracts; existing
ready/done ack ends its lifetime before reuse. No cross-Task split is claimed.
Actual complete B16 four-output and1024-output logits/tokens/KV now pass
bitwise. Teacher512 passes; structural-queue request-reset C3 is5/5 fresh
processes and C-2 compares120 common files. Positive regular/tail service prices
select distinct actual queues; both selected phases pass1024 greedy and
teacher512. Priced-queue five-process admission and performance remain open;
this implementation is not enabled by default.


## Current reconciliation, 2026-10-26

The complete chain and A01–A10 status are consolidated in `AS2_pipeline.md`.
Earlier dated experiments above retain their original scope; they are not
current all-model pass claims. Seven dense long numerical cells pass against
SDPA/fused Norm; formal final correctness/performance remains0/10.

The no-lock wrapper now distinguishes unavailable telemetry from known low HBM,
keeps unknown external load as null, and records the current selected PCI ID.
A cached ACL mapping is labelled verified only when its PCI identity agrees.
Two CPU tests verify prelaunch failure classification and owned-descendant
timeout without touching the unrelated sentinel. An actual DCMI-8005 attempt
launched no child; subsequent healthy ACL inventory on each card launched no
compute kernel. Current physical0 is D7:00.0 and physical1 is55:00.0, unlike the
older enumeration; archived old/current mappings retain that distinction.


Experimental FIFO observation reuse is exposed only by explicit `-fifo` priced
compile modes. CPU proof and real Qwen plans retain publications, payload
acquires, MIX acknowledgements, queues, owner identity and epoch encoding.
`verified-run` Qwen0.6 B1/G8: original/elided same-binary short captures match457
files/request; fast/diagnostic matches401 common files. Five fresh fast processes
with10 reset requests pass; all401 files/request are bitwise stable. Full1024
greedy tokens/logits/56 checked KV slices and teacher512 match independent
SDPA/fused Norm exactly. See `runs/as2/search/fifo_device_validation_v1.json`.
No TPOT admission or default change is claimed; other model wait modes are untested.

Management follow-up: exact `/usr/local/sbin/npu-smi info` succeeds in project
and ordinary shells. Prior15s timeout records are retained and do not establish
compute/device failure. Sourcing CANN changes common-library resolution, so the
query child now prefers installed driver libraries; compute children are unchanged.
This does not eliminate every intermittent timeout or prove its cause. Queries
have a configurable30s budget (max60s), bounded1s reaping and own PID/birth
identity;7/7 CPU safety tests pass, including unrelated-process preservation.
Native CPU tools still link only CPU dependencies; the explicit unchanged Probe
archive compiles, with no additional runtime Probe claim.

C3 dedup now permits an explicit minimum payload size, never below64KiB. Only
completed manifest-registered numeric payloads with exact SHA/stamp/ownership
checks are linked.2286 paths are preserved,763637760 allocated bytes reclaimed;
post-dedup C3 again passes5/5. The brief96GiB soft-budget warning is retained,
not hidden by changing accounting or deleting failed evidence.

## Qwen1.7 B16 observed domain and sourced queues

**verified-run:** actual M1024/N2048/K6144 eager tiling4162 and installed
StreamK scheduler produce full-K regular columns and a1792:2048 N-tail,
seven local slices878×6+876. The same logical MIX owner publishes once;
AIV0 ordered FP32 partial sum → BF16 → BF16 residual is explicit.
Decode uses actual CTG3×K2048 partial/combine domains, not a hidden host loop.
Structural and EFT-selected queues both pass64/1024 and teacher512 bitwise.
Structural fast/diagnostic matches204 files; C3 is5/5 processes/10 reset
requests. This is not five fresh selected-EFT processes or formal TPOT.
`search/qwen17_b16_*prices*` keeps real regular/tail service measurements and
control/GM provenance; decode profile reaches past1086. Both phases compare
two legal queue/placement plans and select heterogeneous_rank_eft. Outer
Body/geometry/G search is incomplete. New v17 binary also protects Llama1B B16.
Additional independent baseline status is in `AS2_baselines.md`.

## B16 active decode diagnostics and current device blocker

**verified-code/run (CPU):** phase binding emits optional `active_bytes.bin`,
hashed in the binding receipt. ACL Host preflight checks its count and bounds
against shared allocations. Diagnostic decode can copy a phase-sized prefix
at most1MiB when a larger shared prefill allocation otherwise hides it.
The fast path, queues, Body arithmetic and device synchronization are unchanged.
Legacy bindings without this optional table retain their previous behavior.
C-2 permits these extra files only with the exact bound extent; mandatory
fast outputs still compare exactly.17 CPU request/diagnostic tests pass.

The explicit `as2_last_forward --matching-prefix-rows` option reports which
independent batch rows have identical prior greedy feedback. It is diagnostic
only, lists excluded rows, rejects an empty comparable set and cannot admit
full-batch correctness. Ordinary comparison still rejects divergent feedback.

**verified-run:** Llama3B B16 full-K bootstrap runs4 complete model forwards,
one launch each, but fails5/64 tokens (first row3/output1), leaving the long
cell incomplete. Retained image `native_b16_fullk_v18` and the failure remain.
The diagnostic-only rebuild `native_b16_active_capture_v19` compiles and loads
the new Host tables, but cannot initialize physical0/logical0: ACL507033,
TsdOpen/CreateSession failure. No kernel is launched in that attempt.
Evidence: `jobs/llama3b_b16_active_request4_v19/{stderr.log,after.json,cann/}`.
The query itself succeeds and reports physical1 Critical with HBM0/0.
New device submissions stop; no driver reset or unrelated process action.
## Llama8B B16 selective three-part implementation, 2026-10-27

`verified-code/run`: opt-in GEMM variant3 keeps16×128×128 geometry and the
original CTG output-tile domain. For M1024/N4096, only N3584:4096 uses ordered
FP32 local K slices. Q/O use1366/1366/1364; down uses4779/4779/4778. Ordinary
N0:3584 retains full-K accumulation. This is one indivisible MIX task, not a
cross-AIC split-K claim. Config: `as2_llama8b_b16_prefill_tail3.json`.

One AIC owner writes disjoint FP32 partials into24576B/group/stage GM scratch;
AIV0 combines in increasing K order, BF16 RINT, then the residual Body preserves
its separate BF16 Add. Q store epilogue4 has no invented residual. Existing
ready/done/reverse-ack ends scratch lifetime before the next owned tile; only
the AIC owner publishes the logical completion. AIV1 remains an independent
queue lane. No new hardware flags, direct UB route, cache shortcut or global
lock is introduced. Original route C3 evidence is retained; no new five-process
configuration claim is made from the numerical run alone.

Host and device guards reject missing capabilities, wrong geometry, shape,
workspace and epilogue. Request source composition unions registered phase
capabilities only when runtime text and geometry agree; fallback macro defaults
cannot override the real generated geometry. Task/FIFO/dependency tables remain
the actual solver output. Full32-layer B16 four-output and1024-output requests
pass tokens, full final logits and64 last-position KV tensors bitwise against
independent SDPA/fused Norm. The four-output saved intermediates also match.
Selected priced queues also pass full1024 bitwise. All four benchmark request
token hashes match that numerical pass; exploratory timing is recorded in
`AS2_focus_comparison.md`. Formal teacher512/C2/C3/statistical Gates remain open.

## Parked wider QK experiment, 2026-10-28

`verified-code`: optional Attention variant7 uses CATLASS32x256x128 instead
of32x64x128 for QK; L0K32, FP32 scaling, fold softmax, PV, task ownership and
GM ready/done/reverse-ack remain unchanged. Its identity enters materialization,
resource/price contracts and request capability composition. The explicit
`cube-fp32-wide-qk` candidate requires `TILEMEGA_LLM_WIDE_QK=1`; ordinary
images exclude it and reject its Stage. No default or target capability enables it.

`verified-run`: BiSheng builds; the initial pack accidentally substituted CPU
RoPE coefficients for the canonical verified NPU table. Four-output tokens
match, but18 KV slices fail the frozen policy (164 elements, max_abs0.0546875).
This confounded comparison does **not** establish a QK Body arithmetic failure.
The1024-output diagnostic times out600s with other load observed. Runtime
logs show stream progress269→523→762, so this is not established deadlock.
No numerical pass, timing admission or portable performance claim is made.
Own PID/context exits; subsequent two-card queries show HealthOK.
Evidence: `runs/as2/failures/opt_qk256_v1.json`. The candidate stays disabled;
failure source/binary/logs and unchanged tolerance are retained. The packing
reuse guard now requires the same verified RoPE table, before writing inputs;
all required packed input hashes are checked on the corrected experiment.

## Valid-key QK candidate, 2026-10-28

`verified-code`: explicit `cube-fp32-valid-qk` / variant8 preserves QK32x64x128,
L0K32, physical score stride1088, original last physical tile, scaling,
Newton QK norm, capacity-fold softmax and full-capacity PV. Only entirely
unconsumed QK64 tiles are skipped. `ServingSoftmax` copies valid scores only
and builds the masked tail locally; omitted scores never flow into computation.
Prepare still initializes the original K/V tail. GM visibility, ownership,
participants, completion counts and buffer lifetime are unchanged.

The64/1024 request reduces QK64 calls per logical Attention task from17391
to9711 (44.16% fewer; a code-derived count, **not** a latency claim). No host
operator, separate attention launch, shortened context or BF16 change is used.
Implementation identity and actual logical GM traffic enter the native Plan
and measured-price path. Outer Body selection is still limited/manual.

`verified-run`:229 packed input hashes match the canonical inputs. Qwen0.6 B16
four-output and full1024 diagnostic requests pass bitwise tokens, final logits
and all56 last-position KV tensors against independent SDPA/fused Norm.
Fresh G8 control calibration passes5/5 processes on physical1; no G14 admission.
Evidence: `correctness/opt_validqk_b16_profile1024_numeric_v2.json`,
`search/opt_validqk_g8_costs_v3.json`. Real sourced costs regenerate both phases,
compare structural/EFT queues and preserve their selected order; repriced
numerical admission and idle fast TPOT remain pending at this code checkpoint.

## Derived FP32 KV state, 2026-10-28 continuation

`verified-code`: explicit variant9 (`cube-fp32-cached-kv`) uses two additional
request-lived FP32 planes/layer, derived from the authoritative rounded BF16
K/V. Prefill owns disjoint16-position writes while retaining its private
current-task preparation; decode appends one row and reads the derived prefix.
Request reset clears both representations. Existing three-phase owner/AIV0
GM ready/done/reverse-ack and all visibility fences remain unchanged. Metadata
and paired-phase validation reject aliasing, missing reset, mixed variants,
wrong extent/dtype and standalone execution before launch. Initially restricted
to Qwen0.6 B16 initially. Qwen1.7 B16 now independently passes actual priced
full1024/teacher512 bitwise, C2 (148 files) and C3 (5/5 processes,10 requests).
Both remain non-default: Qwen0.6 final timing is inconclusive under its frozen
noise guard; Qwen1.7 has pilot data only. Five-dense B16 metadata shapes now
pass CPU contracts; three new Llama candidates still require device admission.

`verified-run`: repriced full64/1024 logits/tokens/56 last KV outputs and
teacher512 are bitwise equal to the independent reference. Fast/diagnostic
full1024 C2 passes92 common files. C3 passes5/5 fresh processes/ten reset
requests,67 outputs crossing past127/128/129. One forward remains one launch.
Arena grows4565844480→8558850560B; cost/Plan expose this storage and avoided
prefix conversion. The bounded native outer search actually compares Body8/9
and selects9 using positive measured prices; TPOT/default admission is separate.
Exact evidence and source/image identities: `AS2_TPOT_optimization.md` and
`runs/as2/search/opt_attention_selected_abi_equivalence_v1.json`.
# Strided SwiGLU decode candidate, 2026-10-29

`verified-code/run`: `SwiGluTile.h` loads an entire16×128 interleaved gate/up
tile with strided DMA, separates U16 groups in UB, and preserves FP32 opmath,
BF16 SiLU rounding and BF16 product rounding. `LlmVectorBody` borrows existing
disjoint arrays; UB remains106496B. MTE2→V and V→MTE3 completion remain explicit.
The existing GM ready/done/reverse-ack protocol and single AIC event owner are
unchanged. No direct UB handoff or second cooperating AIV is implied.

The explicit variant4 maps to geometry1; absent feature, unknown shape/count,
partial-K or wrong epilogue fails before ACL. Qwen1.7 B16 native decode binds
all28 gate/up stages; down split3 and CTG/ISL semantics remain unchanged.
Its actual priced full1024, teacher512 and full fast/diagnostic run pass
bitwise. Five new processes complete10 reset requests with148 equal payloads
per request. Image `ea9aaefb42caeb71ee2b6f2aee91e5b493e848c460f8ca9add1bd28d19967275`.
These proofs do not admit other models or a performance default. Pilot and
frozen paired results are indexed in `AS2_TPOT_optimization.md`.

`verified-code/CPU`: additional opt-in B16 decode families are Qwen0.6
M16/N6144/K1024 (28 layers), Llama1B N16384/K2048 (16), Llama3B
N16384/K3072 (28), Llama8B N28672/K4096 (32). Each native reimport preserves
its original CTG/ISL and down/KV decomposition. Host and device reject unknown
families/partial-K; Host checks the complete tile count. These four candidates
share a single newly compiled CANN implementation catalogue,
`21e6b516c8ea43127cc9526dc58bb50836aa27b1a67cca3e09b28423fa531f3e`.
Their numerical/pricing/TPOT admission is pending. Qwen1.7's frozen study
continues using its original image and immutable header snapshots.
