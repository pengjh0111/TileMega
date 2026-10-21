# AS-2 native runtime/body evidence

All device evidence here: physical 1→logical 0, Ascend950PR_9579, CANN 9.1.0,
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

Verified-run: full28-layer Llama3B GQA3 request executes four forwards with
6,735,441,920B shared arena (>4GiB addresses), valid task/launch receipts and
exact greedy tokens. Against independent SDPA/fused eager, logits NRMSE0.036681
with11020 failures and49 failing final KV slices. Correctness remains fail.

## ABI4 cooperative FP32 attention (experimental, verified-run)

`LlmCubeBody` borrows one CATLASS resource bank across BF16 tiles and full-FP32
QK/PV, draining each MMAD before reuse. G8/head128/GQA2 is the current legal
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
