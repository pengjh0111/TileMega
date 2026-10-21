# AS-2 native runtime/body evidence

All device evidence here: physical 1→logical 0, Ascend950PR_9579, CANN 9.1.0,
BiSheng clang15.0.5, pinned CATLASS 2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52.
No device lock/exclusive setting, reset, other-process operation or CUDA test.
These are bounded probes; no TileMega LLM kernel forward has run.

## Runtime and resource scope

- `tools/probes/ascend950/as2_timer.cpp` compares warmed 4M/16M system-counter
  loops with ACL events: 4.00795/16.0083 ms, differential12.0003 ms. The initial
  1M-tick first-launch experiment failed its admission limit and is retained.
  Frequency 1 GHz is verified here, not inferred from the AI Core clock.
- `Target/Ascend/ProgressPrimitives.h::WaitProgress` uses a supplied counter
  deadline and abort. AS-2 progress probes use 2e9 ticks per wait. This is not
  the legacy 100000-poll limit re-labelled as seconds.
- `as2_progress.cpp`: G2 **5/5**, G4 **5/5 fresh processes**, each16 epochs and
  four directions. AIC producers execute CATLASS identity MMAD/FIX; both AIV
  lanes produce MTE2/MTE3 payloads. Consumers pre-read cached old data, publish
  preload, wait for payload completion/publish, invalidate/acquire and check
  exact values. Ready, preload and reverse acknowledgements have single owners.
- The old DSB and entire-DCache acquire remain the correctness baseline.
  This is not a minimal-cache-maintenance result. New G8/14/28, mode4, BF16
  direct, UB→L1 and MIX logical-task owner/participant completion are unverified.
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
preserves BF16 product/add cast points. Model-specific coefficient generation
and integration of scaled RoPE are not yet implemented by this body path.

AttentionChunk transfers up to32 contiguous K/V rows in each DMA. Broadcast and
AR/RA vector reductions compute QK and weighted PV; it emits FP32(m,l,o).
AttentionMerge rescales all unnormalized contributions before division. K/V
head mapping is integer division; ratio3 is exercised. This all-valid-key body
does not yet implement the serving stage's KV append, QK norm/RoPE, causal
prefill query domain, all-masked/empty domain or its single-LSE ABI.

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
