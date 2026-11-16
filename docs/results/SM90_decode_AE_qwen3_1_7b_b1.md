# Qwen3-1.7B SM90 native decode A-E ablation

Verified scope: BS=1, BF16, ISL/OSL=64/1024. D reuses the SM90R3 main `measured.tpot_ms=1.6754311115689438`; the four other arms each have three accepted fresh timing processes on one physical H800. The D reference is held fixed; no solve or separate D-arm job was submitted.

The frozen D artifact was built from `4b9d1362e648e2ee2cc7f96da9422f97a6d36c45`. All 606 recorded source files and the target JSON matched when recovered. The experiment uses branch `sm90-decode-ae-r3`, retaining D task-body headers and adding only a sequence execution adapter. The preliminary merged-version builds were not timed and are excluded.

The fixed pipeline is export bridge -> BuildModelPlan -> imported semantics -> MaterializeSm90Dense with pinned coordinates and task ranges -> typed native Program -> nvcc/identity -> ServingEngine L1 decode-loop -> request-wall TPOT. --tmexec-config bypasses solve.

Verified CPU selection: independently minimize existing Gemm(pre=false) and complete Attention costs over deduplicated FamilyCandidates TN and AttentionCandidates chunks plus D values, filtering with both policies' Legal checks and available calibrated terms. All other D parameters stay fixed; slot_bytes follows 2*TN*TK_D. No Structure/derive, whole-program search or GPU candidate sweep is used for L. Exact ties prefer D. The historical calibration reports zero stale body headers.

**L equals D:** TN=(32,16,48,16,128), chunks=16. A=B configurations, and C has the same configuration and byte-identical generated CUDA source as historical D. Thus this case has no decomposition treatment difference; measured A/B or C/D deviations cannot establish a decomposition gain.

Basic static retains cyclic/grid-stride ownership, typed stage order, body-internal LA reductions, fusion and embedding handoff. Its only policy change is group_wait=false, replacing QKV-to-attention group release with the aggregate grid boundary. Ownership/order are not independent optimizations in this experiment.

E retains the D tiles, chunks, LA attention merge, folded argmax and head-to-embedding body. It replays initial/steady CUDA graphs over ordinary typed-stage kernels with NoTail and streamed=false. Only absolute step 0 runs the embedding prologue; steady steps start with QKV. Final is conditional on kPostHead (false in frozen R3), so no ArgmaxStage is forced. Kernel completion orders stages; body-internal synchronization is retained. E/B preserve the same stage computations and aggregate boundaries, so TE/TB is also reported.

Timing uses the historical prefill binary and checkpoint algorithm/seed 0, L1 decode-loop execution, one warmup plus three timed requests at each N=1 and N=1024. Per-process TPOT=(median wall N1024 - median wall N1)/1023; the arm statistic is the median of three fresh processes. Order is A B C E / B C E A / C E A B. Existing GPU guards reject interference and retain retry logs. The current recorded idle-power guard is reused without changing GPU clocks or collecting new calibration. No C1/C2, token/logit comparison, HF, sanitizer, synchronization stress or profiling was run. The common driver checks output shape only.

| Arm | Decomposition | Policy | Execution | TPOT ms | Three process TPOTs ms | Range % | New/reused |
|---|---|---|---|---:|---|---:|---|
| A | operator-local L (=D) | basic static: aggregate grid barriers | megakernel | 1.699087408 | [1.6954664916913835, 1.699087407616846, 1.6993427634446197] | 0.228138 | new |
| B | TileMega-selected D | basic static: aggregate grid barriers | megakernel | 1.698308488 | [1.698080157380431, 1.6994076911102842, 1.6983084877767327] | 0.078168 | new |
| C | operator-local L (=D) | TileMega: QKV group release | megakernel | 1.674134351 | [1.6741343509292737, 1.6774068895375986, 1.6737898220887355] | 0.216056 | new |
| D | TileMega-selected D | TileMega: QKV group release | megakernel | 1.675431112 | [1.6761825522152982, 1.6701529110906572, 1.6754311115689438] | 0.359886 | reused |
| E | TileMega-selected D | kernel completion; D intra-stage bodies | stage-granular CUDA Graph replay | 2.096251860 | [2.0962518602152733, 2.105456420331046, 2.0947963284507356] | 0.508531 | new |

| Ratio | Value | Interpretation |
|---|---:|---|
| TA/TB | 1.000458644 | decomposition; L=D, so no decomposition treatment difference |
| TC/TD | 0.999226014 | decomposition; L=D, with historical D denominator |
| TA/TC | 1.014905050 | policy: QKV-to-attention release only |
| TB/TD | 1.013654621 | policy with historical D denominator |
| TA/TD | 1.014119528 | joint comparison with historical D denominator |
| TE/TD | 1.251171621 | overall execution organization with historical D denominator |
| TE/TB | 1.234317484 | same stage computations and aggregate inter-stage boundaries |

D denominators refer to another timing session. Matching source and task bodies do not remove cross-session variability; per-process values, clock observations and full precision ratios are retained in the companion CSVs. Results are performance measurements, with correctness explicitly outside this collection. Two A driver failures were excluded: one guard API mismatch before timing and one missing metadata field after timing. Both attempts and their raw records are retained; exclusion was independent of observed performance. No separate D-arm job was submitted; the prescribed C arm has the same configuration as D because L=D. Raw provenance: `/root/TileMega-x1/runs/sm90_decode_ae`.

Reproduction commands are in `configs/serving/sm90_ae/README.md` on `sm90-decode-ae-r3`. Source recovery and build/identity receipts retain the complete original and failed-attempt evidence under runs/.
