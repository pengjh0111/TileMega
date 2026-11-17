# Explicit-stage realization: Qwen3-1.7B, SM90

Base: `7496665a83f857022f934c6474475be5f6862586`, published
`sm90-decode-mechanism-ae-v1`. New branch: `sm90-explicit-stage-ae-v1`.
Only BS1/TP1/BF16/ISL64/OSL1024 on H800 is in scope.

| Arm | Attention | Realization | Execution | Measurement |
|---|---|---|---|---|
| A2 | S1 | P0 explicit stages | Megakernel | 3 new processes |
| B2 | S16 | P0 explicit stages | Megakernel | 3 new processes |
| C | S1 | P1 selected | Megakernel | Reuse 1.7686245229777098 ms |
| D | S16 | P1 selected | Megakernel | Reuse 1.6754311115689438 ms |
| E2 | S1, identical to A2 | P0, identical to A2 | CUDA Graph | 3 new processes |

The coordinates retain all previous GEMM geometries and packing. P0 is selected
by the explicit emit mode and manifest
`realization=explicit_stages_qwen_b1_v1`, not by changing the meaning of P1's
schema defaults. In particular, the coordinate `attention.merge=la` retains
the original partial/LSE format, while the explicit realization moves the
merge to a separate stage. The same realization replaces folded final argmax.
`native_contract.py` validates the generated explicit signatures and stage list
and reports the actual explicit operations in the artifact identity.

Static implementation locations (no numerical verification is claimed):

| Operation | Location and decision |
|---|---|
| RMS statistics | `include/tilemega/Codegen/tasks/Sm90DenseExplicit.cuh`, `ExplicitRms`: original per-lane m16n8k16 accumulation and WK reduction order; one FP32 inverse. |
| GEMM normalization | `include/tilemega/Backend/Sm90DenseGemm.h`, `ExternalNorm`: skip fragment squares/reduction, consume explicit inverse; `Sm90DenseEpilogue.h` keeps the original output-scaling/BF16 boundary. InputNorm remains true only to retain scaling. Gamma is still folded into weights, not applied twice. |
| Residual / SwiGLU | `ExplicitOutput`: kStore GEMMs write one reused BF16 workspace, followed by independent residual or the original ServingEpilogueValue activation. Original BF16(accumulator), BF16(SiLU), and BF16(product/sum) boundaries are preserved. |
| Q/K norm, RoPE, append | `ExplicitAttentionPrepare` invokes the original attention Prepare once per KV group and writes prepared Q in-place. `Sm90DenseAttentionBody.cuh::Prepare<ExplicitStages>` only loads prepared Q/new KV. |
| Attention compute | Original Produce, Block, online softmax and Finish arithmetic remain. Original range formula covers history plus exactly one owner of the current row. |
| S16 merge | `ExplicitAttentionMerge` calls the original Merge on the unchanged partial/LSE layout; `ExplicitStages` suppresses the original last-arriver merge. S1 has no merge stage. |
| Head | `ExplicitHeadPartialFinish` only publishes per-CTA value/index. `ExplicitArgmax` performs the final reduction and tie-breaking. Independent embedding runs at the beginning of every decode step. |
| Native stages | `lib/Codegen/Codegen.cpp`, explicit branch: one shared `ExplicitLayer` tuple, 12 stages for S16 / 11 for S1, plus embedding and three terminal stages. |
| Megakernel | `ExplicitStageProgram`: one aggregate counter per actual stage, immutable operand-bank publication by the predecessor, shared arena reuse after stage completion, original grid-stride body task mapping. It does not use P1's five-stage QKV/attention indices or parity assumptions. |
| Graph | `Sm90DenseExplicitSequence.cuh`: expands the same tuple through existing `KernelSequence` stage kernels; includes all explicit operations. Capture/upload occurs in engine setup before measured requests. |

S16: 340 actual native stages per decode step. S1: 312. The original semantic
CTG contains 143 stages and is deliberately not rebuilt into a generic
explicit-stage IR. Its count printed by the compiler is not the executed
stage count. The generated NativeProgram and the audit receipt are authoritative.
The graph also retains the existing one-per-step state-advance kernel and
one-per-request sequence-begin kernel; these update step/iteration state and
are part of execution organization, not additional model computation.

All stages launch the original 132-CTA grid with 160 threads, residency1,
cluster1 and 217664 shared bytes. GEMM task sizes/slots/MMA/register stages
are unchanged. Explicit vector output tasks cover 256 output elements each;
RMS has one row task and attention prepare/merge have eight KV-group tasks.
These domains are fixed before timing, not selected from measured speeds.
No artificial work, sleeps, redundant computation or unrelated barriers are added.

P1 keeps its original emitter/runtime and default template branches. The
historical C/D binaries and results are reused without rebuilding or timing them.

Build from this branch on the original host (same dependency gitlinks):

```bash
cmake -S . -B runs/sm90_explicit_ae/build -G Ninja \
  -DMLIR_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/mlir \
  -DLLVM_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/llvm \
  -DTILEMEGA_ISL_BUILD_DIR=/root/TileMega/build-isl \
  -DTILEMEGA_POLYLIB_BUILD_DIR=/root/TileMega/build-polylib \
  -DTILEMEGA_BARVINOK_BUILD_DIR=/root/TileMega/build-barvinok \
  -DTILEMEGA_TARGET_ARCH=sm_90 -DTILEMEGA_BUILD_TESTS=OFF -DTILEMEGA_BUILD_VERIFY=OFF
cmake --build runs/sm90_explicit_ae/build --target tilemega -j 12
python3 - <<'PY'
from pathlib import Path
import lzma
p=Path('runs/sm90_explicit_ae');p.mkdir(parents=True,exist_ok=True)
(p/'decode.export.json').write_bytes(lzma.decompress(Path(
    'configs/serving/sm90d1/inputs/qwen3_1_7b/decode.export.json.xz').read_bytes()))
PY
CUDA_VISIBLE_DEVICES='' runs/sm90_explicit_ae/build/tools/tilemega compile \
  runs/sm90_explicit_ae/decode.export.json runs/sm90_explicit_ae/build-v1/B2.so \
  --tmexec-config configs/serving/sm90_explicit_ae/qwen3_1_7b_b1/B2.json \
  --serving decode --emit explicit-serving --batch 1 \
  --runtime-target configs/targets/sm_90_h800.json --nvcc-split-compile 4
```

Inspect B2's emitted stage tuple, ExternalNorm/ExplicitStages signatures,
unmodified packing recipes and compile identity first. The retained original
audit is `runs/sm90_explicit_ae/B2-first-audit.json`. Then compile A2 with the
same command, replacing B2 with A2. Compile E2 similarly but use
`--emit explicit-kernel-sequence`. No solve or candidate sweep is involved.
Use a new directory for fresh builds; the compiler refuses to overwrite evidence.

To generate that first audit after B2 compilation and before A2/E2 compilation:
`python3 scripts/sm90_explicit_ae_freeze.py --run runs/sm90_explicit_ae --b2-first`.
It compares original buffer/packing recipes against the retained prior B source
on this host, without rebuilding or timing B/C/D.

```bash
python3 scripts/sm90_explicit_ae_freeze.py --run runs/sm90_explicit_ae
python3 scripts/sm90_explicit_ae_collect.py \
  --spec runs/sm90_explicit_ae/spec.json --out runs/sm90_explicit_ae/timing --gpu 0
PYTHONPATH=/root/TileMega-mechanism-ae/runs/sm90_mechanism_ae/plot-packages \
  python3 scripts/sm90_explicit_ae_report.py --run runs/sm90_explicit_ae
```

The collector preserves the previous guard/checkpoint/prompt/prefill and
request-wall protocol. Order: `A2 B2 E2 / B2 E2 A2 / E2 A2 B2`.
Each process has one warmup and three requests at both N1 and N1024;
TPOT is the difference between their median request walls divided by 1023.
The arm statistic is the median of three independent processes. All accepted
values are retained regardless of ordering. Existing PID-namespace guard
limitations are disclosed in results. No clocks or power settings are changed.

Only static implementation checks, compilation/occupancy checks and nine
accepted performance processes (with retained guard retries) are performed. No C1/C2, HF, token/logit comparison,
sanitizer, sync stress, profiling, other workload, calibration or sweep is run.
All C/D comparisons are explicitly cross-session. E2/A2 is execution
organization with matched computation, not a pure launch-latency measurement.
P0/P1 is execution realization, not owner/order search on a fixed task graph.

New results use `docs/results/SM90_explicit_stage_AE_qwen3_1_7b_b1*`.
Earlier CSVs are preserved. Full raw requests, identities, generated CUDA,
compile logs, guard attempts and binaries remain in the ignored run directory.
