# Qwen3-1.7B SM90 mechanism-removal experiment

This experiment starts from `sm90-decode-ae-r3` commit
`4a84be39988967cb0a3c8f30c532bd25a6587960`. Its only baseline is
`configs/serving/sm90_ae/qwen3_1_7b_b1/D.json`, with historical TPOT
**1.6754311115689438 ms**. D is never compiled or timed by this recipe.
The task-body implementation is the historical D version, not current mainline.

| Arm | attention.chunks | group_wait | emb | Execution |
|---|---:|---|---|---|
| A | 1 | false | stage | Megakernel |
| B | 16 | false | stage | Megakernel |
| C | 1 | true | la | Megakernel |
| D | 16 | true | la | Historical megakernel result |
| E | 1 | false | stage | Typed-stage kernels and CUDA Graph |

All other coordinates are exactly D's. A and E files must be byte-identical.
Operator-local denotes the prescribed S1 baseline, not a measured or searched
local optimum. Only two planner mechanisms are removed/restored: KV-group
early release after QKV and fused Head-to-embedding handoff. No independent
ownership/order optimization is claimed.

Verified static implementation inspection: `ImportSm90Dense` reconstructs
attention task ranges from chunks; fixed `MaterializeSm90Dense` retains
resource and schema checks. Codegen emits `Attention<...,Chunks,...>` and
QKV `Placement` with GroupTiles 0 or 16. The original attention body writes
context directly for S1, and partial results plus last-arriver merge for S16.
`Head<...,Embed=false>` retains folded argmax while disabling embedding.
`Program<...,Embed=false>` executes embedding every step. For E, the inherited
preserve-body sequence branch emits the identical Program/body prefix,
captures those stages in order, and replays the graph per decode step.
`kPostHead=false` prevents an unnecessary argmax stage. No backend, codegen,
or runtime adaptation is needed beyond selecting these existing paths.

The requested `scripts/sm90_decode_ae.py` does not exist at the specified base.
Its relevant existing components are `sm90_decode_ae_collect.py` and
`sm90_decode_ae_measure.py`; both are reused unchanged. New experiment helpers
generate the fixed matrix, compile/audit it and publish the required results.
The previous experiment's local cost-model selector and report are not used.

Run from this branch on the original host, with matching CUTLASS/barvinok
gitlinks available (the collection host uses symlinks to those exact commits):

```bash
cmake -S . -B runs/sm90_mechanism_ae/build -G Ninja \
  -DMLIR_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/mlir \
  -DLLVM_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/llvm \
  -DTILEMEGA_ISL_BUILD_DIR=/root/TileMega/build-isl \
  -DTILEMEGA_POLYLIB_BUILD_DIR=/root/TileMega/build-polylib \
  -DTILEMEGA_BARVINOK_BUILD_DIR=/root/TileMega/build-barvinok \
  -DTILEMEGA_TARGET_ARCH=sm_90 \
  -DTILEMEGA_BUILD_TESTS=OFF -DTILEMEGA_BUILD_VERIFY=OFF
cmake --build runs/sm90_mechanism_ae/build --target tilemega -j 12
python3 scripts/sm90_decode_mechanism_ae.py prepare --run runs/sm90_mechanism_ae
python3 scripts/sm90_decode_mechanism_ae.py build --run runs/sm90_mechanism_ae \
  --compiler runs/sm90_mechanism_ae/build/tools/tilemega
python3 scripts/sm90_decode_mechanism_ae.py audit --run runs/sm90_mechanism_ae \
  --prefill /root/TileMega/runs/sm90r2/build/sm90r3-final-prefill/qwen3_1_7b-B1/prefill.so
python3 scripts/sm90_decode_ae_collect.py \
  --spec runs/sm90_mechanism_ae/spec.json \
  --out runs/sm90_mechanism_ae/timing --gpu 0
```

The compile helper issues, for A/B/C/E only:

```text
tilemega compile decode.export.json ARM.so --tmexec-config ARM.json \
  --serving decode --emit serving --batch 1 \
  --runtime-target configs/targets/sm_90_h800.json --nvcc-split-compile 4
```

Only E replaces `--emit serving` with `--emit kernel-sequence`. No solve or
candidate sweep is called. The audit checks fixed coordinates, task-body
hashes in each artifact, attention/Head/Program specializations, QKV release
parameters, A/E source equivalence and successful compilation before freezing
the timing spec. For a new collection, use a new run directory; the build
helper refuses to overwrite retained binaries.

The existing guard pins one physical H800 for all 12 accepted processes:
`A B C E / B C E A / C E A B`. It retains its NUMA lock, idle admission,
interference checks and code-75 retries. Its namespace-limited PID acceptance
is disclosed in the process records. No clocks or power settings are changed.
The existing request-wall driver uses historical prefill, name-seeded random
checkpoint seed0, L1 decode-loop, one warmup and three measured requests at
each N1 and N1024. Per-process TPOT is the difference between the two median
request times divided by 1023; each arm is the median of three fresh processes.
There are no extra correctness, sanitizer, stress, profiling or calibration jobs.

Report and plots (matplotlib is isolated from the timing Python environment):

```bash
/workspace/llm-stack/envs/vllm/bin/python -m pip install \
  --target runs/sm90_mechanism_ae/plot-packages matplotlib
PYTHONPATH="$PWD/runs/sm90_mechanism_ae/plot-packages" \
  python3 scripts/sm90_decode_mechanism_ae_report.py \
  --run runs/sm90_mechanism_ae \
  --historical-result /root/TileMega-x1/runs/sm90_decode_ae/D.result.json
```

Published CSV/Markdown/plots/provenance use the stem
`docs/results/SM90_decode_mechanism_AE_qwen3_1_7b_b1`.
Raw generated CUDA, identities, binaries, all request records, process IDs,
failed attempts and telemetry remain under the ignored run directory.
