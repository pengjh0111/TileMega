# Fixed SM90R3 Qwen3-1.7B A-E decode ablation

The authoritative experiment source is branch `sm90-decode-ae-r3`, based on
`4b9d1362e648e2ee2cc7f96da9422f97a6d36c45`. This is the source version of D's
historical 1.6754311115689438 ms result. Recovery verified all 606 recorded
source files, the target JSON and CUTLASS/barvinok commits against D's artifact.
The current merged `tilemega` branch is a different task-body version and is
not used for these measurements.

`qwen3_1_7b_b1/D.manifest.json` preserves the main experiment's original
timing and artifact provenance. `D.json` is its bare `.coordinates`.
`A.json` and `C.json` share one operator-local L, and differ only in group_wait.
`B.json` is D with group_wait=false. `E.json` equals D, with execution selected
by `--emit kernel-sequence`. D is never built or timed by this recipe.

Verified CPU selection produced L=D in this case: TN=(32,16,48,16,128),
chunks=16. Candidate scores are calibrated model predictions, not isolated GPU
optima. `L.selection.json` records legality and cost terms; the companion
candidate CSV retains scores at full double precision. No Structure/derive,
whole-program search or GPU candidate sweep is used to select L.

The sequence adapter retains the historical task-body headers byte for byte.
It preserves LA attention merge, folded argmax and head-to-embedding handoff.
The initial graph includes the embedding prologue; steady graphs begin at
QKV. R3's sequence adapter has kPostHead=false, and the generic sequence emits
Final only when kPostHead is true. Cross-stage ordering uses kernel completion.
NoTail and streamed=false preserve the prescribed non-streaming body entry.

Example commands on the original host, run from this experiment branch:

```bash
cmake -S . -B runs/sm90_ae_repro/build -G Ninja \
  -DMLIR_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/mlir \
  -DLLVM_DIR=/root/TileMega-integration/runs/integration_dnn_r6_20261116/mlir-prefix/lib/cmake/llvm \
  -DTILEMEGA_ISL_BUILD_DIR=/root/TileMega/build-isl \
  -DTILEMEGA_POLYLIB_BUILD_DIR=/root/TileMega/build-polylib \
  -DTILEMEGA_BARVINOK_BUILD_DIR=/root/TileMega/build-barvinok \
  -DTILEMEGA_TARGET_ARCH=sm_90 \
  -DTILEMEGA_BUILD_TESTS=OFF -DTILEMEGA_BUILD_VERIFY=OFF
cmake --build runs/sm90_ae_repro/build --target tilemega sm90-decode-ae-local -j 12
python3 - <<'PY'
from pathlib import Path
import lzma
p = Path('runs/sm90_ae_repro')
(p / 'decode.export.json').write_bytes(lzma.decompress(Path(
    'configs/serving/sm90d1/inputs/qwen3_1_7b/decode.export.json.xz').read_bytes()))
PY
runs/sm90_ae_repro/build/sm90-decode-ae-local \
  configs/targets/sm_90_h800.json "$PWD" \
  configs/serving/sm90_ae/qwen3_1_7b_b1/D.json \
  runs/sm90_ae_repro/decode.export.json runs/sm90_ae_repro/local
for arm in A B C E; do
  execution=serving
  if [ "$arm" = E ]; then execution=kernel-sequence; fi
  runs/sm90_ae_repro/build/tools/tilemega compile \
    runs/sm90_ae_repro/decode.export.json "runs/sm90_ae_repro/$arm.so" \
    --tmexec-config "configs/serving/sm90_ae/qwen3_1_7b_b1/$arm.json" \
    --serving decode --emit "$execution" --batch 1 \
    --runtime-target configs/targets/sm_90_h800.json --nvcc-split-compile 4
done
```

The retained collection spec is
`/root/TileMega-x1/runs/sm90_decode_ae/historical-spec.json`. It binds each
binary and the unchanged historical prefill by SHA256, the resolved historical
body version, and the guard policy. For a fresh collection, create a new spec
with the rebuilt binary paths/hashes and a new output directory; do not alter
the retained spec or overwrite prior artifacts. The original collection command:

```bash
python3 scripts/sm90_decode_ae_collect.py \
  --spec /root/TileMega-x1/runs/sm90_decode_ae/historical-spec.json \
  --out /root/TileMega-x1/runs/sm90_decode_ae/timing --gpu 2
python3 scripts/sm90_decode_ae_report.py \
  --run /root/TileMega-x1/runs/sm90_decode_ae --publish-root "$PWD"
```

The collector fixes one physical H800 and rotates A B C E / B C E A /
C E A B. Every arm has three fresh processes. The lightweight driver uses
the historical prefill/checkpoint algorithm and seed 0, L1 decode-loop,
one warmup and three timed requests per N=1 and N=1024. TPOT is
`(median(wall N1024)-median(wall N1))/1023`, followed by a median across the
three processes. Interference invalidates only the affected attempt, which is
retained and retried on the same GPU. No C1/C2, token/logit comparison, HF,
sanitizer, synchronization stress or profiling is run.

`guard_policy.json` reuses already recorded idle-power thresholds from the
merged repository. It affects idle admission only, not model compilation,
calibrated cost terms or timing statistics. The historical target remains
unchanged, and no GPU clocks or power settings are changed.
