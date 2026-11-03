# FP32 QK phase completion probe

`verified-code/run`, physical950PR1/55:00.0, logical0, CANN9.1.0, pinned
CATLASS2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52. G1 only; this is a
component, not an imported/solved model or a performance/default admission.

The existing `GemmTile<float,float,32,64,128,true>` uses
`MmadPingpong<Ascend950,true,false,1,false,2,2,2,2>`: UnitFlag enabled,
one L0C bank. The pinned `block_mmad_pingpong_tla.hpp` enforces that pairing
and passes unit0b11 to its output copy. The TLA basic kernel keeps one
BlockMmad alive across coordinates and completes its phase afterwards.
No change to these internal flags/policies or whole-operator scheduler is
introduced by the probe.

`as2_qk_phase_drain.cpp` compares the actual legacy per-tile FIX_S completion
with one FIX_S completion after all disjoint score tiles in the phase. Both
include the MMAD constructor/destructor inside their diagnostic timer, keep
one worker resource outside, and return only after output completion. There
is no asynchronous external Task retirement, credit queue or reduced precision.
The two scalar-completion schedules use identical FP32 input/accumulator/output
math with HF32 disabled. Physical score row stride remains1088.

Two bounded cases, four reuse epochs, five fresh wrapper processes:

| Rows | D | Valid keys | Output tile columns | Result |
|---|---:|---:|---:|---|
|2|128|129|3×64|5/5 bitwise/FP32/guards pass|
|4|64|1087|17×64|5/5 bitwise/FP32/guards pass|

Independent NumPy FP32 reference tolerance is frozen before execution:
atol=rtol=2e-5. Maximum observed absolute error1.7881393432617188e-7;
zero nonfinite/failed elements. Uncomputed capacity suffix and512B guards
remain untouched. The last tile's initialized invalid K rows are zero,
matching the bounded cached-KV preparation contract; no uninitialized read
is accepted. Each process executes **two component launches**, one/case.

Ticks are diagnostic component intervals under observed shared load. They
exclude cross-lane softmax/PV, dispatcher/metadata, whole-model scheduling and
request iteration; they are not task prices or an end-to-end speedup claim.
Model integration must expose an implementation/capability, retain the final
phase completion before publication, reprice through CTG/Solver/Plan, and
pass full correctness/C2/C3/request timing. G1 proof is not G8 model admission.

Reproduction, after sourcing the private `scripts/as2_cpu_env.sh`:

```sh
nice -n 10 cmake -S tools/probes/ascend950 \
  -B "$AS2_WORK/build/taskbody_qk_phase_20261103_v1"
nice -n 10 cmake --build "$AS2_WORK/build/taskbody_qk_phase_20261103_v1" \
  --target as2_qk_phase_drain --parallel 1
python -m tilemega.bench.as2_run --physical-device 1 \
  --required-free-mib 2048 --timeout-s 60 --output-reserve-bytes 268435456 \
  --out "$REPO/runs/as2/jobs/NEW_NAME" -- \
  "$AS2_WORK/build/taskbody_qk_phase_20261103_v1/as2_qk_phase_drain" \
  "$REPO/runs/as2/probes/qk_phase_drain_20261103_v1/inputs" \
  "$REPO/runs/as2/probes/NEW_OUTPUT"
```

Input generator/version/seed/hashes, predeclared policy, source snapshots,
all five receipts/FP32 error records and raw timers:
`runs/as2/probes/qk_phase_drain_20261103_v1/`.
Binary SHA256:
`e5b1532034e11600716fdb32b042889f98cdf50bceabe8e01a72fa283c4b8c4d`.
Compile logs/private finite five-process driver remain under `tilemega-as2`.
No device lock/reset, other-process operation or CUDA regression is used.

Subsequent actual G8 CTG/Solver/Plan integration and its inconclusive full
request timing are documented in
[`AS2_cache_runtime_20261103.md`](../../../docs/results/AS2_cache_runtime_20261103.md#integrated-qk-phase-candidate).
The standalone G1 probe above remains a separate, narrower evidence claim.
