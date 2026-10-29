# Bounded control-only MIX topology diagnostic

`as2_topology` is separate from the older `as2_progress` payload litmus. It
performs no Cube work or cross-core wait. Each actor writes a cache-line-separated
record of its role, raw block/subblock ID and TileMega worker ID. The host checks
all expected records, inactive slots and guards. One launch requests G MIX blocks.
Read-only `aclrtGetDeviceInfo`, `aclrtGetDeviceResLimit` and
`aclrtGetStreamResLimit` observations are flushed before launch and retained
in the result. Unsupported queries retain their ACL error and a null value.
No resource limit is set or reset; reported limits are not claimed as free cores.

`verified-run`: CANN9.1.0/BiSheng15.0.5 compiles this target for dav-3510.
`verified-run`: G8/G14 observe24/42 unique workers and expected subblock IDs
on physical1/logical0. Read-only device and stream limits both return28 Cube,
56 Vector cores. G14 remains disabled for model plans. ID coverage cannot establish simultaneous
residency, FIFO progress, DMA visibility or a usable larger model topology.

Build one target with the existing private environment:

```bash
source scripts/as2_env.sh
nice -n 10 cmake -S tools/probes/ascend950 \
  -B /home/d50064638/junhuip/tilemega-as2/build/opt_l1k256_probe_v1 \
  -DASC_DIR=/usr/local/Ascend/cann-9.1.0/lib64/cmake
nice -n 10 cmake --build \
  /home/d50064638/junhuip/tilemega-as2/build/opt_l1k256_probe_v1 \
  --target as2_topology --parallel 1
```

CLI: `as2_topology <8|14> <inventoried_hardware_groups> <new_output>`.
The third argument must be a new directory below `runs/as2` with an existing
parent. Use the no-lock single-card `as2_run` wrapper with a finite timeout;
run G8 first and G14 only after reviewing its result. Keep the inventoried
capacity and physical/logical card identity in the job receipt. No other own
NPU job may overlap. On timeout, retain the receipt and stop related tests;
do not reset the shared device or enable larger model capabilities.

After successful ID coverage, a separate `as2_gemm` diagnostic accepts
`M16 N1792 K128 implementation1 input output G8|G14`. All fourteen output
tiles are independent, cover every requested AIC and perform no cross-core
wait. This distinguishes Cube/flag-local failures from the older message-passing
litmus. It cannot admit G14 payload visibility or model residency. The optional
group argument leaves all existing default-G2 probes unchanged. Build in a
new private directory to preserve the earlier selected probe binary.

`verified-run`: independent Cube G8/G14 runs pass the frozen BF16/CPU-FP32
policy (max_abs0.003326893) and produce identical BF16 payload hashes. Each
uses one launch. Evidence: `search/opt_independent_cube_workflow_v1.json`.

`as2_arrival` then holds every actor at a scalar-only all-arrival condition.
Every actor has separate512B entered/abort/completion cells and one writer;
the verified1GHz system counter bounds each wait at2e9 ticks. It never waits
inside a Cube/DMA operation. Timeout and peer-abort states are retained, and
the host returns failure rather than resetting the device. One G8 sanity and
**five fresh G14 processes pass**, with all42 actors completing, no aborts,
correct mapping and intact guards. Maximum per-actor elapsed across G14 trials
is16.569–22.165us; this is a control diagnostic, not a hardware throughput cost.
Evidence: `search/opt_arrival_workflow_v1.json`.

These results establish bounded scalar-wait liveness in these processes.
They do not establish physical-core affinity or four-direction Cube/DMA payload
visibility. The old G14 payload timeout is retained; G14 model capability
stays off until a new bounded stale-cache/buffer-reuse message-passing test
passes five fresh processes. The old failed probe was not retried unchanged.
