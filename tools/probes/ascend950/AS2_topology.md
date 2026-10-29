# Bounded control-only MIX topology diagnostic

`as2_topology` is separate from the older `as2_progress` payload litmus. It
performs no Cube work or cross-core wait. Each actor writes a cache-line-separated
record of its role, raw block/subblock ID and TileMega worker ID. The host checks
all expected records, inactive slots and guards. One launch requests G MIX blocks.

`verified-run`: CANN9.1.0/BiSheng15.0.5 compiles this target for dav-3510.
Device results are not yet collected at this code checkpoint. G14 remains
disabled for model plans. Eventual ID coverage cannot establish simultaneous
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
