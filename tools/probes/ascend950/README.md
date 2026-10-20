# AS-1 native probes

Source `scripts/as1_env.sh`. All commands below use the pinned private CATLASS
checkout and write outside the source tree. Select targets individually; do not
build `all` or launch a test matrix.

```bash
cmake -S tools/probes/ascend950 -B "$AS1_WORK/build/probes" \
  -DCMAKE_BUILD_TYPE=Release -DCATLASS_ROOT="$AS1_WORK/third_party/catlass"
cmake --build "$AS1_WORK/build/probes" --target as1_topology --parallel 1 --verbose
python3 python/tilemega/bench/as1_run.py --physical-device 0 \
  --out runs/as1/jobs/topology_new --timeout-s 90 \
  -- "$AS1_WORK/build/probes/as1_topology" 2 16
```

Each invocation requires a new output directory. Physical device 0 and its
logical ID 0 were verified in `runs/as1/inventory/acl_device_0.json`.

| Target | Arguments | Checks |
|---|---|---|
| `as1_topology` | groups (1–4), epochs (1–16) | MIX ID bijection, bounds, one launch, repeated worker work |
| `as1_entry_resources` | `cube` or `vector` | explicit AIC/AIV entry; vector touches 212 KiB UB |
| `as1_entry_oversize` | **compile only, never execute** | 1 MiB UB compiler diagnostic; host also rejects it |
| `as1_gm_visibility` | direction (0–4), elements (32 or 1024) | two groups, 16 buffer-reuse epochs, pre-read old cache, bounded waits |
| `as1_aiv_body` | kind (1 RMSNorm, 4 SiLU, 10 Add), rows (1/4), width (128/129), input dir, output file | BF16 inputs/output, FP32 vector arithmetic, exact tail DMA and guarded output |
| `as1_catlass43` | M N K | official 950 basic GEMM math |
| `as1_catlass50` | M N K | official 950 GEMV/tail math |
| `as1_catlass64_add` | M N K | official MIX GEMM + Add through GM |
| `as1_catlass64_add_ub` | M N K | optional official same-group direct route; disabled until tested |

GM directions: 0 AIC→AIC, 1 AIC→AIV, 2 AIV→AIC, 3 AIV→AIV,
4 all four sequentially in one launch. Producer group 0 and consumer group 1
are deliberately different. AIC production invokes TileMega's coordinate-driven
CATLASS `GemmTile`, not CATLASS's full-operator scheduler. Consumers announce
their old-data pre-read before production. Each epoch requires producer ready
and consumer acknowledgement; payload validation is bit-exact. A wait exceeding
100000 polls publishes abort. A host timeout only stops this invocation's group.

Safe sample copies are generated in the build directory. CMake removes the
upstream `aclrtResetDevice` teardown and caps its AIC group request at two;
the pinned checkout and sample arithmetic remain unchanged. A missing expected
source pattern rejects configuration. Check **`Compare success`**, since the
upstream sample exit code alone does not prove numerical success.

These are probes, not CTG/Solver end-to-end workloads. The exact tested scope,
failures and raw evidence index are in `docs/results/AS1_probe.md`.

For one AIV body, use `as1_correctness prepare-aiv`, execute `as1_aiv_body`
through the wrapper, then `as1_correctness compare-aiv`. Each invocation admits
one operator and shape. Six individual regular/tail cases passed; numerical
receipts are indexed in `runs/as1/probes/aiv_bodies_summary.json`. These do not
stand in for the multi-stage CTG/Solver execution.
