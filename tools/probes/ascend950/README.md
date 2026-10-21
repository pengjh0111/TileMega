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

## AS-2 contextual cost probe

Build only `as2_native_cost` with the private AS-2 CANN environment, `nice -n10`
and parallel1. Run it through `tilemega.bench.as2_run`, physical1/logical0 in
the recorded environment. It uses validated G8,32 bounded metadata/publication
iterations, an AIC/AIV cross-group round trip, and a512MiB AIV copy working set.
It checks8192 payload samples and reports raw1GHz timer ticks. The wait deadline
is2e9 counter ticks, unlike the older AS-1 poll-count protocol above.

`as2_native_prices costs` requires at least five fresh, successful, nonoverlapping
process receipts with the same binary/device. `prices` combines that summary
with actual diagnostic2 Body profiles, checks model/config/shape/implementation/G
identity, and leaves shared-load measurements labelled contended. Half-roundtrip
latency is a symmetric estimate; logical GM/L2 copy throughput is not peak HBM.
`as2_compile_llm emit-priced` compares two real legal FIFO/placement plans with
these prices. Decode context bounds survive request relocation and are checked
before ACL. Neither a successful probe nor a predicted makespan passes model
correctness or the final TPOT Gate. See `docs/results/AS2_pipeline.md`.

## AS-2 FP32 attention arithmetic

`as2_f32_mmad` is a component probe, not a generated full-model executor. Build
only this target with parallel1. Arguments are input directory, new output,
M (1 or32) and mode:0 QK,1 PV,2 reciprocal softmax,3 vector-division softmax.
Its one MIX block performs one coordinate-driven Cube tile or AIV0 softmax;
FP32 Cube explicitly disables HF32. No CATLASS full-operator scheduler is used.

`as2_f32_mmad.py prepare/reference/compare` freezes actual independently observed
Q/K operands and compares Cube against torch_npu eager and CPU FP32/FP64.
`prepare-pv` needs the no-lock NPU wrapper: it computes actual causal FP32
probabilities/PV with static capacity1088; the recorded example has64 valid KV.
`prepare-softmax` only packs those already observed independent probabilities.
Keep the exact binary/headers before changing a build. Current bounded checks
cover QK M32/M1,N64,K128; PV M32,N128,K1088; and causal softmax M32,capacity1088.
All three vector-division/Cube components match independent NPU outputs bitwise.
This is not proof of other shapes, integrated MIX ownership, or full-model C-1.
