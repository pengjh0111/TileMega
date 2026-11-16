# DNN1 forward calibration reproduction

The checked-in target contains 1006 measured keys and 3018 physical-card medians.
The generators below perform CPU work only. CUDA sources, build snapshots,
binaries, queues and measurement receipts belong under ignored `runs/` paths.

First derive an explicit method inventory from the frozen target:

```sh
python3 scripts/dnn1_forward_probe_catalog.py --target configs/targets/sm_90_h800.json --out runs/reprobe/catalog.json
```

Every inventory entry retains its original `constant_id`, exact selector strings,
unit and source recipe. The earlier recipes use:

```sh
python3 scripts/dnn1_forward_recipe_probe.py --recipe tm32 --source-root . --out runs/reprobe/tm32 --build
```

Recipes are `base`, `live16`, `float4`, `tm32`, `mma32_math`, `dwpw`,
`residual_scaled`, `gated_dw`, `global_pool`, `global_pool_unique`, and
`maxpool_flat`. `base` and `float4` share probe source; the production Split-K
header distinguishes their implementation epochs. `live16` retains the full
original source catalog; select only `^dw_8_8_64_16_s[12]$` when queuing that
supplement. For `float4`, select `^split_`. The remaining recipe catalogs select
the measured subset directly. Resource-infeasible catalog entries remain
infeasible and are excluded from queues.

Pool Finish, complete row-statistics Finish/store and cached ScalePanel use:

```sh
python3 scripts/dnn1_forward_primitive_probe.py keys --family pool --target configs/targets/sm_90_h800.json --out runs/reprobe/pool_required.json
python3 scripts/dnn1_forward_primitive_probe.py generate --required runs/reprobe/pool_required.json --source-root . --out runs/reprobe/pool --build
```

Use families `rowstat` and `sca` for the other two methods. Complete SCA requires
the actual TM32 implementation. `--tm 64 128 --legacy-pending-tm32` reproduces
the older four-key recipe with explicit unmeasured TM32 records; `--tm 32`
produces the separate two-key recipe. SCA generation requires CPU PyTorch for
its independent BF16 oracle, including negative zero and padded positive zero.
Pool requirements may instead be derived from explicit producer inventories
and layouts; rowstat requirements from five-model shape inventories; SCA
requirements from generated model CUDA. See the generator's `keys --help`.

After building, prepare queues with the existing measurement workflow:

```sh
python3 scripts/dnn1_constants.py queue --build runs/reprobe/tm32/build/build.json --out runs/reprobe/tm32/measure --gpus 0,5,7 --batch-processes --strict-processes --python scripts/dnn1_idle_floor_python.py
```

Queues require the established GPU ownership, NUMA affinity and scheduler locks.
Use the existing owner/memory guard policy and five fresh executable processes
per case per card. GPU6 is excluded. Queue generation does not run measurements.
Collect each accepted recipe with `dnn1_constants.py collect`. Complete and
subset conversion both validate unchanged raw files, physical sessions and five
fresh processes; pending methods receive no fabricated values.

Compose the fifteen explicit collected directories into a new calibration epoch:

```sh
python3 scripts/dnn1_calibration_epoch.py --base-target configs/targets/sm_90_h800.json --out runs/reprobe/epoch --recipe base=runs/reprobe/base/collected --recipe live16=runs/reprobe/live16/collected --recipe float4=runs/reprobe/float4/collected --recipe tm32=runs/reprobe/tm32/collected --recipe mma32_math=runs/reprobe/mma32_math/collected --recipe dwpw=runs/reprobe/dwpw/collected --recipe residual_scaled=runs/reprobe/residual_scaled/collected --recipe gated_dw=runs/reprobe/gated_dw/collected --recipe global_pool=runs/reprobe/global_pool/collected --recipe global_pool_unique=runs/reprobe/global_pool_unique/collected --recipe pool_finish=runs/reprobe/pool_finish/collected --recipe rowstat_finish=runs/reprobe/rowstat_finish/collected --recipe maxpool_flat=runs/reprobe/maxpool_flat/collected --recipe sca_panel_64_128=runs/reprobe/sca64/collected --recipe sca_panel_32=runs/reprobe/sca32/collected
python3 scripts/dnn1_dictionary_constants.py --source runs/reprobe/epoch/target.json --output runs/reprobe/epoch/dictionary_target.json
```

The composition preserves all source-recipe views, exact doubles and constant
indices. It replaces only the declared base keys and writes a separate target;
its `DNN1_constants.csv` also applies unchanged to the schema3 view. The input
target's non-forward prefix and native constants are preserved.

Verified CPU evidence: all eleven earlier generated recipe sources and all four
Pool/Rowstat/SCA source variants match their measured CUDA sources byte for byte.
The portable composer reproduced all 3018 published doubles from the accepted
raw recipes. This establishes method and selector coverage for the frozen epoch.
Historical production header snapshots and raw measurement receipts remain in
`runs/`; generating the same CUDA source against current headers does not
recreate an earlier binary. New header implementations require new measurements.
In particular, register-chain Apply needs distinct composite measurements;
adding the older single-step costs overcounts its shared-memory passes.

## Composite register Apply

After the production epilogue gains `kRegisterPointwise`, inventory its actual
emitted plan choices and generate distinct ordered-chain probes:

```sh
python3 scripts/dnn1_chain_apply_probe.py keys --generated-cu runs/plans/*/plan.so.cu --out runs/reprobe/chain_required.json
python3 scripts/dnn1_chain_apply_probe.py generate --required runs/reprobe/chain_required.json --source-root runs/accepted-register-chain/source --out runs/reprobe/chains --build
```

The source root must contain the register-chain implementation. Each timed
interval calls `Chain::ApplyRegister` once, retains operation/vector order and
explicit residual scale-pointer mode, and excludes initialization and FP32
output drainage. An independent CPU float oracle checks the resulting live
values. Live row/column tails and spatial residual pitches remain separate
keys. Finish/store, row statistics and pool side outputs need their own terms.
First-channel/first-row address transfer is stated explicitly; shuffled
residuals and nonzero offsets require separate mapped probes and are rejected.
This inventory covers the provided emitted choices, including TM32 when
present; it does not fabricate costs for other solver choices. Its CPU build
and source checks establish probe readiness. Performance requires accepted
five-process measurements on all three physical cards.
