# User-directed SM90R2 closeout

stated: after correcting known D/E implementation defects, measure TileMega
end to end directly. Reuse the frozen external baselines. Do not iterate on
calibration to chase unmet E-3 metrics or claim a validated solver final.

The measurement scope is five dense models at B1/2/4/8/16, the retained
Qwen3-30B-A3B at those five batches, and all five retained DNN at BF16 B1.
Dense programs use the current native full-K static path. Retained MoE/DNN
artifacts keep their actual original source HEADs; neither path is optimized
in this closeout. Every configuration is frozen before timing.

LLM protocol: ISL64/OSL1024, greedy BF16, one warmup and three timed
requests for N1 and N1024 per fresh process; three fresh processes,
five for Llama-3.1-8B. DNN protocol: 50 warmups and 200 CUDA event samples,
three fresh processes. GPU6 is excluded and GPU5 is functional-only.

`run_sm90r2_closeout.py` binds the explicit inputs and uninstrumented artifacts.
It does not replace the original strict solver-final runner. C-1, C-2, HF,
external anchors and full ctest are not performed in the closeout. Existing
timed-request token shape, telemetry and greedy-repeat evidence remain
measurement validity checks; no additional GPU execution is needed for them.

Reported ratios use historical frozen baselines and do not establish
same-session improvement. E-3, complete shared-service pricing, and
solver-final reconstruction remain unmet unless separately established.

For dense reproduction, build the host compiler, then use
`scripts/build_sm90r2.py --phase prefill` with this directory's build configs.
Build each decode with `scripts/build_sm90r2_native.py --config CELL.json
--coordinates CELL.tmexec.json --prefill-library REBUILT_PREFILL.so`.
Both builders also take `--compiler` and `--tag`. The explicit prefill avoids
requiring an ignored historical reference binary for the CPU ABI check.

`scripts/prepare_sm90r2_closeout.py --tag TAG --out NEW_DIRECTORY` verifies
the existing 50 artifacts and emits manifests, per-card queues, and scheduler
commands without starting GPU work. `--model-gpus` accepts JSON overrides for
available timing cards; keep every fresh process for a cell on one card.
All queues retain the physical lock, guard and NUMA binding. Authorized
concurrent timing is marked `NUMA_CONCURRENCY_TRIAL=1`, `VERIFIED=0`.
After execution, collect the completed state files with
`scripts/collect_sm90r2_closeout.py --state STATE [--state STATE ...] --out OUT`.
Raw receipts and generated preparations stay under `runs/sm90r2/`.
