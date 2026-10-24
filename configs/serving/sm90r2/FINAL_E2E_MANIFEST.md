# TileMega-only final E2E artifact contract

`python scripts/run_sm90r2_final_e2e.py` provides independent `validate`, `queue`,
`run`, and `collect` actions. It does not change the frozen development runner,
pair runner, or selection callback. **Verified by CPU contract checks:** its
command contains no HF, C-1, C-2, external-engine, or ctest invocation. No final
GPU run or artifact acceptance is established by these checks.

The result records `user_final_e2e_amendment`, with C-1/C-2/HF/external baselines
and full ctest marked `not_performed`. It does not synthesize correctness passes,
G-SOLVER, G-PERF, or a final selection. Existing development correctness evidence
inside the solver receipt remains historical selection evidence.

This single-arm result reports absolute metrics and does not establish G-PERF.
The separate [TileMega reference pair workflow](FINAL_PAIR_MANIFEST.md) adds the
required same-session, same-card TM-SM90R1 comparison without external engines.

## Future manifest, schema 1

Every `FILE` below is `{"path": "...", "sha256": "<64 hex digits>"}`. Paths are
absolute or relative to the manifest directory. Each `ARTIFACT` is an object
with five required FILE entries: `binary`, `identity`, `plan`,
`generated_source`, and `sass`. Identity, plan, and generated source must be the
binary's actual `.identity.json`, `.plan.json`, and `.cu` sidecars. The other
sidecars required by `tilemega.build.identity.verify`, including ptxas resources
and build command, must remain available. `sass` is a captured `cuobjdump
--dump-sass` file; validation independently disassembles the bound binary on CPU
and compares its instruction encodings.

The required top-level fields are:

| Field | Value |
|---|---|
| `schema`, `kind` | `1`, `"sm90r2_final_artifacts"` |
| `model`, `batch` | One of the five dense models; 1, 2, 4, 8, or 16 |
| `source_head` | Full final committed source SHA |
| `source_digest` | Common artifact identity `source.source_digest` |
| `compiler` | FILE for the compiler used for all three artifacts |
| `config` | FILE for the serving config with model, batch, execution, prompt/model hashes, and reference prefill coordinates |
| `model_config` | FILE for the local BF16 model `config.json` |
| `prompt` | FILE for frozen rectangular ISL64 token IDs |
| `target` | FILE for the exact native target used by selection/decode |
| `coordinates` | FILE for the selected native tmexec JSON |
| `solver_receipt` | FILE for completed `.search/receipt.json` |
| `e3_validation` | FILE for independent seven-cell E-3 evidence, as specified below |
| `rebuild` | FILE for passing `.rebuild/comparison.json` |
| `prefill_reference_plan` | FILE for the immutable reference prefill plan; no reference binary is timed |
| `guard_policy` | FILE for the SM90R2 scheduler/guard policy |
| `prefill` | ARTIFACT for explicit newly rebuilt prefill |
| `selected_candidate` | ARTIFACT for the measured winning solver candidate |
| `decode` | ARTIFACT for the final rebuilt solver output |
| `protocol` | `{"isl":64,"osl":1024,"warmup":1,"repeats":3,"dtype":"bfloat16","sampling":"greedy"}` |

All three artifact identities must bind the same clean committed source set,
source digest, and compiler. The run checkout must be at that HEAD with matching
compiler/runtime files and E2E helpers. All calibrated native header hashes must
match that source. An old mixed-source development compiler does not satisfy
this contract. Traced or boundary-diagnostic artifacts are rejected.

Both decode artifacts must have `native_origin=solver` and the exact selected
coordinates/export/target. Selection must be measured and converged without
budget exhaustion, with a legal winner and its accepted binary-bound measured
receipt. The solver must report `e3_validated=true`, `development_only=false`;
both its embedded calibration and the bound target must have `fit_status=fitted`.
The embedded calibration must equal the target after the existing C++ receipt
serializer's `%.10g` round trip; the target SHA256 independently binds all original
unrounded bytes. No price, coefficient, or threshold is rewritten by this check.
The final plan must contain that exact solver receipt. The rebuild
receipt must bind both binaries, generated sources, coordinates, workload, and
instruction encodings with a passing, non-diagnostic comparison. The runner
also compares actual candidate/final SASS encodings itself.

The E-3 aggregate is a JSON object with `schema=1`,
`kind="sm90r2_e3_validation"`, matching `source_head` and `target_sha256`,
`fit_status="fitted"`, `independent_validation=true`, `solver_gate_pass=true`,
and `cells`. It must be assembled from audited independent evidence; this runner
does not produce E-3 measurements or promote an initial fit. There must be exactly
one row for each of the seven development cells (`cell` names such as
`qwen3_0_6b_b1`). Each row contains:

- `class_errors_pct`: signed errors for qkv, attention, o, gate_up, down, and
  lm_head, using the corrected per-step class totals; every absolute error ≤15%.
- `step_error_pct`: signed whole-step prediction error, absolute value ≤10%.
- `kendall_tau` ≥0.7, `global_solver_top3_contains_fastest=true`, and
  `distinct_programs` ≥8.
- `class_evidence`, `step_evidence`, `ranking_evidence`: FILE records for the
  underlying audit artifacts; these paths are relative to the E-3 aggregate.

All numeric errors must be finite. The runner checks these unchanged thresholds
and every evidence-file hash; it does not replace the audit that derives errors
from measurements. Historical F87/F113 development selections are ineligible.

Prefill is explicit; there is no D1 prefill path default. Its source must be the
same final HEAD/compiler. Its complete plan, excluding only `model`,
`runtime_target`, and `target_spec`, must equal the bound reference prefill plan.
The reference plan must also reproduce `config.coordinates.prefill`: pg,
requested residency, attention implementation/block size, sorted GEMM geometries,
and the SHA256 of `json.dumps(plan['gemms'], sort_keys=True)` with default Python
separators. This preserves the reference coordinates while rebuilding at the
new HEAD. A target's native-only calibration addition may change its hash.

## Staging and collection

These are future-use forms; preparing this implementation launches no jobs:

```text
python scripts/run_sm90r2_final_e2e.py validate --manifest MANIFEST
python scripts/run_sm90r2_final_e2e.py queue --manifest MANIFEST --gpu 1 --out runs/sm90r2/FINAL_PREP
```

`queue` creates a new inactive directory with `queue/queue_final_e2e.json` and a
bound preparation receipt for one cell: three fresh processes, or five for
Llama-8B. It does not publish jobs or start a scheduler. The operator separately
admits that queue to the existing guarded scheduler, using the declared device,
`TILEMEGA_GPU_LOCK=/tmp/tilemega_gpu<i>.lock`, and bound guard policy.

Each job runs the physical CUDA/NVML identity wrapper under `numactl`. `run`
requires the held per-card flock, scheduler timing marker, wrapper parent PID,
physical UUID/PCI, policy hash, explicit matching device variables, and actual
CPU affinity within that GPU's NUMA node. GPU6 is excluded; GPU5 is functional
only and cannot time this runner. Current external GPU restrictions still apply.

Only the TileMega `r16_run` harness is invoked, under the existing seed-0 BF16
random-checkpoint supervisor. It runs N1 and N1024 with warmup1/repeats3, L1
prefill, and native device-loop decode. The process receipt remains
`awaiting_outer_guard` until collection verifies the scheduler completed each
fresh process, both wrapper and outer guard succeeded, checkpoint identities
match, and the physical GPU stayed the same.

```text
python scripts/run_sm90r2_final_e2e.py collect --manifest MANIFEST --preparation runs/sm90r2/FINAL_PREP --state SCHEDULER_STATE --out runs/sm90r2/FINAL_RESULT.json
```

Collection preserves token-file paths/hashes and validates their table sizes.
TTFT is median N1 wall time; E2E is median N1024 wall time;
TPOT is `(E2E - TTFT)/1023`; output throughput is `batch*1024/E2E`.
The cohort reports medians of those process metrics, exact generated-token
counts, and namespace ownership status without upgrading provisional ownership.
Raw harness metadata remains intact, including its generic diagnostic selection
label; the separate amendment receipt states this run's actual scope.
