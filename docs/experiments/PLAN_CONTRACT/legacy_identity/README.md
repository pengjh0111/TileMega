# EX-E1 / H2: the legacy plan is bit-identical at codegen

Gate E1-a asks that a Plan of `legacy_grid_stride` with `W = 1` produce exactly
what the pre-plan generator produced.  This directory holds the codegen half of
that evidence.  The host-materialization half of H2 — `schedule`,
`schedule_offsets`, `task_waits` and the event tables — is the `p0` cells of
`../mode_identity/`, where the legacy plan's dumps are byte-identical to the
baseline host's on both models at both seq.

Reproduce with `../run.sh`, which prints the E1-a and E1-b verdicts.

## Method (verified, 2026-09-12, this session)

Both `.cu` sets come from the same build directory `build-portable`, differing
only in the commit the generator was built from:

* `baseline/` — round-2 baseline `6c359e2b`, the tree before any Plan work.
* `plan/` — the Plan contract tree (`bbe813e5`, `157b5960`, `e7b4c5cc`).

Command, for `model in {gqa2, mha4}`:

```
build-portable/tools/tilemega-compile \
  docs/experiments/SEQSCAN/raw/export/${model}.json  OUT/${model}.cu \
  --variants docs/experiments/OWNERSHIP/plan_structured.json
```

## Result — PASS (verified)

| pair | diff |
|---|---|
| `gqa2.cu` | empty (sha256 `017a39b9…` on both sides) |
| `mha4.cu` | empty (sha256 `1be74406…` on both sides) |

`sha256.txt` carries every hash.

### Positive control

An empty diff is only evidence if the emitter is alive. `plan_emission_*.diff`
compares this tree's plan-free source against the same source generated with
`"balanced_placement": true` added to the single variant — currently the only
route by which the frontend emits a non-legacy `mode`. Exactly one line changes
per model, and it is the variant initializer:

```
- {…, 15u},
+ {…, 15u, nullptr, true, true, nullptr, {2u, nullptr, 0u, 1u, 0u}},
```

i.e. `RuntimePlanDesc{mode = balanced (2), params = none, param_count = 0,
window = 1, policy = aot}`. `plan/{gqa2,mha4}_balanced.cu` are those sources.
The baseline tree has no `RuntimePlanDesc`, so there is no baseline counterpart
to diff a balanced source against; the control is within this tree by
construction.
