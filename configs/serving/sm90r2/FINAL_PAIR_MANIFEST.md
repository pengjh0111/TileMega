# Optional final TileMega/reference pairs

`run_sm90r2_final_pair.py` closes the TileMega-reference comparison separately
from the absolute-metrics runner. The candidate still requires the complete
[final artifact contract](FINAL_E2E_MANIFEST.md), including independent E-3,
solver origin, same committed source, rebuilt prefill, and rebuild/SASS evidence.
The frozen TM-SM90R1 artifacts enter only the comparison arm; they never become
eligible final solver output. No external engine, anchor, HF, C-1, C-2, or full
ctest is run. `user_final_e2e_amendment` continues to mark the waived checks
`not_performed`.

**Verified on CPU:** contract tests cover reference identity/source/hash/geometry
mismatches, inactive queue construction, whole-pair outer-guard rejection,
bookend completeness, threshold calculations, and near-threshold escalation.
The actual frozen Qwen B1 reference index/config/binaries have also passed the
read-only reference-binding check. These checks establish no GPU timing,
accepted final candidate, or G-PERF result.

## Pair manifest, schema 1

Every FILE is `{"path":"...","sha256":"<64 hex digits>"}`, absolute or
relative to this pair manifest. Required fields:

| Field | Value |
|---|---|
| `schema`, `kind` | `1`, `"sm90r2_final_pair"` |
| `candidate_manifest` | FILE for the strict final artifact manifest |
| `reference_index` | FILE with the exact committed `configs/serving/sm90d1/reference/artifacts/index.json` bytes |
| `reference_config` | FILE for this cell's frozen reference config |
| `reference_prefill`, `reference_decode` | FILE for each explicitly selected indexed reference binary |
| `processes` | 3, or 5 for Llama-8B or a near-threshold retry |

The index is checked against the candidate source commit's repository copy and
must retain source HEAD `966f0b9ab3936964c7bc4ca9c64ad467552747b0`.
Each reference binary is independently checked with the artifact identity
verifier and must match the index's config/phase, binary SHA, artifact ID, source
HEAD, and frozen coordinate signature. Reference and candidate use identical
model config, prompt IDs, batch, checkpoint seed, and execution protocol.
The candidate's prefill-reference plan must be this indexed prefill plan.

## Inactive queue and execution

```text
python scripts/run_sm90r2_final_pair.py validate --manifest PAIR_MANIFEST
python scripts/run_sm90r2_final_pair.py queue --manifest PAIR_MANIFEST --gpu 1 --out runs/sm90r2/PAIR_PREP
```

Preparation writes an inactive single-cell queue, session ID, and immutable
queue/manifest/runner bindings. It neither publishes the queue nor starts a
scheduler. The operator separately admits it to the existing guarded scheduler
with the declared GPU, per-card lock, and bound guard policy.

The sequence is a reference/reference blank pair, then 3 or 5
reference/candidate pairs, then another reference/reference blank pair. Each arm
uses its own fresh GPU process and the final warmup1/repeats3, greedy BF16,
ISL/OSL64/1024 protocol. All pairs use the same physical card and NUMA binding;
GPU6 remains excluded and GPU5 functional-only. Both arms of each pair stay
inside one guard and physical wrapper. The pair root directly calls the strict
candidate runner function, keeping the wrapper parent-PID check valid. A failed
pair retries both arms in a new scheduler attempt directory.

After the scheduler completes the entire session:

```text
python scripts/run_sm90r2_final_pair.py collect --manifest PAIR_MANIFEST --preparation runs/sm90r2/PAIR_PREP --state SCHEDULER_STATE --out runs/sm90r2/PAIR_RESULT.json
```

Collection requires successful whole-pair guards/wrappers, unique GPU-process
PIDs across the session, matching checkpoints, one physical UUID, and unchanged
per-arm measurements, token hashes, commands, and candidate provenance. Raw
receipts preserve failed attempts. Namespace-limited ownership stays provisional.

## Decision

Paired change is the median of `(candidate_TPOT/reference_TPOT - 1)*100` across
the 3 or 5 comparison pairs. Each process TPOT follows the absolute runner's
N1/N1024 protocol. The result includes both arms' median TPOT/TTFT/E2E/output
throughput, both main-arm process ranges, bookend blank range, and the range of
all unchanged-reference process medians in this session, including bookends.
Every range is `(max - min)/median * 100`.

The decision threshold remains
`max(0.5%, same-session unchanged-reference range, reference range, candidate range)`.
A change within that threshold is indistinguishable; otherwise its sign gives
faster or slower. A 3-process result with absolute change no more than twice the
threshold is conservatively marked `needs_five_processes`; this schedules
nothing automatically and does not change the threshold. Prepare a new complete
5-process session before resolving that case. Results describe this cell only
and never assert the global G-PERF gate.
