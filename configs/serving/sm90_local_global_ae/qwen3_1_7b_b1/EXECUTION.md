# Frozen experiment execution

Evidence: the local domain, all 196 compiled/measured shortlisted candidates,
and the frozen winners are verified by the linked records. The README is the
immutable pre-measurement boundary/protocol declaration; its hash in S_local
must not be rewritten after end-to-end timing.

## Actual chain

This experiment deliberately uses the permitted model-specific offline path:

1. Frontend export semantics fix the local boundaries. Shared frontend native
   ABI buffer/weight recipes are reused as metadata, not as local winners.
2. Complete-operator H800 measurements select S_local. S_TM is extracted from
   the actual historical D generated source and manifest, both hash checked.
3. `sm90_local_global_plan.py` consumes the frozen structure, constructs its
   task dependencies, and generates Basic and critical-frontier candidates.
4. `sm90_local_global_emit.py` writes the selected owner/order/wait/publication
   tables and compatible native body aliases into the generated CUDA program.
5. The common `StaticTaskProgram` consumes those arrays on GPU. Native serving
   ABI weight recipes rebuild tile packing, map boxes and temporary bindings.

No solve, Structure/derive call, whole-network structure search or MLIR
writeback is used. There is no claim that the old generic native solver already
implemented this planner. Historical D used an additive local model and fixed
native policy templates. The new executor therefore requires D_test.

The planner uses measured S_local complete-operator time divided by task waves,
the existing calibrated native operator model for S_TM, fixed untuned auxiliary
operator estimates, and calibrated publication/handoff costs. These are planning
estimates, not TPOT measurements. It compares the complete frontier replay with
Basic and retains Basic on an exact tie. Both structures tie in this run. The
S_local frontier candidate moves 672 owners; S_TM moves none. Both actual plans
retain Basic, so A/C and B/D respectively have identical executed task programs.
Frontier candidates and replay evidence remain in `runs/.../audit-v1/`.

## Static executor constraints and audit

Every ordinary queue entry names one compute task. The coordinate is supplied
to every GEMM/attention producer address, consumer address and task count. All
stages use the same interface and dispatch method in A–D. No entry invokes a
whole-stage traversal with its original cyclic task count.

The terminal head is a constrained group: each track calls the head body once
with its complete assigned coordinate list, including a zero-task call for an
idle track. This preserves register accumulation across head tiles and exactly
one original folded finish per track. It does not introduce a second reduction,
full logits tensor or duplicate embedding. B/D retain fused head embedding;
A/C/E retain separate embedding. Their different fusion boundaries belong to S.

Rings and operand banks keep their existing offsets. An explicit coordinate
context occupies the otherwise unused bytes at the end of the 1024-byte
header. No cross-task preissue is used; original D also had weight_stream=false.
The selected aggregate plans finish prior consumers before temporary, norm,
QKV/context and activation storage reuse. Kernel completion replaces the same
aggregate unit boundaries in E. Native head/attention body completion remains.
The final step boundary acquires head completion before the next token reuses x.

Static audit verifies task coverage, data/order acyclicity, all logical
dependencies being enforced by the actual waits and track sequence, event
producer counts, map/recipe aliases, complete KV coverage for past=64..1086,
paired signatures and compiled coordinate/queue symbols. This is not numerical
or dynamic race verification. No correctness suite was added or run.

All arms use 132 resident tracks, 160 threads/CTA, residency=1, cluster=1 and
the common 232448-byte dynamic shared-memory resource ceiling. Per-task MMA,
TMA, slots and register stages are exactly those frozen in S. No SM/grid
reduction, dummy work or performance-based arm selection is used.

## Reproduce the frozen matrix in the original environment

The metadata templates are retained historical frontend artifacts, with hashes
recorded in each `.emit.json`:

* S_TM: `/root/TileMega-m/runs/sm90r3/solve/FINAL/qwen3_1_7b-B1/decode.so.cu`
* S_local ABI metadata: `/root/TileMega-explicit-ae/runs/sm90_explicit_ae/build-v1/A2.so.cu`

The local emitter replaces all implementation aliases, tile packing recipes,
map boxes and required buffer sizes; it does not reuse A2's configuration or
measurements. GateUp receives its own activation map because its selected TK
differs from QKV. The standard checkpoint, prompt and unchanged prefill artifact
are used by the thin timing driver.

Use a fresh runs directory for a new collection. Preserve the published frozen
selection and original measurement artifacts:

```bash
git switch sm90-local-global-ae-v1
python3 scripts/sm90_local_global_plan.py --out runs/local-global-replay/plans
python3 scripts/sm90_local_global_emit.py emit \
  --plans runs/local-global-replay/plans --out runs/local-global-replay/build
python3 scripts/sm90_local_global_emit.py build --out runs/local-global-replay/build
python3 scripts/sm90_local_global_audit.py \
  --plans runs/local-global-replay/plans --build runs/local-global-replay/build \
  --out runs/local-global-replay/audit
python3 scripts/sm90_local_global_collect.py \
  --spec runs/local-global-replay/audit/spec.json \
  --out runs/local-global-replay/timing --gpu 2
```

NVCC, CUTLASS and original artifact paths must be available as recorded in
`runs/sm90_local_global_ae/`. The timing Python is
`/workspace/llm-stack/envs/vllm/bin/python`. Independent processes run in
ABCDE / BCDEA / CDEAB order, with N1/N1024 each receiving one warmup and three
timed requests. The original guard retries interference, retaining every
attempt. It reports provisional ownership because of the container PID
namespace; that limitation is retained, not silently upgraded to full host
ownership verification.

The local microbenchmarks were compiled at
`dc8ae63c7a6cf43557e97ab460d7e40827302b32`, before the explicit coordinate API
was added. Exact generated shards, commands, binary hashes and source header
hashes are in `runs/sm90_local_global_ae/local-v1/`; all candidate JSONL samples
are in `local-timing-v1/attempt-42/measurements/`. Re-run those recorded compile
commands against that source revision if reproducing the original local
measurements. Do not replace the frozen S_local using later network results.

## Files

* `S_local.json`, `S_TM.json`: complete selected coordinates and body contracts.
* `A.json`–`E.json`: structure/plan signatures, execution mode and archive links.
* `plans/A.json`–`plans/D.json`: complete actual plans, with all task records,
  per-track sequences and event producer sets. JSON whitespace is removed but
  no plan data is omitted. E uses A's plan.
* `docs/results/SM90_local_global_AE_qwen3_1_7b_b1/local_candidates.csv`:
  every candidate, preselection decision, measured score where available and
  winner; complete samples remain at the referenced raw paths.
* `runs/sm90_local_global_ae/audit-v1/consistency.json`: executed static checks.
* Final TPOT/process/ratio CSV, provenance and plots use the new
  `SM90_local_global_AE_qwen3_1_7b_b1` stem. Previous experiment files are retained.
