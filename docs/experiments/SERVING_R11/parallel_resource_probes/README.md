# Exact top-3 resource probes, lowered serially and compiled concurrently

Source for the cached-floor measurements: `e04fd5d77`. The Llama B=1
`python -m tilemega build` run reused calibration and export but missed both
plan keys. Its prefill/decode solve wall times were **788.426/641.505 s**;
both still fail the 600 s budget. The prefill cumulative piece-pricing and
resource-compile phases were 289.017/272.976 s, and decode's were
65.363/229.585 s. These are not additive wall-clock partitions. They also
cannot isolate the floor-cache speedup because artifact hits and load changed.

The subsequent code change splits each resource probe into an MLIR/ISL
lowering step on the calling thread and a generated CUDA compilation/query
step. The skeleton top-3 path first prepares all three sources and then
compiles/queries them concurrently. The explicit three-candidate Llama B=1
decode smoke in `smoke.cases.json` completed; the three real occupancy
queries all returned **resident=1** and the `megakernel_compile` wall-clock
phase was **83.923 s** (one batch), versus 229.585 s cumulative for three
sequential queries in the previous full decode solve. The candidates differ
in attention block choice, so this is evidence of concurrency and correct
occupancy for the smoke, not an exact paired speedup. The candidate test
selected rank 1, L2, mean **3.937 ms** on random BF16 data; it is not a
full-request or EV-2 performance result. Four host solver regression cases
(`stage_flow`, `flow_runtime_release`, `skeleton_search_isolation`,
`plan_skeleton`) passed after the code change.

An initial direct C++ invocation omitted `PYTHONPATH` and stopped after the
successful resource queries when the Python measurement command could not
import `tilemega`. The completed smoke reran with `PYTHONPATH` set. Both
fail and success are retained in the work directory; the files here are
from the successful invocation.

The full `python -m tilemega build` was then repeated with the parallel
path. Llama B=1 prefill solved in **597.918 s** (`megakernel_compile`
99.670 s), and decode in **493.758 s** (`megakernel_compile` 86.758 s).
Both plan records mark the 600 s budget as passing. The 2.082 s prefill
margin is narrow; these two plans do not establish G-7 across the reduced
four-cell matrix. `prefill_parallel.*` and `decode_parallel.*` retain the
full source's phase and occupancy evidence. TF-1/EV-2 remain open.

The two cached winners also completed a real-weight 64+1024-token request:
Llama B=1 E2E median **4.0844 s**, TTFT **4.106 ms**,
`E2E/ΣT_floor = 1.573`. All three timed token sequences matched; L1 and
L2 had **0/1024** mismatches. HF teacher-forced C-1 passed
(1024/1024 gaps ≤ 0.5; maximum 0.125). Both selected `.so` SASS audits
reported FP64 count zero. `full_run/` contains the 1,023-step curve, guard
rounds, plan manifests, SASS audits, cache decisions and full check output.
This is one diagnostic cell with `handoff=off`, without a same-session vLLM
arm; it is not EV-2 or the four-cell performance comparison.
