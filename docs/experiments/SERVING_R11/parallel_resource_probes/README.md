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
from the successful invocation. The full 600 s plan budget remains unproven
with the parallel path, and TF-1/EV-2 remain open.
