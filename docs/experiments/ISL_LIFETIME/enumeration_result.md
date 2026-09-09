# Endpoint enumeration: exact union enumeration without disjointization

✅ While rebuilding the original 4×4096 split8 scene for condition9, the
baseline compiler ran for **1089 seconds without completion**. A read-only
gdb stack sample places it in isl_map_make_disjoint called by
isl_set_foreach_point from FitWaitWindowSymbolic's endpoint enumeration.
This was CPU set processing, not a hung GPU kernel. The owned compiler
process was interrupted; no fixture or model data was deleted.

`lib/Analysis/CouplingRelation.cpp:222` now enumerates each basic set and
deduplicates the resulting finite points. For finite sets this is exactly
the union, not a relaxation or sampling. It avoids partitioning overlapping
pieces into disjoint polyhedra first. Scoped reference checks remain active.
`TILEMEGA_ISL_COMPONENT_ENUMERATION=0` selects the old algorithm for isolated
comparison; there is no automatic fallback or enumeration budget cutoff.

✅ The same full wide split8 input finishes in **17.184903416 seconds**, with
96 tasks / 114 couplings / 60 stages, 114 symbolic windows and 0 fallback.
1089s is an interrupted lower bound, not the old completion time. These are
compiler measurements, not end-to-end GPU performance statistics.
Evidence: `../REALMODEL/condition9/codegen_stack.txt`,
`interrupted_codegen.txt`, `component_enumeration.txt` in that directory.

✅ Both BF16 reference models generated with `plan_structured.json` are
byte-for-byte equal between the saved pre-change compiler and the changed
compiler (`cmp` succeeds). SHA256 for **both** versions:

| Model | SHA256 |
|---|---|
| gqa2 | 017a39b966658b8b5ce3b22cef6528366068806a2503a53110395fe19c5c9b9c |
| mha4 | 1be74406debdc487b29ad0bdac4a0757d96a683135cac16caf1c9332a8c71cd2 |

Control binary was saved before the change at
`/tmp/tilemega-points.PYKsfW/compiler_before`; temporary before/after CUDA
files are in the same directory. For reproduction, build one compiler with
the above macro=0 and another with macro=1, and pass each the identical
`SEQSCAN/raw/export/{gqa2,mha4}.json` and `OWNERSHIP/plan_structured.json`.
The compiler's stdout path is not part of the CUDA byte comparison.

The unit test independently enumerates 32 eight-piece overlapping integer
relations by bounded host loops and compares exact point sets; an explicit
overlap case checks duplicate elimination. No GPU race claim comes from this
CPU test. Full build/policy/CTest evidence is in `enumeration_build.txt` and
`enumeration_ctest.txt`; condition9 GPU acceptance is reported separately.
