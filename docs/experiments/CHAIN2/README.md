# R4 cost-aware extension and queue placement

All switches remain off by default. `ChainRequest::cost_aware_extend` rejects
an edge unless its calibrated hop is strictly greater than the successor's
`task_ns`. The calibration is `SIMULATOR/hop_ns.tsv` evaluated at the producer's
fan-out, approximately 1234 ns on sm_89, rather than a newly invented literal.
The successor uses the same cost-model weight as chain extraction. Equality
rejects. Existing queue capacity formulas and hard legality checks are retained.

The simple price test alone does not satisfy D-b. `minimal_variant.cpp` is a
reproducible isolated diagnostic containing that test without the final fill
changes; `build_variant.py` links it without replacing the shared library.
The final implementation also ranks ready tasks by remaining critical work,
prices sibling CTA sharing on the same SM in `finish_on`, and breaks equal
finish-time ties toward fewer path hops. The selected experiment uses four
**existing** feedback rounds to carry realized queue blocking into extraction.
These are explicit changes to the initial D recipe, authorized by the user's
instruction to solve a mistaken premise rather than stop all R4 work.
`design.json` froze this recipe before any Chain2 GPU timing.

The GPU comparison has four arms: rotate; the completed cost-aware recipe;
the same four-feedback-round recipe with cost awareness disabled; and the
unchanged R3 chain with feedback disabled. All use configuration A so the
measured protocol matches the hop calibration. The matched control separates
the feedback setting from the cost-aware implementation. No `ChainDP` code is
changed and no window becomes default.

`final/replay/path/*_path.tsv` records each simulator critical-path node and
edge. The verifier counts cross-worker edges directly, checks edge types
against worker IDs and requires replayed chain source bytes to match the
measured materialized source. It does not accept `predicted.tsv` as a D-b
verdict. The replay uses the same local geometry and manifest as the final
solver run. Raw edge-price exclusions are stored in exact-price histogram
buckets in `rejected_extensions.tsv`; their counts are unique DAG edges in
the selected extraction pass, not repeated DP visits. Each bucket records
hop, queue cost, multiplicity and an example producer/successor.

The sm_120 runner regenerates geometry, calibration and all materialized
Plans on its target. Its original/feedback-matched/cost-aware arms expose
whether a cheaper hop changes the relative value of the extension test.
Only CPU self-checks and sm_120 compilation have been performed here.
