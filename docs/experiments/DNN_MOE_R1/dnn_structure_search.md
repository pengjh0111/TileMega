# DNN structural selection

verified: `SearchDnnStructures` rebuilds the same exported graph for each
structural choice and runs the existing geometry, placement, residency and page
search on that graph. Coordinates are individual eligible DW/PW pairs,
individual `(LayerNorm representative, consumer GEMM)` edges, none/greedy/L2
reuse, and C′=4/8 for small-channel inputs. Disabled features remain disabled.
Two starts (eligible rewrites enabled/disabled), strict improvements, memoized
states and a shared wall budget bound coordinate descent. The complete key,
errors and each state's retained top-M geometry predictions are recorded in
`<output>.structures.json`; the selected key and sidecar hash enter the manifest
and identity. Final materialization replays page/lookahead/handoff choices too.

verified: the synthetic two-pair, two-LN-edge, RGB/residual graph admits 11
distinct structural states and generates final CUDA. Its host test checks mixed
choices, exact edge provenance and rebuilt body probes. The fixed case selects
C′=4, both fusion pairs and both deferred edges. This is predicted selection;
no latency measurements or global optimality proof are claimed. Native execution
of this combined case and the public CLI's real resource probe are pending.
`results/CI7_dnn_structure_host.json` seals the host evidence.

verified: this integration exposed and repaired three independent defects:
exact DM scalar pricing retained legacy q coordinates while arithmetic used m/n;
nominal output work counted unwritten halo storage; and C′=4 descriptors required
an eight-channel stride. The old negative stride test used four, which the DM-1
spec explicitly permits; it now rejects two and separately accepts four. A stale
test also declared the already implemented DW/PW private body unavailable.
Numerical tolerances were unchanged. Eight default LLM CUDA outputs remain
byte-identical to reference (`T1_default_cuda_host.json`).

verified: eight focused host regressions pass. The larger semantic-lifting test
hit its 240-second timeout; a diagnostic stack remains in enumerated ISL relation
construction. This is retained as incomplete host evidence, not counted as a pass.

inferred: per-plan non-GEMM resource sources improve admission over the old
decoder-only wrapper by instantiating the actual LN/embedding/pool/attention/DW
bodies and finite DW chains. Narrow GEMM probes still compile a generic body;
the final megakernel resource query remains necessary for complete admission.
No new empirical body fits were invented. Only the selected structure family is
finally materialized; other families retain predictions, since measured second
level selection was removed by the user's scope change.
