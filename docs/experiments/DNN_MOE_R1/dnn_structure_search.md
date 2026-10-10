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
no latency measurements or global optimality proof are claimed. One fixed synthetic B=2 input now passes the unchanged elementwise tolerance
and L1/L2 bit equality in the selected combined megakernel
(`DN_dnn_structure_native.json`). The public CLI's actual resource probes also
compile; its separate search-only run evaluates 14 structural states.
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

verified: the combined native check exposed vector LN2d affine packing and
compact RGB host admission defects. Both spellings C and1,C,1,1 now pack
identically; a four-channel physical pitch is admitted. The earlier failures
remain recorded. The combined kernels spill; per-kernel resources are sealed
in the native receipt. No new process matrix or timing run was performed.

verified: pool LA versus independent-stage selection now participates in the
structural starts and coordinate scan. The selected mask is encoded in generated
CUDA, manifest and identity; shared proof qualification and inferred arrival
pricing are described in `dm_reduction_search.md`. The focused DNN final-CUDA
check passes in `CI7_dm_reduction_selection_host.json`; the earlier combined
DNN native receipt is retained without repeating its input check.
