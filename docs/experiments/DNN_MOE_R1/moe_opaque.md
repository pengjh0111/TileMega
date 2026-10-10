# Opaque MoE control

verified: `compile --moe-opaque 1` is accepted for decoder MoE regions/plans with
`--pg l2` or `pages`. The manifest, build identity and CG module attribute record
the control. It is absent from default generated CUDA. The control applies at
runtime construction, after split-K expansion; it does not change L-sem or claim
that the added synchronization is an inherent data dependency.

inferred: each region spans its router's optional explicit RMSNorm through its
combine. Construction installs full-stage predecessor edges at entry, internally,
and at exit, including expanded split-K reducers. Existing counted contracts remain
present. LA producers/reducers within the region are restored to ordinary stages.
The standard runtime graph and placement legality checks see these extra edges.

inferred: the paged loader waits for the predecessor's completed stage before
advancing. Lookahead uses an acquire probe that returns without changing its cursor
when the predecessor is incomplete, including when the next stage has no pages.
Loader-required aggregate events are retained independently of compute FIFO wait
elision. The actual loader and the lookahead cursor use the same gate.

verified: the host region-boundary test covers explicit/deferred norm, multi-step
dispatch, adjacent regions and malformed/missing boundaries. One fixed synthetic
T=17 grouped region passes nonpaged and paged L1/L2 numerical and bit checks
(`results/MO_opaque_generated_smoke.json`). Both have exact outputs/routes for this
fixture. No latency was collected. These are one-process sm_89 checks, not 50-process
synchronization evidence or a complete decoder/model gate. Resource/spill records
are retained in the artifact identities. The enabled 65,536-byte lookahead path also passes on the same fixed fixture;
this exercises the nonblocking lookahead gate as well as actual-loader waits.

stated: no empirical choice between template, opaque or dynamic controls is made
under the user's no-performance-testing scope. Opaque barriers are exclusive to
this explicitly requested control, not a substitute for the default executor.
