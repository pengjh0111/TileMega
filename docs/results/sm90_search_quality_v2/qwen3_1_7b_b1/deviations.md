# Paper-alignment notes

Paper artifact: /root/Prompt/tilemega.pdf, SHA-256 7430b2df06be5bdb1706fd0ef0631b1295b39d3c94a7576ce38997628f1af3bb.

| Paper passage | SQ2 realization | Suggested paper wording |
|---|---|---|
| §5.1, “All rates are measured on the target.” | Measure production GEMM and attention stage bodies on H800. GEMM markers use a calibration-only ledger at the existing epilogue marker. The reported stage-body interval begins after ring initialization. Kernel entry and GPU-event elapsed are also retained. | State the instrumentation and report the stage-body timing boundary. |
| §5.2, “TileMega constructs finite candidate domains from supported decompositions, compatible tile implementations, and target resource limits.” | Use native dense `FamilyCandidates` and `AttentionCandidates` at both residencies. Record the product upper bound, Legal-only estimate, and post-search-admissibility estimate. | Label the legal-space count as a Monte Carlo estimate for the actual rule-pruned dense domain. |
| §7.3, “Basic megakernel (B) selects each operator’s decomposition” | SQ2 B-iso tests the isolated configuration selection rule. SQ2 selection-quality runs keep comparison candidates on the frozen execution path or record their actual plan binding. | Distinguish the B-iso selection-control result from the complete Fig. 9 Basic-planner arm. |

Further rows will record the actual replay realization, capacity handling, and any measurement anomalies once these phases run.

| §5.3, “TileMega selects the owner with the earliest estimated completion” (Eq. 5) | For each shortlisted native configuration, compare predecessor owners and eight evenly spaced tracks with a task-index offset, using grouped CTG prefixes and current queue tails. A separate verifier checked the computed owners, typed-stage order, replay queue order, and replay score against the native template. All 24 matched. | Report that Eq. (5) did not change the chosen native execution template or ordering for this case. Avoid implying support for lowering a different native owner table or measuring a distinct placement gain. |
| §7.3, “Compiling a single candidate megakernel, by contrast, takes 26–67 s” | Measure finalist compilation and occupancy for this case and report parallel wall time plus per-command work. The audit-only compilation and measurement of P are additional evaluation work. | Keep production search, finalist work, audit work, and exhaustive extrapolations separate. |

### GPU outage and audit dispatch

The first audit round had 190 valid processes before the original GPU outage.
One original H800 UUID remained absent after reboot. Its two surviving lanes
then waited for almost two hours while occupied. Following user authorization,
the resumed queue also admits current devices 0 and 1. The original frozen
device assignments and all prior receipts are retained. The dispatch amendment
and UUID binding are recorded in gpu_failover.json.

The resumed audit offers the next candidate in the frozen shuffle or reverse
to an authorized available H800. A guard rejection requeues only that item.
The same six-sample GPU guard, correctness checks, binaries, and request-wall
protocol apply. The 43 documented CUDA infrastructure errors are retained
as failed attempts and are eligible for a fresh process. Already valid
processes are preserved. Audit and confirmation results can span H800 UUIDs.
Device placement depends on availability. This change affects the audit
transport and carries no change to the paper's hierarchical-search method.
