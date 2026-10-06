# Mandatory implementation checkpoint after Phase A

Phase A is accepted in summary.md. The development checkpoint is merged into `/root/TileMega`. Main sources are frozen during the Phase-B build and validation queue.

| Item | Implementation | Validation / remaining work |
|---|---|---|
| AT-1 | Ec32 legality; small-chunk L1 ordinal permutation | Pure host permutation/coverage test; L2 logical queues unchanged. Ec variant selection and C-1 remain. |
| AT-2 | Nonpaged attention merge LA, separate L1/L2 ticket banks | Handoff host tests and single-process monotonic arrival test pass; three real-model smokes pass; 50 processes pending. |
| AT-3a | PV transposed MMA, `attention_impl=pvswap` | Five-architecture compile and position-coded numeric suite pass; model C-1/timing pending. |
| SK-1 | Required seed_fill and nonpaged combine LA | Host proof cases pass; Llama B16 split-32 model smoke passes; 50-process checks pending. |
| GV-1 | Direct register streaming, TN8/16/32, row/tiled input, existing epilogue | Standalone five-architecture compile/numeric pass; solver/codegen integration committed, production dispatch and two B1 model smokes pass. Small-TN residual with DN SS output is not supported yet and must not enter selection. |
| RW-3 | Register ping-pong in paged/tiled MMA loops, default off | Standalone and paged numeric checks pass; full model/resource/performance checks pending. |
| RA-1 | Attention noinline compile/CLI knob, default off | Five-architecture compile and numeric pass; resource/performance comparison pending. |
| EP-1 | Dedicated row scratch, parallel argmax, default off | Five-architecture compile and negative/tie/tail numeric pass. L1 lm_head tail distribution integrated; full matrix harness validation queued. |
| SL-6 | Configurable measure_top, integrated past objective and halving with identity checks | Python unit tests pass; Ec/implementation variants and required GEMV family integrated; global hard budget remains a documented deviation. |

## Nonpaged LA reasoning (inferred; reliability not accepted)

Every producer converges after its output stores, performs a device fence, and participates in the compute barrier. Thread 0 then performs an acquire/release ticket increment. The last increment acquires the release sequence of all producers before a compute barrier broadcasts the dedicated last flag. Only that CTA reads partials and calls the existing merge/combine arithmetic.
L1's producer-stage barrier occurs after the inline reducer completes; the elided reducer has no task or barrier. L2 publishes the reducer's own fine/aggregate events only after the reduction, so its consumers wait for the correct producer. FIFO elision excludes dynamic elided producers.
L1 and L2 have independent iteration counters and separate monotonic ticket banks. Tickets are never reset. For attention, all capacity chunks participate, including inactive chunks; merge still reads only active partials. This is the simplest fixed producer cardinality for changing past within an iteration stream, and is an explicit implementation detail. Full fan-in and step completion order reuse after the previous reduction.
The existing handoff proof rejects external readers of partial workspaces. The new path retains this check. These arguments need the required model C-1/C-2 and 50-process evidence before any synchronization conclusion.

## Selection details and limits

The multi-past score averages a piecewise-linear interpolation at all 1024 integer pasts 64..1087; beyond 1000 it holds the last measured value. This endpoint rule was unspecified and is recorded explicitly. Pilot rounds retain half (rounded up), at least three; final candidates receive three fresh alternating rounds. All pasts/rounds must have the same execution identity. Trace identities are rejected. Exit 75 propagates without rejecting the candidate.
Attention execution permutation is restricted to L1 decode Ec<=64. It changes execution ordinals to chunk-major so the active prefix balances over CTAs, while preserving g-major logical task/event numbering. Paged loader, consumer and lookahead, and nonpaged prefetch, use the same permutation. L2 uses the solver's existing logical schedule.
Default feature switches stay off pending the registered Phase-B measurements. Phase B is queued behind frozen-source numerical and per-cell smoke gates; Phase C/D remain evidence-dependent.
