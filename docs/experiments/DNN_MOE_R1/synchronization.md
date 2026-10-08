# DM-1 synchronization evidence

verified: the sparse table, counted release/acquire, and weighted last-arriver
primitives each pass 50/50 fresh sm_89 processes. Every process exercises 32
epochs per path. Five targets (sm_80/89/90/100/120) compile without spills.
`results/CI5_dependency_primitives.json` records transitive input hashes,
compiler commands, per-kernel resources, binary hashes, and each process log.

inferred: table padding contributes no dependency; each consumer traverses only
its row's intervals. Counted producers publish all contributing warp stores
before a release RMW. The consumer's acquire observes that release sequence
before reading partials. Counts accumulate across launches with the target
`static_count * (iteration + 1)`; overflow is rejected. Weighted last-arriver
uses the same release sequence, and only an arrival ending at the static total
executes the reducer and resets its private ticket. Zero-contribution arrivals
cannot execute a reducer. The original unweighted interface is unchanged.

verified: the synthetic oracle checks sparse and empty table rows, poisoned
partial buffers, runtime permutations of target counters, variable arrival
weights, exactly one reducer callback per target/epoch, and zero tickets after
reduction. These checks retain independent host expectations.

Pending: integration with full L1/L2 and paged execution; real TaskBody 50-process
checks; WAR/WAW poisoning, dispatch, binding-aware pages, empty virtual-task
notification, and the remaining paths of DM-1 section 8.A. No full section 8.A
path or model gate is claimed complete by these primitive tests.

verified: the full stage materializer and its wait/notification functions pass
150/150 new sm_89 processes, 50 each for kappa=1/4/16
(`results/CI5_stage_dependencies.json`). Every process alternates L1/L2 for
16 epochs with poisoned partials. Table rows remain sparse/empty, grouped fine
events preserve their producer counts, and runtime permutations accumulate
exact counted totals. The two modes produce bitwise equal synthetic outputs.
Five architectures compile; all recorded kernels have zero spill stores/loads.
The separately sealed host adapter checks pass 7/7
(`results/CI5_table_materialization_host.json`).

inferred: counted stages use separate L1/L2 counter banks, prebound in host
parameter tables. Switching execution mode cannot satisfy a wait with the
other mode's earlier arrivals. The placement DAG conservatively includes all
potential counted producers; device waits use the target's static contribution
count. This affects scheduling legality rather than introducing a stage wait.

Pending: real TaskBodies and their gates, and paged counted execution.
The stage experiment uses synthetic task bodies
through the production materializer/wait functions, not DNN/MoE model bodies.

verified: automatic bound CG/codegen transport passes 9/9 host checks
(`results/CI5_bound_codegen_host.json`). CG validates injective task
linearizations, exact bound relation equivalence, counts, canonical intervals
and table stride. Multiple input edges between the same stages are unioned
before choosing an exact window/table, including a sparse table whose extra
input fills its hole. B=1/2/8 and deliberate attribute corruption are covered.

verified: the generated table descriptors compile on sm_80/89/90/100/120;
all recorded kernels have no spill stores/loads. A conflicting batch range
is rejected by the generated ABI contract
(`results/CI5_bound_codegen_architectures.json`). This semantic fixture is
compile-only: its access relation is synthetic and is not numerically executed
as a dense GEMM. Real DNN table-body gates remain pending.

inferred: symbolic metric verification must rederive in the same symbolic
space before binding. A finite nonzero-fiber enumeration and Barvinok's
symbolic expression may differ in their zero-valued coordinate support;
the existing structural polynomial comparison does not canonicalize all
floor identities between them. Parameter-only image counts use a scalar
polynomial space so physical-reread subtraction remains well typed.

verified: the production PageStream/ring and native C ABI execute synthetic
forward dense plans (50/50), prefill dense plans (50/50), and prefill
QKV/attention/o_proj plans (50/50), sixteen poisoned epochs per process.
L1/L2 outputs are bitwise equal and pass independent FP32/BF16-store oracles.
Forward/prefill activation reads stay compute-side; prefill attention emits no
loader pages and writes context directly without a merge stage. TM=16/32/64/128
and five architectures compile (`results/CI5_paged_phases.json`,
`results/CI5_paged_attention.json`). Resource records retain every spill.
This covers the core forward/prefill page path in §8.A.10; binding-aware
pages/new-body workspace requirements and model gates remain pending.

verified: initial handwritten smoke fixtures failed before the final checks:
a missing finite-epilogue dispatcher trapped, and an invalid prefill merge read
unwritten partials. Signed-input generation also required an explicit cast of
the unsigned epoch before subtraction. The fixtures now follow the generated
prefill direct-context contract and use signed test inputs. Criteria and
independent oracle equations were not changed. Original failed logs remain in
`runs/dm1-ci5-paged-phases-smoke` and `runs/dm1-ci5-paged-attention-smoke`.

verified: binding/page primitives pass 50/50 fresh processes, 32 poisoned
epochs per process, and five architectures compile with zero spills
(`results/CI5_binding_page_primitives.json`). The nonblocking ready probe
precedes dispatch publication; active records select independently checked
expert pages, stale empty records emit no page, all virtual tasks notify, and
the two-slot ring wraps repeatedly. Host capacity/address checks pass 3/3
(`results/CI5_binding_host.json`). The installed vLLM version was appended to
the native-test identities after completion, without changing binaries; both
old and enriched artifact IDs are retained. This synthetic single-CTA probe
does not complete production binding-aware PageStream or expert/model gates.

verified: monotonic weighted/unweighted last-arriver primitives pass 50/50
fresh sm_89 processes, 32 epochs for each primitive and each of two separate
banks. Five architectures compile with zero spills
(`results/CI5_epoch_last_arriver.json`). All four compute warps write poisoned
partials before publication; exactly one callback reads them per target/epoch.
Zero contributions neither increment the ticket nor execute the callback.

inferred: the release RMW follows every writer fence and compute convergence;
the final acquire observes earlier arrivals before reduction. Static totals
define unique epoch boundaries, and tickets are never reset. The runtime must
protect partial storage across epochs and give L1/L2 separate banks. Full
nonpaged event publication, elided-stage skipping, and §8.A.11 remain pending.

verified: nonpaged access-proved selection/lowering passes 11/11 host checks
(`results/CI5_nonpaged_handoff_host.json`). The opt-in path emits
`TILEMEGA_NONPAGED_LA=1`; absent/false settings emit no added CUDA text.
It selects attention merge and split-K combine, excludes norm recompute and
argmax, and rejects escaped split-K partials. Device integration and §8.A.11
are pending. This is host evidence, not a synchronization claim.

verified: production binding-aware PageStream/Task/GEMM paths pass 100/100
fresh sm_89 processes (50 each for SYNC_V3=0/1), sixteen poisoned epochs per
process (`results/CI5_binding_pagestream.json`). Window/table binding events,
kappa=0/1/4/16, parked lookahead before dispatch, empty virtual notifications,
expert source pages and split-K stage/LA are covered. Native page execution
matches the independent numerical oracle and L1/L2 outputs match bitwise.
Sixteen builds cover five architectures; StreamProbe spills are retained in
the per-kernel resource records. Expectations were not changed.

Pending: rowgather, real router/dispatch/experts, model gates, and default-path
G-REG. This fixture supplies
synthetic dispatch records and dense A rows arranged in virtual order.

verified: native nonpaged split-K LA passes 100/100 fresh processes, fifty
each for SYNC_V3=0/1 (`results/CI5_nonpaged_handoff_native.json`). Sixteen builds
cover sm_80/89/90/100/120 and kappa=0/1/4/16; every kernel has zero spill
stores/loads. Every process runs sixteen poisoned epochs in L1 and L2,
compares independent-stage/LA outputs bitwise, then checks a four-step L1
loop. Elided-stage barriers remain zero and elided stages are absent from L2
queues. Eleven incompatible ownership fixtures are rejected. This establishes
the split-K core of §8.A.11; attention and new-body LA gates remain pending.

inferred: each execution mode owns a separate monotonic 64-bit ticket bank.
Only the final arrival reads partials and publishes the reducer's L2 events.
L1 skips the reducer and its barrier; the owner's barrier follows all reducer
work. Lookahead skips elided stages. Default nonpaged-LA-off declarations and
launch arguments stay behind the original paths; G-REG still requires checking.

verified: the first production smoke failed during compilation because
ticket preparation preceded GEMM construction and the prefetch fixture lacked
its stride macro. Corrected retries retain the original reference and
assertions; the failed logs remain under
`runs/dm1-ci5-nonpaged-handoff-smoke-production`.

verified: the L1 binding wait resolves an elided producer to its unique earlier
LA owner; the owner's barrier replaces the absent reducer barrier. The
fresh-process repeat passes 100/100, fifty each for SYNC_V3=0/1, and all five
targets compile (`results/CI5_elided_binding_owner.json`). Each process checks
32 poisoned owner-barrier epochs with the elided stage's events untouched,
then repeats the production window/table/page/empty/split-K checks. Missing,
duplicate and chained owner relations are checked. The added owner probe has
zero spills; thirteen builds retain StreamProbe spills. Real dispatch and
expert/model gates remain pending.

inferred: L2 continues waiting on the reducer's fine events, which the LA
callback publishes. L1 instead follows strictly earlier owners until it
reaches a non-elided stage. Its barrier follows completion of all reducer
work. Default non-DM PageStream field layouts, constructor arguments and
operand signatures now stay behind the original paths; G-REG is pending.
