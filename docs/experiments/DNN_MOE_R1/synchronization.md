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
