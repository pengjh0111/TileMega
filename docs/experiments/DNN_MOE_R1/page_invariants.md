# DM-1 page sequence and workspace invariants

inferred: a CTA's loader and compute group traverse the same placed task queue
in L2, or the same stage/grid-stride task order in L1. Elided stages are absent
from both traversals. For a tiled GEMM task, each side advances its private
sequence by `ceil(ceil(K_count/TK) * TN * TK * 2 / page_bytes)`. M ownership
does not affect this B-page count. Both sides use the same split-K invocation.

inferred: forward GEMM and prefill QKV/o_proj activations are produced inside
the launch. They are read by the compute group's activation pipeline after
the task's dependencies become ready. They never enter the loader page list.
Prefill attention uses the existing compute-side K/V pipeline and contributes
zero loader pages. Its block extent remains the full cache capacity, with
independent causal query blocks writing disjoint context rows. The original
prefill contract has no partial-output merge. Decode retains its historical
KV page stream and reduction path.

inferred: sequence `s` owns slot `s % pages`, parity `(s / pages) % 2`, and an
exact generation tag `s`. The loader acquires the empty slot before changing
its generation or data. All loader lanes publish the full barrier after their
copies; every compute lane arrives at the empty barrier after consumption.
AwaitFull checks the generation before parity, preventing an old completion
from satisfying a wait after a ring wrap. Ring initialization is the sole
barrier shared by all 160 threads; task work uses the compute-only barrier.

inferred: ring control, the dedicated LA flag, task workspace and page data
occupy disjoint regions. Task workspace is the maximum of activation pipeline,
epilogue scratch, attention shared storage and declared new-body requirements.
Activation and epilogue scratch alias only after the activation pipeline
drains. PageLayout checks this partition against TargetSpec's shared-memory
limit; an explicitly requested ring count either fits or is rejected.

inferred: the lookahead cursor advances through the same no-producer byte
stream as the loader. Prefetched and loaded counters measure bytes, not tasks
or K iterations. A lookahead refill may cross its requested threshold by at
most one emitted range (at most one page), then stops until loader progress.
Prefetch is a cache hint: it neither publishes full pages nor changes compute
readiness. It cannot make a produced activation eligible for an early read.

Pending binding-aware extension: the weight payload has no in-launch producer,
but its source address depends on dispatch. A lookahead cursor must stop at an
unready binding without waiting and retain that cursor position. Blocking it
could prevent the current page from reaching compute, which must run the
dispatch producer. The actual loader task may wait for its binding's release
event. Loader and compute must apply the same valid/empty predicate, so empty
virtual tasks advance neither sequence and still publish completion. These
requirements are not yet claimed as verified implementation.

verified: forward dense GEMMs, prefill dense GEMMs and prefill attention each
pass 50/50 fresh sm_89 processes, sixteen poisoned epochs per process. Native
C ABI L1/L2 outputs match bitwise and meet independent FP32-reference numerical
criteria. TM=16/32/64/128 are covered for GEMMs; all five architectures compile.
`results/CI5_paged_phases.json` and `results/CI5_paged_attention.json` retain
identities, source hashes, process logs and all kernel resource/spill lines.
These synthetic plans verify the core page paths; binding-aware full PageStream,
real DNN/MoE TaskBodies and model gates remain separate checks.
