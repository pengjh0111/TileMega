# Fused Runtime Event Projection

Design before implementation; not GPU evidence.

The A2 runtime graph is the input, including split and attention expansion.
A selected adjacent pair supplies an exact consumer-runtime-task to
producer-runtime-task relation derived from physical ownership. The consumer
defines the replacement task space. Producer work may repeat when inverse
fanout exceeds one; that remains charged by the mixed task price.

Build the relation new task -> old phase tasks. Rewire external dependencies
by composing it on both sides of the original runtime C. Remove the selected
internal edge before composition, not every resulting self-edge (which could
hide a cycle). Rebuild stage-local event groups from the resulting C, preserve
aggregate policies on external edges, and deduplicate the (worker,event)
image before counting waits. The removed producer stage reduces task refs
and stage terms directly; do not additionally subtract a fixed fused-edge
rebate. Same-worker singleton polls are elided by the same ownership rule as
A2. No new fence discount is invented while its rate is uncalibrated.

The API validates complete consumer/producer coverage, one producer per
consumer, stage adjacency, and no overlapping phase ownership. A producer
whose result remains externally read requires unique inverse ownership;
replicated external writes are rejected. Runtime lowering must later consume
the same replacement and phase maps. A CPU projected event difference alone
does not establish executable fusion or pass the GPU sign gate.
