# Dynamic virtual task control

verified: `--moe-dynamic 1 --pg l2` emits the dynamic L2 control. Expert
GEMM tasks share one saturating claim counter per expanded stage. Other
stages retain their static owners. Workers advance in stage order and wait
on the claimed task's complete dependency list. Empty virtual tasks still
execute the ordinary completion notification.

inferred: the counter's iteration interval contains exactly the stage's task
count. A successful CAS consumes one ticket; saturation consumes none.
Sequential launches require no counter clearing kernel. Queue-local FIFO and
previous-wait elision are disabled because the claiming worker can differ
from the original owner. Placement-specific event sharding is also excluded
from this control; logical task/group event counts are unchanged. There is
no added inter-worker stage barrier.

verified: the host cursor check demonstrates that a fast worker can claim
more than its original share, with complete, unique task coverage and static
entry/exit ownership. The generated 17-token, 16-expert fixture passes the
HF operator tolerance on one fixed synthetic input, with L1/L2 bit equality.
Two invocations of that same input check persistent counter advancement.
Maximum error is 0.0078125 and both routing sets agree completely.
`results/MO_dynamic_generated_smoke.json` joins the result and resources to
artifact `84d1157d5f56a1ee7ba929be627e5fa74eb815fd24223fd26b7ce0003ed9a4e0`.

verified: the L2 entry uses 255 registers and has recorded spills. This is
one process on sm_89; it is not the original 50-process synchronization gate.
No performance measurement was made. Architecture compilation and default
LLM SASS invariance for this addition remain unverified.
