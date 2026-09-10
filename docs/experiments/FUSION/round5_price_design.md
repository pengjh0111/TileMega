# Mixed task pricing contract

Implementation design, not a GPU performance claim. B1.1 pricing extends the
existing TaskInstanceNs evaluator; it must not approximate a mixed task by
assigning all its arithmetic to either TC or CUDA.

1. Retain the producer and consumer physical R/W maps alongside their exact
   composition. Only internal tensors change memory space. External reads
   remain global, and externally needed intermediate writes remain global.
2. Price each phase through the same instance evaluator as an unfused task.
   GEMM nominal mainloop work remains the calibrated issued-work domain;
   its output is redirected to shared. SIMT uses its exact read/write sets.
   Local input/output traffic is charged against shared bandwidth, not L2.
3. The TaskBody flow identifies which operand loads become entirely local;
   only those nodes lose global-memory latency. A mixed local/global load
   phase stays global. CTA reduction and boundary barriers remain charged.
4. A fused task is indexed by the consumer. Sum phase time for each task,
   then take the maximum in each wave, with that wave's active occupancy.
   Repeated producer instances therefore really consume time. Report the
   unfused producer price times `(fanout-1)` separately; do not add it again.
5. Resource legality remains max phase scratch plus the exact intermediate
   tile, max registers, with a caller-supplied whole-kernel residency pin.
   Neither task count nor intermediate allocation can retain the old
   producer parallelism when the consumer requires a full reduction row.

The price API does not invent event counts: candidate-specific L-sched
projection must provide before/after runtime event metrics. It also does not
claim that a model estimate of shared traffic is a compiled fused TaskBody.
Mixed pricing, interval DP, L-task rewrite and GPU acceptance stay separate
ledger entries until each is executed.
