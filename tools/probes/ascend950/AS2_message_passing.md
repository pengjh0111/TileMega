# Bounded payload visibility, separate from model admission

`as2_message_passing` tests G8/G14 under the conservative existing DSB/full
DCache acquire protocol. It is not a TPOT benchmark or an optimized memory
protocol. No target/model Group limit changes as a side effect of this probe.

One MIX launch contains twelve explicit phases: all four cross-group role
directions at32,1056 and4096 FP32 elements.1056 elements are33x32, exercising
the Cube M tail. Each phase reuses each producer buffer for eight epochs.
AIC producers compute a CATLASS BlockMmad identity product and drain FIX;
AIV producers complete MTE2/MTE3. Each consumer first loads and checks the
previous payload through Scalar/DCache, then publishes `armed`. Producers
wait for every consumer's prior ack and current armed before overwriting.
Consumers wait for ready, explicitly acquire, and check both Scalar reads and
a separate DMA path. AIC consumers use MTE2/L1/MMAD/FIX, AIV consumers use UB.
The reverse ack is published only after those reads and output DMA complete.
Both AIV lanes contribute; AIV→AIC fan-in waits for two distinct producers.

Empty roles participate in every phase rendezvous, preventing early entry
into a different hardware phase. Every control field is512B separated and
single-writer, including one abort word per actor. Waits check peer abort and
the locally verified1GHz system counter, with2e9 ticks per wait. Hardware DMA
itself has no software preemption guarantee; the Host wrapper retains a120s
timeout and stops only its own process group. A timeout stops related testing;
it does not authorize a reset or another trial of the same failing path.

CLI: `as2_message_passing <8|14> <new runs/as2 output> [abort-worker]`.
The optional actor injects a software abort after the first rendezvous;
`expected_abort` requires every other actor to observe it without timing out.
This does not inject a hardware fault. Each launch starts with new zero state.
Host CLI rejects larger G, invalid actors and escaped/existing paths before ACL.

Compile with the reviewed private environment, one target/parallel1/nice10:

```bash
source scripts/as2_env.sh
nice -n 10 cmake -S tools/probes/ascend950 \
  -B /home/d50064638/junhuip/tilemega-as2/build/opt_message_passing_v1 \
  -DASC_DIR=/usr/local/Ascend/cann-9.1.0/lib64/cmake
nice -n 10 cmake --build \
  /home/d50064638/junhuip/tilemega-as2/build/opt_message_passing_v1 \
  --target as2_message_passing --parallel 1
```

Run through the existing no-lock single-card wrapper, first G8, then review
before G14. Preserve controls and complete payload/consumer records even on
a bounded device failure. `as2_message_passing_audit.py <output>` independently
checks extents, all protocol counters, old checksums, final producer/consumer
payloads and guard holes. Five fresh normal processes per new G are required
for a synchronization claim; an injected-abort process is separate evidence.

`verified-run`,2026-10-30: CANN9.1.0/BiSheng15.0.5, pinned AS-1 CATLASS,
physical0 PCI D7:00.0→logical0. G8 and G14 each pass **5/5 fresh processes**;
one G14 software-abort process also passes with all42 actors observing abort,
no deadline. Device numerical counters, independent Host final-payload oracle,
old checksums and all guards pass. Concurrent external contexts were observed
and did not block these functional tests. Each process uses one bounded launch.
Evidence: `runs/as2/search/opt_message_passing_workflow_v1.json`.

The first G8 run failed one diagnostic Finished record (7 versus expected8),
with all24 actors at final barrier24 and no numerical/guard/timeout errors.
Its source/image/records remain under `failures/opt_message_passing_g8_completion_v1`.
The fix publishes Finished exactly once, with ordered publication after all
epochs/acks; the expected count is unchanged. Rapid unordered diagnostic stores
are a suspected cause, not an established hardware fault. The compiler's earlier
host-lambda rejection is also retained. Neither failed attempt counts as a pass.

This establishes this FP32 payload protocol, not model-speed improvement,
BF16 direct handoff, fixed physical-core affinity, or cache-maintenance removal.
