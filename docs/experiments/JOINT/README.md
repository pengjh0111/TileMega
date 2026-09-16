# R5 joint L2 search

Baseline: a001b0ac53197551fe1e5e115528a9ea40d24783. The phase fork is committed
in 495e7c10 before this implementation. ChainDP remains unchanged as the L1
solver and initializer; its fresh result is PHASE/chain_dp/.

The finite catalog includes M={8,16,32,64,128}, N={4,8,16,32,64,128},
K={4,8,16,32,64}, split={1,2,4,8,16,32}, kappa={1,2,4}, low residency and
the register/smem seed's natural limit. The reference L1 center is 32x16x16s2;
its lower neighbors are explicitly rejected by the BF16 backend. The actual
R4 tile 128x128x16s3 is separately retained. Runtime compilation is authoritative
for residency; an over-resident request is rejected before launching.

`search.py screen` retains every point, including backend rejection and
capacity deferral. Its priority is max(estimated work/grid, weighted stage-DAG
critical path), never the L1 sum of stages. Phase-scaled costs are inferred,
with 70% tile-area and 30% operand-size scaling of the mainloop; this is an
explicit approximation to be checked by the measured ranking. The theoretical
whole-device arithmetic bound is deliberately weak, and the conservative
unknown path bound is zero. Neither is substituted for the corrected measured
floor. Capacity limits are not mislabeled as admissible pruning.

EX-S1c fails its simultaneous speed/ranking requirement. The declared degraded
search keeps three distinct geometries by that priority plus L1 and R4 seeds,
then evaluates every retained kappa/residency combination. `project.cpp` imports
each configuration's own CG, projects task ownership and split rewriting, and
materializes legacy, rotate, balanced, EFT, wavefront and chain. It reads the
emitted stage schedule, expands it around split combiners, and checks Plan
legality. Requested-event relations supply group readiness, stage publication
flags, singleton locality and sigma wait lifting for the publication-aware
simulator. Coarse top-k=3 includes boundary ties; only eligible placements get
full simulation. Every other placement remains visible as un-simulated.

`JointSearch.h` and rank.cpp evaluate the retained catalog against makespan,
retaining max(work_lb,cp_lb) pruning and explicit capacity accounting. Cached
project evaluations are inputs to this pass, not GPU timings. Exact predicted
ties in top-3 are broken by previously unused geometry, then residency. This
rule was set before any GPU candidate measurement. It prevents three identical
cost estimates from testing only the same tile at three residency levels.

`measure.py` compiles the predicted top-3. Five independent rotated pilot rounds
choose an arm; choice.json is frozen before 25 new confirmatory rounds. The
control is a fresh rotate + R3 B protocol, C1/C2/C3 off and W=1. Each process
records commands, exit status and binary hashes. Selected correctness uses 50
new processes. Materialized EFT/chain Plans are pinned to theta and residency;
SEQSCAN and sm_120 must re-solve them locally, never reuse a foreign Plan.

S3-c's report distinguishes rank within the compiled top-3 from the unmeasured
catalog. The latter has no empirical rank; a top-3% claim for it is unavailable
under this explicitly degraded candidate universe. No gate is relaxed.

The raw post-split runtime dependency seed is exported beside new trace dumps.
This is essential for split-K: pre-rewrite generated windows refer to different
stage ids. TRACE_V2/analyze.py prefers that actual seed and retains old behavior
for historical dumps. It does not change Plan semantics or synchronization.

Pilot repairs retained separately: an initial stage-order approximation was
replaced with the emitted schedule; a stage-sum priority was replaced with the
stage DAG path; real-width's metadata export.json was corrected to model.json,
which contains the actual export bridge. These were CPU preparation errors,
not correctness failures in an existing GPU configuration.
