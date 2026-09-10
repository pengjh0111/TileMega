# Logical Candidate Adaptation

Verified: `verify_candidates.py` scans adjacent L-task identities at BF16
tile 128x128x16, stages=3, split=1, seq=4, past=3. The earlier runtime-stage
scan is retained in `../production_candidates`; its zero accepted candidates
was an adapter limitation, not a proof that fusion opportunities are absent.

| Model | New logical candidates | Existing runtime fusion | Rejected |
|---|---:|---:|---:|
| gqa2 | 2 | 1 | 30 |
| mha4 | 4 | 1 | 62 |

The new candidates are RoPE -> KVAppend. Each passes exact symbolic
sum(wait)=sum(fanout), with zero recompute tasks at this configuration.
The existing candidate is the first layer's projection -> residual add;
it already executes inside the GEMM epilogue and earns no new runtime event
discount. Later residual adds have multiple graph producers and are rejected.
GEMM -> RMSNorm is not synthesized by dropping the intervening residual add.
Tile incompatibility is rejected, not repaired by pretending the producer
still has its original N-axis parallelism.

Code: `lib/Solver/TaskModel.cpp` (`DeriveLogicalFusionCandidate`,
`ComposeModelCandidate`), `tools/tilemega-fusion-probe.cpp`. Logical derivation
does not use scalar runtime ownership; that projection follows L-task fusion.
The historical runtime-task pricing API retains its default ownership path.

Verified: all candidate paths report `remaining=0`; 23 scalar/fusion rejection
branches report before=after=0 (`errors.txt`). New checks include absent task,
nonadjacency and reversed order. These are CPU analysis checks, not GPU
synchronization evidence.

Unverified/unfinished: mixed-task timing, interval DP, L-task rewrite and two
GPU calibration fusions. Logical legality alone does not satisfy those gates.
