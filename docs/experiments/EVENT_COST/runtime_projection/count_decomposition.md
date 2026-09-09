# A2 exact counting: decomposition experiments

These are CPU derivation experiments, not GPU correctness processes and
not evidence that A2's complete matrix is closed.

## Methods and evidence

All methods retain symbolic seq and the requested past domain. The two
experimental projection switches default OFF until the full matrix has
been evaluated. Neither samples theta to replace a symbolic function.

- `RuntimeProjectionOptions::partition_worker_counts` partitions the
  event image by the finite worker id, counts each disjoint one-dimensional
  event image, then sums them. `TILEMEGA_PROJECTION_PARTITION_WORKERS=0`
  preserves joint (worker,event) counting as a control.
- `QuasiPolynomial::Sum`, under `TILEMEGA_BALANCED_QP_SUM=1`, scales identical
  exact QPs and combines the remaining terms in a balanced tree. Only
  integer/rational symbolic additions are reordered, not FP64 prices.
- `RuntimeProjectionOptions::split_count_periods` invokes the existing
  ISL exact `isl_pw_qpolynomial_split_periods` operation before combining
  counts. Its limit is the actual grid size. The bundled implementation at
  `third_party/barvinok/isl/isl_polynomial.c:4814` splits a domain when a
  floor has at most that many values. This is not numerical root search.

| scope / method | result |
|---|---|
| gqa2 split1, symbolic seq/past, original joint counts | ✅ 129.296 s; 15 cells exact |
| gqa2 split2, symbolic seq/past, original joint counts | ✅ 167.112 s; 15 cells exact |
| gqa2 split4, symbolic seq/past, original joint counts | ⚠️ 600 s timeout |
| gqa2 split4, concrete past0 or past3, original joint counts | ⚠️ each 600 s timeout |
| gqa2 split4, past0, worker partition only | ⚠️ 600 s timeout |
| gqa2 split4, past0, partition + balanced sum | ⚠️ 600 s timeout |
| gqa2 split4, past0, partition + balanced sum + period split | ✅ 157.920 s; all 5 seq counters exact |

The final pilot checks 10 task_refs/waits values against independent
repaired-runtime logs. Those logs come from the ongoing fresh-process
matrix; this CPU comparison is **not** another synchronization test.
Raw folders are `repaired_matrix`, `repaired_k4_p0`, `repaired_k4_p3`,
`partitioned_k4_p0`, `balanced_partition_k4_p0`, and
`period_partition_k4_p0`. Their manifests identify tool hashes and commands.
Timeouts are not cardinality mismatches. The first original runner did not
persist the timed-out third command's partial output; later runners do.

✅ `partition_stack.txt` / `partition_stack_deep.txt` locate the partition
experiment inside `isl_pw_qpolynomial_union_add_` → `QuasiPolynomial::Sum`.
This motivated reducing repeated piecewise-domain simplification. Attaching
the debugger briefly paused that process, so timings are exploratory, not
paired performance claims. `unpartitioned_stack.txt` has mismatched on-disk
debug symbols after a rebuild and is **invalid evidence**; do not use it.

The full symbolic-past period-split matrix is running separately in
`period_symbolic_matrix`; the one positive pilot does not certify it.
Worker partition alone did not solve the timeout, so it remains opt-in.

## Validation detour: structural zero is not functional zero

✅ A new test initially failed `SemanticallyEqual` for
`floor((S+3)/4)+floor(S/7), 0<=S<=63` versus ISL's split representation.
After subtraction the only remaining piece is that expression on **S=0**.
It is identically zero there, but the existing `is_zero` structural test
does not simplify it. This is not a changed count or relaxed expectation.

The corrected test checks the full difference function's global min/max
are both exactly zero, in addition to all 64 individual values, for split
limits 1,4,16,64. `period_isl_diagnostic.stderr` preserves the initial
assertion; `period_isl_range.stderr` preserves the full-domain zero proof.
No production equality rule was weakened. The pre-existing semantic-equality
API's incompleteness is recorded, not claimed repaired.

✅ Invalid split limit is rejected with reference count before=0/after=0;
normal completion reports `ISL_CONTEXT remaining=0`. Projection unit tests
cover both worker-count strategies, both period settings, both split orders
and kappa=0,1,2. Full CTest after these changes is **29/29**, and check-policy
passes (`COST_MODEL/round5_period_ctest.txt`, `round5_period_policy.txt`).
These logs predate the subsequent A3 local-reduction addition.

## Complete repaired symbolic matrix

✅ Partition + balanced sum + exact period splitting completed both models
at split1/2/4/8/16 with **S and P left symbolic**. Evaluation at the fifteen
requested seq/past cells per query yields 150 cells, then the independent
7500-process checker matches **15000/15000** task-ref/wait counters.
The first batch was interrupted after query eight; its stale status and raw
files are preserved, with `period_symbolic_matrix/recovery.md` explaining the
boundary. The two missing queries ran in `period_symbolic_continuation/`.
All ten queries exit0 and explicitly report zero ISL references.

| model | split1 | split2 | split4 | split8 | split16 |
|---|---:|---:|---:|---:|---:|
| gqa2 seconds | 201.808 | 225.274 | 228.430 | 234.100 | 267.866 |
| mha4 seconds | 306.536 | 344.942 | 361.338 | 388.242 | 409.745 |

These are CPU completion times, not paired speedup claims; independent CPU
work overlapped the continuation. The frozen executable hash and exact
commands are in each batch manifest. No unpartitioned/partition-only timeout
was erased, no numeric sampling replaced a symbolic query, and the controls
remain opt-in. Other ownership/variant GPU gates remain open.
