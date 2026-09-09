# Round 5 A3–A5: work domains and declarations (in progress)

This is not A6 acceptance. No new unified price or ranking is claimed.
The original GEMM per-stage bit-pattern gate remains unchanged.

## A3 counterexample, decision and validation

✅ `lib/Analysis/TaskWork.cpp:29` constructs element access relations from
`BuildReadMap`/`BuildWriteMap`, then counts them with barvinok. Physical
accesses intersect tensor origins/extents; nominal accesses retain the full
allocated tile. No operator-kind FLOP/byte formulas appear in this derivation.
Reduction axes come from iteration dimensions absent from the semantic output
index map (`TaskWork.cpp:103`), not from a declared reduction-axis list.

The original proposed equality mixes two domains. For BF16 GEMM with
M=4, N=K=512, tile=128×128×16, a task physically reads
4×512 activation elements plus 128×512 weight elements. Dividing their
two-byte volume by 32 mainloop iterations gives **4224 B/iteration**.
The nominal tile is (128+128)×512 elements, or **8192 B/iteration**,
which is what historical `MainloopBytes` prices.

| seq | physical B/iteration | nominal B/iteration | legacy MainloopBytes |
|---:|---:|---:|---:|
| 4 | 4224 | 8192 | 8192 |
| 128 | 8192 | 8192 | 8192 |
| 512 | 8192 | 8192 | 8192 |

✅ User approved continuing with two explicitly named domains, recording the
reasoning: physical R/W for actual accessed elements and future fusion;
nominal work for reproducing historical collective pricing. This is **not**
a tolerance change, and does not relabel padded elements as physical loads.
These are unique-element footprints, not measured cache-transaction counts.
The future unified evaluator must choose its domain explicitly, preserving
the A6 historical GEMM bit gate.

Two implementation corrections were needed during the probe:

1. `QuasiPolynomial::Sum` seeded `{0}` in scalar space. Adding per-task
   `[m,n] -> work` then failed with an ISL space mismatch. It now seeds the
   first operand; the empty sum is still scalar zero. Permanent coordinate
   sum/empty sum tests are in `test/unit/isl_relation_test.cpp`.
2. Production and reference GEMM semantics omitted external weight reads.
   `SemanticLifting.cpp:222` and `ReferenceModels.cpp:94` now add the actual
   `(n,k)` weight operand. Its empty producer creates no new dependency edge.
   `TILEMEGA_COMPLETE_GEMM_READS=0` retains the historical omission solely as
   a control. A missing operand is not compensated with a solver constant.

✅ Repeated operands naming the same tensor are unioned before cardinality
(`TaskWork.cpp:77`); different tensor identities are disjoint. Inconsistent
layouts for the same identity are rejected, not approximated. The probe
checks repeated-read deduplication and missing-identity/incompatible-layout
errors, with zero surviving ISL references.

✅ Reproducible command:

```sh
TILEMEGA_ISL_AUDIT=1 build-portable/tools/tilemega-task-work-probe \
  docs/experiments/SEQSCAN/raw/export/gqa2.json \
  docs/experiments/SEQSCAN/raw/export/mha4.json
```

Raw `task_work_domains.tsv` / `.stderr`: **70/70 gqa2 + 140/140 mha4**
GEMM count/nominal-read checks at seq=1,4,128,512,2048 using production export
semantics and launch granularity. The manual tail counterexample remains
visible. `task_work_lifting.stderr` also reports `ISL_CONTEXT remaining=0`;
the pre-existing semantic comparison passes without changing its expected
differences. `task_work_isl.stderr` covers the new primitive regressions.

⚠️ Scope: these 210 cells are not 1077 configurations, not a CG serialization
round trip, and not A6 prices. Non-GEMM semantic reads (including RMSNorm
weights and RoPE partner/frequency accesses), split per-task reduction
extent, complete CG work attributes and their solver consumption still need
completion. `TILEMEGA_TASK_WORK=0` rejects derivation rather than fabricating
fallback values. A3 remains open; dependent A6/B gates remain closed.

### Split-local reduction work

✅ `TaskWork::task_reduce_extent` is distinct from the complete semantic
`reduce_extent`. `TaskWork.cpp` identifies a reduction axis by its absence
from the result index map, matches that axis to the semantic operand index,
and reads its span from the instantiated access relation. Thus a split
contribution uses its actual chunk span; an unsplit reduction uses the full
span. No GEMM-kind formula is inserted into work derivation. Non-unit or
inconsistent indexing is rejected explicitly rather than approximated.

`task_work_local.stderr` records 25/25 split×seq checks (split=1,2,4,8,16;
seq=1,4,128,512,2048), with K=512 kept as the complete operator extent and
512/split as the local extent. The existing 210 production-export checks
and two malformed-input checks still pass; `ISL_CONTEXT remaining=0`.
This closes the original prototype's full-K-as-local-K error, not the
remaining A3/A6 integration. A chunk not divisible by the collective tile_k
issues a padded final iteration, which needs its own explicit work domain.

✅ `TaskWorkOptions::reduction_tiles` now accepts the inner tile supplied by
implementation traits. Only the nominal read relation is padded; the
physical relation and `task_reduce_extent` are unchanged. No operator-kind
formula supplies the padding. `nominal_task_reduce_extent` records issued
work, which can exceed a contribution's actual reduction interval.
For K=1536/split16/tile_k64, the contribution spans 96 K elements, whereas
two nominal iterations span 128. The test proves nominal BF16 mainloop
bytes/iteration = 32768 and physical reads unchanged. `task_work_inner.*`
preserve this counterexample and the five error-path checks (zero residual
ISL references). The complete work gate passed **4308/4308 configuration
groups** (1077 × two models × BF16/FP32), with **1,357,020 individual work
bit checks** across all GEMM stages and seq=1,4,128,512,2048. It checks task
counts, nominal mainloop bytes/iteration, and nominal iteration counts using
`memcmp` of doubles in `tools/tilemega-task-work-gate.cpp`. Both dtypes use
the complete archived FP32 1077-shape universe; no new BF16 runtime success
or ranking is inferred from that configuration list. Raw data and independent
coverage/hash verification are `task_work_full_gate.{tsv,stderr,json}` and
`verify_task_work_gate.py`; all four group markers and zero ISL references
are required. **This is not A6's CostBreakdown gate.** Combiner work and production CG
serialization/consumption remain open.

### Normalization scale access

✅ `SemanticLifting.cpp` now records the actual RMSNorm scale operand from
`stage.operands[1]`, indexed by the output's hidden coordinate. The reference
semantics records the same weight read. `TILEMEGA_COMPLETE_NORMALIZATION_READS=0`
retains the old omission as an independent control, not a pricing fallback.
As with GEMM weights, the input has no task producer and creates no extra
event edge. `task_work_norm.*` checks 20 gqa2 + 40 mha4 cases: one task per
token, two widths of unique reads (activation and scale), one width of
writes, and reduction span=width. The semantic comparison still passes
without changing expected differences; both tools report zero ISL references.

✅ Both models, default and ownership-variant generation, remain **4/4
byte-identical** to the preserved pre-refactor compiler. Raw hashes and logs
are in `EVENT_COST/runtime_projection/norm_codegen_control`. This verifies
that adding the external read did not alter generated dependencies; it is
not a GPU test or proof that incomplete RoPE reads are already fixed.

✅ After the normalization-read change, the rebuilt CTest executables pass
29/29 and policy passes (`round5_norm_ctest.txt`, `round5_norm_policy.txt`).
This evidence predates the independent FP32-partial calibration preparation.

## A4 arithmetic declaration audit

The single table is `lib/Analysis/OpArithmetic.cpp:9`. Work is represented
as an integer QP numerator and a concrete rational denominator until numeric
evaluation; widths are fixed model/implementation axes, seq remains symbolic.
FMA counts as two FLOP, scalar arithmetic as one; comparisons and conversion
are excluded. Semantic DAG work does not silently include TaskBody replication.

| Signature | FLOP/output | transcendental/output | Basis |
|---|---:|---:|---|
| GEMM | 2K | 0 | K multiply-adds; BF16 MMA, existing FP32 SIMT |
| Attention | 4T | T/W | QK and PV each 2T; T exponentials amortized over W |
| RMSNorm | (4W+1)/W | 1/W | squares, sum, mean/epsilon, normalization/weight |
| RoPE | 7/2 | 1 | angle multiply, four products, two adds per pair; sin/cos |
| SiLU | 3 | 1 | negate/add/divide; exp |
| mul / add | 1 | 0 | one scalar operation; unit residual coefficient |
| SwiGLU | 4 | 1 | SiLU followed by gate/up product |
| KVAppend | 0 | 0 | data movement only, with explicit reason |
| sum | R−1 | 0 | reduction DAG; zero-seeded implementation overhead separate |
| Softmax | (3W−1)/W | 1 | subtract, sum, divide; exp |
| LayerNorm | (8W+1)/W | 1/W | two-pass mean/variance plus affine normalization |
| GeLU tanh | 8 | 1 | explicit tanh approximation DAG |
| MoE router | (3W−1)/W | 1 | softmax arithmetic; selection comparisons separate |

✅ `op_audit.txt` records 14 complete declarations and four rejected error
paths (unknown name, missing total, invalid width, missing reason), with
zero ISL references. Four standalone implementations are explicitly absent:
Softmax, LayerNorm, GeLU, and the placeholder MoE body. They are not given
zero prices: `RequireArithmeticImplementation` rejects them.

⚠️ Attention's 4T is dense semantic work, not the exact issued instruction
count of the causal SIMT body. Causal QK bounds, softmax overhead and repeated
reduction work must be accounted for before A6 acceptance. Merely attaching
the `arithmetic` key to CG is not price consumption. The solver-level missing
signature gate awaits A6. Independent switch: `TILEMEGA_OP_ARITHMETIC`.

## A5 resources (implementation in progress)

`TaskResources.h` gives the SIMT bodies their storage, threads and stages
from one trait. `TaskSmem` uses the same types and takes the actual maximum;
no semantic storage is removed. BF16 non-GEMM thread count now reads the
backend's 128-thread trait rather than the legacy FP32 256-thread assumption.
`TILEMEGA_TASK_TRAIT_COSTS=0` / `--legacy-task-traits` retains that control.
CUDA contract compilation has passed; full price/regression comparisons and
final-header GPU controls remain distinct. The ongoing A2 GPU sweep uses
frozen binaries built before these trait-only edits, as its manifest records.

✅ The FP32 traits OFF/ON controls produce byte-identical
`predictions_{gqa2,mha4}.tsv` (1077 rows each) and `summary.tsv` in
`traits_control/f32_{old,new}`. Full-model FP32 ranking remains
rho=.9432/.9421, top1/3/10=1/3/6 for both. This checks the printed price/rank
outputs; it is not A6's per-stage raw-double bit gate.
The BF16 old/new files are also retained, but they use the historical
770/462-row screened subset and **cannot establish BF16 ranking acceptance**.
For their first gqa2 32×16×16s2/split1 point, other_ns changes
7546.62→7126.19 from the thread trait; GEMM/combine/barrier columns are
unchanged. This is a model change, not a measured speedup.

✅ Final headers were compiled into ten split binaries. All ten match the
running sweep binaries' disassembled device instructions and resource
records exactly. See `EVENT_COST/split_order_final_headers/device_comparison.json`
and `compare_device_builds.py`. This is a device-code comparison, not new
fresh-process evidence or a host-byte comparison. The additional host
`max_worker_task_refs` field scans schedule offsets only for reporting.

✅ Full CTest 29/29, policy PASS, target-audit five targets/zero failures;
raw `round5_ctest.txt`, `round5_policy.txt`, `round5_target_audit.txt`.
Two command detours are recorded: the stale `build/` configuration rejects
MLIR=OFF, so verification used the established `build-portable/`; target-audit
does not implement `--help` and interpreted it as a repo path, so the proper
repo-root invocation was rerun successfully. Neither was a GPU correctness
failure. These results predate the later balanced-QP-sum experiment, which
has separate logs and must not inherit this acceptance automatically.
