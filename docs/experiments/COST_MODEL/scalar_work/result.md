# A6 scalar work consumption and differences

✅ Source: `lib/Solver/ScalarTaskWork.cpp:10` composes semantic element
accesses with `ProjectScalarTaskOwnership` (`RuntimeProjection.cpp:36`).
`TaskModel.cpp:84` supplies those QPs, the audited arithmetic signature, and
TaskBody resources to `CostModel::TaskCostNs` (`CostModel.cpp:408`).
The actual `TaskStageNs` / Evaluate / DP paths consume this entry; the old
`NonGemmStageNs` switch is retained only as an explicit historical control.
`TILEMEGA_UNIFIED_TASK_COST` is independently selectable (currently default
OFF pending the complete A6 acceptance record). `TILEMEGA_SCALAR_TASK_WORK=0`
rejects the explicit derived-scalar path, not a silent fallback.

## What is derived

Runtime task count is domain cardinality of the projected write relation.
Reads of the same tensor are unioned before barvinok counting. RoPE includes
both partner halves and unique frequency reads; attention includes causal K
but all V loads, matching the current body. KV element ownership copies both
append and retained prefix ranges; tile ownership preloads the prefix on the
host, so its task does not claim those loads. Writes union both regions in
the same logical output space; cardinality is not their overlapping task-count
sum. See the per-stage `read_elements`, `write_elements`, `tasks` columns.

Arithmetic remains the single semantic schema. Attention supplies
`4*total*write_elements` FP32 SIMT FLOPs and `total/head_dim` exponentials
per output element; it does not use the tensor-core lane. These are declared
semantic counts, not a claim of measured issued-instruction counts.

The TaskBody's load/reduction/publication/store structure is exposed as a
DAG in `ScalarDataflow.h:47`. Memory depth is its longest dependent global
memory phase chain, not a per-kind latency integer in CostModel. The actual
RMS block-reduction halving loop gives one publication plus log2(threads)
barriers: eight at 128 threads, nine at 256. Attention has two publications,
as its current RunTask does. This is an explicitly audited structural trait,
not automatic compiler extraction or a latency fit.

Scalar stages use the same residency-sized waves and tail occupancy rule as
the collective model. Because predication/tails vary by task coordinate,
each wave costs its largest task at that wave's occupancy. Each work quantity
stays a QP until coordinate binding and theta substitution. Nine lanes and
max are retained. Scalar BF16 tasks may use the scalar shared-memory pipe;
the disabled BF16 GEMM ldmatrix lane is not reactivated.

## BF16 differences (ns)

Seq128/past3, element ownership, residency2. Every runtime scalar stage in
both models is listed separately in `verified/*s128_p3.tsv`, including all
layers. The repeated semantic classes have the same derivation below:

| Stage class | Model | Old | Derived | Ratio | Reason |
|---|---|---:|---:|---:|---|
| RMSNorm | both | 658.559063 | 1584.633492 | 2.406213 | Actual three dependent memory phases and eight barriers; old two/two |
| RoPE Q | both | 329.214500 | 285.225718 | .866383 | Unique partner/frequency reads, dtype bytes, output tail and arithmetic max |
| RoPE K | gqa2 | 271.963699 | 256.379564 | .942698 | Same derivation; two KV heads instead of four |
| RoPE K | mha4 | 329.214500 | 285.225718 | .866383 | Four KV heads |
| KV append K/V | gqa2 | 265.994949 | 278.815462 | 1.048198 | Physical appended plus retained-prefix addresses, not a padded chunk-byte constant |
| KV append K/V | mha4 | 531.989897 | 525.579641 | .987950 | More heads/task waves; same prefix-union derivation |
| Attention | both | 15064.537878 | 7185.677225 | .476993 | Two-byte storage, causal K, actual two publications, SIMT FLOPs; lanes take max, not sum |
| SwiGLU | both | 1190.418898 | 1038.338769 | .872247 | Actual output/read sets and SiLU/multiply arithmetic; dtype bytes and lane max |

Tile ownership changes task grouping and frequency-read reuse; its full
per-stage tables are in the same TSVs, not inferred by scaling element
ownership. Prefix-dominated seq1/past512 and BF16 seq512/past512 are covered.
The new attention price **decreases** despite adding FLOPs: correcting the
old four-byte assumption and memory/barrier overcounts outweighs that term
here. We do not claim the missing-FLOP fix must increase every price, nor
that these predictions establish GPU timing accuracy at long contexts.

### Long-context diagnostic (A7.3's independent FLOP/price diagnostic)

✅ BF16 seq512/past512, original plan, residency2. Element ownership:

| Model | Old total ns | New total ns | Old attention ns | New attention ns | Old / new attention fraction |
|---|---:|---:|---:|---:|---:|
| gqa2 | 1188769.413264 | 725248.767403 | 853234.258052 | 384459.779592 | .717746 / .530107 |
| mha4 | 2555096.325426 | 2850322.868866 | 1706468.516103 | 1974947.404051 | .667869 / .692886 |

The model's live footprints are 41943040 and 94371840 bytes, respectively;
the unchanged SDCM cache model gives hit probabilities 1 and
.49883845476461874. Unlike the old scalar template, derived task pricing
includes the DRAM lane. This explains why the long-context models need not
move in the same direction. Tile ownership fractions are .532952 and
.695156; complete logs under `attention_fraction/` retain both variants.
Adding footprint/cache diagnostics produces byte-identical per-stage TSVs
to the preceding run. These are predictions, not measured occupancy or GPU
timings, and do not claim A7 chunk implementation/acceptance.

## Independent checks and failures retained

✅ Four BF16 (seq,past) pairs × two models plus three FP32 pairs × two
models = 14/14 legal CPU cases, each in both ownership modes. For every
scalar stage, runtime projection and write-domain task counts are identical.
At first/middle/last task coordinates, a separate TaskBody index emulator
checks the **exact write set** and unique input-read counts. Logs report
their actual task-point check counts and `ISL_CONTEXT remaining=0`.
This is CPU index emulation, not GPU address instrumentation or a new
50-process synchronization claim.

`run_scalar_work.py` saves immutable per-case hashes and refuses to replace
existing logs; resume validates the frozen executable. Two attempted inputs
were rejected before pricing and retained: BF16 past2048 exceeds its exported
512 bound; historical FP32 past512 exceeds its [1,8] domain. No domain was
widened. BF16 long-context uses prompt's alternative seq512/past512;
FP32 regression cases stay in the original [1,8] domain.

✅ `tools/tilemega-scalar-error-probe.cpp` exercises 13 rejection paths,
all before=0/after=0: ownership width/rank/kind/tile/rotation, missing
signature/flow, wrong coordinates, bad reduction width, cyclic/empty flow,
unbound theta, missing semantic input. `errors.txt` includes each diagnostic.
The current suite is 33/33; policy passes and target-audit is 5 targets,
0 failures. The pre-existing `build/` MLIR-OFF cache remains invalid;
these successful commands use `build-portable/`.

❌ First scalar pricing reused the GEMM alpha+beta*tile-area fit. Its BF16
intercept is -344.946 ns; on small scalar tasks it yielded negative prices,
which a zero-initialized wave max hid as zero. The invalid results are
retained as `invalid_collective_setup.*`. The fix is structural: a SIMT
TaskBody has no collective accumulator-initialization phase. Its arithmetic,
memory phases and barriers are now charged directly; the GEMM fit remains
unchanged in the GEMM branch. Nonpositive/nonfinite scalar task prices throw,
and the wave maximum starts at -infinity, so this mistake cannot silently
turn into zero again. No fitted expected value or tolerance was changed.

Ranking controls: [unified_rank](../unified_rank/result.md). Full per-stage
GEMM bit acceptance remains separate: [GEMM gate](../gemm_price_gate/result.md).
