# V-G — shared-memory union and occupancy

Evidence labels: ✅ measured; ⚠️ configuration projection; ❌ conjecture.

## Results

| Kernel | Largest branch | Emitted SHM | REG | CTA/SM |
|---|---:|---:|---:|---:|
| dispatch 2 | 2,048 B | 2,048 B | 10 | 6 |
| dispatch 3 | 3,072 B | 3,072 B | 12 | 6 |
| dispatch 5 | 5,120 B | 5,120 B | 10 | 6 |
| dispatch 8 | 8,192 B | 8,192 B | 10 | 6 |
| nested dispatch | 8,192 B | 8,192 B | 10 | 6 |
| loop-carried, explicit union | 8,192 B | 8,192 B | 23 | 6 |
| loop-carried, escaping separate storage | 8,192 B | 36,864 B | 11 | 2 |

✅ The max rule holds when mutually exclusive TaskBody storage has one explicit
union lifetime, including nested control flow and a value carried beyond the
branch. It degrades to the sum when distinct storage objects remain live and
their pointers escape; the compiler cannot infer mutual exclusion from the
dispatch alone. The union can raise registers, as the loop-carried case shows.

The portable variant bound is not a fixed task count. For target `T`, launch
shape `B`, and variant storage sizes `S_i`, a union is feasible when

```text
max_i(S_i) + fixed_overhead <= T.res.max_dynamic_smem_per_cta
and occupancy(T, B, REG_codegen, max_i(S_i) + fixed_overhead) >= required_CTA_per_SM.
```

Thus sm_90 and sm_120 accept the largest prefix of variants satisfying that
function under their respective 232,448 B and 101,376 B checked-in block
budgets; the answer changes with `TargetSpec`, not merely with task-type count.
⚠️ These two projections were not runtime-measured on migration hardware.

## Reproduction

```bash
docs/experiments/V_G/run.sh
cat docs/experiments/V_G/raw/resources.jsonl
cat docs/experiments/V_G/raw/occupancy.txt
```
