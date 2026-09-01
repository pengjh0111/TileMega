# V-D — CUTLASS compile-time traits

Evidence labels: ✅ measured; ⚠️ not run; ❌ inference.

The probe instantiated 100 real SM80 cp.async `CollectiveMma` types without
generating kernels and queried `sizeof(Collective::SharedStorage)`, candidate
legality under the sm_89 budget, and `size(TiledMma{})`.

- ✅ Wall time: 17.625790 s for 100 candidates.
- ✅ 79 candidates fit the configured shared-memory budget; 21 did not.
- ✅ MMA size was 128 threads for this candidate family.

Three candidates were then emitted as real kernels:

| Tile | Trait SHM | ptxas SHM | Error |
|---|---:|---:|---:|
| 64×64×32 | 24,576 B | 24,576 B | 0 B |
| 64×96×32 | 30,720 B | 30,720 B | 0 B |
| 96×96×32 | 36,864 B | 36,864 B | 0 B |

✅ Traits can replace real kernel compilation for shared-storage and basic
template-legality pruning in this family, with zero observed shared-memory
error. They cannot predict register allocation, spills, final inlining, or
whole-megakernel occupancy, so survivors still require true compilation.

## Reproduction

```bash
docs/experiments/V_D/run.sh
cat docs/experiments/V_D/raw/traits_compile_seconds.txt
cat docs/experiments/V_D/raw/ptxas_compare.txt
```
