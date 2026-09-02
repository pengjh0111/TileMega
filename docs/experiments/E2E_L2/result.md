# L2 fine-grained events vs. L1 global barrier (P3.5 acceptance)

Evidence labels: ✅ executed/observed; ⚠️ scope limitation; ❌ conjecture.

This experiment does not re-run the frontend/codegen pipeline itself; it reads
the L2 numbers off the same generated binaries produced by
`docs/experiments/E2E_GEN/run.sh` (2-layer GQA) and
`docs/experiments/P3_GENERALIZATION/run.sh` (4-layer MHA), both of which now
emit an `l2_ms`/`l2_over_l1` line and an `l2` hash alongside `l05`/`l1`. Raw
logs are copied into `raw/` under this directory; the source-of-truth raw data
(including the 50-fresh-process logs) stays in `E2E_GEN/raw/` and
`P3_GENERALIZATION/raw/` where the pipelines that produced it live.

## What L2 is here

`CouplingGraphToCUDA::Lower` emits one `StageDependency` per coupling whose
producer stage precedes its consumer stage. `tilemega_stage_kernel` waits on
those specific producer-stage event counters (`WaitDependencies` in
`ModelHarness.cuh`) instead of one grid-wide barrier per stage
(`GridBarrier`, which L1 uses unconditionally). Both paths run inside the same
compiled binary and are selected by the harness at the top level, so L1 and L2
share every TaskBody, every table, and every numeric code path — only the
synchronization primitive differs.

Every dependency currently uses `StageDependency::Map::kAll`: a consumer CTA
waits for the producer stage's entire launch grid to arrive, not just the
producer CTAs the coupling `C` actually identifies. This is intentional and
recorded in `lib/Codegen/Codegen.cpp` and skeleton §1.5.1: proving that
`blockIdx.x` names the same tile in two independently compiled TaskBody
specializations requires an explicit CTA→task ownership entry in the TaskBody
ABI, which does not exist yet. `kAll` is the I2-safe relaxation
(`C' ⊇ C`); `kIdentity` (the tight path) is wired into `WaitDependencies` and
`ActiveBlocks` but never emitted by the generator in this round.

## Correctness

| Model | L2 vs L1 | Fresh processes |
|---|---|---|
| 2-layer GQA (179 task / 222 coupling / 24 stage) | 0 mismatch, max_abs `0`, hash `5245714bc5d3ab4d` for l05/l1/l2 | ✅ 50/50, 0 hang, 0 error |
| 4-layer MHA (355 task / 444 coupling / 60 stage / 11 guard, `kv_heads == heads`) | 0 mismatch, max_abs `0`, hash `fd15fa2e89cdb915` for l05/l1/l2 | ✅ 1/1 (single-process acceptance run; see `docs/experiments/P3_GENERALIZATION/result.md`) |

L2 is bitwise identical to L1 on both models. This is expected: `kAll`
dependencies impose a superset of the synchronization L1's global barrier
already provides, so the two schedules can only differ in *when* a CTA is
allowed to proceed, never in what it reads — no L1 barrier point is ever
crossed by an L2 wait early, because `kAll` always waits for the same set of
producer CTAs the barrier would have waited for.

## Performance

Medians from the 50-fresh-process run (2-layer GQA) and the single acceptance
run (4-layer MHA):

| Model | l1 median | l2 median | l2/l1 |
|---|---:|---:|---:|
| 2-layer GQA | 1.095856 ms | 1.295360 ms | **1.182×** |
| 4-layer MHA | 2.160576 ms | 2.928160 ms | **1.355×** |

L2 is slower than L1 on both models, not faster. This is the direct, expected
consequence of `kAll`: per-edge events replace one barrier per stage with
*several* per-edge waits per stage (one per incoming coupling), each of which
still waits for a whole producer grid — so L2 pays strictly more
synchronization events than L1 for the same coverage. The event *semantics*
(§2.3's `EventTensor(e) = image(C_κ)`, `wait(e) = |C_κ⁻¹(e)|`) are correct and
machine-checked (`event_synthesis_test`, `containment_test`), but realizing a
performance win requires the `kIdentity` fast path, which needs the TaskBody
ABI extension noted above. That is out of this round's scope (the task
explicitly deprioritizes performance optimization behind acceptance B); it is
recorded here and in skeleton §1.5.1 rather than hidden.

## Resources

Generated build now links three device kernels (`tilemega_l1_kernel`,
`tilemega_l2_kernel`, `tilemega_stage_kernel`) against the same `TaskSmem`
union and the same `ModelSpec` tables; ptxas reports 168 (l1) / 153 (l05) /
142 (stage-loop) registers for the 2-layer GQA build, zero spills, 49,536 B
dynamic shared memory — unchanged from the L1-only baseline in
`docs/experiments/E2E_GEN/result.md`. Adding L2 did not change L1's or L0.5's
resource footprint because they remain separate kernel entry points sharing
one TaskBody/table implementation, not a new specialization.

## Conclusion

L2 event synthesis is implemented, verified correct (bitwise-identical output,
50/50 fresh processes on the primary model, a second structurally different
model passing independently), and generalizes across both accepted models
without any model-structure constant in the generated source. It is currently
a *correctness* deliverable, not a *performance* one: the conservative `kAll`
relaxation the generator emits is strictly safe under I2 but not the tightest
event graph §2.3 allows, and today it is measurably slower than the L1 global
barrier it replaces. Closing that gap is scoped to the TaskBody ABI work
recorded as residual debt, not to this round.
