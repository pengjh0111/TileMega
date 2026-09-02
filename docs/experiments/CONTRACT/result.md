# Part 5 — the implementation contract and its consistency check

Reproduce with `build-portable/impl_contract_test` (CPU only). Every check
below is paired with the mutation it must reject; a check with no failing
counterpart is not evidence.

## 5.1 What an implementation declares

`tilemega.implementation` carries exactly the fields §5 asks for — this is
the §2.7 score matmul at the granularity `Tm = Tn = 128, Tk = 16`:

```mlir
tilemega.implementation @impl_qkv_17_ctl for @score {
  access = [
    #tilemega.access_map<{coordinates = ["h", "q", ""], operand = "q_rope",
                          spans = array<i64: 1, 128, 128>}>,
    #tilemega.access_map<{coordinates = ["h", "kv", ""], operand = "k_cache",
                          spans = array<i64: 1, 128, 128>}>],
  alignment = array<i64: 1, 1>, arch_required = 80 : i64,
  backend = "cutlass.sm80_cpasync.simt_f32", cluster = array<i64: 1, 1, 1>,
  smem_bytes = 49536 : i64, stages = 3 : i64, threads = 256 : i64,
  tile = array<i64: 128, 128, 16>}
```

(✅ verified: printed by the test itself, not transcribed by hand.)

`regs_est` is **absent**, not zero. CUTLASS's `constexpr` traits carry no
register count (Part 4 §4.3), so an always-present field could only hold a
guess; the attribute appears once a tier-3 ptxas log fills it.

## 5.2 The solver picks `(g, impl)` as one decision

`SelectImplementation` (`lib/Solver/ImplementationContract.cpp`) ranks the
tier-1 legal set with the tier-2 key, and the tile it picks is what binds the
`g` symbols the task space's access maps are written in. So the contract's
declared spans are *the ones the chosen granularity induces*, not a second
independently authored copy — the failure mode Part 5 exists to prevent
cannot be introduced by the emitter, only by a later edit or a hand-written
implementation. Measured on the score matmul with problem (4, 512, 128):

```
selected g: 16x16x16 s2, impl cutlass.sm80_cpasync.simt_f32
            threads=256 smem=4352 arch=80
  operand q_rope   h:1 q:16 *:128
  operand k_cache  h:1 kv:16 *:128
```

(`*` is an axis the task covers whole — the contracted `d` axis.)

## 5.3 The checks, each with the mutation it catches

✅ verified, `impl_contract_test`, exit 0:

| mutation | rejected with |
|---|---|
| operand coordinates transposed | `impl_qkv_17_ctl operand q_rope axis 1: declared coordinate '', task space indexes it by 'q'` |
| tile span halved | `impl_qkv_17_mistiled operand k_cache axis 1: declared span 64, task space touches 128` |
| declared smem restated (−4 B) | `declared traits contradict cutlass.sm80_cpasync.simt_f32: smem 4348 != 4352` |
| one operand dropped | `contract declares 1 operands, the task space has 2` |
| transposition, **through the IR** | `error: 'tilemega.implementation' op impl_qkv_17_transposed operand q_rope axis 1: declared coordinate '', task space indexes it by 'q'` |
| tile mismatch, **through the IR** | `error: 'tilemega.implementation' op impl_qkv_17_mistiled operand k_cache axis 1: declared span 64, task space touches 128` |

The last two are the same defects reaching `mlir::verify()` on a module whose
`tilemega.task_space @score` carries `index_map`; each is erased afterwards
and the module verifies again, so the rejection is attributable to the
mutation and not to leftover state. The smem row mutates the contract the
solver picked (`16x16x16 s2`, 4352 B), hence the smaller numbers.

### Why the transposition negative is run at Tm = d = 128

The transposition test deliberately uses the §2.7 granularity, where the two
axes a swap exchanges are **the same width**: `q_rope` is `(h, q, d)` with
spans `1, 128, 128`. The test asserts the equality before mutating —

```cpp
REQUIRE(control.operands[0].span[1] == control.operands[0].span[2]);
REQUIRE(transposed.operands[0].span == control.operands[0].span);
```

— so a swap changes *no extent at all* and the span comparison provably
cannot be what rejects it. Only the driving-coordinate comparison can. Run at
the solver's own pick (`16x16x16`) the spans differ (16 vs 128) and the
negative would have passed for the wrong reason; that is the version this
test started as, and it is why the assertion is there.

This is the F-17 shape exactly: CUTLASS's B operand is logically `(N, K)`, and
getting it the other way round once produced 6143 mismatching elements. Two
things make it worth a verifier rather than a comment:

- the wrong access *range* is invisible in the task graph — both forms read a
  128×128 rectangle of the same tensor;
- under L1 the global barrier between stages hides a wrong range whenever the
  producer happens to have finished anyway. It becomes a probabilistic failure
  only once L2 replaces the barrier with per-producer events, which is the
  worst possible time to discover it.

## 5.4 The verifier's own limit

[!] The IR-level check runs **only when the task space carries `index_map`**.
`tilemega.task_space` declares it optional, and the FX importer does not emit
it yet: the importer builds task spaces from FX nodes and never constructs an
`OperatorNode`, so it has no derived access relation to serialize. On a task
space without `index_map` the verifier still rejects a restated cost
(`VerifyTraits`) but the access pattern is **unfalsifiable** — which is
precisely the state F-17 was found in. Wiring the importer through
`BuildReadMap` is the remaining work; it is recorded as a gap rather than
hidden behind a check that silently passes.

The trait half has no such hole: `VerifyTraits` recomputes threads, shared
memory, alignment and minimum architecture from the backend's own closed form
for the declared tile, and that closed form is `static_assert`ed against the
instantiated CUTLASS collective (Part 4 §4.2). An implementation may choose a
shape; it may not restate what the shape costs.
