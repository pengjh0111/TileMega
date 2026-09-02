# Proposed changes to `TileMega_skeleton.md`

Only new, not-yet-settled proposals remain here. Confirmed findings F-1 through
F-26 are incorporated directly into the skeleton (see §1.5.1 for the residual
technical debt those findings left open, which is tracked there rather than
here — this file is for changes to the skeleton's design text, not a general
TODO list).

## Resolved and removed from this file

"Generalize semantic stage formation" (the P1.4 explicit two-layer Llama rule)
is resolved: `lib/Frontend/ModelPlan.cpp` replaced it with a declarative
pattern over the decoder-layer dataflow shape — no parameter names and no ATen
target literals — and two structurally different models (2-layer GQA, 4-layer
MHA) pass through it in both their composite and Core ATen forms, generating
byte-identical code (`docs/experiments/SEMANTIC/`). See F-25 and `TileMega_skeleton.md` §1.5.1 for what this does and does not
generalize over — it is not moved here because the remaining gap (a
structurally distinct model family needing a new `ModelPlan.cpp` rule) is
recorded as residual debt, not as an open skeleton-design question.

## TaskBody ABI: CTA-to-task ownership map

- Location: §5.3 TaskBody ABI contract.
- Proposal: give every TaskBody specialization a way to publish, for a given
  `blockIdx.x`, which task-space coordinate(s) it owns this launch — e.g. a
  `constexpr` or device-queryable `OwnedTask(Params, stage, blockIdx.x)` — so
  that `CouplingGraphToCUDA::Lower` can prove `StageDependency::Map::kIdentity`
  is sound for a coupling whose derived `C` is the identity relation, instead
  of always falling back to `Map::kAll`. Today the fallback is I2-safe (F-24)
  but is why L2 events are consistently slower than the L1 barrier they
  replace (median `1.16x`-`1.36x` across the two accepted models); this ABI
  change is what a real fine-grained-event performance win needs.
- Evidence: ✅ `Map::kIdentity` is already read by `WaitDependencies` and
  `ActiveBlocks` in `ModelHarness.cuh` — the consumer side of this contract
  exists and is untested only because the producer side (the ABI fact this
  proposes) is never emitted; ✅ `isIdentityRelation`-style structural
  identity detection on `C` was prototyped and works, but is insufficient by
  itself (a semantically identity `C` does not prove `blockIdx.x` is the same
  physical CTA in two independently compiled TaskBody specializations without
  this ABI fact); ❌ no measurement yet of the performance win this would
  unlock, since it isn't implemented.
- Confidence: medium — the mechanism is clear, but this changes every
  TaskBody's launch contract, and the four existing dispatch shapes (one CTA
  per row/token/head/tile, grid-stride tile loops in RoPE/KVAppend/
  Elementwise, one CTA per GEMM tile_n) would each need their own ownership
  rule; none of that is designed yet, only motivated.

## Stabilize export-bridge schema independently of torch

- Location: L4 frontend interchange contract.
- Proposal: version each metadata field and add golden archives from multiple
  supported torch releases. Keep all private API access in one Python adapter
  and reject unknown versions rather than guessing.
- Evidence: ✅ torch 2.13 `_guards_code` is handled and four guards import
  correctly on the 2-layer model, and the same adapter now also serializes the
  4-layer MHA model unchanged (F-25); ⚠️ no other torch release has been
  tested.
- Confidence: high for the need, undecided support window.
