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

"TaskBody ABI: CTA-to-task ownership map" is resolved: every dispatched TaskBody declares `Ownership` (`TaskOwnership`, `include/tilemega/Codegen/tasks/TaskBase.h`), `ActiveBlocks` dispatches to it, and windowed edges depend on it (F-34, F-58, F-68). Its stated motivation — that `kAll` made L2 slower than L1 — was superseded by F-32, F-35, F-79 to F-86 and F-126.

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
