# Proposed changes to `TileMega_skeleton.md`

Only new, not-yet-settled proposals remain here. Confirmed findings F-1 through
F-21 are incorporated directly into the skeleton.

## Generalize semantic stage formation

- Location: Phase 1 frontend policy after P1.4.
- Proposal: replace the current explicit two-layer Llama rule with a rule table
  over module-stack identity, dataflow cuts, side effects, and TaskBody fusion
  legality. Preserve a diagnostic mode that prints why every task entered a
  stage.
- Evidence: ✅ the explicit rule deterministically maps the V-H graph from 179
  tasks to 24 stages; ⚠️ no second exported architecture has exercised it.
- Confidence: medium.

## Stabilize export-bridge schema independently of torch

- Location: L4 frontend interchange contract.
- Proposal: version each metadata field and add golden archives from multiple
  supported torch releases. Keep all private API access in one Python adapter
  and reject unknown versions rather than guessing.
- Evidence: ✅ torch 2.13 `_guards_code` is handled and four guards import
  correctly; ⚠️ no other torch release has been tested.
- Confidence: high for the need, undecided support window.
