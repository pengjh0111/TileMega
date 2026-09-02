# TileMega development conventions

Rules that apply to every change in this repository. They are conventions, not
suggestions; follow them without being asked again.

## Commit messages

One line, no body. Format:

```
<area>: <imperative summary>
```

`<area>` is the subsystem the change belongs to, lower case: `analysis`,
`codegen`, `backend`, `dialect`, `runtime`, `docs`, `test`, `build`,
`experiments`. Keep the whole line under 72 characters.

```
analysis: split L-sem out of the operator graph
experiments: add the partition oracle sweep
docs: record the oracle result in FINDINGS
```

Do not add a detailed description, a bullet list of what changed, a rationale
paragraph, or trailers. The diff and `docs/FINDINGS.md` carry that. Do not add
co-author or tool-attribution trailers of any kind.

## Comments in implementation code

Comment the non-obvious: an invariant, a correctness argument, a measured
number that justifies a choice, a deliberate deviation. Do not narrate what the
code already says, do not restate a function's name in prose, and do not leave
running commentary about the editing process.

Prefer one short comment on the surprising line over a paragraph above the
function. Headers may carry a longer block when it documents an ABI or a
skeleton reference; implementation files should stay lean.

## Evidence

Claims in `docs/` and in commit messages are marked by how they were
established: verified (measured or executed here), stated (asserted by a
source, not re-checked), inferred (reasoned, not observed). Any synchronization
or race claim is backed by at least 50 fresh processes with the pass rate
reported. Never move an expected value to match an implementation; record the
difference instead.
