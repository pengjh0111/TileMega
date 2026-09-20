# D-b — top-k quality under §6 C1-c's caliber, six cells

C1-c asks how good the search's own shortlist is, not how well the cost model
ranks:

```
ratio = (fastest measured among the top three)
      / (fastest measured among every candidate the search evaluated)
```

so the denominator needs a buildable source per evaluated candidate.
`tools/tilemega-compile --dump-evaluated 1` writes one per candidate, each
carrying that candidate's own best plan -- the residency and placement the
search would have taken had it won -- so the comparison is between choices, not
between arbitrary plans. 12 candidates per cell (capacity 12), 25 rounds, arms
rotated inside each round, one fresh process per round.

## Results

`topk.tsv`, recomputed by `../verify.py` from the same logs.

| cell | ratio | coverage | best measured candidate |
|---|---|---|---|
| gqa2_s4 | 1.0141 | 12/12 | `32x16x64s2split8kappa4r4` |
| gqa2_s128 | 1.0000 | 12/12 | `32x16x64s2split4kappa1r4` |
| mha4_s4 | 1.0007 | 12/12 | `32x16x64s2split32kappa4r4` |
| mha4_s128 | 1.0000 | 12/12 | `32x16x64s2split4kappa1r4` |
| real_s4 | 1.0000 | 6/12 | `64x128x16s2split16kappa4r2` |
| real_s128 | 1.0000 | 9/12 | `64x128x16s2split4kappa4r2` |

✅ **Verified: six of six cells at or below 1.05.** Four cells sit at exactly
1.0000 -- the shortlist holds the fastest measured candidate outright -- mha4_s4
is within 0.07%, and gqa2_s4, the only cell above 1%, loses 1.41%.

## Coverage, and why two cells are not 12/12

A candidate that disagreed with the golden output stopped being timed at the
round where it did, and the exclusion is recorded (`<cell>/excluded.tsv`,
`mismatch.tsv`). §6 asks for exactly this to be said rather than papered over.

✅ **Verified: every exclusion is the same element of the same output.** All
nine excluded arms print `E2E_OUTPUT_DIFF index=0 buffer=73 mismatch=1`, one
element of the logits, `max_abs` 0.039 to 0.047 against a golden value small
enough to make `max_rel` 3906 to 5371 -- the language-model head cancellation
D-a runs into (`../llama/head_cancellation.txt`), not a defect of any one
candidate.

⚠️ **Inferred: which candidates it hits is geometry-correlated.** At seq 4 the
six excluded arms are exactly the `32x16x64` family (`split4` and `split8`, all
three `kappa`); at seq 128 they are exactly the `64x128x16s2split8` triple. A
candidate's reduction order decides which side of the tolerance that one
cancelling element lands on, and `kappa` -- which changes no arithmetic -- never
splits a triple. The shortlist's own three arms were measurable in both cells,
so the numerator is never the excluded set.

## Counterfactuals on the same measurements

Both re-spend the three shortlist slots under a pure ordering rule; no new
pricing, no re-ranking of anything the search did.

| cell | as shipped | three distinct geometries | distinct geometries, ties broken by larger kappa |
|---|---|---|---|
| gqa2_s4 | 1.0141 | 1.0245 | 1.0000 |
| gqa2_s128 | 1.0000 | 1.0000 | 1.0000 |
| mha4_s4 | 1.0007 | 1.1060 | 1.0000 |
| mha4_s128 | 1.0000 | 1.0000 | 1.0000 |
| real_s4 | 1.0000 | 1.0047 | 1.0000 |
| real_s128 | 1.0000 | 1.0057 | 1.0057 |

⚠️ **Inferred: the slots are spent along an axis the cost model prices at
zero.** Candidates tie exactly on `(floor_ns, predicted_ns)` across `kappa`, and
the retention test at `include/tilemega/Solver/CompilerSearch.h:170-180` then
keeps whichever arrived first, so a shortlist can hold one geometry three times.
Breaking those exact ties toward the larger `kappa` -- a free choice, since the
model prices them identically -- gives 1.0000 in five cells. Collapsing `kappa`
and taking three *distinct* geometries instead is worse in four cells (1.1060 on
mha4_s4), so "more geometries" is not the rule; "break exact ties by something
the device can tell apart" is. **Reported, not implemented**: §6 sets the gate
at the measured ratio, which is already met, and changing the retention rule
would move every shipped plan.

## What is and is not in the tree

Committed: every round log and its manifest, each evaluated candidate's
generated source (`auto.cu.cand<N>.cu`), the search tables, the exclusion
records and these tables. Not committed: the per-candidate binaries
(`<cell>/bin/`), the residency probe artifacts (`<cell>/auto.cu.resources/`) and
the per-candidate CG dumps (`auto.cu.cand<N>.mlir`, 54 MiB) -- all three
regenerate from the commands below, and the sources that were actually built are
kept.

## Reproducing

```
python3 topk.py search   # solve each cell with --dump-evaluated 1
python3 topk.py build    # one binary per evaluated candidate
python3 topk.py measure  # 25 rotated rounds, fresh process each
python3 topk.py report   # topk.tsv, mismatch.tsv and the lines above
```
