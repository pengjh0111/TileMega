# B3 — segmented geometry inside a finite theta interval (§5.4)

One `tilemega-compile` invocation per model produces both arms, so the two are
the same solve and not two invocations:

| arm | source | what it is |
|---|---|---|
| `fixed` | `auto.cu.fixed.cu` | one variant over the whole interval, at the geometry the search chose at the theta point |
| `segmented` | `auto.cu` | the best cut of the interval into two variants, each carrying its own sub-range of the plan table |

Both arms advertise the whole interval in the macros -- the 15 `#define
TILEMEGA_*` lines are identical between the two sources -- so what differs is
the plan table the runtime selects by `seq`, not the launch geometry.

The invocation adds `--seq-begin 1 --segments 2 --segment-candidates 4` to the
§7.1 command, on the SEQSCAN export of the model (`solve.json`, exit 0 in
772.4 s on gqa2); `segments.py` runs it, then the two verifiers, then the
rounds. Everything below is recomputed from the raw logs by
`../verify.py` (gates `B3 intra-interval campaign, <model>`).

## What the solve reports

`SEGMENT_SUMMARY` on gqa2, interval `[1,16]`, 4 candidate geometries priced at
all 16 integer points (`auto.cu.segments.tsv`, 64 rows):

| quantity | geometry | summed predicted ns over the interval |
|---|---|---|
| `winner` — what the point search picked at theta | `32x16x64s2split8` | 2.68866e+06 |
| `fixed` — best single geometry over the interval | `32x16x64s2split16` | 2.67977e+06 |
| `segmented` — best two-segment cut, `cut=14` | `split16` on `[1,13]`, `split8` on `[14,16]` | 2.67944e+06 |

✅ **Verified: the cut lands exactly where the two curves cross.** At `seq=13`
`split16` prices 166892 ns against `split8`'s 166986; at `seq=14` it is 167552
against 167544. The cut is at 14 because that is the first point where `split8`
becomes the cheaper of the two, which is what a segmenting search should find.

⚠️ **Inferred: the benefit is small because the curves are nearly parallel.**
Over the whole interval the best cut beats the best single geometry by 330 ns in
2.68e+06, 0.012%, and beats the point winner by 0.34%. `split16` and `split32`
price *identically* at every one of the 16 points -- the same exact tie the D-b
counterfactual found along `kappa` -- so the candidate set holds fewer distinct
curves than it holds candidates.

## Legality (the S5 ISL path)

`segment_proof` runs one fresh process per integer point of the interval
(`proof/p<seq>/proof.log`, `proofs.tsv`, `coverage.tsv`):

✅ **Verified: 16/16 points proved, `failed=0` on every one.** Each point
reports `bijective=1 dense=1 acyclic=1 resident=1` for its own segment's
geometry, the same certificate S5 uses for a single geometry, applied per
segment rather than per interval.

## Endpoint and interior materialization

`segment_check` re-solves every point's table on the graph of the segment that
owns it and compares the serialized plan (`check.log`, `check.json`):

✅ **Verified: `segments=2 points=16 endpoints=4 interior=12 interval=1..16
kappa=1 residency=4 serialization=byte_identical`, and `diff=0` on all 16
`SEGMENT_MATERIAL` lines.** Endpoints and interior points are checked by the
same rule; 5084 to 5152 plan nodes per point.

## Correctness

`correctness.tsv`, one fresh process per round, `TILEMEGA_WARMUP=0
TILEMEGA_REPEAT=1`:

✅ **Verified: gqa2 500/500 fresh processes.** Five seqs inside the interval
(1, 4, 13, 14, 16 -- both sides of the cut and both endpoints) times two arms
times 50 rounds. One binary per arm across all five seqs: segmented
`632ba5073543a6c4`, fixed `f254ff53eeef2c8c`.

## Benefit, measured

`timing.tsv`, 20 paired rounds per seq, arms rotated inside each round, median
of `l2_ms`:

| seq | segment | segmented ms | fixed ms | segmented/fixed |
|---|---|---|---|---|
| 1 | 0 | 0.416736 | 0.419840 | 0.9926 |
| 4 | 0 | 0.746384 | 0.747440 | 0.9986 |
| 13 | 0 | 1.872000 | 1.844960 | 1.0147 |
| 14 | 1 | 1.854656 | 1.855296 | 0.9997 |
| 16 | 1 | 2.113424 | 2.113536 | 0.9999 |

⚠️ **Inferred: no measurable benefit on this model, and none was predicted.**
The predicted gain over the whole interval is 0.34% against the point winner;
the measured ratios straddle 1.0 with a 1.5% spread between seqs, so the
mechanism is verified correct and its benefit is below what this cell can
resolve. A model whose candidate curves cross steeply inside the interval is
where a gain would show; §5.4 asks for the number, not for a threshold.

## mha4

The same campaign on mha4 (88 projected stages against gqa2's 44) is the
second interval in this directory. Its state is whatever `../verify.py`
recomputes from `mha4_i1_16/` -- the proof is the long pole at roughly an hour
per point.
