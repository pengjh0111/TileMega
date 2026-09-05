# Element ownership experiment

Evidence status: ✅ measured on RTX 4090 (`sm_89`), BF16 Tensor Core model,
25 fresh-process paired rounds.  Confidence intervals are a deterministic
20,000-resample bootstrap of the within-round percentage delta; the last
column is the paired Wilcoxon signed-rank normal approximation.  Raw logs are
recreated under `raw/`; compact results are in [`paired_s4.tsv`](paired_s4.tsv)
and [`paired_s128.tsv`](paired_s128.tsv).

The first controlled experiment changed only RoPE from flattened
grid-stride chunks to `(token, head)` tiles. At `seq=128`, all/identity/window
moved from 20/3/15 to 18/5/15, exact polls fell 13.27%, and L2 improved 3.23%
on GQA and 6.40% on MHA, while L0.5 was statistically flat. See
[`rope_only_s128.tsv`](rope_only_s128.tsv) and
[`rope_poll_profile.tsv`](rope_poll_profile.tsv). That met the
predeclared “narrowable + material poll transfer + acceptable body cost” rule,
so the ownership was promoted to all structurally compatible bodies:

- RoPE owns `(token, head)`;
- KVAppend owns `(new-token, kv-head)` (the pre-existing cache prefix is host
  state and is copied during reset, not falsely represented as append work);
- activation owns one token row;
- split-K combine owns one GEMM output tile.

The final GQA dependency mix changes from **20 all / 3 identity / 15 window**
to **10 all / 7 identity / 21 window**.  At `seq=128`, exact event reads fall
from 482,316 to 282,636 (**41.40% fewer**) while the fully relaxed comparison
set grows slightly with the new task partition.  The exact/relaxed poll ratio
therefore moves from 0.999925 to 0.520240; see
[`poll_profile.tsv`](poll_profile.tsv).

The body-only cost is visible in L0.5: +0.88–1.23% at `seq=4` and
+1.00–1.25% at `seq=128`.  The dependency-driven L2 gain more than pays it:

| seq | GQA L2 | MHA L2 |
| ---: | ---: | ---: |
| 4 | **1.63% faster** | **1.88% faster** |
| 128 | **13.57% faster** | **16.32% faster** |

This is a Place result, but not the previously rejected cache-locality Place
objective.  It changes the ownership contract by which a task body maps work
to CTAs, enabling a correct inverse image for synchronization.  It does not
claim that the same CTA should receive producer and consumer work for cache
reuse.

No flattened-index alternative was needed: the direct ownership experiment
passed all three decision criteria.  Remaining `kAll` edges are semantic
whole-reduction or incompatible domains, not a conservative fallback.
