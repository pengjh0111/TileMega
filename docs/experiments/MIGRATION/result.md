# Part 4.4 — does an sm_89-calibrated cost model still rank on a 5090?

> **2026-09-05 protocol update.** ⚠️ The numerical tables below are the
> historical FP32 + element-chunk baseline and are retained for provenance;
> they are not evidence for the current executable. `run_on_sm120.sh` now
> rebuilds both accepted models from BF16 ExportedPrograms with structured
> ownership, runs 50 fresh processes, and records the runtime variant plus
> L0.5/L1/L2 medians. Before building, `tilemega-migrate --probe` compares
> `TargetSpec::Probe()` with the configured host target and requires usable
> cluster capability. A 4090, a mismatched target file, or a target without
> clusters exits 3; no architecture number is accepted on trust. ❌ The
> refreshed arm remains unverified here because this host is sm_89. Successful
> 5090 output is `raw/summary.tsv`, with 50/50 required for both models.
> `bash -n` passes, MLIR is found from an existing TileMega build cache, and a
> fresh self-contained configure/build reaches `tilemega-migrate --probe` on
> this host. The probe then exits 3 because it is the sm_89 calibration target;
> this is the intended hard failure, not a skipped arm.
>
> **The rank-transfer arm is restored, on the BF16 validation set.** The
> regenerated script had dropped it, which would have left this experiment
> answering only "does it still run there" and not the question it exists for.
> `tilemega-migrate` now takes `--dtype f32|bf16` (default `bf16`) and the
> sweep directory, the register table and the generated source travel together
> with the dtype, so the transfer is scored against the implementation that is
> actually shipped rather than the retired FP32 one. Granularity is no longer a
> `-D` override, so each subset point is regenerated as its own single-interval
> runtime plan. Registers are still read from the migration machine's own ptxas
> logs, never carried over.
>
> ✅ **The pre-repair BF16 sm_89 baseline (`raw/summary_sm89_baseline.txt`)**, subset of
> top-50 + 50 random, seed 20260904:
>
> | model | n | MAPE % | Spearman | top10 | optimum_rank |
> |---|---:|---:|---:|---:|---:|
> | gqa2 | 100 | 41.99 | **0.6369** | 0 | 39 |
> | mha4 | 100 | 37.93 | **0.6599** | 0 | 39 |
>
> ⚠️ These rows predate F-74. The repaired full-sweep BF16 model is now
> ρ 0.8942 / 0.8834, above its analytic baseline but below the FP32 target.
> `run_on_sm120.sh` always scores with the current checked-in coefficients, so
> a returned 5090 dataset must be compared with a freshly regenerated sm_89
> subset, not with the historical numbers above.

```
bash docs/experiments/MIGRATION/run_on_sm120.sh        # on the migration target
./build-portable/tools/tilemega-migrate --repo .        # the sm_89 baseline, no GPU
```

## Why this GPU pair

The 4090 (sm_89) and the 5090 (sm_120) are the closest thing to a controlled
experiment the consumer line offers for the term this repository's cost model
was hardest to fit:

| | RTX 4090 | RTX 5090 |
|---|---:|---:|
| SMs | 128 | **170** |
| max smem / SM | 102 400 B | 102 400 B ⚠️ |
| max dynamic smem / CTA | 101 376 B | 101 376 B ⚠️ |
| regs / SM | 65 536 | 65 536 ⚠️ |
| max threads / SM | 1 536 | 1 536 ⚠️ |
| Thread Block Cluster | no | yes |

⚠️ stated: the sm_120 row is `configs/targets/sm_120.json`, which was written
from the published architecture specification and has **not** been probed here.
`TargetSpec::Probe()` on the migration target is what the script actually runs
against, and the run records `nvidia-smi` output next to the result so the two
can be compared.

Occupancy per SM is therefore unchanged and the grid width is not: 128 → 170
SMs moves §2.2(d)'s wave-quantization term and essentially nothing else. That
is the single-variable version of "does calibration transfer", and it is the
term with the most leverage on rank — the COST_MODEL ladder shows `+waves`
lifting Spearman from 0.9095 to 0.9450 on gqa2 and 0.9070 to 0.9435 on mha4,
the largest single step after split-K.

## What is measured

A 100-configuration subset of the 1077-point ORACLE sweep, chosen from the
**committed sm_89 measurements** with a fixed seed so the subset is a property
of the repository and not of the run:

* the **50 fastest** by measured `l05_ms` — the band a solver actually has to
  order correctly;
* **50 drawn uniformly** from the remaining 1027 with `mt19937(20260904)` — so
  the correlation is not computed over a truncated range.

`raw/subset_gqa2.txt` and `raw/subset_mha4.txt` are that list. The script
recompiles exactly those on the migration target with the same nvcc line
ORACLE used, measures them, and re-reads registers from **that machine's**
ptxas logs — a register count is a tier-3 quantity and is never carried across
targets.

Three arms are scored, all with the **unchanged sm_89 coefficients**:

| arm | calibration | resources | what it answers |
|---|---|---|---|
| `sm_89 measured / sm_89 model` | sm_89 | sm_89 | the baseline this subset scores at |
| `migrated measured / sm_89 model` | sm_89 | sm_89 | what happens if nobody tells the model the GPU changed |
| `migrated measured / transfer model` | sm_89 | sm_120 | the cross-compilation path: new `caps` and `res`, old `calib` |

The third is the one that matters. It is what `TargetSpec` is for: capabilities
and resource budgets come from the target file, coefficients come from wherever
they were measured, and §2's design rule says business code never compares
architecture numbers. The check is whether that separation actually holds up.

## Baseline ✅ verified (no GPU needed; from committed data)

`raw/summary_sm89_baseline.txt`:

| model | arm | n | MAPE % | Spearman | top1 | top3 | top10 | optimum rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gqa2 | sm_89 measured / sm_89 model | 100 | 27.48 | **0.9144** | 0 | 0 | 0 | 13 |
| mha4 | sm_89 measured / sm_89 model | 100 | 25.75 | **0.9095** | 0 | 0 | 2 | 8 |

Two things to read correctly before any migration number is compared to these.

**The subset is harder than the full sweep, by construction.** Over all 1077
points the full model reaches ρ = 0.9450 / 0.9435 and puts 3 of its top 10
inside the measured top 3 % (`COST_MODEL/raw/summary.tsv`). Here half the
points are the 50 fastest, which ORACLE measured as a band within ±10 % where
two 25-process replicates disagree on the winner — so the subset deliberately
removes most of the easy ordering. ρ = 0.91 on it is the same model.

**The top-3 % hit rate is 0 on this subset and that is not a regression.** At
n = 100 the 3 % band is the 3 fastest configurations of a set whose top 50 are
already inside measurement noise of each other; no model that does not
reproduce noise can hit it. The migration metrics to compare are therefore
**Spearman**, **`optimum_rank`** (where the measured winner lands in the
model's ordering) and **top10**, and the top-3 % column is reported for
continuity with §2.4 rather than as the criterion.

## Migration result

❌ **not run — no sm_120 hardware in this environment.** `tilemega-migrate
--probe` refuses on the calibration GPU and returns 3:

```
PROBE sm_89: SMs=128 smem/SM=102400 smem/CTA=101376 regs/SM=65536 cluster=false TMA=false tcgen05=false
tilemega-migrate: this is the calibration target (sm_89, 128 SMs); a migration check needs a different GPU
```

✅ verified that the refusal fires — the guard reads `TargetSpec::Probe()` and
compares against the calibration target's own `arch_tag` and SM count, not a
hard-coded architecture number, so it also refuses on a second 4090 and admits
any part that differs. Like `CLUSTER/run_on_cluster_gpu.sh`, it hard-fails
rather than measuring a degraded configuration and reporting it as the result.

When it does run, `raw/summary_sm120.txt` gains the two migrated rows and the
loss is `baseline − migrated` in each column. The prediction on record, to be
falsified rather than confirmed:

> ❌ inferred. The `migrated / sm_89 model` arm loses rank because it prices a
> 128-SM wave on a 170-SM grid: every configuration whose grid is between 129
> and 170 CTAs is charged two waves where the hardware runs one, and those are
> exactly the small-tile split-K configurations that ORACLE's top 50 is made
> of. The `transfer` arm should recover most of it, because `num_sms` is the
> only input that moved. If it does **not**, the separation between `calib` and
> `res` is not real and the coefficients are absorbing something about the
> grid width — which would be a finding about the model, not about the 5090.

## What this check cannot answer

* Nothing here recalibrates. A 5090 `tilemega-calibrate` run would produce a
  different and better model; it would also answer a different question. The
  question asked is whether the *shipped* coefficients survive a port.
* sm_120 has Thread Block Cluster, DSMEM and TMA but **no tcgen05** and no
  Blackwell-datacentre L1.5/LRC layer, so six of the nine resource lanes are
  live there — the same six as on sm_89 (`tilemega-target-audit`). A pass here
  must not be recorded as "all nine dimensions validated"; see
  `CLUSTER/result.md` §7.4a.
* The subset is 100 of 1077 points on two models. It bounds rank loss on the
  configurations a solver would actually consider; it is not a claim about the
  full space.
