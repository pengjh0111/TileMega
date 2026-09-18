# R6 final requirements and documentation audit

Audit basis: the external `/root/Prompt/TileMega_R6_prompt.md`, SHA256
`908c09131f8b395c6dfdf3e9329db5a684baf965822e528cfc4c5c2bf551793c`.
The audited delivery is `f698ccbd0b87c306a258aedff2c72cbb20eae5dd`.
This follow-up changes documentation only; the final source/document revision
and subsequent evidence-only commit are resolved by `../sass_identity/manifest.json`.

**Verified conclusion:** all requested groups have implementations, measurements
or an explicitly recorded stop/degraded result. No additional unrecorded task
omission was found in this audit. This is not a claim that every R6 requirement
is fulfilled: J-b/J-c/J-d/J-e and maximal A1 admission remain failures; the
limited C3/W2/S5 domains and B1 causal interpretation remain explicit below.
An allowed stop is an unfinished acceptance result, not a completed mechanism.

## Task-by-task disposition

| Requirement | Delivered evidence | Completion boundary / controlling rule |
| --- | --- | --- |
| C1 unified access-derived task prices | `lib/Solver/CostModel.cpp`, `ScalarTaskWork.cpp`, `COSTMODEL/stagekind_audit.txt`; F-201/F-207 | Implemented, including combine and retained-prefix corrections. No StageKind pricing branches in the reviewed cost path. |
| C2 all three former stage-price callers | CostModel, ChainDP, CouplingInterfaceDP; C-a and 49 CTest evidence | Implemented; retired NonGemmStageNs is absent. |
| C3 K-loop decomposition and calibration | `COSTMODEL/raw_kloop/`, F-190; C-e/C-f | Implemented for instrumented GEMM K-loops. FORK6 is GEMM-scoped; SIMT exposed waits are unmeasured. Reported scope degradation, not evidence for a whole-pipeline 25% wait share. |
| C acceptance | C-a through C-f, report §2/§5 | Recorded PASS in the declared measurement scope; full/coarse rho 0.896800825593/0.876160990712. The C3 scope limitation remains even though its recorded gates pass. |
| J1 binding bound, inner placements, pruning | `JOINT2/bounded_search`, `bounded_boxes`, F-204/F-212, J-outer | Implemented: nonzero work/CP/queue lower bounds and all six placements for every evaluated geometry. Capacity 12, fixed calibrated domain and global kappa are disclosed degraded search, permitted by §9.3 when J2 misses budget. No global-optimum claim. |
| J2 preparation reuse and full evaluation | `COSTMODEL/closure_effects/evaluations.tsv`; F-206/F-209/F-210 | Implemented but not budget-complete: reference 1555.545 us exceeds 1000 us; real 2028.897 us is below 10000 us. §9.2/§9.3; preparation remains separately reported. |
| J3 supported Fuse upper bound | `JOINT2/bounded_fuse/`, F-213 | Complete for the existing legal adjacent RoPE/KVAppend family. Maximum model upper share 0.015426, whole-pair envelope cap 0.041874, both below 10%. No new fusion decision or broader-family proof. |
| J research and hard gates | Six fresh paired/correctness campaigns, report §6, F-212 | J-a 2/2 and J-f PASS; J-b/J-c/J-d/J-e FAIL. Measurements complete; qualified search acceptance remains locally open under §9.2. No threshold was changed. |
| W1 schema and materialized carrier | CG schema/verifier, `PlacementSolvePass.h`, report §7 | Implemented parametric maps and resident/grid constraints, CG module attribute carrier, legacy syntax retained. |
| W2 point and theta-interval writer | `WRITEBACK/interval_closure`, F-202/F-211 | Implemented finite seq [1,5] producer, all six placements per point, fixed geometry chosen at upper endpoint. 250/250 one-binary processes. General interval geometry optimization is not delivered and is disclosed under §9.3. |
| W3 one-command driver and W gates | `WRITEBACK/full_roundtrip`, `legacy_closure`, report §7 | Complete for tested point/finite carrier: 15/15 full execution tables, 10 fresh direct/CG executions, 12 legacy tables, 600/600 default SEQSCAN, 49 CTest. W pass does not imply A1 numerical admission. |
| B1 six-cell attribution | `REBASE/bounded_raw`, `bounded_analysis`, F-214 | 7500/7500 process records and 60/60 traces complete. All 1500 full arms pass; unsafe arms are timing probes. Requested ratios/floors/L1 and real-width tables are reported. |
| B1 causal interpretation | `REBASE/bounded_w1_identity`, `bounded_w1_repeat` | Original identical-kernel timing discrepancy does not reproduce in the additional 100 processes. Cause unresolved; only the affected causal interpretation stops under §9.2. Original samples and CIs are retained. |
| S5 templates, interval legality and portability | `SYMBOLIC/complete`, `bounded_certificates`, `queue_roundtrip`, `cross_grid`; F-205/F-211 | Complete in declared finite domains: four families, seq [1,128], grids 256/340, current wavefront winners at actual grids, 55/55 native queues. Three EFT winners remain outside these four families under S-c/§9.3; no worse winner substituted. No unbounded-grid or cross-architecture proof. |
| A1 configs, coverage and extension cost | `MODELS/sources/manifest.json`, coverage/dimensions/extension TSVs; report §10 | Complete public-source audit, explicit Llama public-copy provenance, 11 actual TaskKinds versus prompt assumption 16, 15 conditional extension sites. Missing operators remain excluded by §2/§7.3. |
| A1 maximal supported end-to-end subset | `MODELS/covered_llama*`, `covered_geometry_probes`, `diagnostic`; F-203 | Supported graph assembled and generated: 98 inputs/66 outputs/209 stages. Five geometries fail the same final residual element. Numerical admission and accepted timing remain stopped under §9.2; old independent MLP 50/50 is not a substitute. Existing-backend numerical repair is not prohibited by the missing-operator exclusion. |
| D1 three documentation repairs | STATUS G1; skeleton §4.4.2 and changelog | Complete; original text preserved and R3–R6 design rows added. Only the two permitted skeleton locations differ from R6 baseline. |
| Four sm_120 runners | Four `run_sm120.sh`, `JOINT2/sm120_common.sh`, `selfcheck_sm120_slices` | Present with parameter/artifact documentation, capability/environment/disk/error guards, local recalibration/solve path and SELF_CHECK PASS. No local sm_120 execution claimed (§9.5 H7). |
| Raw verifier | `JOINT2/verify.py`, `sass_identity/verify.log` | All 30 gates executed; 25 PASS, 5 FAIL, including 4 hard failures, exit 1. It runs all gates before exiting and does not read summary conclusions. |
| Provenance and delivery | Report §1, source/identity manifest, commit history | Prompt hash, baseline, order, source identity and remote delivery recorded. Final documentation correction requires a regenerated identity child, not reuse of an earlier source stamp. |

## Final-report coverage

All fourteen required sections exist in `../summary.md`:

| R6 §14 item | Report location and audit result |
| --- | --- |
| 1 baseline, prompt SHA, commits, H4 | §1; all four required dependency orderings recorded and checked |
| 2 each gate with numbers/evidence | §2; 30 rows, failed gates retained |
| 3 complete verify output | §3; byte-identical to archived raw `sass_identity/verify.log` |
| 4 stop/degradation ledgers | §4; scope, cause, unlock and inferred effort present |
| 5 C results, grep, calibration, FORK6 | §5; original line retained, denominator limitation explicit |
| 6 J research/ratios/ranking/budget/Fuse | §6; six placement orders and exact threshold decision present |
| 7 W setAttr/one command/round trip | §7; actual source and command, full carrier equality evidence |
| 8 B1 attribution and real-width | §8; complete tables and historical 0.6705 comparison; causal limitation retained |
| 9 S interval/sample/unfit/portability | §9; finite proof boundaries and unchanged unfit winners explicit |
| 10 A configs/coverage/extensions/end-to-end | §10; source URLs/hashes, tables, command and failed maximal result |
| 11 deviations | §11 plus §4; historical/external evidence distinguished |
| 12 exclusions | §12; every §2 excluded group listed, invariants retained |
| 13 failed causes and next steps | §13 plus §4; code locations, constants and work estimates |
| 14 R7 priorities and unsolved decisions | §14; dependencies/effort and decisions still outside solver control |

## Cross-document checks and correction

- FINDINGS F-190 through F-214 are present with unique IDs. F-199/F-200 are
  attributed upstream sm_120 observations, not local reruns. Final continuation
  findings supersede earlier round checkpoints without deleting them.
- TODO has the required EX-S1/S3/S5/V1/E4 updates and new EX-W/EX-A rows and
  detailed entries. Its EX-A detail still described the pre-continuation
  independent-MLP state; this audit adds a local v2.1 correction linking F-203
  and the actual maximal-graph stop. The ledger and final report were already
  correct. This was a documentation omission, not a prompt exclusion.
- STATUS G1/G8/G9/G11 and Place/Fuse contain the final continuation annotations.
  Historical numbers remain historical; they are not substituted for final data.
- The skeleton baseline diff is confined to §4.4.2 and the changelog.
- The full verifier output remains the completed run at the audited delivery.
  This follow-up does not rerun GPU experiments or change any performance data,
  source, verifier or gate. Targeted C-a/W-a/W-b/H4/H7 checks were rerun during
  this audit; H2 and its source hashes are regenerated after the documentation
  commit. No fresh full 30-gate run is claimed for documentation-only changes.

## Remaining work is not all prohibited work

Explicitly out of scope: full EX-V1 anchored decode sweep, new Fuse decisions,
shared-memory pipeline/union-lifetime redesign, missing A1 operators, EX-E5,
EX-S4 and L5. These are intentionally not implemented in R6.

Locally stopped or degraded work: unmet search gates/budget, the B1 timing-band
cause, maximal A1 numerical admission, GEMM-only phase scope and finite
search/interval/template coverage. These have controlling §9.2/§9.3 records,
but are not declared solved or inherently impossible. Report §4/§13 specifies
how to unlock them. Thus the accurate completion statement is: the requested
implementation/evidence workflow has been carried through its documented
acceptance boundaries; full single-inference research closure remains partial.
