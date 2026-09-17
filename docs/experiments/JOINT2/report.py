#!/usr/bin/env python3
"""Render the R6 report after all raw-data gates have been evaluated.

This presentation tool may read derived tables. verify.py deliberately does
not; its complete output is supplied through --verification.
"""
import argparse,csv,json,re,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;EX=HERE.parent;REPO=HERE.parents[2]
BASE='bad8a0d9b17804b73afe00a6d545dcea72cc6cbb'
PROMPT='908c09131f8b395c6dfdf3e9329db5a684baf965822e528cfc4c5c2bf551793c'
def rows(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def table(headers,data):
 def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in data])
def f(x,n=6):return f'{float(x):.{n}f}'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verification',type=Path,required=True);a=ap.parse_args();verification=a.verification.read_text()
 joint=rows(HERE/'comparisons_gqa2_mha4_real.tsv');assert len(joint)==6
 rebase=rows(EX/'REBASE/attribution_gqa2_mha4.tsv')+rows(EX/'REBASE/attribution_real.tsv');assert len(rebase)==60
 out=[];add=out.append
 add('# TileMega R6 — solver pipeline closure and real-model anchoring\n')
 add('The production point-solve/writeback path and the bounded symbolic templates are implemented. Performance acceptance is **not fully closed**: the verifier below retains every failed hard gate. These results do not establish that all single-inference decisions or all real-model operators are solved. The selected configurations remain opt-in; their reference regressions are not a delivered performance improvement.\n')
 add('## 1. Provenance and auditable order\n')
 add(f'Baseline: `{BASE}`. Branch: `tilemega`. External prompt: `/root/Prompt/TileMega_R6_prompt.md`; SHA256 `{PROMPT}`. Machine: RTX 4090 / sm_89. The final artifact-only child commit records its source parent in `sass_identity/manifest.json`; that parent is the final source/document revision for this report.\n')
 add('H4 is recomputed by the verifier: C1/C2 precede J1; the separately committed FORK6 precedes the Fuse/R7 decision; every frozen selection exists in its recorded commit before its B1 processes start; W2 precedes S5.\n')
 history=subprocess.check_output(['git','log','--reverse','--format=%h %s',BASE+'..HEAD'],cwd=REPO,text=True).strip()
 add('Commits already present when this report was rendered (the subsequent report commit and evidence-only stamp are resolved through the manifest):\n\n```text\n'+history+'\n```\n')
 add('## 2. Gate results\n')
 gate=[]
 for line in verification.splitlines():
  m=re.match(r'^(\S+) (PASS|FAIL) \[(\w+)\] (.*)$',line)
  if m:gate.append([m[1],m[2],m[3],m[4]])
 assert len(gate)==27
 add(table(['Gate','Result','Type','Measurement and raw evidence'],gate)+'\n')
 add('## 3. Complete verification output\n\n```text\n'+verification.rstrip()+'\n```\n')
 add('The nonzero result is intentional whenever hard gates remain failed. All gates execute before exit. The verifier reads raw processes, evaluated cost rows, proof logs and full Plan tables; it does not read this report or the derived comparison/attribution tables. A preliminary valid identity stamp permits capture of actual complete verifier output before rendering this report. After the report commit, the final SASS is regenerated and the full verifier is rerun; its output must match the embedded output byte for byte. No expected PASS line substitutes for execution.\n')
 add('## 4. Stalled items and explicit degraded forms\n')
 add(table(['Item','Affected scope','Cause / unlock','Inferred work'],[
 ['C-c/J-c','Unqualified pruning and unrestricted outer search','Coarse rho <0.85; reference full evaluation >1 ms. Repair weights and remaining queue/graph scans, then rerun the frozen 18 points.','3–7 developer-days plus measurement'],
 ['J-b/J-d/J-e','Claim of balanced and non-regressing selected optimum','Incorrect task service prices and shortlist ranking. Derive combine task traffic/iteration work, price serial scalar phases and compiled occupancy.','3–7 developer-days'],
 ['Unfit S5 winners','Automatic interval dispatch for five EFT winners','Outside four implemented template families; retain original materialized winners and expand templates only with equivalence/proof.','2–5 developer-days per additional family'],
 ['B1 causal separation','Real-s128 physical attribution','Unchanged control kernels have two timing bands; preserve signed pairs. Log process-level clocks/power/allocation and rotate five arms within each configuration rather than across all fifty.','1–2 developer-days + fresh GPU campaign'],
 ['A1 full/maximal subset','Full architecture EX-V1 anchor','Missing or mismatched embedding/head, epsilon/RoPE, QK-norm semantics; attention cuts not yet assembled.','5–10 developer-days plus correctness runs']])+'\n')
 add('No global correctness-stop was triggered. New real-s4 split8 candidates failed one CPU-golden value identically in L0.5/L1/L2 and were excluded with input-model SHA256 and theta checks; existing configurations were not relabeled. The admitted split16 selection was frozen after a separate pilot.\n')
 add('Degraded forms: finite outer capacity (9 geometry/kappa combinations, 12 for the rejected real-s4 search), five calibrated tile shapes, retained global κ, point-only production W2 solve rather than automatic interval optimization, finite S5 grid branches, GEMM-only FORK6 wait denominator, common-mode-contaminated real-s128 intervention attribution, and sixteen independent normalized MLP components rather than a maximal whole-model subset. B1 measures solver-selected diagnostic configurations even where J gates fail; it is not called a re-established optimal baseline.\n')
 add('## 5. C — unified costs, K-loop calibration and remaining errors\n')
 add('`DeriveTaskWork → TaskMemoryTraffic → TaskInstanceNs` replaces the three NonGemmStageNs call sites. Backend TaskBody traits/dataflow supply resources and phase structure. `StageKind` remains in parsing/ownership projection, not the four core cost-evaluation files.\n\n```sh\nrg -n "StageKind|NonGemmStageNs" lib/Solver/{CostModel,ChainDP,CouplingInterfaceDP,TaskModel}.cpp\n# no matches\nrg -n "StageKind|NonGemmStageNs" lib/Solver\n# Complete output and reviewed non-price roles: COSTMODEL/stagekind_audit.txt\n```\n')
 add('18-point full Spearman 0.884416925 (R5 threshold 0.880288958); coarse 0.766769866 (gate 0.85). Historical 68-dump replay absolute relative errors: p50 4.539964%, p90 10.847552%, max 13.827710%, versus R5 4.54/10.85/13.83%. These are fresh CPU reevaluations of historical calibration evidence, not fresh GPU speedup claims.\n')
 add('```text\nFORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4\n```\n')
 add('The denominator is instrumented GEMM mainloops. SIMT exposed wait is unmeasured; the all-path-mainloop lower-bound denominator gives 0.161782. The prescribed rule is applied to the measured GEMM scope, with this deviation explicit. Per-cell iteration/fixed p50 ns: gqa2 s4 494.574/84.156, s128 519.355/219.341; mha4 s4 504.267/52.239, s128 318.945/67.038; real s4 266.040/52.492, s128 538.262/102.693. Raw: `../COSTMODEL/raw_kloop/`.\n')
 add('The task-zero audit (`../COSTMODEL/selected_prices/comparison.tsv`, reconstructed from price rows and fresh trace slots) finds combine measured/predicted medians 23.282/25.320/31.399/12.885 for gqa2 s4, mha4 s4, real s4/s128. This is the remaining `PlacementSolvePass.h::projected.combine` whole-stage/wave conversion. Non-combine medians are 1.449/1.902/1.449/1.313/2.213/2.007. Attention’s thread-zero softmax loop also needs backend-declared serial iteration work. The retired branches are unified; **every runtime-task price is not yet fully unified/calibrated**.\n')
 add('## 6. J — binding objective, fresh comparisons and Fuse bound\n')
 data=[];config=[]
 for r in joint:
  name=r['cell'];cell=REPO/json.loads((HERE/'cells.json').read_text())[name];chosen=next(x for x in rows(cell/'auto.cu.top3.tsv') if 'top'+x['rank']==r['arm']);arm=r['arm']
  data.append([name,f(r['l2_ms']),f(r['control_ratio'])+' ['+f(r['control_ci_lo'])+','+f(r['control_ci_hi'])+']',f(r['champion_ratio'])+' ['+f(r['champion_ci_lo'])+','+f(r['champion_ci_hi'])+']',f(r[arm+'_queue_over_cp']),f(r['l2_l1_ratio']),f(r['measured_floor_ratio']),r['predicted_top1_rank']+'/3'])
  config.append([name,chosen['key'],chosen['placement'],f(float(r[arm+'_cp_ns'])/1000,3),f(float(r[arm+'_queue_lb_ns'])/1000,3),r[arm+'_nodes'],f(r[arm+'_node_mean_ns'],3)])
 add(table(['Cell','L2 ms','/ fresh R5 control [95% CI]','/ fresh R5 champion [95% CI]','queue/semantic CP','L2/L1','L2/floor','Predicted top1 measured rank'],data)+'\n')
 add(table(['Cell','Selected geometry / split / κ / residency','Placement','CP µs','Queue µs','CP nodes','Mean node ns'],config)+'\n')
 add(table(['Cell','Fresh R5 champion CP µs','Selected CP µs','Fresh R5 champion queue/CP','Selected queue/CP'],[[r['cell'],f(float(r['champion_cp_ns'])/1000,3),f(float(r[r['arm']+'_cp_ns'])/1000,3),f(r['champion_queue_over_cp']),f(r[r['arm']+'_queue_over_cp'])] for r in joint])+'\n')
 add('Historical R5 real-width queue/CP was 1.7589/1.4174; current selected trace ratios are 1.849117/1.701968. These ratios remain above the frozen J-b line.\n')
 add('Controls and candidates are twenty-five rotated, same-session, fresh-process rounds; choice uses the separate five-round pilot and is frozen before confirmation. The zero-sync union bound used by search is distinguished from semantic CP in J-b; replacing J-b’s denominator with a queue-containing CP would make that gate tautological. Minimizing max(CP,queue) itself does not enforce queue/semantic-CP ≤1.\n')
 ranks=[]
 for r in joint:
  name=r['cell'];cell=REPO/json.loads((HERE/'cells.json').read_text())[name];chosen=next(x for x in rows(cell/'auto.cu.top3.tsv') if 'top'+x['rank']==r['arm'])
  predicted=sorted([x for x in rows(cell/'auto.cu.search.tsv') if x['candidate']==chosen['key'] and x['status']=='ok'],key=lambda x:(float(x['floor_ns']),float(x['predicted_ns'])))
  measured=sorted([x for x in rebase if x['cell']==name and x['config'] in ['legacy_grid_stride','rotate','balanced','eft','wavefront','chain']],key=lambda x:float(x['l2_ms']))
  ranks.append([name,' < '.join(x['placement'] for x in predicted),' < '.join(x['config'] for x in measured)])
 add(table(['Cell','Predicted six-placement order at selected geometry','Fresh B1 full-time order'],ranks)+'\n')
 add('Full simulator evaluation peaks at 2480.201 µs for a reference point and 2873.300 µs for real-width; graph preparation separately peaks at 145561.384/242049.983 µs. The reference 1 ms gate still fails; real-width meets 10 ms. Preparation is not hidden as free.\n')
 bounds=rows(HERE/'fuse_upper/bounds.tsv');add(table(['Cell','Supported pairs','Removed nodes','Removed bytes','Fixed upper ns','Traffic upper ns','Total upper ns','/ measured floor'],[[r[k] for k in ['cell','supported','eliminated_nodes','removed_global_bytes','fixed_upper_ns','traffic_upper_ns','optimistic_upper_ns','upper_share']] for r in bounds])+'\n')
 add('```text\n'+(HERE/'fuse_upper/decision.txt').read_text().strip()+'\n```\n')
 add('Only the existing legal adjacent RoPE→KVAppend family is bounded. Fixed upper=.232×separate envelope; traffic upper is the unified-path reduction with internal storage free, external traffic/arithmetic retained; total is capped at the separate envelope. No added barrier, copy, recomputation or occupancy loss is charged. This is an optimistic model bound under its wave assumptions, not an executed fusion or proof about all possible fusion families. The common .232 fixed fraction is an extrapolation, not per-pair phase data; even eliminating the entire modeled pair envelope caps the largest share at .034358, below .10. The existing Fuse direction gate remains unchanged.\n')
 add('## 7. W — one-command import, solve, writeback and code generation\n')
 command=json.loads((EX/'MODELS/llama_mlp/direct_pt2.command.json').read_text())['command'];add('Executed from the repository root:\n\n```sh\n'+' '.join(command)+'\n```\n')
 add('The `.pt2` is regenerated with `MODELS/export_mlp.py`; large random fixtures/archives are not committed. Exact command/source/bridge/resource evidence is in `../MODELS/llama_mlp/`. The generated `direct_pt2.cu` equals the source tested in fifty fresh processes. Geometry, split, global κ, residency and placement come from the solver.\n\n```sh\nrg -n "placement->setAttr" include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h\n# mode, params, window, policy, resident_only, grid_map, resident_limit_map\n```\n')
 add('Physical EFT worker/slot arrays are stored in CG module attributes, trading module size for self-contained round trips. `WRITEBACK/roundtrip_gqa2_s4/{cg_plan,host_plan}.tsv`: 2696 nodes, diff empty. Matched baseline/current host archives produce 12/12 identical legacy schedule/waits/events files. Default SEQSCAN is 600/600; selected reference subset adds 400/400. All 49 CTest cases pass. W2 production is a point solve; automatic interval winner selection remains partial.\n')
 add('## 8. B1 — attribution at solver-selected geometry\n')
 add('Six placements and four additional protocol variants, five arms each, twenty-five rotated fresh rounds per cell. C3(a) is intentionally inert at W=1. C1/C2/C3 are cumulative flags; no window is enabled. Unsafe probes may fail numerical comparison, but full arms must PASS.\n')
 add(table(['Cell','Config','Full L2 ms','Wait ms','Notify ms','Fence ms','Barrier ms','Protocol/barrier','/ selected','/ legacy','L2/L1','Floor µs','L2/floor','Nonpositive barrier pairs'],[[r['cell'],r['config'],*[f(r[k]) for k in ['l2_ms','wait_ms','notify_ms','fence_ms','barrier_ms','protocol_barrier','relative_selected','relative_legacy','l2_l1']],f(float(r['floor_ns'])/1000,3),f(r['measured_floor']),r['nonpositive_barrier_pairs']] for r in rebase])+'\n')
 attribution=[]
 for name in [r['cell'] for r in joint]:
  by={r['config']:r for r in rebase if r['cell']==name};rot=by['rotate'];off=by['protocol_off']
  attribution.append([name,f(rot['relative_legacy'])+' ['+f(rot['relative_legacy_lo'])+','+f(rot['relative_legacy_hi'])+']',f(1/float(off['relative_selected']))+' ['+f(1/float(off['relative_selected_hi']))+','+f(1/float(off['relative_selected_lo']))+']',*[f(by[k]['relative_selected']) for k in ('c1','c2','c3')]])
 add(table(['Cell','Rotate / legacy [95% CI]','R3 B on / off [95% CI]','+C1 / B','+C1+C2 / B','+C1+C2+C3(a) / B'],attribution)+'\n')
 add('Placement ratios keep the R3 B protocol fixed. Protocol ratios keep the solver-selected placement and geometry fixed; the on/off row is the reciprocal of the retained paired off/on statistic, with interval endpoints reversed. These are conditional contributions, not multiplicative independent factors. R1’s historical pooled rotate/legacy was .6705; R3’s historical default-placement cumulative B reductions were 5.16/3.48/4.79/3.74%. R4 reduced the protocol/barrier diagnostic by 36% while full time increased. None of those historical percentages substitutes for the fresh configuration-specific rows here.\n')
 add('All medians and paired 95% intervals are retained in `REBASE/attribution_*.tsv`; raw logs and trace dumps are under `REBASE/raw/`. Wait=full−nowait, notify=nowait−neither, fence=full−nofence, barrier=L1full−L1nosync. Signed differences are not clamped: an unsafe arm can change execution/overlap, so these are intervention differences rather than disjoint nonnegative physical service times. Real-s128 also exhibits two timing bands in the unchanged L0.5/L1 controls. Full/nofence L1 SASS is identical, so that control variation cannot be assigned to the removed L2 fence. Negative barrier pairs are retained with their original signs, including the ratio; a zero denominator makes the complete ratio statistic undefined instead of deleting a pair. This corrects an extra analyzer positivity assumption, not a frozen R6 gate or its measurement formula. Causal attribution is explicitly degraded; next diagnostics need per-process clocks/power/allocation traces and arm rotation within each configuration to keep paired controls close in time. The historical rotate/legacy 0.6705 is compared with the fresh `rotate` rows above, not reused as a current measurement.\n')
 add('## 9. S5 — proven domain and cross-grid comparison\n')
 add('Source graph and geometry are in `SYMBOLIC/complete/provenance.json`. Four families, integer seq [1,128], grids 256/340; 166 exhaustive ISL certificate pieces. Wavefront G340 uses singleton certificates for every integer in the interval, not inference from five samples. Forty endpoint/interior native/template tables and forty CG serialize/read evaluations agree. One carried CG contains the two constant-grid branches per family; these are not extra kernel variants.\n')
 add('Five EFT champions do not fit the four families; mha4 s128 fits rotate. No worse template replaces a measured winner. Fresh six-catalog solves on the same CG choose wavefront at G256 and EFT at G340 (`SYMBOLIC/cross_grid/`). Actual SM count/resident witness are unchanged. This is finite CPU portability evidence, not unbounded variable-divisor Presburger support or cross-architecture performance.\n')
 add('## 10. A1 — model sources, coverage and executable subset\n')
 dims=rows(EX/'MODELS/dimensions.tsv');add(table(list(dims[0]),[list(r.values()) for r in dims])+'\n')
 add('Sources: Qwen official public config; Meta official download returned 401, so Llama uses an explicitly identified public redistributor config cross-checked against Meta SKU/implementation. Exact URLs and SHA256 are in `MODELS/sources/manifest.json`. No pretrained weights or HF runtime dependency.\n')
 coverage=rows(EX/'MODELS/coverage.tsv');add(table(list(coverage[0]),[list(r.values()) for r in coverage])+'\n')
 sites=rows(EX/'MODELS/extension_sites.tsv');add(table(list(sites[0]),[list(r.values()) for r in sites])+'\n')
 add('There are 11 current TaskKind values, not 16. A new distinct kind/ownership can touch 15 conditional sites; new prices come through semantics/traits rather than per-operator Solver formulas. The executed subset is sixteen independent, normalized Llama-width MLP components with distinct random weights and 32 boundary inputs, not a sequential decoder or maximal covered whole graph. Correctness 50/50, sixteen output checks each. Warmup=0/repeat=1 L2 median 2.016880 ms (range 1.973248–2.034688), L1 median 3.050624 ms; these are single-launch diagnostic observations, not full-model steady-state gains.\n')
 add('## 11. Deviations and reasons\n')
 add('The explicit degraded forms in §4 remain open. The requested conventional source filenames include header-only production JointSearch/placement passes in this checkout; symbol names were used. S5’s large-map parser was reused inside the evidence driver without changing the proof. The global timer has 1024 ns ticks; zero durations are preserved. Source calibration replay uses historical GPU traces exactly as requested for replay gates, while J/B performance controls are fresh. sm_120 runners were compile/guard self-checked here and were not run on sm_120. The report and SASS stamp use a parent/child commit manifest to avoid claiming that an earlier code revision was the final one.\n')
 add('## 12. Excluded work confirmation\n')
 add('No EX-V1 full anchored decode sweep; no Fuse outer-search decision or fused implementation; no EX-E4 shared-memory pipeline or TaskSmem union-lifetime change; no A1 missing-operator implementation; no EX-E5, EX-S4 or L5 serving. Plan execution semantics, W=1, monotonic epochs, release rules and legality checks remain unchanged. Only the skeleton changelog and §4.4.2 were edited. User-owned edits in PLACE_EFT2/summary.md and SYNC_V2/sass_identity/meta.tsv were preserved.\n')
 add('## 13. Failed gates: causes and concrete next changes\n')
 add('C-c/J-d: bound weights and shortlist fine ordering do not resolve task duration correctly. Fix the combine `whole_stage / waves` conversion in PlacementSolvePass.h, and add serial iteration/active-lane terms to backend ScalarDataflow for AttentionChunkTaskBody’s thread-zero loops. PriceTaskInstances currently passes active_ctas=1; fit service dilation under actual compiled residency. J-c: retain compressed/grouped graph readiness and replace remaining per-plan initialization/queue rebuild scans; cache only immutable shape/target/theta inputs, measuring preparation separately. J-b: the binding objective does not mathematically enforce queue/semantic-CP≤1; if retained as an admissibility constraint, explicitly search its feasible set and remeasure the latency tradeoff, without changing this round’s gate. J-e: the same weight defects select extra split nodes and poorer geometry/placement; verify repair against the unchanged fresh-control protocol. Do not call an R5 fallback a completed new solution.\n')
 add('## 14. R7 priorities and decisions still outside the solver\n')
 add(table(['Priority','Block','Dependency / action','Inferred effort'],[
 ['0','Finish price/ranking closure','Access-derived combine work, serial scalar phases, residency service and budget; preserve the frozen gates.','3–7 developer-days + fresh campaigns'],
 ['1','A1 semantic/operator gaps and maximal covered graph','Exact epsilon/RoPE, embedding/head, QK-norm ownership, compose covered attention boundaries.','5–10 developer-days'],
 ['2','EX-V1 full anchored main benchmark','Depends on corrected anchor and qualified search; decode seq 1/4/16/64, complete ablations.','3–5 developer-days + 1–3 GPU-days'],
 ['3','EX-E4 step 2 design and bounded prototype','FORK6 rule1 on GEMM scope; distinguish intra-K-loop waits from first-slot prefetch; explicitly redesign §8.6 before shared buffering.','4–8 developer-days'],
 ['defer','Fuse in outer search','Current legal-family model upper <10%; expand rejected ownership families and reprice before changing the decision.','2–5 developer-days for an expanded bound'],
 ['later','EX-E5/S4 → L5 serving','After single-inference correctness/resource contracts and cost quality; separate serving-state/lifetime design.','1–2 weeks for E5/S4, then 2–4 weeks for L5']])+'\n')
 add('Automatic decisions now include geometry, split-K, global κ, residency, placement and slot order for the evaluated point/candidate domain. Still outside complete solver control: fusion partitioning, cross-task shared-memory pipeline selection, per-stage κ, interval winner/variant selection for unfit materialized schedules, search-domain/capacity policy, and missing architecture import/semantics. Therefore single-inference closure is partial even though the concrete import→solve→CG→CUDA channel now executes.\n')
 (HERE/'summary.md').write_text('\n'.join(out))
if __name__=='__main__':main()
