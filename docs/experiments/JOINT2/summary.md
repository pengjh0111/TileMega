# TileMega R6 — solver pipeline closure and real-model anchoring

The production point and finite-interval solve/writeback paths and the bounded symbolic templates are implemented. Performance acceptance is **not fully closed**: the verifier below retains every failed hard gate. These results do not establish that all single-inference decisions or all real-model operators are solved. The selected configurations remain opt-in; their reference regressions are not a delivered performance improvement.

## 1. Provenance and auditable order

Baseline: `bad8a0d9b17804b73afe00a6d545dcea72cc6cbb`. Branch: `tilemega`. External prompt: `/root/Prompt/TileMega_R6_prompt.md`; SHA256 `908c09131f8b395c6dfdf3e9329db5a684baf965822e528cfc4c5c2bf551793c`. Machine: RTX 4090 / sm_89. The final artifact-only child commit records its source parent in `sass_identity/manifest.json`; that parent is the final source/document revision for this report.

Concurrent remote commits c1588be6 and 8f4951c1 were preserved in merge 61165f0a. Their sm_120 experiments and script edits are external contributions, not new measurements or out-of-scope implementations performed by this R6 continuation. Their F-199/F-200 entries retain their IDs; this continuation’s formerly conflicting F-199..F-208 entries are now F-201..F-210, with references updated. Both pre-existing user-owned dirty files retain their exact original bytes. Audit: closure/upstream_integration/manifest.json.

H4 is recomputed by the verifier: C1/C2 precede J1; the separately committed FORK6 precedes the Fuse/R7 decision; every frozen selection exists in its recorded commit before its B1 processes start; W2 precedes S5.

Commits already present when this report was rendered (the subsequent report commit and evidence-only stamp are resolved through the manifest):

```text
567cb81df solver: cost tasks from derived access relations
dbfb4051d solver: retire the per-operator stage formulas
e3ee9419e trace: split the mainloop into loop and fixed work
7fc201bdd experiments: calibrate the K loop fixed term
6344daa59 solver: minimize the binding bound, not the path
294c146dd solver: reuse prepared readiness across plans
2d388552c cg: carry parametric placement parameters
c8d58a187 cg: write the solved placement back to the graph
71ce688e7 solver: count worker ownership without sorting groups
80cb9be0f tools: solve before generating the megakernel
5bd862ecb experiments: audit operator coverage for real models
b16e839e3 tools: query compiled occupancy before choosing the grid
39113b7af docs: correct the place channel status and objective
a88ec2853 solver: prove placement legality over intervals
66aced8e0 solver: price loop service from measured backend work
4711e065c experiments: check legacy tables and the complete seqscan subset
fa6d208ad solver: carry causal event prices into placement evaluation
d08e0a795 tools: retain the solved shortlist for direct measurement
adca051ae solver: batch access cardinalities across a task wave
dc824af3f solver: reuse prices for identical derived task work
0bef657e8 frontend: lower the supported real model MLP regions
7ade63a25 solver: carry derived traffic into instance pricing
4772fba17 experiments: freeze the reference search choices
e6730cc96 solver: cache immutable task prices across resident plans
887e91d14 solver: exclude numerical failures within their model and theta
32549cbfe experiments: freeze the admissible real width small seq choice
b67988f68 solver: prove bounded placement templates across resident grids
1a8d0f249 experiments: compare legacy tables using matching host archives
d61ee9260 tools: import torch exports inside the solve command
988277397 experiments: freeze the real width long seq search choice
eaedcd14e experiments: bound fixed work and traffic removed by fusion
99c1a8140 experiments: locate task price errors in selected configurations
b41aa9e09 experiments: rebase five completed cells at solved geometry
14347b322 experiments: verify symbolic plans across intervals and grids
1c01faae8 experiments: add local regeneration runners for sm_120
1ea48486a experiments: record five cells and selected plan sequence checks
265b5bc3c docs: distinguish verified closure from remaining price gaps
4fbb9fa0b experiments: rebuild price and attribution checks from raw data
42fe3eac8 experiments: complete the six cell search confirmation
1342ecffc experiments: retain signed intervention pairs and audit controls
1408f0a9a experiments: render the round six closure report
2a5f25763 experiments: finish the solved configuration ablations
e5f7663d4 docs: record the round six closure results
d95e1e2b8 experiments: retain signed reference intervention pairs
3bea714a9 experiments: check default sass before rendering the report
2c948cefa experiments: record the complete real width attribution
fe9290f41 docs: finalize the round six report and gate evidence
d626aa95f experiments: stamp the sass identity at head
c1588be61 docs: record round three's sm_120 execution and fix the SEQSCAN timeout
8f4951c11 experiments: fix the cluster runner bugs and record the F-200 residency hazard
041b7b819 solver: derive combine prices from typed task accesses
7e8fda2fe frontend: retain supported regions across explicit model cuts
5104d4185 experiments: replay prices with the measured task ownership
f9f7fb319 cg: carry solved placements across finite theta intervals
d2050b8ab tools: solve intervals and fingerprint large model archives
00edea9fe experiments: validate every point of the solved interval
04da5c3db solver: reuse immutable readiness and forward task order
3dde40950 solver: exclude failed geometries before spending the budget
f1f58ba7d experiments: compare interval tables with executed host queues
167761edb experiments: record hardware context around paired processes
c8b78c043 experiments: replay historical paths through prepared readiness
85053f669 docs: record the repaired prices and interval writeback coverage
50b140b0c experiments: separate readiness setup from full evaluation time
f0a53d8b3 experiments: freeze corrected-price gqa2 configurations
5e7ab8b62 solver: prepare task bounds and reuse changed event rows
1611afc38 experiments: freeze the bound-ranked gqa2 s4 choice
0039d4a9f experiments: audit the connected model numerical failure
b1c11ba2d solver: keep recurrence state per worker where possible
de76675d5 experiments: sample hardware during paired process runs
ce4c6a496 experiments: retain cold timings for the compact recurrence
f6843dc38 docs: record the corrected outer bounds and cold budget
542571e75 docs: distinguish repaired omissions from remaining gate failures
6a1e5b4df experiments: freeze the bound-ranked mha4 s4 choice
d3abc3198 experiments: recheck legacy tables and all sequence cells
1f8f9e2e3 experiments: freeze the bound-ranked gqa2 s128 choice
1a0dc22b0 experiments: self-check the updated target model runner
fd794ee28 experiments: verify the complete continuation evidence scope
25ff9ae35 experiments: freeze the bound-ranked mha4 s128 choice
d428c0f8b experiments: freeze the bound-ranked real s4 choice
78dc9e67f experiments: certify current winners at their actual grid
138d5ee83 experiments: prove the current reference placement domains
34c1d7dd8 docs: record current-grid proofs and completed legacy checks
7fe8163cc experiments: prove the real-width wavefront interval
11d4851e6 experiments: confirm the bound-ranked gqa2 s4 candidate
8605ab128 docs: identify the rule controlling model numerical admission
c3637ea37 experiments: report every physical task price by backend kind
c5e0313e8 docs: include the maximal model command and exact stop rule
4481f8561 experiments: localize the remaining outer-search preparation cost
1cfca7abc experiments: derive incremental protocol effects from raw pairs
ebcccefae experiments: match the original pooled placement comparison
14c0e3cba experiments: confirm the bound-ranked mha4 s4 candidate
6fc86c8c4 experiments: validate all selected sequence and past cells
0bc12e861 solver: share consecutive successor intervals
2ebba228e experiments: measure interval storage and verify exact predictions
6f3c7623f docs: record interval storage equivalence and remaining budget gap
87fc843ff experiments: self-check target runners after interval storage
96107ef75 experiments: confirm the bound-ranked gqa2 s128 candidate
5ad408aac experiments: recalibrate and revalidate on the target host
d2eb8899a experiments: retain the fifth bound-ranked confirmation
c8ecdd732 solver: derive prefix traffic from the graph state effect
2ae6ec79e experiments: verify strict cost dispatch and portable runners
4d575936d docs: record state-effect costs and target portability checks
6b84321bc analysis: stream verified rectangular relation pieces
ebb9432cb experiments: retain exact relation-streaming evidence
ef20e1196 docs: distinguish the active closure from its prior checkpoint
3839031dd analysis: enumerate dense dependency slices exactly
d5e5130e0 docs: record exact dependency slice validation
c127d4cb2 experiments: pin the dependency slice search tools
c04d21702 experiments: self-check the final target runners
71c834454 docs: reserve finding IDs for concurrent target evidence
61165f0ac docs: reconcile concurrent architecture findings
c5380184c docs: attribute concurrent target evidence without overwriting edits
685c94a8c experiments: archive the completed real s128 bound search
f415f42a0 docs: expose the full search preparation cost
f52bf9cac docs: track the completed bound search and pending measurements
ad2c2e9db experiments: freeze the bound-ranked real s128 choice
de4f93c22 experiments: compare complete tables through both plan carriers
3b39dcc19 experiments: compare native and symbolic execution queues
3f5d79efb docs: require complete carrier and queue evidence
4a5b3a437 experiments: preserve all selected task prices and fit witnesses
d651cb6d4 experiments: preserve the real s128 confirmation pairs
d9da962d6 experiments: archive all solved-config ablation pairs
0bf008bee docs: record the corrected search and fusion bounds
aac91c523 experiments: preserve every ablation trace and probe control
948bdfb79 experiments: check the identical-kernel timing controls
d86529d64 experiments: stamp the pre-report sass identity
```

## 2. Gate results

| Gate | Result | Type | Measurement and raw evidence |
| --- | --- | --- | --- |
| C-a | PASS | hard | command=rg -n StageKind\|NonGemmStageNs lib/Solver/CostModel.cpp lib/Solver/ChainDP.cpp lib/Solver/CouplingInterfaceDP.cpp lib/Solver/TaskModel.cpp lib/Solver/ScalarTaskWork.cpp; output=''; whole-tree command=rg -n StageKind lib/Solver; reviewed non-price roles={'ModelDescription.cpp': 'model parsing', 'AlignmentPropagation.cpp': 'alignment constraints', 'RuntimeProjection.cpp': 'runtime ownership and dependency projection', 'AttentionWork.cpp': 'attention semantic/resource contract validation'}; full grep evidence=COSTMODEL/stagekind_audit.txt |
| C-b | PASS | hard | n=18 rho=0.896800825593 required=0.880288958; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation) |
| C-c | PASS | hard | n=18 rho=0.876160990712 required=0.85; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation) |
| C-d | PASS | report | n=68 relative_error p50=0.045399639 p90=0.108475523 max=0.138277099; docs/experiments/COSTMODEL/closure_replay/replay.tsv |
| C-e | PASS | report | FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4; (iteration_ns,fixed_ns)=[(494.5739772318567, 84.15595062406445), (519.3551879204905, 219.34076673175684), (504.26682039442346, 52.23934537324856), (318.9449407298156, 67.03824809284495)]; scope=instrumented GEMM mainloops, SIMT wait unmeasured; COSTMODEL/raw_kloop/*/phase/selected/dump |
| C-f | PASS | hard | 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s128/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s128/correctness; models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity |
| J-a | PASS | research | real_s4 0.547671353 [0.539770996,0.568571846] docs/experiments/JOINT2/bounded_search/real_s4/measure; real_s128 0.944310839 [0.943726539,0.981891257] docs/experiments/JOINT2/bounded_search/real_s128/measure |
| J-b | FAIL | hard | real_s4 queue/semantic_CP=2.317659352 /root/TileMega/docs/experiments/JOINT2/bounded_search/real_s4/trace/top2/dump; real_s128 queue/semantic_CP=2.182635901 /root/TileMega/docs/experiments/JOINT2/bounded_search/real_s128/trace/top1/dump |
| J-c | FAIL | hard | reference full_max_us=1555.545 budget=1000 at mha4/s512/legacy_grid_stride; prepare_max_us=178093.298; real full_max_us=2028.897 budget=10000 at real/s128/legacy_grid_stride; prepare_max_us=139360.244; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv |
| J-outer | PASS | hard | gqa2_s4 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/gqa2_s4; gqa2_s128 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/gqa2_s128; mha4_s4 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/mha4_s4; mha4_s128 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/mha4_s128; real_s4 outer=90 inner=288 admissible=all; docs/experiments/JOINT2/bounded_search/real_s4; real_s128 outer=90 inner=252 admissible=all; docs/experiments/JOINT2/bounded_search/real_s128 |
| J-d | FAIL | hard | 2/6 top1 in measured top2; gqa2_s4 predicted_top1=3/3; gqa2_s128 predicted_top1=1/3; mha4_s4 predicted_top1=3/3; mha4_s128 predicted_top1=2/3; real_s4 predicted_top1=3/3; real_s128 predicted_top1=3/3 |
| J-e | FAIL | hard | gqa2_s4 1.090909091 [1.062000000,1.107498689] docs/experiments/JOINT2/bounded_search/gqa2_s4/measure; gqa2_s128 1.330033937 [1.308370044,1.354607899] docs/experiments/JOINT2/bounded_search/gqa2_s128/measure; mha4_s4 1.175841796 [1.164324182,1.226326021] docs/experiments/JOINT2/bounded_search/mha4_s4/measure; mha4_s128 1.309298156 [1.301310044,1.314176000] docs/experiments/JOINT2/bounded_search/mha4_s128/measure |
| J-f | PASS | hard | 50/50 docs/experiments/JOINT2/bounded_search/gqa2_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/gqa2_s128/correctness; 50/50 docs/experiments/JOINT2/bounded_search/mha4_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/mha4_s128/correctness; 50/50 docs/experiments/JOINT2/bounded_search/real_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/real_s128/correctness; selected SEQSCAN 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s4_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s4_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s128_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s128_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s4_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s4_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s128_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s128_p512/correctness |
| J-g | PASS | report | FUSE6 enter_r7=0 maximum_bound_share=0.015426 cells=6; gqa2_s4 supported=2 upper_ns=1263.712247 share=0.015050 docs/experiments/JOINT2/bounded_fuse/gqa2_s4_selected.tsv; gqa2_s128 supported=2 upper_ns=1302.737889 share=0.006267 docs/experiments/JOINT2/bounded_fuse/gqa2_s128_selected.tsv; mha4_s4 supported=4 upper_ns=2527.424495 share=0.015426 docs/experiments/JOINT2/bounded_fuse/mha4_s4_selected.tsv; mha4_s128 supported=4 upper_ns=2761.578343 share=0.006421 docs/experiments/JOINT2/bounded_fuse/mha4_s128_selected.tsv; real_s4 supported=4 upper_ns=2652.823643 share=0.001168 docs/experiments/JOINT2/bounded_fuse/real_s4_selected.tsv; real_s128 supported=4 upper_ns=11425.094939 share=0.001999 docs/experiments/JOINT2/bounded_fuse/real_s128_selected.tsv |
| W-a | PASS | hard | rg -n setAttr include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h => ['48:module->setAttr(kPlacementTableAttr,b.getDictionaryAttr({', '55:placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));', '56:placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));', '57:placement->setAttr("window",b.getI64IntegerAttr(1));', '58:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '59:placement->setAttr("resident_only",b.getBoolAttr(true));', '63:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));', '64:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));', '66:module->setAttr("tilemega.solved_placement",b.getStringAttr(selected.name));', '67:module->setAttr("tilemega.solved_kappa",b.getI64IntegerAttr(options.kappa));', '68:module->setAttr("tilemega.solved_residency",b.getI64IntegerAttr(options.residency));', '69:module->setAttr("tilemega.solved_seq",b.getI64IntegerAttr(options.dims.seq));', '70:module->setAttr("tilemega.solved_past",b.getI64IntegerAttr(options.dims.past));', '71:module->setAttr("tilemega.solved_grid",b.getI64IntegerAttr(grid));', '72:module->setAttr("tilemega.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));', '217:module->setAttr("tilemega.event_cost_calibrated",mlir::BoolAttr::get(module.getContext(),', '296:placement->setAttr("mode",b.getStringAttr("eft"));', '297:placement->setAttr("params",b.getDenseI64ArrayAttr({}));', '298:placement->setAttr("window",b.getI64IntegerAttr(1));', '299:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '300:placement->setAttr("resident_only",b.getBoolAttr(true));', '305:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),map));', '306:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),map));', '312:module->setAttr(kPlacementTableAttr,table.getDictionary(module.getContext()));', '314:module->setAttr("tilemega.solved_seq_begin",b.getI64IntegerAttr(begin));', '315:module->setAttr("tilemega.solved_seq_end",b.getI64IntegerAttr(end));', '316:module->setAttr("tilemega.solved_placement",b.getStringAttr("finite_theta_interval"));']; docs/experiments/WRITEBACK/gqa2_resident_auto.mlir |
| W-b | PASS | hard | command=['build-portable/tools/tilemega-compile', 'docs/experiments/MODELS/llama_mlp/exported_program.pt2', 'docs/experiments/MODELS/llama_mlp/direct_pt2.cu', '--solve', 'docs/experiments/COSTMODEL/event_fit/target.json', '--seq', '4', '--past', '3', '--search-capacity', '3', '--search-domain', 'docs/experiments/COSTMODEL/event_fit/search_domain.json', '--dump-cg', 'docs/experiments/MODELS/llama_mlp/direct_pt2.mlir', '--hop-curve', 'docs/experiments/SIMULATOR/hop_ns.tsv']; output=docs/experiments/MODELS/llama_mlp/direct_pt2.cu byte_equal_to_50_process_tested_source |
| W-c | PASS | hard | 12/12 byte-identical tables; WRITEBACK/legacy_closure/identity |
| W-d | PASS | hard | point + interval CG/direct solver/generated/host arrays identical; interval nodes=[546, 580, 614, 648, 682]; full direct-injection versus CG path=15/15 schedule/waits/events byte comparisons, fresh executions=10/10; docs/experiments/WRITEBACK/interval_closure; docs/experiments/WRITEBACK/full_roundtrip |
| W-e | PASS | hard | CTest=49 docs/experiments/JOINT2/closure/finite_slices/ctest.log; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s2048_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s2048_p512 |
| W-interval | PASS | hard | 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s1; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s2; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s3; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s4; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s5; one binary, all six placements at every integer point; geometry selected at upper endpoint, fixed past/grid |
| B1 | PASS | report | gqa2_s4 1250/1250; gqa2_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7567567567567567, 0.041728, 0.026720000000000008, 0.023552000000000003, 0.07168, 0.9674502712477397]; selected_nonpositive_barrier_pairs=0; gqa2_s4/wavefront cp_ns=82944.000 queue_ns=75776.000 measured_floor=1.691358 l2_l1=0.713263 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/wavefront__full/dump; gqa2_s4/eft cp_ns=83968.000 queue_ns=79872.000 measured_floor=1.743902 l2_l1=0.748210 relative_selected=(1.049965776865161, 1.0428441203281678, 1.052154195011338) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/eft__full/dump; gqa2_s4/rotate cp_ns=79872.000 queue_ns=41984.000 measured_floor=1.782051 l2_l1=0.723958 relative_selected=(1.0145985401459854, 1.008482142857143, 1.0220588235294117) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/rotate__full/dump; gqa2_s4/chain cp_ns=80896.000 queue_ns=43008.000 measured_floor=1.860759 l2_l1=0.768041 relative_selected=(1.0735294117647058, 1.0714285714285714, 1.0811791383219955) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/chain__full/dump; gqa2_s4/legacy_grid_stride cp_ns=81920.000 queue_ns=104448.000 measured_floor=1.803922 l2_l1=0.963731 relative_selected=(1.343065693430657, 1.322544642857143, 1.35067305498517) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/legacy_grid_stride__full/dump; gqa2_s4/balanced cp_ns=74752.000 queue_ns=121856.000 measured_floor=6.453256 l2_l1=4.376361 relative_selected=(5.6007292616226065, 5.579710144927536, 5.634942528735633) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/balanced__full/dump; gqa2_s4/protocol_off cp_ns=79872.000 queue_ns=73728.000 measured_floor=1.951122 l2_l1=0.759344 relative_selected=(1.1160386029411766, 1.1071428571428572, 1.1176470588235294) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/protocol_off__full/dump; gqa2_s4/c1 relative_previous=wavefront:(0.999776835527784, 0.9927007299270073, 1.0059742647058822); gqa2_s4/c1 cp_ns=81920.000 queue_ns=77824.000 measured_floor=1.704297 l2_l1=0.710124 relative_selected=(0.999776835527784, 0.9927007299270073, 1.0059742647058822) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c1__full/dump; gqa2_s4/c2 relative_previous=c1:(1.0, 0.9924242424242424, 1.0011423349326023); gqa2_s4/c2 cp_ns=81920.000 queue_ns=73728.000 measured_floor=1.700000 l2_l1=0.709257 relative_selected=(0.9974724264705883, 0.9869981751824818, 1.0004597701149427) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c2__full/dump; gqa2_s4/c3 relative_previous=c2:(1.0025339783460032, 0.9993106617647058, 1.0073529411764706); gqa2_s4/c3 cp_ns=82944.000 queue_ns=77824.000 measured_floor=1.688657 l2_l1=0.711340 relative_selected=(1.0, 0.9927007299270073, 1.0006696428571429) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c3__full/dump; gqa2_s128 1250/1250; gqa2_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[1.0480769230769231, 0.09222399999999997, 0.023552000000000017, 0.030399999999999983, 0.09936000000000003, 1.165703275529865]; selected_nonpositive_barrier_pairs=0; gqa2_s128/eft cp_ns=203776.000 queue_ns=209920.000 measured_floor=1.521951 l2_l1=0.948993 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/eft__full/dump; gqa2_s128/wavefront cp_ns=188416.000 queue_ns=197632.000 measured_floor=1.533679 l2_l1=0.899696 relative_selected=(0.9515224358974359, 0.948671679197995, 0.965309805494285) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/wavefront__full/dump; gqa2_s128/rotate cp_ns=201728.000 queue_ns=178176.000 measured_floor=1.664975 l2_l1=0.993638 relative_selected=(1.0511182108626196, 1.0354113237324916, 1.0541401273885351) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/rotate__full/dump; gqa2_s128/legacy_grid_stride cp_ns=191488.000 queue_ns=225280.000 measured_floor=1.413352 l2_l1=1.033861 relative_selected=(1.0, 0.990843949044586, 1.0479611240228184) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/legacy_grid_stride__full/dump; gqa2_s128/chain cp_ns=201728.000 queue_ns=187392.000 measured_floor=1.695431 l2_l1=1.018579 relative_selected=(1.071186440677966, 1.0541259982253772, 1.073482428115016) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/chain__full/dump; gqa2_s128/balanced cp_ns=163840.000 queue_ns=611328.000 measured_floor=3.586265 l2_l1=7.124075 relative_selected=(6.928226363008973, 6.844560104093684, 7.247457627118644) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/balanced__full/dump; gqa2_s128/protocol_off cp_ns=183296.000 queue_ns=185344.000 measured_floor=1.805767 l2_l1=0.976119 relative_selected=(1.038469126060367, 1.0169491525423728, 1.0477707006369426) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/protocol_off__full/dump; gqa2_s128/c1 relative_previous=eft:(0.990675756968117, 0.9872204472843449, 1.0063694267515924); gqa2_s128/c1 cp_ns=202752.000 queue_ns=209920.000 measured_floor=1.507317 l2_l1=0.940663 relative_selected=(0.990675756968117, 0.9872204472843449, 1.0063694267515924) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c1__full/dump; gqa2_s128/c2 relative_previous=c1:(1.000404694455686, 0.9968968968968969, 1.0033373786407767); gqa2_s128/c2 cp_ns=201728.000 queue_ns=207872.000 measured_floor=1.527094 l2_l1=0.942339 relative_selected=(0.9935897435897436, 0.9898162939297124, 0.9968152866242039) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c2__full/dump; gqa2_s128/c3 relative_previous=c2:(1.0, 0.9968051118210862, 1.0032362459546926); gqa2_s128/c3 cp_ns=205824.000 queue_ns=208896.000 measured_floor=1.519455 l2_l1=0.942249 relative_selected=(0.9945859234008422, 0.9875199680511182, 1.0095541401273886) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c3__full/dump; mha4_s4 1250/1250; mha4_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.8672566371681416, 0.08569600000000002, 0.057344000000000006, 0.048831999999999987, 0.145056, 0.9904719698648352]; selected_nonpositive_barrier_pairs=0; mha4_s4/wavefront cp_ns=164864.000 queue_ns=165888.000 measured_floor=1.759259 l2_l1=0.751851 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/wavefront__full/dump; mha4_s4/eft cp_ns=161792.000 queue_ns=159744.000 measured_floor=1.821005 l2_l1=0.757488 relative_selected=(1.0070175438596491, 1.0035087719298246, 1.0104166666666665) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/eft__full/dump; mha4_s4/rotate cp_ns=166912.000 queue_ns=86016.000 measured_floor=1.791028 l2_l1=0.767728 relative_selected=(1.0206185567010309, 1.0091250670960816, 1.024561403508772) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/rotate__full/dump; mha4_s4/chain cp_ns=161792.000 queue_ns=84992.000 measured_floor=1.930380 l2_l1=0.804749 relative_selected=(1.0677168799912482, 1.0547945205479452, 1.0709429824561403) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/chain__full/dump; mha4_s4/legacy_grid_stride cp_ns=162816.000 queue_ns=210944.000 measured_floor=1.621359 l2_l1=0.960280 relative_selected=(1.1707236842105264, 1.1685982943363218, 1.192982456140351) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/legacy_grid_stride__full/dump; mha4_s4/balanced cp_ns=156672.000 queue_ns=248832.000 measured_floor=13.064815 l2_l1=8.365372 relative_selected=(11.112280701754386, 11.048060344827586, 11.161184210526317) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/balanced__full/dump; mha4_s4/protocol_off cp_ns=162816.000 queue_ns=157696.000 measured_floor=2.007469 l2_l1=0.811558 relative_selected=(1.1162790697674416, 1.1022336769759449, 1.1196271929824562) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/protocol_off__full/dump; mha4_s4/c1 relative_previous=wavefront:(0.993984468992672, 0.9862068965517242, 0.9964912280701755); mha4_s4/c1 cp_ns=164864.000 queue_ns=158720.000 measured_floor=1.763975 l2_l1=0.747553 relative_selected=(0.993984468992672, 0.9862068965517242, 0.9964912280701755) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c1__full/dump; mha4_s4/c2 relative_previous=c1:(1.0007708402158353, 0.9902224132373484, 1.0038554747741795); mha4_s4/c2 cp_ns=166912.000 queue_ns=160768.000 measured_floor=1.746741 l2_l1=0.749712 relative_selected=(0.9974755789704753, 0.9851645287228108, 1.0002193944712592) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c2__full/dump; mha4_s4/c3 relative_previous=c2:(0.9982442664325688, 0.9829241071428573, 1.0034865983874481); mha4_s4/c3 cp_ns=166912.000 queue_ns=165888.000 measured_floor=1.742331 l2_l1=0.748484 relative_selected=(0.9908865886588659, 0.9743421052631579, 0.9974844143060264) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c3__full/dump; mha4_s128 1250/1250; mha4_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9809223930357474, 0.17203200000000002, 0.05286400000000002, 0.05926399999999998, 0.20684800000000003, 1.0901741293532337]; selected_nonpositive_barrier_pairs=0; mha4_s128/eft cp_ns=400384.000 queue_ns=430080.000 measured_floor=1.530655 l2_l1=0.964126 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/eft__full/dump; mha4_s128/wavefront cp_ns=369664.000 queue_ns=406528.000 measured_floor=1.556596 l2_l1=0.927849 relative_selected=(0.9625194704049844, 0.9597818677573279, 0.9641674780915287) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/wavefront__full/dump; mha4_s128/chain cp_ns=397312.000 queue_ns=387072.000 measured_floor=1.767800 l2_l1=1.029889 relative_selected=(1.0671170295489891, 1.0652616279069769, 1.0701053042121687) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/chain__full/dump; mha4_s128/legacy_grid_stride cp_ns=346112.000 queue_ns=416768.000 measured_floor=1.663237 l2_l1=1.015038 relative_selected=(1.0532671146168078, 1.0512382995319813, 1.0578124999999998) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/legacy_grid_stride__full/dump; mha4_s128/rotate cp_ns=342016.000 queue_ns=336896.000 measured_floor=1.988024 l2_l1=0.995315 relative_selected=(1.034321372854914, 1.0311526479750779, 1.0376862401402278) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/rotate__full/dump; mha4_s128/balanced cp_ns=304128.000 queue_ns=1272832.000 measured_floor=4.870475 l2_l1=9.081006 relative_selected=(9.41699066874028, 9.39641472868217, 9.44461778471139) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/balanced__full/dump; mha4_s128/protocol_off cp_ns=362496.000 queue_ns=396288.000 measured_floor=1.744590 l2_l1=1.001202 relative_selected=(1.0514820592823713, 1.0482115085536547, 1.0541407465007777) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/protocol_off__full/dump; mha4_s128/c1 relative_previous=eft:(0.9955279020027222, 0.9933999805881781, 0.996875); mha4_s128/c1 cp_ns=398336.000 queue_ns=430080.000 measured_floor=1.521801 l2_l1=0.959581 relative_selected=(0.9955279020027222, 0.9933999805881781, 0.996875) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c1__full/dump; mha4_s128/c2 relative_previous=c1:(0.9124999999999999, 0.9119229264475743, 0.9155355922273216); mha4_s128/c2 cp_ns=402432.000 queue_ns=431104.000 measured_floor=1.384798 l2_l1=0.960956 relative_selected=(0.9082426127527216, 0.9065420560747665, 0.9107910156249999) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c2__full/dump; mha4_s128/c3 relative_previous=c2:(1.0981687342247999, 1.0960548885077188, 1.0999570999570998); mha4_s128/c3 cp_ns=397312.000 queue_ns=425984.000 measured_floor=1.540865 l2_l1=0.961479 relative_selected=(0.9968992248062016, 0.9965493779160186, 0.999951550387597) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c3__full/dump; real_s4 1250/1250; real_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9629777023468133, -0.01123200000000013, 0.030719999999999636, 0.035136000000000056, 0.19257599999999986, 0.10754495042849974]; selected_nonpositive_barrier_pairs=0; real_s4/wavefront cp_ns=974848.000 queue_ns=2274304.000 measured_floor=1.121117 l2_l1=0.923458 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/wavefront__full/dump; real_s4/eft cp_ns=973824.000 queue_ns=2271232.000 measured_floor=1.128945 l2_l1=0.928815 relative_selected=(1.0057802743436064, 1.0048260733312442, 1.0060265166733628) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/eft__full/dump; real_s4/rotate cp_ns=1007616.000 queue_ns=1927168.000 measured_floor=1.364008 l2_l1=0.952442 relative_selected=(1.0313378867014866, 1.0308988764044944, 1.0314291808338458) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/rotate__full/dump; real_s4/legacy_grid_stride cp_ns=1011712.000 queue_ns=2350080.000 measured_floor=1.162092 l2_l1=0.989109 relative_selected=(1.0709520465489568, 1.0705168392807882, 1.0715146645239053) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/legacy_grid_stride__full/dump; real_s4/chain cp_ns=971776.000 queue_ns=1951744.000 measured_floor=1.378443 l2_l1=0.974495 relative_selected=(1.055020080321285, 1.0546230084054697, 1.0552699551119693) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/chain__full/dump; real_s4/balanced cp_ns=808960.000 queue_ns=3873792.000 measured_floor=6.033835 l2_l1=8.468043 relative_selected=(9.167268351383875, 9.154379584926955, 9.171649749695746) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/balanced__full/dump; real_s4/protocol_off cp_ns=978944.000 queue_ns=2219008.000 measured_floor=1.177276 l2_l1=0.938926 relative_selected=(1.0249096022498996, 1.0239696802369358, 1.0256178307137562) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/protocol_off__full/dump; real_s4/c1 relative_previous=wavefront:(1.0021203718806069, 1.0016070711128968, 1.0031388107673764); real_s4/c1 cp_ns=972800.000 queue_ns=2265088.000 measured_floor=1.128391 l2_l1=0.925699 relative_selected=(1.0021203718806069, 1.0016070711128968, 1.0031388107673764) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c1__full/dump; real_s4/c2 relative_previous=c1:(1.0005507090378862, 1.0000125159578463, 1.0012159045326916); real_s4/c2 cp_ns=975872.000 queue_ns=2278400.000 measured_floor=1.122219 l2_l1=0.926132 relative_selected=(1.002836523376216, 1.0023324638844302, 1.003151366010446) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c2__full/dump; real_s4/c3 relative_previous=c2:(0.9999124649859945, 0.9995994943616315, 1.000262815378454); real_s4/c3 cp_ns=974848.000 queue_ns=2273280.000 measured_floor=1.124718 l2_l1=0.925961 relative_selected=(1.002748381793897, 1.002296571457256, 1.0030885897950985) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c3__full/dump; real_s128 1250/1250; real_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9678012537729277, 0.523968, 0.12288000000000032, 0.16771199999999986, 0.3002560000000001, 2.0940270648623427]; selected_nonpositive_barrier_pairs=3; real_s128/eft cp_ns=2606080.000 queue_ns=5701632.000 measured_floor=1.153556 l2_l1=0.986411 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/eft__full/dump; real_s128/wavefront cp_ns=2719744.000 queue_ns=6149120.000 measured_floor=1.078435 l2_l1=0.995621 relative_selected=(1.0090834665315194, 1.0065113024548582, 1.0098069738480697) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/wavefront__full/dump; real_s128/rotate cp_ns=2816000.000 queue_ns=5491712.000 measured_floor=1.213547 l2_l1=0.999551 relative_selected=(1.0131832141191106, 1.0122766122766123, 1.0137028962939896) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/rotate__full/dump; real_s128/chain cp_ns=2778112.000 queue_ns=6243328.000 measured_floor=1.122996 l2_l1=1.053027 relative_selected=(1.0655005057578586, 1.0499698302709437, 1.0683478785402256) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/chain__full/dump; real_s128/legacy_grid_stride cp_ns=2864128.000 queue_ns=6403072.000 measured_floor=1.068607 l2_l1=1.032248 relative_selected=(1.0412644036125818, 1.0133680235323814, 1.04839338066992) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/legacy_grid_stride__full/dump; real_s128/balanced cp_ns=2104320.000 queue_ns=18079744.000 measured_floor=4.474782 l2_l1=12.135989 relative_selected=(12.3082312231577, 12.283934786785881, 12.33038727348373) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/balanced__full/dump; real_s128/protocol_off cp_ns=2561024.000 queue_ns=5681152.000 measured_floor=1.182949 l2_l1=1.007333 relative_selected=(1.0230132643535086, 1.0211414227895392, 1.024302850911357) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/protocol_off__full/dump; real_s128/c1 relative_previous=eft:(0.9993870581710984, 0.9867642478978511, 1.0007637375466998); real_s128/c1 cp_ns=2733056.000 queue_ns=5966848.000 measured_floor=1.101253 l2_l1=0.985420 relative_selected=(0.9993870581710984, 0.9867642478978511, 1.0007637375466998) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c1__full/dump; real_s128/c2 relative_previous=c1:(1.0055593959633529, 1.0038880248833593, 1.0070093457943925); real_s128/c2 cp_ns=2588672.000 queue_ns=5692416.000 measured_floor=1.160640 l2_l1=0.990922 relative_selected=(1.0047488715047284, 1.0041545997006744, 1.0062579746564202) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c2__full/dump; real_s128/c3 relative_previous=c2:(0.9990707759021218, 0.9979891724671307, 1.0006196746707978); real_s128/c3 cp_ns=2604032.000 queue_ns=5689344.000 measured_floor=1.161267 l2_l1=0.991205 relative_selected=(1.0043684266853468, 1.0037290242386574, 1.0049813200498132) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c3__full/dump; placement_pool {'group': 'reference', 'cells': 4, 'pairs': 100, 'median_ratio': 0.8959319950530888, 'ci_lo': 0.8708708708708708, 'ci_hi': 0.9763663220088626, 'bootstrap_draws': 20000, 'seed': 20260906}; placement_pool {'group': 'real', 'cells': 2, 'pairs': 50, 'median_ratio': 0.9635739126503015, 'ci_lo': 0.9631847425184086, 'ci_hi': 0.967151369350876, 'bootstrap_draws': 20000, 'seed': 20260906}; supplemental_W1 c3_a/c2_a=(1.0002433800623054, 0.9965976475162829, 1.002112713251994); supplemental_W1 c2_b/c2_a=(1.0, 0.9740680713128038, 1.0020028332763422); supplemental_W1 c3_b/c3_a=(0.998732449297972, 0.9707723372034326, 1.0018984568952929); six C2/C3 W1 kernel sets byte-identical; supplemental controls=100/100; original 7500 samples retained; causal timing-band origin unresolved; REBASE/bounded_raw/*/measure |
| S-a | PASS | hard | legacy_grid_stride/G256 proved=128/128; legacy_grid_stride/G340 proved=128/128; rotate/G256 proved=128/128; rotate/G340 proved=128/128; band/G256 proved=128/128; band/G340 proved=128/128; wavefront/G256 proved=128/128; wavefront/G340 proved=128/128; gqa2_s4 binary_variants=1; gqa2_s128 binary_variants=1; mha4_s4 binary_variants=1; mha4_s128 binary_variants=1; real_s4 binary_variants=1; real_s128 binary_variants=1; SYMBOLIC/complete/proofs.tsv and JOINT2 selected raw trace resource records |
| S-b | PASS | hard | 40/40 endpoint/interior pi/sigma byte comparisons; actual queue-vector comparisons=55/55 including fitted current winners; SYMBOLIC/complete/ and SYMBOLIC/queue_roundtrip/ |
| S-c | PASS | hard | gqa2_s4=wavefront; gqa2_s128=outside_these_four_template_families; mha4_s4=wavefront; mha4_s128=outside_these_four_template_families; real_s4=wavefront; real_s128=outside_these_four_template_families; SYMBOLIC/bounded_fit/*/fit.tsv and complete native/symbolic/selected tables; point witnesses do not assert impossibility in every quasi-affine family |
| S-current | PASS | hard | gqa2_s4 family=wavefront grid=512 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/gqa2_s4; gqa2_s128 outside four families; original winner retained; mha4_s4 family=wavefront grid=512 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/mha4_s4; mha4_s128 outside four families; original winner retained; real_s4 family=wavefront grid=256 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/real_s4; real_s128 outside four families; original winner retained |
| S-d | PASS | report | 16 template/champion/fresh-solve comparisons recomputed from complete raw tables; 40 CG round trips; CPU only; SYMBOLIC/cross_grid/ |
| A1-subset | FAIL | report | 0/50 docs/experiments/MODELS/covered_llama_admitted2/correctness missing=49; maximal connected graph checked_outputs=66 mismatches=[(0, 1)]; docs/experiments/MODELS/covered_llama_admitted2/correctness/r0.log; earlier independent MLP subset=True 50/50 docs/experiments/MODELS/llama_mlp/correctness does not satisfy maximal-graph admission; no tolerance or output changes |
| H2 | PASS | hard | models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity |
| H4 | PASS | hard | continuation derived combine before bound repair; interval writer before current winner audits; C1 before J1; C2 before J1; FORK6 before Fuse R7 decision; W2 before symbolic proof implementation; gqa2_s4 selection 1611afc3 precedes all B1 measurements; gqa2_s128 selection 1f8f9e2e precedes all B1 measurements; mha4_s4 selection 6a1e5b4d precedes all B1 measurements; mha4_s128 selection 25ff9ae3 precedes all B1 measurements; real_s4 selection d428c0f8 precedes all B1 measurements; real_s128 selection ad2c2e9d precedes all B1 measurements |
| H7 | PASS | report | four local compile/guard checks; no sm_120 execution claim; docs/experiments/COSTMODEL/selfcheck_sm120_slices; docs/experiments/JOINT2/selfcheck_sm120_slices; docs/experiments/REBASE/selfcheck_sm120_slices; docs/experiments/MODELS/selfcheck_sm120_slices |

## 3. Complete verification output

```text
C-a PASS [hard] command=rg -n StageKind|NonGemmStageNs lib/Solver/CostModel.cpp lib/Solver/ChainDP.cpp lib/Solver/CouplingInterfaceDP.cpp lib/Solver/TaskModel.cpp lib/Solver/ScalarTaskWork.cpp; output=''; whole-tree command=rg -n StageKind lib/Solver; reviewed non-price roles={'ModelDescription.cpp': 'model parsing', 'AlignmentPropagation.cpp': 'alignment constraints', 'RuntimeProjection.cpp': 'runtime ownership and dependency projection', 'AttentionWork.cpp': 'attention semantic/resource contract validation'}; full grep evidence=COSTMODEL/stagekind_audit.txt
C-b PASS [hard] n=18 rho=0.896800825593 required=0.880288958; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation)
C-c PASS [hard] n=18 rho=0.876160990712 required=0.85; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation)
C-d PASS [report] n=68 relative_error p50=0.045399639 p90=0.108475523 max=0.138277099; docs/experiments/COSTMODEL/closure_replay/replay.tsv
C-e PASS [report] FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4; (iteration_ns,fixed_ns)=[(494.5739772318567, 84.15595062406445), (519.3551879204905, 219.34076673175684), (504.26682039442346, 52.23934537324856), (318.9449407298156, 67.03824809284495)]; scope=instrumented GEMM mainloops, SIMT wait unmeasured; COSTMODEL/raw_kloop/*/phase/selected/dump
C-f PASS [hard] 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s128/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s128/correctness; models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity
J-a PASS [research] real_s4 0.547671353 [0.539770996,0.568571846] docs/experiments/JOINT2/bounded_search/real_s4/measure; real_s128 0.944310839 [0.943726539,0.981891257] docs/experiments/JOINT2/bounded_search/real_s128/measure
J-b FAIL [hard] real_s4 queue/semantic_CP=2.317659352 /root/TileMega/docs/experiments/JOINT2/bounded_search/real_s4/trace/top2/dump; real_s128 queue/semantic_CP=2.182635901 /root/TileMega/docs/experiments/JOINT2/bounded_search/real_s128/trace/top1/dump
J-c FAIL [hard] reference full_max_us=1555.545 budget=1000 at mha4/s512/legacy_grid_stride; prepare_max_us=178093.298; real full_max_us=2028.897 budget=10000 at real/s128/legacy_grid_stride; prepare_max_us=139360.244; docs/experiments/COSTMODEL/closure_effects/evaluations.tsv
J-outer PASS [hard] gqa2_s4 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/gqa2_s4; gqa2_s128 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/gqa2_s128; mha4_s4 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/mha4_s4; mha4_s128 outer=90 inner=360 admissible=all; docs/experiments/JOINT2/bounded_search/mha4_s128; real_s4 outer=90 inner=288 admissible=all; docs/experiments/JOINT2/bounded_search/real_s4; real_s128 outer=90 inner=252 admissible=all; docs/experiments/JOINT2/bounded_search/real_s128
J-d FAIL [hard] 2/6 top1 in measured top2; gqa2_s4 predicted_top1=3/3; gqa2_s128 predicted_top1=1/3; mha4_s4 predicted_top1=3/3; mha4_s128 predicted_top1=2/3; real_s4 predicted_top1=3/3; real_s128 predicted_top1=3/3
J-e FAIL [hard] gqa2_s4 1.090909091 [1.062000000,1.107498689] docs/experiments/JOINT2/bounded_search/gqa2_s4/measure; gqa2_s128 1.330033937 [1.308370044,1.354607899] docs/experiments/JOINT2/bounded_search/gqa2_s128/measure; mha4_s4 1.175841796 [1.164324182,1.226326021] docs/experiments/JOINT2/bounded_search/mha4_s4/measure; mha4_s128 1.309298156 [1.301310044,1.314176000] docs/experiments/JOINT2/bounded_search/mha4_s128/measure
J-f PASS [hard] 50/50 docs/experiments/JOINT2/bounded_search/gqa2_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/gqa2_s128/correctness; 50/50 docs/experiments/JOINT2/bounded_search/mha4_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/mha4_s128/correctness; 50/50 docs/experiments/JOINT2/bounded_search/real_s4/correctness; 50/50 docs/experiments/JOINT2/bounded_search/real_s128/correctness; selected SEQSCAN 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s4_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s4_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s128_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/gqa2_s128_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s4_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s4_p512/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s128_p0/correctness; 50/50 docs/experiments/JOINT2/bounded_seqscan/mha4_s128_p512/correctness
J-g PASS [report] FUSE6 enter_r7=0 maximum_bound_share=0.015426 cells=6; gqa2_s4 supported=2 upper_ns=1263.712247 share=0.015050 docs/experiments/JOINT2/bounded_fuse/gqa2_s4_selected.tsv; gqa2_s128 supported=2 upper_ns=1302.737889 share=0.006267 docs/experiments/JOINT2/bounded_fuse/gqa2_s128_selected.tsv; mha4_s4 supported=4 upper_ns=2527.424495 share=0.015426 docs/experiments/JOINT2/bounded_fuse/mha4_s4_selected.tsv; mha4_s128 supported=4 upper_ns=2761.578343 share=0.006421 docs/experiments/JOINT2/bounded_fuse/mha4_s128_selected.tsv; real_s4 supported=4 upper_ns=2652.823643 share=0.001168 docs/experiments/JOINT2/bounded_fuse/real_s4_selected.tsv; real_s128 supported=4 upper_ns=11425.094939 share=0.001999 docs/experiments/JOINT2/bounded_fuse/real_s128_selected.tsv
W-a PASS [hard] rg -n setAttr include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h => ['48:module->setAttr(kPlacementTableAttr,b.getDictionaryAttr({', '55:placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));', '56:placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));', '57:placement->setAttr("window",b.getI64IntegerAttr(1));', '58:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '59:placement->setAttr("resident_only",b.getBoolAttr(true));', '63:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));', '64:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));', '66:module->setAttr("tilemega.solved_placement",b.getStringAttr(selected.name));', '67:module->setAttr("tilemega.solved_kappa",b.getI64IntegerAttr(options.kappa));', '68:module->setAttr("tilemega.solved_residency",b.getI64IntegerAttr(options.residency));', '69:module->setAttr("tilemega.solved_seq",b.getI64IntegerAttr(options.dims.seq));', '70:module->setAttr("tilemega.solved_past",b.getI64IntegerAttr(options.dims.past));', '71:module->setAttr("tilemega.solved_grid",b.getI64IntegerAttr(grid));', '72:module->setAttr("tilemega.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));', '217:module->setAttr("tilemega.event_cost_calibrated",mlir::BoolAttr::get(module.getContext(),', '296:placement->setAttr("mode",b.getStringAttr("eft"));', '297:placement->setAttr("params",b.getDenseI64ArrayAttr({}));', '298:placement->setAttr("window",b.getI64IntegerAttr(1));', '299:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '300:placement->setAttr("resident_only",b.getBoolAttr(true));', '305:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),map));', '306:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),map));', '312:module->setAttr(kPlacementTableAttr,table.getDictionary(module.getContext()));', '314:module->setAttr("tilemega.solved_seq_begin",b.getI64IntegerAttr(begin));', '315:module->setAttr("tilemega.solved_seq_end",b.getI64IntegerAttr(end));', '316:module->setAttr("tilemega.solved_placement",b.getStringAttr("finite_theta_interval"));']; docs/experiments/WRITEBACK/gqa2_resident_auto.mlir
W-b PASS [hard] command=['build-portable/tools/tilemega-compile', 'docs/experiments/MODELS/llama_mlp/exported_program.pt2', 'docs/experiments/MODELS/llama_mlp/direct_pt2.cu', '--solve', 'docs/experiments/COSTMODEL/event_fit/target.json', '--seq', '4', '--past', '3', '--search-capacity', '3', '--search-domain', 'docs/experiments/COSTMODEL/event_fit/search_domain.json', '--dump-cg', 'docs/experiments/MODELS/llama_mlp/direct_pt2.mlir', '--hop-curve', 'docs/experiments/SIMULATOR/hop_ns.tsv']; output=docs/experiments/MODELS/llama_mlp/direct_pt2.cu byte_equal_to_50_process_tested_source
W-c PASS [hard] 12/12 byte-identical tables; WRITEBACK/legacy_closure/identity
W-d PASS [hard] point + interval CG/direct solver/generated/host arrays identical; interval nodes=[546, 580, 614, 648, 682]; full direct-injection versus CG path=15/15 schedule/waits/events byte comparisons, fresh executions=10/10; docs/experiments/WRITEBACK/interval_closure; docs/experiments/WRITEBACK/full_roundtrip
W-e PASS [hard] CTest=49 docs/experiments/JOINT2/closure/finite_slices/ctest.log; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/gqa2_s2048_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy_closure/seqscan/mha4_s2048_p512
W-interval PASS [hard] 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s1; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s2; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s3; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s4; 50/50 docs/experiments/WRITEBACK/interval_closure/correctness/s5; one binary, all six placements at every integer point; geometry selected at upper endpoint, fixed past/grid
B1 PASS [report] gqa2_s4 1250/1250; gqa2_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7567567567567567, 0.041728, 0.026720000000000008, 0.023552000000000003, 0.07168, 0.9674502712477397]; selected_nonpositive_barrier_pairs=0; gqa2_s4/wavefront cp_ns=82944.000 queue_ns=75776.000 measured_floor=1.691358 l2_l1=0.713263 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/wavefront__full/dump; gqa2_s4/eft cp_ns=83968.000 queue_ns=79872.000 measured_floor=1.743902 l2_l1=0.748210 relative_selected=(1.049965776865161, 1.0428441203281678, 1.052154195011338) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/eft__full/dump; gqa2_s4/rotate cp_ns=79872.000 queue_ns=41984.000 measured_floor=1.782051 l2_l1=0.723958 relative_selected=(1.0145985401459854, 1.008482142857143, 1.0220588235294117) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/rotate__full/dump; gqa2_s4/chain cp_ns=80896.000 queue_ns=43008.000 measured_floor=1.860759 l2_l1=0.768041 relative_selected=(1.0735294117647058, 1.0714285714285714, 1.0811791383219955) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/chain__full/dump; gqa2_s4/legacy_grid_stride cp_ns=81920.000 queue_ns=104448.000 measured_floor=1.803922 l2_l1=0.963731 relative_selected=(1.343065693430657, 1.322544642857143, 1.35067305498517) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/legacy_grid_stride__full/dump; gqa2_s4/balanced cp_ns=74752.000 queue_ns=121856.000 measured_floor=6.453256 l2_l1=4.376361 relative_selected=(5.6007292616226065, 5.579710144927536, 5.634942528735633) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/balanced__full/dump; gqa2_s4/protocol_off cp_ns=79872.000 queue_ns=73728.000 measured_floor=1.951122 l2_l1=0.759344 relative_selected=(1.1160386029411766, 1.1071428571428572, 1.1176470588235294) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/protocol_off__full/dump; gqa2_s4/c1 relative_previous=wavefront:(0.999776835527784, 0.9927007299270073, 1.0059742647058822); gqa2_s4/c1 cp_ns=81920.000 queue_ns=77824.000 measured_floor=1.704297 l2_l1=0.710124 relative_selected=(0.999776835527784, 0.9927007299270073, 1.0059742647058822) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c1__full/dump; gqa2_s4/c2 relative_previous=c1:(1.0, 0.9924242424242424, 1.0011423349326023); gqa2_s4/c2 cp_ns=81920.000 queue_ns=73728.000 measured_floor=1.700000 l2_l1=0.709257 relative_selected=(0.9974724264705883, 0.9869981751824818, 1.0004597701149427) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c2__full/dump; gqa2_s4/c3 relative_previous=c2:(1.0025339783460032, 0.9993106617647058, 1.0073529411764706); gqa2_s4/c3 cp_ns=82944.000 queue_ns=77824.000 measured_floor=1.688657 l2_l1=0.711340 relative_selected=(1.0, 0.9927007299270073, 1.0006696428571429) dump=docs/experiments/REBASE/bounded_raw/gqa2_s4/trace/c3__full/dump; gqa2_s128 1250/1250; gqa2_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[1.0480769230769231, 0.09222399999999997, 0.023552000000000017, 0.030399999999999983, 0.09936000000000003, 1.165703275529865]; selected_nonpositive_barrier_pairs=0; gqa2_s128/eft cp_ns=203776.000 queue_ns=209920.000 measured_floor=1.521951 l2_l1=0.948993 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/eft__full/dump; gqa2_s128/wavefront cp_ns=188416.000 queue_ns=197632.000 measured_floor=1.533679 l2_l1=0.899696 relative_selected=(0.9515224358974359, 0.948671679197995, 0.965309805494285) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/wavefront__full/dump; gqa2_s128/rotate cp_ns=201728.000 queue_ns=178176.000 measured_floor=1.664975 l2_l1=0.993638 relative_selected=(1.0511182108626196, 1.0354113237324916, 1.0541401273885351) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/rotate__full/dump; gqa2_s128/legacy_grid_stride cp_ns=191488.000 queue_ns=225280.000 measured_floor=1.413352 l2_l1=1.033861 relative_selected=(1.0, 0.990843949044586, 1.0479611240228184) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/legacy_grid_stride__full/dump; gqa2_s128/chain cp_ns=201728.000 queue_ns=187392.000 measured_floor=1.695431 l2_l1=1.018579 relative_selected=(1.071186440677966, 1.0541259982253772, 1.073482428115016) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/chain__full/dump; gqa2_s128/balanced cp_ns=163840.000 queue_ns=611328.000 measured_floor=3.586265 l2_l1=7.124075 relative_selected=(6.928226363008973, 6.844560104093684, 7.247457627118644) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/balanced__full/dump; gqa2_s128/protocol_off cp_ns=183296.000 queue_ns=185344.000 measured_floor=1.805767 l2_l1=0.976119 relative_selected=(1.038469126060367, 1.0169491525423728, 1.0477707006369426) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/protocol_off__full/dump; gqa2_s128/c1 relative_previous=eft:(0.990675756968117, 0.9872204472843449, 1.0063694267515924); gqa2_s128/c1 cp_ns=202752.000 queue_ns=209920.000 measured_floor=1.507317 l2_l1=0.940663 relative_selected=(0.990675756968117, 0.9872204472843449, 1.0063694267515924) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c1__full/dump; gqa2_s128/c2 relative_previous=c1:(1.000404694455686, 0.9968968968968969, 1.0033373786407767); gqa2_s128/c2 cp_ns=201728.000 queue_ns=207872.000 measured_floor=1.527094 l2_l1=0.942339 relative_selected=(0.9935897435897436, 0.9898162939297124, 0.9968152866242039) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c2__full/dump; gqa2_s128/c3 relative_previous=c2:(1.0, 0.9968051118210862, 1.0032362459546926); gqa2_s128/c3 cp_ns=205824.000 queue_ns=208896.000 measured_floor=1.519455 l2_l1=0.942249 relative_selected=(0.9945859234008422, 0.9875199680511182, 1.0095541401273886) dump=docs/experiments/REBASE/bounded_raw/gqa2_s128/trace/c3__full/dump; mha4_s4 1250/1250; mha4_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.8672566371681416, 0.08569600000000002, 0.057344000000000006, 0.048831999999999987, 0.145056, 0.9904719698648352]; selected_nonpositive_barrier_pairs=0; mha4_s4/wavefront cp_ns=164864.000 queue_ns=165888.000 measured_floor=1.759259 l2_l1=0.751851 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/wavefront__full/dump; mha4_s4/eft cp_ns=161792.000 queue_ns=159744.000 measured_floor=1.821005 l2_l1=0.757488 relative_selected=(1.0070175438596491, 1.0035087719298246, 1.0104166666666665) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/eft__full/dump; mha4_s4/rotate cp_ns=166912.000 queue_ns=86016.000 measured_floor=1.791028 l2_l1=0.767728 relative_selected=(1.0206185567010309, 1.0091250670960816, 1.024561403508772) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/rotate__full/dump; mha4_s4/chain cp_ns=161792.000 queue_ns=84992.000 measured_floor=1.930380 l2_l1=0.804749 relative_selected=(1.0677168799912482, 1.0547945205479452, 1.0709429824561403) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/chain__full/dump; mha4_s4/legacy_grid_stride cp_ns=162816.000 queue_ns=210944.000 measured_floor=1.621359 l2_l1=0.960280 relative_selected=(1.1707236842105264, 1.1685982943363218, 1.192982456140351) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/legacy_grid_stride__full/dump; mha4_s4/balanced cp_ns=156672.000 queue_ns=248832.000 measured_floor=13.064815 l2_l1=8.365372 relative_selected=(11.112280701754386, 11.048060344827586, 11.161184210526317) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/balanced__full/dump; mha4_s4/protocol_off cp_ns=162816.000 queue_ns=157696.000 measured_floor=2.007469 l2_l1=0.811558 relative_selected=(1.1162790697674416, 1.1022336769759449, 1.1196271929824562) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/protocol_off__full/dump; mha4_s4/c1 relative_previous=wavefront:(0.993984468992672, 0.9862068965517242, 0.9964912280701755); mha4_s4/c1 cp_ns=164864.000 queue_ns=158720.000 measured_floor=1.763975 l2_l1=0.747553 relative_selected=(0.993984468992672, 0.9862068965517242, 0.9964912280701755) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c1__full/dump; mha4_s4/c2 relative_previous=c1:(1.0007708402158353, 0.9902224132373484, 1.0038554747741795); mha4_s4/c2 cp_ns=166912.000 queue_ns=160768.000 measured_floor=1.746741 l2_l1=0.749712 relative_selected=(0.9974755789704753, 0.9851645287228108, 1.0002193944712592) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c2__full/dump; mha4_s4/c3 relative_previous=c2:(0.9982442664325688, 0.9829241071428573, 1.0034865983874481); mha4_s4/c3 cp_ns=166912.000 queue_ns=165888.000 measured_floor=1.742331 l2_l1=0.748484 relative_selected=(0.9908865886588659, 0.9743421052631579, 0.9974844143060264) dump=docs/experiments/REBASE/bounded_raw/mha4_s4/trace/c3__full/dump; mha4_s128 1250/1250; mha4_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9809223930357474, 0.17203200000000002, 0.05286400000000002, 0.05926399999999998, 0.20684800000000003, 1.0901741293532337]; selected_nonpositive_barrier_pairs=0; mha4_s128/eft cp_ns=400384.000 queue_ns=430080.000 measured_floor=1.530655 l2_l1=0.964126 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/eft__full/dump; mha4_s128/wavefront cp_ns=369664.000 queue_ns=406528.000 measured_floor=1.556596 l2_l1=0.927849 relative_selected=(0.9625194704049844, 0.9597818677573279, 0.9641674780915287) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/wavefront__full/dump; mha4_s128/chain cp_ns=397312.000 queue_ns=387072.000 measured_floor=1.767800 l2_l1=1.029889 relative_selected=(1.0671170295489891, 1.0652616279069769, 1.0701053042121687) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/chain__full/dump; mha4_s128/legacy_grid_stride cp_ns=346112.000 queue_ns=416768.000 measured_floor=1.663237 l2_l1=1.015038 relative_selected=(1.0532671146168078, 1.0512382995319813, 1.0578124999999998) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/legacy_grid_stride__full/dump; mha4_s128/rotate cp_ns=342016.000 queue_ns=336896.000 measured_floor=1.988024 l2_l1=0.995315 relative_selected=(1.034321372854914, 1.0311526479750779, 1.0376862401402278) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/rotate__full/dump; mha4_s128/balanced cp_ns=304128.000 queue_ns=1272832.000 measured_floor=4.870475 l2_l1=9.081006 relative_selected=(9.41699066874028, 9.39641472868217, 9.44461778471139) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/balanced__full/dump; mha4_s128/protocol_off cp_ns=362496.000 queue_ns=396288.000 measured_floor=1.744590 l2_l1=1.001202 relative_selected=(1.0514820592823713, 1.0482115085536547, 1.0541407465007777) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/protocol_off__full/dump; mha4_s128/c1 relative_previous=eft:(0.9955279020027222, 0.9933999805881781, 0.996875); mha4_s128/c1 cp_ns=398336.000 queue_ns=430080.000 measured_floor=1.521801 l2_l1=0.959581 relative_selected=(0.9955279020027222, 0.9933999805881781, 0.996875) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c1__full/dump; mha4_s128/c2 relative_previous=c1:(0.9124999999999999, 0.9119229264475743, 0.9155355922273216); mha4_s128/c2 cp_ns=402432.000 queue_ns=431104.000 measured_floor=1.384798 l2_l1=0.960956 relative_selected=(0.9082426127527216, 0.9065420560747665, 0.9107910156249999) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c2__full/dump; mha4_s128/c3 relative_previous=c2:(1.0981687342247999, 1.0960548885077188, 1.0999570999570998); mha4_s128/c3 cp_ns=397312.000 queue_ns=425984.000 measured_floor=1.540865 l2_l1=0.961479 relative_selected=(0.9968992248062016, 0.9965493779160186, 0.999951550387597) dump=docs/experiments/REBASE/bounded_raw/mha4_s128/trace/c3__full/dump; real_s4 1250/1250; real_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9629777023468133, -0.01123200000000013, 0.030719999999999636, 0.035136000000000056, 0.19257599999999986, 0.10754495042849974]; selected_nonpositive_barrier_pairs=0; real_s4/wavefront cp_ns=974848.000 queue_ns=2274304.000 measured_floor=1.121117 l2_l1=0.923458 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/wavefront__full/dump; real_s4/eft cp_ns=973824.000 queue_ns=2271232.000 measured_floor=1.128945 l2_l1=0.928815 relative_selected=(1.0057802743436064, 1.0048260733312442, 1.0060265166733628) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/eft__full/dump; real_s4/rotate cp_ns=1007616.000 queue_ns=1927168.000 measured_floor=1.364008 l2_l1=0.952442 relative_selected=(1.0313378867014866, 1.0308988764044944, 1.0314291808338458) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/rotate__full/dump; real_s4/legacy_grid_stride cp_ns=1011712.000 queue_ns=2350080.000 measured_floor=1.162092 l2_l1=0.989109 relative_selected=(1.0709520465489568, 1.0705168392807882, 1.0715146645239053) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/legacy_grid_stride__full/dump; real_s4/chain cp_ns=971776.000 queue_ns=1951744.000 measured_floor=1.378443 l2_l1=0.974495 relative_selected=(1.055020080321285, 1.0546230084054697, 1.0552699551119693) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/chain__full/dump; real_s4/balanced cp_ns=808960.000 queue_ns=3873792.000 measured_floor=6.033835 l2_l1=8.468043 relative_selected=(9.167268351383875, 9.154379584926955, 9.171649749695746) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/balanced__full/dump; real_s4/protocol_off cp_ns=978944.000 queue_ns=2219008.000 measured_floor=1.177276 l2_l1=0.938926 relative_selected=(1.0249096022498996, 1.0239696802369358, 1.0256178307137562) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/protocol_off__full/dump; real_s4/c1 relative_previous=wavefront:(1.0021203718806069, 1.0016070711128968, 1.0031388107673764); real_s4/c1 cp_ns=972800.000 queue_ns=2265088.000 measured_floor=1.128391 l2_l1=0.925699 relative_selected=(1.0021203718806069, 1.0016070711128968, 1.0031388107673764) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c1__full/dump; real_s4/c2 relative_previous=c1:(1.0005507090378862, 1.0000125159578463, 1.0012159045326916); real_s4/c2 cp_ns=975872.000 queue_ns=2278400.000 measured_floor=1.122219 l2_l1=0.926132 relative_selected=(1.002836523376216, 1.0023324638844302, 1.003151366010446) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c2__full/dump; real_s4/c3 relative_previous=c2:(0.9999124649859945, 0.9995994943616315, 1.000262815378454); real_s4/c3 cp_ns=974848.000 queue_ns=2273280.000 measured_floor=1.124718 l2_l1=0.925961 relative_selected=(1.002748381793897, 1.002296571457256, 1.0030885897950985) dump=docs/experiments/REBASE/bounded_raw/real_s4/trace/c3__full/dump; real_s128 1250/1250; real_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9678012537729277, 0.523968, 0.12288000000000032, 0.16771199999999986, 0.3002560000000001, 2.0940270648623427]; selected_nonpositive_barrier_pairs=3; real_s128/eft cp_ns=2606080.000 queue_ns=5701632.000 measured_floor=1.153556 l2_l1=0.986411 relative_selected=(1.0, 1.0, 1.0) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/eft__full/dump; real_s128/wavefront cp_ns=2719744.000 queue_ns=6149120.000 measured_floor=1.078435 l2_l1=0.995621 relative_selected=(1.0090834665315194, 1.0065113024548582, 1.0098069738480697) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/wavefront__full/dump; real_s128/rotate cp_ns=2816000.000 queue_ns=5491712.000 measured_floor=1.213547 l2_l1=0.999551 relative_selected=(1.0131832141191106, 1.0122766122766123, 1.0137028962939896) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/rotate__full/dump; real_s128/chain cp_ns=2778112.000 queue_ns=6243328.000 measured_floor=1.122996 l2_l1=1.053027 relative_selected=(1.0655005057578586, 1.0499698302709437, 1.0683478785402256) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/chain__full/dump; real_s128/legacy_grid_stride cp_ns=2864128.000 queue_ns=6403072.000 measured_floor=1.068607 l2_l1=1.032248 relative_selected=(1.0412644036125818, 1.0133680235323814, 1.04839338066992) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/legacy_grid_stride__full/dump; real_s128/balanced cp_ns=2104320.000 queue_ns=18079744.000 measured_floor=4.474782 l2_l1=12.135989 relative_selected=(12.3082312231577, 12.283934786785881, 12.33038727348373) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/balanced__full/dump; real_s128/protocol_off cp_ns=2561024.000 queue_ns=5681152.000 measured_floor=1.182949 l2_l1=1.007333 relative_selected=(1.0230132643535086, 1.0211414227895392, 1.024302850911357) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/protocol_off__full/dump; real_s128/c1 relative_previous=eft:(0.9993870581710984, 0.9867642478978511, 1.0007637375466998); real_s128/c1 cp_ns=2733056.000 queue_ns=5966848.000 measured_floor=1.101253 l2_l1=0.985420 relative_selected=(0.9993870581710984, 0.9867642478978511, 1.0007637375466998) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c1__full/dump; real_s128/c2 relative_previous=c1:(1.0055593959633529, 1.0038880248833593, 1.0070093457943925); real_s128/c2 cp_ns=2588672.000 queue_ns=5692416.000 measured_floor=1.160640 l2_l1=0.990922 relative_selected=(1.0047488715047284, 1.0041545997006744, 1.0062579746564202) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c2__full/dump; real_s128/c3 relative_previous=c2:(0.9990707759021218, 0.9979891724671307, 1.0006196746707978); real_s128/c3 cp_ns=2604032.000 queue_ns=5689344.000 measured_floor=1.161267 l2_l1=0.991205 relative_selected=(1.0043684266853468, 1.0037290242386574, 1.0049813200498132) dump=docs/experiments/REBASE/bounded_raw/real_s128/trace/c3__full/dump; placement_pool {'group': 'reference', 'cells': 4, 'pairs': 100, 'median_ratio': 0.8959319950530888, 'ci_lo': 0.8708708708708708, 'ci_hi': 0.9763663220088626, 'bootstrap_draws': 20000, 'seed': 20260906}; placement_pool {'group': 'real', 'cells': 2, 'pairs': 50, 'median_ratio': 0.9635739126503015, 'ci_lo': 0.9631847425184086, 'ci_hi': 0.967151369350876, 'bootstrap_draws': 20000, 'seed': 20260906}; supplemental_W1 c3_a/c2_a=(1.0002433800623054, 0.9965976475162829, 1.002112713251994); supplemental_W1 c2_b/c2_a=(1.0, 0.9740680713128038, 1.0020028332763422); supplemental_W1 c3_b/c3_a=(0.998732449297972, 0.9707723372034326, 1.0018984568952929); six C2/C3 W1 kernel sets byte-identical; supplemental controls=100/100; original 7500 samples retained; causal timing-band origin unresolved; REBASE/bounded_raw/*/measure
S-a PASS [hard] legacy_grid_stride/G256 proved=128/128; legacy_grid_stride/G340 proved=128/128; rotate/G256 proved=128/128; rotate/G340 proved=128/128; band/G256 proved=128/128; band/G340 proved=128/128; wavefront/G256 proved=128/128; wavefront/G340 proved=128/128; gqa2_s4 binary_variants=1; gqa2_s128 binary_variants=1; mha4_s4 binary_variants=1; mha4_s128 binary_variants=1; real_s4 binary_variants=1; real_s128 binary_variants=1; SYMBOLIC/complete/proofs.tsv and JOINT2 selected raw trace resource records
S-b PASS [hard] 40/40 endpoint/interior pi/sigma byte comparisons; actual queue-vector comparisons=55/55 including fitted current winners; SYMBOLIC/complete/ and SYMBOLIC/queue_roundtrip/
S-c PASS [hard] gqa2_s4=wavefront; gqa2_s128=outside_these_four_template_families; mha4_s4=wavefront; mha4_s128=outside_these_four_template_families; real_s4=wavefront; real_s128=outside_these_four_template_families; SYMBOLIC/bounded_fit/*/fit.tsv and complete native/symbolic/selected tables; point witnesses do not assert impossibility in every quasi-affine family
S-current PASS [hard] gqa2_s4 family=wavefront grid=512 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/gqa2_s4; gqa2_s128 outside four families; original winner retained; mha4_s4 family=wavefront grid=512 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/mha4_s4; mha4_s128 outside four families; original winner retained; real_s4 family=wavefront grid=256 proved=128/128 samples=5/5; docs/experiments/SYMBOLIC/bounded_certificates/real_s4; real_s128 outside four families; original winner retained
S-d PASS [report] 16 template/champion/fresh-solve comparisons recomputed from complete raw tables; 40 CG round trips; CPU only; SYMBOLIC/cross_grid/
A1-subset FAIL [report] 0/50 docs/experiments/MODELS/covered_llama_admitted2/correctness missing=49; maximal connected graph checked_outputs=66 mismatches=[(0, 1)]; docs/experiments/MODELS/covered_llama_admitted2/correctness/r0.log; earlier independent MLP subset=True 50/50 docs/experiments/MODELS/llama_mlp/correctness does not satisfy maximal-graph admission; no tolerance or output changes
H2 PASS [hard] models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity
H4 PASS [hard] continuation derived combine before bound repair; interval writer before current winner audits; C1 before J1; C2 before J1; FORK6 before Fuse R7 decision; W2 before symbolic proof implementation; gqa2_s4 selection 1611afc3 precedes all B1 measurements; gqa2_s128 selection 1f8f9e2e precedes all B1 measurements; mha4_s4 selection 6a1e5b4d precedes all B1 measurements; mha4_s128 selection 25ff9ae3 precedes all B1 measurements; real_s4 selection d428c0f8 precedes all B1 measurements; real_s128 selection ad2c2e9d precedes all B1 measurements
H7 PASS [report] four local compile/guard checks; no sm_120 execution claim; docs/experiments/COSTMODEL/selfcheck_sm120_slices; docs/experiments/JOINT2/selfcheck_sm120_slices; docs/experiments/REBASE/selfcheck_sm120_slices; docs/experiments/MODELS/selfcheck_sm120_slices
R6_VERIFY gates=30 hard_failures=4 exit=1
```

The nonzero result is intentional whenever hard gates remain failed. All gates execute before exit. The verifier reads raw processes, evaluated cost rows, proof logs and full Plan tables; it does not read this report or the derived comparison/attribution tables. A preliminary valid identity stamp permits capture of actual complete verifier output before rendering this report. After the report commit, the final SASS is regenerated and the full verifier is rerun; its output must match the embedded output byte for byte. No expected PASS line substitutes for execution.

B1 PASS certifies complete raw-data coverage and the requested recomputed tables. It does not close the identical-kernel causal discrepancy documented below; that interpretation remains locally stopped.

## 4. Stalled items and explicit degraded forms

| Item | Affected scope | Cause / unlock | Inferred work |
| --- | --- | --- | --- |
| J-c | Unrestricted outer search | Full cold reference evaluation still exceeds 1 ms; grouped readiness and per-worker recurrence preserve predictions. Replace remaining preparation/allocation costs, retaining cold timing. | 2–4 developer-days plus benchmark |
| Unmet J gates (see §2) | Qualified selected optimum | The implemented binding objective and all-six inner catalog cannot compensate for inaccurate per-task prices or finite search coverage. Diagnose actual failed cells; no gate is relaxed. | 3–7 developer-days plus paired runs |
| B1 C2/C3 causal attribution | Mechanism-specific interpretation only | Identical W=1 kernels had a 9.8% original timing difference; the independent double-label repeat gives a ratio near one. Retain all data; isolate module placement and process/runtime context before assigning cause. | 1–2 developer-days + diagnostic pairs |
| S5 winners outside four families | Unbounded symbolic extrapolation | Keep each exact winning materialized table; finite interval writeback works, but a general fitted expression is not proved. | 2–5 developer-days per family |
| A1 maximal covered graph | Accepted anchor correctness and timing | First V GEMM BF16 midpoint rounding propagates along residual chain. New-graph admission stops under §9.2 after five numerical failures; unlock requires a backend accuracy fix and unchanged-golden revalidation. | 2–5 developer-days plus 50-process revalidation |

No existing-configuration global correctness stop was triggered. The maximal covered Llama graph fails one checked element in five geometries; its failed result is retained. The earlier independent-MLP 50/50 does not satisfy this larger graph. Other groups proceed independently.

Degraded forms: finite outer capacity of twelve geometry/kappa candidates; fixed five-shape calibrated domain; global κ; interval geometry chosen at the upper endpoint, followed by exact placement solves at each integer point; finite S5 grid branches; GEMM-only FORK6 wait denominator; and A1 numerical admission failure. B1 measures frozen solver-selected configurations even if J gates fail and is not claimed to establish an optimum. The first continuation search (`closure_search`) was superseded after detecting zero outer bounds; its partial measurements are diagnostics only. Final performance evidence uses `bounded_search` and `REBASE/bounded_raw`.

## 5. C — unified costs, K-loop calibration and remaining errors

`DeriveTaskWork → TaskMemoryTraffic → TaskInstanceNs` replaces the three NonGemmStageNs call sites. Backend TaskBody traits/dataflow supply resources and phase structure. `StageKind` remains in parsing/ownership projection, not the five core cost/access-evaluation files.

```sh
rg -n "StageKind|NonGemmStageNs" lib/Solver/{CostModel,ChainDP,CouplingInterfaceDP,TaskModel,ScalarTaskWork}.cpp
# no matches
rg -n "StageKind|NonGemmStageNs" lib/Solver
# Complete output and reviewed non-price roles: COSTMODEL/stagekind_audit.txt
```

18-point full Spearman 0.896800825593 (R5 threshold 0.880288958); coarse 0.876160990712 (gate 0.85). Historical 68-dump replay absolute relative errors: p50 4.539964%, p90 10.847552%, max 13.827710%, versus R5 4.54/10.85/13.83%. These are fresh CPU reevaluations of historical calibration evidence, not fresh GPU speedup claims.

```text
FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4
```

The denominator is instrumented GEMM mainloops. SIMT exposed wait is unmeasured; the all-path-mainloop lower-bound denominator gives 0.161782. The prescribed rule is applied to the measured GEMM scope, with this deviation explicit. Per-cell iteration/fixed p50 ns: gqa2 s4 494.574/84.156, s128 519.355/219.341; mha4 s4 504.267/52.239, s128 318.945/67.038; real s4 266.040/52.492, s128 538.262/102.693. Raw: `../COSTMODEL/raw_kloop/`.

The continuation removes the remaining whole-stage/waves combine conversion. Combine tasks derive semantic reduction and typed physical traffic, then use TaskInstanceNs like other tasks; tail coordinates and FP32 partial versus BF16 residual bytes are explicit. Replay now respects each archived binary’s physical ownership flags and evaluates actual task coordinates instead of repeating task zero. F-201 records the independent element-enumeration checks. These repairs close the earlier implementation omissions, while remaining predictive error is still reported rather than assumed solved.

A later C-a audit also removed the retained-prefix StageKind selector from ScalarTaskWork.cpp in favor of the CG state effect. This corrective cleanup occurred after the original ordered C1/C2 and J1 implementation commits; it is disclosed rather than retroactively claimed to have existed at search start. Independent prefix-element enumeration and all forty replay predictions remain identical, so it does not replace or invalidate the already frozen plans.

| Cell | Backend kind | Combine | Physical tasks | Median stage measured/price | Total measured / total price |
| --- | --- | --- | --- | --- | --- |
| gqa2_s4 | kRMSNorm | 0 | 16 | 1.050288 | 1.181574 |
| gqa2_s4 | kGemm | 0 | 4608 | 1.428406 | 1.389348 |
| gqa2_s4 | kGemm | 1 | 512 | 0.967964 | 1.325068 |
| gqa2_s4 | kRoPE | 0 | 48 | 1.128686 | 1.199229 |
| gqa2_s4 | kKVAppend | 0 | 32 | 1.144865 | 0.787094 |
| gqa2_s4 | kAttention | 0 | 32 | 2.500856 | 2.401541 |
| gqa2_s4 | kElementwise | 0 | 8 | 2.043364 | 1.958224 |
| gqa2_s128 | kRMSNorm | 0 | 512 | 0.840230 | 0.873873 |
| gqa2_s128 | kGemm | 0 | 8192 | 1.429581 | 1.373455 |
| gqa2_s128 | kGemm | 1 | 2048 | 1.720570 | 1.710291 |
| gqa2_s128 | kRoPE | 0 | 1536 | 1.128686 | 1.075779 |
| gqa2_s128 | kKVAppend | 0 | 1024 | 1.144865 | 0.979396 |
| gqa2_s128 | kAttention | 0 | 1024 | 3.893720 | 4.269823 |
| gqa2_s128 | kElementwise | 0 | 256 | 2.043364 | 1.971527 |
| mha4_s4 | kRMSNorm | 0 | 32 | 0.840230 | 0.853359 |
| mha4_s4 | kGemm | 0 | 10240 | 1.071304 | 1.180667 |
| mha4_s4 | kGemm | 1 | 1152 | 0.967964 | 1.360982 |
| mha4_s4 | kRoPE | 0 | 128 | 1.128686 | 1.155140 |
| mha4_s4 | kKVAppend | 0 | 128 | 1.144865 | 0.831816 |
| mha4_s4 | kAttention | 0 | 64 | 2.540634 | 2.480931 |
| mha4_s4 | kElementwise | 0 | 16 | 2.043364 | 1.958224 |
| mha4_s128 | kRMSNorm | 0 | 1024 | 0.840230 | 0.810281 |
| mha4_s128 | kGemm | 0 | 18432 | 1.429581 | 1.293586 |
| mha4_s128 | kGemm | 1 | 4608 | 1.720570 | 1.654396 |
| mha4_s128 | kRoPE | 0 | 4096 | 1.128686 | 1.088179 |
| mha4_s128 | kKVAppend | 0 | 4096 | 1.144865 | 0.837965 |
| mha4_s128 | kAttention | 0 | 2048 | 4.243823 | 4.428798 |
| mha4_s128 | kElementwise | 0 | 512 | 2.043364 | 1.911663 |
| real_s4 | kRMSNorm | 0 | 32 | 2.412741 | 2.325744 |
| real_s4 | kGemm | 0 | 21504 | 0.920528 | 0.987604 |
| real_s4 | kGemm | 1 | 1344 | 1.704374 | 1.794823 |
| real_s4 | kRoPE | 0 | 640 | 1.105005 | 1.201693 |
| real_s4 | kKVAppend | 0 | 256 | 1.125295 | 0.901115 |
| real_s4 | kAttention | 0 | 512 | 2.911858 | 2.688589 |
| real_s4 | kElementwise | 0 | 16 | 2.583386 | 2.563360 |
| real_s128 | kRMSNorm | 0 | 1024 | 2.412743 | 2.399693 |
| real_s128 | kGemm | 0 | 10752 | 0.823957 | 1.053863 |
| real_s128 | kGemm | 1 | 2688 | 1.798604 | 1.867521 |
| real_s128 | kRoPE | 0 | 20480 | 1.105005 | 0.857836 |
| real_s128 | kKVAppend | 0 | 8192 | 1.125295 | 0.600423 |
| real_s128 | kAttention | 0 | 16384 | 2.320270 | 2.323701 |
| real_s128 | kElementwise | 0 | 512 | 2.483256 | 2.497494 |

This diagnostic joins every physical trace task with its actual coordinate/residency price; no task-zero extrapolation or zero-duration filtering is used. The 1024 ns global-timer quantization remains visible. Kind labels describe backend observations and do not reintroduce Solver per-operator pricing branches. The task-level mismatch is distinct from the historical whole-plan replay error above.

## 6. J — binding objective, fresh comparisons and Fuse bound

| Cell | L2 ms | / fresh R5 control [95% CI] | / fresh R5 champion [95% CI] | queue/semantic CP | L2/L1 | L2/floor | Predicted top1 measured rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 0.134144 | 0.505747 [0.501931,0.505976] | 1.090909 [1.062000,1.107499] | 0.975610 | 0.723757 | 1.597561 | 3/3 |
| gqa2_s128 | 0.302912 | 0.728187 [0.724566,0.734491] | 1.330034 [1.308370,1.354608] | 1.035714 | 0.964549 | 1.457204 | 1/3 |
| mha4_s4 | 0.283648 | 0.537315 [0.531942,0.541616] | 1.175842 [1.164324,1.226326] | 0.981250 | 0.767507 | 1.731250 | 3/3 |
| mha4_s128 | 0.612512 | 0.723301 [0.681507,0.726678] | 1.309298 [1.301310,1.314176] | 1.065990 | 0.963964 | 1.424182 | 2/3 |
| real_s4 | 2.549504 | 0.547671 [0.539771,0.568572] | 0.619535 [0.606925,0.643527] | 2.317659 | 0.925922 | 1.122520 | 3/3 |
| real_s128 | 6.579200 | 0.944311 [0.943727,0.981891] | 0.937300 [0.936062,0.964787] | 2.182636 | 0.986155 | 1.151227 | 3/3 |

| Cell | Selected geometry / split / κ / residency | Placement | CP µs | Queue µs | CP nodes | Mean node ns |
| --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 32x16x64s2split16kappa4r4 | wavefront | 83.968 | 81.920 | 28 | 2998.857 |
| gqa2_s128 | 32x16x64s2split4kappa2r4 | eft | 200.704 | 207.872 | 28 | 7168.000 |
| mha4_s4 | 32x16x64s2split16kappa4r4 | wavefront | 163.840 | 160.768 | 56 | 2925.714 |
| mha4_s128 | 32x16x64s2split4kappa4r4 | eft | 403.456 | 430.080 | 56 | 7204.571 |
| real_s4 | 64x128x16s2split16kappa2r2 | wavefront | 979.968 | 2271.232 | 52 | 18845.538 |
| real_s128 | 64x128x16s2split4kappa1r2 | eft | 2618.368 | 5714.944 | 52 | 50353.231 |

| Cell | Fresh R5 champion CP µs | Selected CP µs | Fresh R5 champion queue/CP | Selected queue/CP |
| --- | --- | --- | --- | --- |
| gqa2_s4 | 91.136 | 83.968 | 0.943820 | 0.975610 |
| gqa2_s128 | 184.320 | 200.704 | 0.905556 | 1.035714 |
| mha4_s4 | 161.792 | 163.840 | 0.955696 | 0.981250 |
| mha4_s128 | 411.648 | 403.456 | 0.910448 | 1.065990 |
| real_s4 | 2000.896 | 979.968 | 1.748721 | 2.317659 |
| real_s128 | 4631.552 | 2618.368 | 1.424497 | 2.182636 |

Historical R5 real-width queue/CP was 1.7589/1.4174; the current ratios and frozen J-b decision are in the tables above.

| Cell | Fresh R5 champion / fresh R5 control [95% CI] |
| --- | --- |
| gqa2_s4 | 0.463320 [0.456657,0.474903] |
| gqa2_s128 | 0.549080 [0.545906,0.552936] |
| mha4_s4 | 0.456493 [0.436927,0.462178] |
| mha4_s128 | 0.551529 [0.521493,0.553666] |
| real_s4 | 0.881976 [0.871251,0.897005] |
| real_s128 | 1.009463 [1.007097,1.033665] |

This independently paired row rechecks the configuration-axis premise in the current session. It is not the quotient of two reported median ratios. The reference speedup magnitude remains visible, while the exact R5 gqa2 s4 point estimate does not repeat; the current raw ratio is used throughout, rather than copying the historical .409039. Different sessions are not paired evidence.

Controls and candidates are twenty-five rotated, same-session, fresh-process rounds; choice uses the separate five-round pilot and is frozen before confirmation. The zero-sync union bound used by search is distinguished from semantic CP in J-b; replacing J-b’s denominator with a queue-containing CP would make that gate tautological. Minimizing max(CP,queue) itself does not enforce queue/semantic-CP ≤1.

| Cell | Predicted six-placement order at selected geometry | Fresh B1 full-time order |
| --- | --- | --- |
| gqa2_s4 | wavefront < eft < rotate < chain < legacy_grid_stride < balanced | wavefront < rotate < eft < chain < legacy_grid_stride < balanced |
| gqa2_s128 | eft < wavefront < rotate < legacy_grid_stride < chain < balanced | wavefront < legacy_grid_stride < eft < rotate < chain < balanced |
| mha4_s4 | wavefront < eft < rotate < chain < legacy_grid_stride < balanced | wavefront < eft < rotate < chain < legacy_grid_stride < balanced |
| mha4_s128 | eft < wavefront < chain < legacy_grid_stride < rotate < balanced | wavefront < eft < rotate < legacy_grid_stride < chain < balanced |
| real_s4 | wavefront < eft < rotate < legacy_grid_stride < chain < balanced | wavefront < eft < rotate < chain < legacy_grid_stride < balanced |
| real_s128 | eft < wavefront < rotate < chain < legacy_grid_stride < balanced | eft < wavefront < rotate < legacy_grid_stride < chain < balanced |

The B1 order above uses marginal full-time medians. Paired ratio medians are reported separately and need not induce the same order; neither is used to replace the independently frozen selection after seeing confirmatory data.

Full cold simulator evaluation (µs), including readiness: reference max 1555.545; real-width max 2028.897. Shared graph preparation and cached-only recurrence are separate columns in COSTMODEL/closure_effects/evaluations.tsv. Cached-only timing never substitutes for the budget gate. Forty predictions across five value columns match the preceding implementation exactly.

The complete real-s128 compiler search took 3641.211 seconds in one fresh CPU process, including outer preparation and occupancy compilation. It evaluated twelve candidates and disclosed seventy-two deferred candidates. This unpaired wall-time diagnostic is not a speedup claim or the per-plan J-c budget; it exposes the remaining end-to-end preparation cost. Raw command, source/binary hashes, output and superseded-search provenance are retained in bounded_boxes/real_s128/ and closure/search_profile/.

Outer candidates now carry nonzero work, semantic-CP and queue pigeonhole lower bounds from the same derived task-cost preparation as the inner catalog. The priority and pruning bound are their maximum. Every evaluated geometry explores all six placements and compiled legal residencies. Raw auto.cu.bounds.tsv files expose evaluated, deferred and pruned candidates; J-outer verifies bounds and complete inner catalogs. The finite budget remains explicit.

| Cell | Supported pairs | Removed nodes | Removed bytes | Fixed upper ns | Traffic upper ns | Total upper ns | / measured floor | Whole-pair envelope cap ns | Cap / measured floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 2 | 16.0 | 8192.0 | 795.82491445376 | 467.88733279999997 | 1263.7122472537599 | 0.015049926725106706 | 3430.27980368 | 0.04085222708269817 |
| gqa2_s128 | 2 | 512.0 | 262144.0 | 809.20952995616 | 493.52835865999987 | 1302.73788861616 | 0.006267019553456743 | 3487.97211188 | 0.01677942249018627 |
| mha4_s4 | 4 | 64.0 | 32768.0 | 1591.64982890752 | 935.7746655999999 | 2527.4244945075197 | 0.015426174893234373 | 6860.55960736 | 0.04187353275976562 |
| mha4_s128 | 4 | 2048.0 | 1048576.0 | 1671.95752190336 | 1089.6208207199998 | 2761.5783426233597 | 0.00642108059575744 | 7206.71345648 | 0.01675668121391369 |
| real_s4 | 4 | 128.0 | 65536.0 | 1634.65794674848 | 1018.16569592 | 2652.82364266848 | 0.0011680108604794578 | 7045.93942564 | 0.0031022543824849244 |
| real_s128 | 4 | 4096.0 | 2097152.0 | 6817.74071750464 | 4607.35422148 | 11425.09493898464 | 0.001999161310939292 | 29386.81343752 | 0.005142099981648114 |

```text
FUSE6 enter_r7=0 maximum_bound_share=0.015426 cells=6
```

Only the existing legal adjacent RoPE→KVAppend family is bounded. Fixed upper=.232×separate envelope; traffic upper is the unified-path reduction with internal storage free, external traffic/arithmetic retained; total is capped at the separate envelope. No added barrier, copy, recomputation or occupancy loss is charged. This is an optimistic model bound under its wave assumptions, not an executed fusion or proof about all possible fusion families. The common .232 fixed fraction is an extrapolation, not per-pair phase data; the separate-envelope cap and the recomputed decision are retained in the table; no prior-round numeric decision substitutes for this evaluation. The existing Fuse direction gate remains unchanged.

## 7. W — one-command import, solve, writeback and code generation

Executed from the repository root:

```sh
build-portable/tools/tilemega-compile docs/experiments/MODELS/llama_mlp/exported_program.pt2 docs/experiments/MODELS/llama_mlp/direct_pt2.cu --solve docs/experiments/COSTMODEL/event_fit/target.json --seq 4 --past 3 --search-capacity 3 --search-domain docs/experiments/COSTMODEL/event_fit/search_domain.json --dump-cg docs/experiments/MODELS/llama_mlp/direct_pt2.mlir --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv
```

The `.pt2` is regenerated with `MODELS/export_mlp.py`; large random fixtures/archives are not committed. Exact command/source/bridge/resource evidence is in `../MODELS/llama_mlp/`. The generated `direct_pt2.cu` equals the source tested in fifty fresh processes. Geometry, split, global κ, residency and placement come from the solver.

```sh
rg -n "placement->setAttr" include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h
# mode, params, window, policy, resident_only, grid_map, resident_limit_map
```

Physical EFT worker/slot arrays are stored in CG module attributes, trading module size for self-contained round trips. `WRITEBACK/roundtrip_gqa2_s4/{cg_plan,host_plan}.tsv`: 2696 nodes, diff empty. Matched baseline/current host archives produce 12/12 identical legacy schedule/waits/events files. The default and selected SEQSCAN process counts and current CTest result are recomputed in §2. W2 also solves every integer seq in [1,5], testing one binary in 250 fresh processes. Each point evaluates six placements; direct solve, serialized CG, generated arrays and executed host queues match byte for byte. Geometry is fixed from the upper-endpoint search, not asserted optimal for the whole interval. The supplemental `WRITEBACK/full_roundtrip/` campaign independently re-solves all five points and compares a direct RuntimePlanDesc injection against the CG-generated carrier: ten fresh executions pass and all fifteen complete schedule/waits/events tables are byte-identical. This closes the earlier validation gap where only worker/slot arrays were compared.

## 8. B1 — attribution at solver-selected geometry

Six placements and four additional protocol variants, five arms each, twenty-five rotated fresh rounds per cell. C3(a) is intentionally inert at W=1. C1/C2/C3 are cumulative flags; no window is enabled. Unsafe probes may fail numerical comparison, but full arms must PASS.

| Cell | Config | Full L2 ms | Wait ms | Notify ms | Fence ms | Barrier ms | Protocol/barrier | / selected | / legacy | L2/L1 | Floor µs | L2/floor | Nonpositive barrier pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | wavefront | 0.140288 | 0.041728 | 0.026720 | 0.023552 | 0.071680 | 0.967450 | 1.000000 | 0.744565 | 0.713263 | 82.944 | 1.691358 | 0 |
| gqa2_s4 | eft | 0.146432 | 0.043328 | 0.029696 | 0.024576 | 0.071392 | 1.025362 | 1.049966 | 0.777365 | 0.748210 | 83.968 | 1.743902 | 0 |
| gqa2_s4 | rotate | 0.142336 | 0.091136 | 0.008192 | 0.020576 | 0.071680 | 1.383936 | 1.014599 | 0.756757 | 0.723958 | 79.872 | 1.782051 | 0 |
| gqa2_s4 | chain | 0.150528 | 0.098336 | 0.009216 | 0.024480 | 0.071680 | 1.514286 | 1.073529 | 0.803642 | 0.768041 | 80.896 | 1.860759 | 0 |
| gqa2_s4 | legacy_grid_stride | 0.188416 | 0.066560 | 0.032768 | 0.045824 | 0.070560 | 1.403986 | 1.343066 | 1.000000 | 0.963731 | 104.448 | 1.803922 | 0 |
| gqa2_s4 | balanced | 0.786368 | 0.612352 | 0.036704 | 0.134048 | 0.054272 | 11.942975 | 5.600729 | 4.163043 | 4.376361 | 121.856 | 6.453256 | 0 |
| gqa2_s4 | protocol_off | 0.155840 | 0.032928 | 0.047392 | 0.016544 | 0.080448 | 0.995270 | 1.116039 | 0.830601 | 0.759344 | 79.872 | 1.951122 | 0 |
| gqa2_s4 | c1 | 0.139616 | 0.040896 | 0.027360 | 0.023296 | 0.071616 | 0.957105 | 0.999777 | 0.743546 | 0.710124 | 81.920 | 1.704297 | 0 |
| gqa2_s4 | c2 | 0.139264 | 0.040896 | 0.027648 | 0.022528 | 0.071680 | 0.956522 | 0.997472 | 0.739130 | 0.709257 | 81.920 | 1.700000 | 0 |
| gqa2_s4 | c3 | 0.140064 | 0.040960 | 0.026944 | 0.023392 | 0.070656 | 0.973182 | 1.000000 | 0.740541 | 0.711340 | 82.944 | 1.688657 | 0 |
| gqa2_s128 | eft | 0.319488 | 0.092224 | 0.023552 | 0.030400 | 0.099360 | 1.165703 | 1.000000 | 1.000000 | 0.948993 | 209.920 | 1.521951 | 0 |
| gqa2_s128 | wavefront | 0.303104 | 0.088000 | 0.023552 | 0.029696 | 0.100352 | 1.112245 | 0.951522 | 0.953931 | 0.899696 | 197.632 | 1.533679 | 0 |
| gqa2_s128 | rotate | 0.335872 | 0.129024 | 0.017856 | 0.026656 | 0.103104 | 1.368457 | 1.051118 | 1.048077 | 0.993638 | 201.728 | 1.664975 | 0 |
| gqa2_s128 | legacy_grid_stride | 0.318400 | 0.076896 | 0.032704 | 0.035040 | 0.091136 | 1.390305 | 1.000000 | 1.000000 | 1.033861 | 225.280 | 1.413352 | 0 |
| gqa2_s128 | chain | 0.342016 | 0.137216 | 0.013344 | 0.040960 | 0.098880 | 1.586284 | 1.071186 | 1.023125 | 1.018579 | 201.728 | 1.695431 | 0 |
| gqa2_s128 | balanced | 2.192384 | 1.538144 | 0.023904 | -0.004160 | 0.070784 | 22.117860 | 6.928226 | 6.894986 | 7.124075 | 611.328 | 3.586265 | 0 |
| gqa2_s128 | protocol_off | 0.334688 | 0.075968 | 0.051200 | 0.023392 | 0.105472 | 1.200536 | 1.038469 | 1.048864 | 0.976119 | 185.344 | 1.805767 | 0 |
| gqa2_s128 | c1 | 0.316416 | 0.092160 | 0.020704 | 0.027488 | 0.100320 | 1.133014 | 0.990676 | 0.996774 | 0.940663 | 209.920 | 1.507317 | 0 |
| gqa2_s128 | c2 | 0.317440 | 0.092320 | 0.021440 | 0.028672 | 0.100352 | 1.142135 | 0.993590 | 0.996885 | 0.942339 | 207.872 | 1.527094 | 0 |
| gqa2_s128 | c3 | 0.317408 | 0.092448 | 0.021504 | 0.027744 | 0.100352 | 1.137953 | 0.994586 | 0.993490 | 0.942249 | 208.896 | 1.519455 | 0 |
| mha4_s4 | wavefront | 0.291840 | 0.085696 | 0.057344 | 0.048832 | 0.145056 | 0.990472 | 1.000000 | 0.854173 | 0.751851 | 165.888 | 1.759259 | 0 |
| mha4_s4 | eft | 0.294624 | 0.085344 | 0.058272 | 0.049152 | 0.144032 | 0.999334 | 1.007018 | 0.859656 | 0.757488 | 161.792 | 1.821005 | 0 |
| mha4_s4 | rotate | 0.298944 | 0.193536 | 0.015104 | 0.043232 | 0.143936 | 1.453237 | 1.020619 | 0.867257 | 0.767728 | 166.912 | 1.791028 | 0 |
| mha4_s4 | chain | 0.312320 | 0.209824 | 0.015584 | 0.053664 | 0.163744 | 1.390046 | 1.067717 | 0.911115 | 0.804749 | 161.792 | 1.930380 | 0 |
| mha4_s4 | legacy_grid_stride | 0.342016 | 0.102752 | 0.094048 | 0.035104 | 0.111808 | 1.557015 | 1.170724 | 1.000000 | 0.960280 | 210.944 | 1.621359 | 0 |
| mha4_s4 | balanced | 3.250944 | 2.917600 | 0.068704 | 0.892928 | 0.144160 | 20.754633 | 11.112281 | 9.443308 | 8.365372 | 248.832 | 13.064815 | 0 |
| mha4_s4 | protocol_off | 0.326848 | 0.061472 | 0.105472 | 0.035072 | 0.158720 | 1.070777 | 1.116279 | 0.943934 | 0.811558 | 162.816 | 2.007469 | 0 |
| mha4_s4 | c1 | 0.290816 | 0.084320 | 0.056320 | 0.047808 | 0.143232 | 0.985255 | 0.993984 | 0.845029 | 0.747553 | 164.864 | 1.763975 | 0 |
| mha4_s4 | c2 | 0.291552 | 0.083808 | 0.058368 | 0.047104 | 0.144160 | 1.000485 | 0.997476 | 0.846055 | 0.749712 | 166.912 | 1.746741 | 0 |
| mha4_s4 | c3 | 0.290816 | 0.083968 | 0.058304 | 0.047104 | 0.141312 | 0.998276 | 0.990887 | 0.832256 | 0.748484 | 166.912 | 1.742331 | 0 |
| mha4_s128 | eft | 0.658304 | 0.172032 | 0.052864 | 0.059264 | 0.206848 | 1.090174 | 1.000000 | 0.949427 | 0.964126 | 430.080 | 1.530655 | 0 |
| mha4_s128 | wavefront | 0.632800 | 0.177504 | 0.059168 | 0.058368 | 0.207872 | 1.137102 | 0.962519 | 0.913235 | 0.927849 | 406.528 | 1.556596 | 0 |
| mha4_s128 | chain | 0.702368 | 0.260928 | 0.045056 | 0.060416 | 0.207488 | 1.476755 | 1.067117 | 1.013525 | 1.029889 | 397.312 | 1.767800 | 0 |
| mha4_s128 | legacy_grid_stride | 0.693184 | 0.199648 | 0.070528 | 0.074624 | 0.206880 | 1.304615 | 1.053267 | 1.000000 | 1.015038 | 416.768 | 1.663237 | 0 |
| mha4_s128 | rotate | 0.679936 | 0.249792 | 0.044064 | 0.061440 | 0.208576 | 1.411548 | 1.034321 | 0.980922 | 0.995315 | 342.016 | 1.988024 | 0 |
| mha4_s128 | balanced | 6.199296 | 4.683776 | 0.030400 | 0.732128 | 0.208896 | 22.453659 | 9.416991 | 8.944880 | 9.081006 | 1272.832 | 4.870475 | 0 |
| mha4_s128 | protocol_off | 0.691360 | 0.137472 | 0.113664 | 0.048352 | 0.216064 | 1.156842 | 1.051482 | 0.997037 | 1.001202 | 396.288 | 1.744590 | 0 |
| mha4_s128 | c1 | 0.654496 | 0.218176 | 0.004256 | 0.109568 | 0.207872 | 1.071527 | 0.995528 | 0.943870 | 0.959581 | 430.080 | 1.521801 | 0 |
| mha4_s128 | c2 | 0.596992 | 0.118784 | 0.046080 | 0.000000 | 0.146432 | 1.124837 | 0.908243 | 0.863034 | 0.960956 | 431.104 | 1.384798 | 0 |
| mha4_s128 | c3 | 0.656384 | 0.176480 | 0.046240 | 0.057344 | 0.206944 | 1.078707 | 0.996899 | 0.946713 | 0.961479 | 425.984 | 1.540865 | 0 |
| real_s4 | wavefront | 2.549760 | -0.011232 | 0.030720 | 0.035136 | 0.192576 | 0.107545 | 1.000000 | 0.933749 | 0.923458 | 2274.304 | 1.121117 | 0 |
| real_s4 | eft | 2.564096 | -0.004096 | 0.037888 | 0.041248 | 0.192512 | 0.181985 | 1.005780 | 0.939141 | 0.928815 | 2271.232 | 1.128945 | 0 |
| real_s4 | rotate | 2.628672 | 0.576608 | 0.023648 | 0.042944 | 0.192512 | 3.116383 | 1.031338 | 0.962978 | 0.952442 | 1927.168 | 1.364008 | 1 |
| real_s4 | legacy_grid_stride | 2.731008 | 0.019808 | 0.056320 | 0.044384 | 0.188320 | 0.399595 | 1.070952 | 1.000000 | 0.989109 | 2350.080 | 1.162092 | 1 |
| real_s4 | chain | 2.690368 | 0.462240 | 0.007264 | 0.042880 | 0.192512 | 2.444764 | 1.055020 | 0.985489 | 0.974495 | 1951.744 | 1.378443 | 0 |
| real_s4 | balanced | 23.373823 | 18.455424 | 0.114880 | 0.863072 | 0.194560 | 95.553477 | 9.167268 | 8.559020 | 8.468043 | 3873.792 | 6.033835 | 0 |
| real_s4 | protocol_off | 2.612384 | 0.004800 | 0.074784 | 0.035584 | 0.215296 | 0.365424 | 1.024910 | 0.956913 | 0.938926 | 2219.008 | 1.177276 | 0 |
| real_s4 | c1 | 2.555904 | -0.003264 | 0.025920 | 0.039936 | 0.190592 | 0.117949 | 1.002120 | 0.936246 | 0.925699 | 2265.088 | 1.128391 | 0 |
| real_s4 | c2 | 2.556864 | -0.003040 | 0.031392 | 0.040640 | 0.192512 | 0.155527 | 1.002837 | 0.936247 | 0.926132 | 2278.400 | 1.122219 | 0 |
| real_s4 | c3 | 2.556800 | -0.002144 | 0.029312 | 0.040704 | 0.193056 | 0.146831 | 1.002748 | 0.936164 | 0.925961 | 2273.280 | 1.124718 | 0 |
| real_s128 | eft | 6.577152 | 0.523968 | 0.122880 | 0.167712 | 0.300256 | 2.094027 | 1.000000 | 0.960371 | 0.986411 | 5701.632 | 1.153556 | 3 |
| real_s128 | wavefront | 6.631424 | 0.493632 | 0.129120 | 0.230272 | 0.289984 | 1.930818 | 1.009083 | 0.963958 | 0.995621 | 6149.120 | 1.078435 | 4 |
| real_s128 | rotate | 6.664448 | 1.152000 | 0.108672 | 0.198016 | 0.288928 | 4.322959 | 1.013183 | 0.967801 | 0.999551 | 5491.712 | 1.213547 | 2 |
| real_s128 | chain | 7.011232 | 0.807776 | 0.130048 | 0.182304 | 0.287520 | 3.079357 | 1.065501 | 1.019620 | 1.053027 | 6243.328 | 1.122996 | 3 |
| real_s128 | legacy_grid_stride | 6.842368 | 0.285696 | 0.281568 | 0.223168 | 0.281600 | 2.018212 | 1.041264 | 1.000000 | 1.032248 | 6403.072 | 1.068607 | 3 |
| real_s128 | balanced | 80.902908 | 61.126654 | 0.534466 | 1.730812 | 0.313312 | 190.968242 | 12.308231 | 11.841745 | 12.135989 | 18079.744 | 4.474782 | 1 |
| real_s128 | protocol_off | 6.720512 | 0.549728 | 0.235232 | 0.138400 | 0.311296 | 2.426131 | 1.023013 | 0.978017 | 1.007333 | 5681.152 | 1.182949 | 1 |
| real_s128 | c1 | 6.571008 | 0.545792 | 0.088064 | 0.167936 | 0.290784 | 2.200946 | 0.999387 | 0.954525 | 0.985420 | 5966.848 | 1.101253 | 1 |
| real_s128 | c2 | 6.606848 | 0.580864 | 0.087040 | 0.168960 | 0.286944 | 2.263164 | 1.004749 | 0.960490 | 0.990922 | 5692.416 | 1.160640 | 1 |
| real_s128 | c3 | 6.606848 | 0.579776 | 0.097216 | 0.164864 | 0.281600 | 1.912620 | 1.004368 | 0.958722 | 0.991205 | 5689.344 | 1.161267 | 6 |

| Cell | Rotate / legacy [95% CI] | R3 B on / off [95% CI] | +C1 / B | +C1+C2 / B | +C1+C2+C3(a) / B |
| --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 0.756757 [0.754098,0.762291] | 0.896026 [0.894737,0.903226] | 0.999777 | 0.997472 | 1.000000 |
| gqa2_s128 | 1.048077 [1.028024,1.058065] | 0.962956 [0.954407,0.983333] | 0.990676 | 0.993590 | 0.994586 |
| mha4_s4 | 0.867257 [0.840841,0.874965] | 0.895833 [0.893154,0.907249] | 0.993984 | 0.997476 | 0.990887 |
| mha4_s128 | 0.980922 [0.977679,0.985207] | 0.951039 [0.948640,0.954006] | 0.995528 | 0.908243 | 0.996899 |
| real_s4 | 0.962978 [0.962144,0.963255] | 0.975696 [0.975022,0.976591] | 1.002120 | 1.002837 | 1.002748 |
| real_s128 | 0.967801 [0.966903,0.986816] | 0.977504 [0.976274,0.979296] | 0.999387 | 1.004749 | 1.004368 |

| Cell | C1 / B [95% CI] | C1+C2 / C1 [95% CI] | C1+C2+C3 / C1+C2 [95% CI] |
| --- | --- | --- | --- |
| gqa2_s4 | 0.999777 [0.992701,1.005974] | 1.000000 [0.992424,1.001142] | 1.002534 [0.999311,1.007353] |
| gqa2_s128 | 0.990676 [0.987220,1.006369] | 1.000405 [0.996897,1.003337] | 1.000000 [0.996805,1.003236] |
| mha4_s4 | 0.993984 [0.986207,0.996491] | 1.000771 [0.990222,1.003855] | 0.998244 [0.982924,1.003487] |
| mha4_s128 | 0.995528 [0.993400,0.996875] | 0.912500 [0.911923,0.915536] | 1.098169 [1.096055,1.099957] |
| real_s4 | 1.002120 [1.001607,1.003139] | 1.000551 [1.000013,1.001216] | 0.999912 [0.999599,1.000263] |
| real_s128 | 0.999387 [0.986764,1.000764] | 1.005559 [1.003888,1.007009] | 0.999071 [0.997989,1.000620] |

Incremental protocol rows above are recomputed from the same round’s two full logs. They are not quotients of separately aggregated median ratios. C2 depends on C1, and C3(a) remains inert at the required W=1.

| Group | Cells | Within-cell pairs | Pooled rotate / legacy [95% CI] |
| --- | --- | --- | --- |
| reference | 4 | 100 | 0.895932 [0.870871,0.976366] |
| real | 2 | 50 | 0.963574 [0.963185,0.967151] |

The reference pooled row directly matches R1’s .6705 statistic: the median of 100 within-cell ratios, with 20,000 bootstrap draws and seed 20260906. It is not a geometric mean or a median of four cell medians. Real-width is pooled separately.

Placement ratios keep the R3 B protocol fixed. Protocol ratios keep the solver-selected placement and geometry fixed; the on/off row is the reciprocal of the retained paired off/on statistic, with interval endpoints reversed. These are conditional contributions, not multiplicative independent factors. R1’s historical pooled rotate/legacy was .6705; R3’s historical default-placement cumulative B reductions were 5.16/3.48/4.79/3.74%. R4 reduced the protocol/barrier diagnostic by 36% while full time increased. None of those historical percentages substitutes for the fresh configuration-specific rows here.

Each statistic uses its own twenty-five pairs; medians need not add. Wait=full−nowait, notify=nowait−neither, fence=full−nofence, barrier=L1full−L1nosync. Signed differences are retained, including negative barrier ratios; a zero denominator makes the ratio undefined without deleting a sample. Raw per-process pre/post and 100 ms GPU telemetry accompanies the new campaign to diagnose the prior unchanged-control timing bands. Unsafe intervention differences are not disjoint nonnegative physical costs. No sample is normalized or filtered using telemetry. R1’s .6705 is a historical comparator only.

| real s128 probe | Kernel | Byte-identical to full |
| --- | --- | --- |
| nofence | L2 | 0 |
| nofence | L1 | 1 |
| l1nosync | L2 | 1 |
| l1nosync | L1 | 0 |

The real-s128 probe audit preserves all 1250 raw processes and all 250 barrier differences, of which 25 are nonpositive. The unchanged L1 kernel under nofence and unchanged L2 kernel under l1nosync are checked by disassembly. Their timing variation is not explained away or subtracted; it limits causal interpretation of signed intervention differences. Evidence: `REBASE/bounded_probe_audit/`, plus all six hardware-context tables in `REBASE/bounded_analysis/`.

| Supplemental mha4 s128 control pair | L2 ratio [95% CI] |
| --- | --- |
| c3_a/c2_a | 1.000243 [0.996598,1.002113] |
| c2_b/c2_a | 1.000000 [0.974068,1.002003] |
| c3_b/c3_a | 0.998732 [0.970772,1.001898] |

All six C2/C3 W=1 kernel sets are byte-identical. The original mha4-s128 C3/C2 ratio 1.098169 does not reproduce in this independent 100-process, four-label control campaign: C3/C2 is 1.000243 [0.996598,1.002113], and both same-binary duplicate-label CIs contain one. Original measurements and CIs remain unchanged. The cause of the original timing band is unresolved; attributing the apparent C2 gain or C3 penalty to the local-dependency mechanism is stopped under §9.2. The next diagnostic must control CUDA module/instruction addresses and allocation/runtime context; telemetry is not used to normalize or delete data. Evidence: `REBASE/bounded_w1_identity/` and `bounded_w1_repeat/`.

## 9. S5 — proven domain and cross-grid comparison

Source graph and geometry are in `SYMBOLIC/complete/provenance.json`. Four families, integer seq [1,128], grids 256/340; 166 exhaustive ISL certificate pieces. Wavefront G340 uses singleton certificates for every integer in the interval, not inference from five samples. Forty endpoint/interior native/template tables and forty CG serialize/read evaluations agree. One carried CG contains the two constant-grid branches per family; these are not extra kernel variants.

| Current winner | Exact template matches |
| --- | --- |
| gqa2_s4 | wavefront |
| gqa2_s128 | outside these four families |
| mha4_s4 | wavefront |
| mha4_s128 | outside these four families |
| real_s4 | wavefront |
| real_s128 | outside these four families |

For every current winner that exactly fits a template, bounded_certificates/ also proves all 128 integer seq values at that winner’s actual grid, with five native-table endpoint/interior comparisons. The unchanged template is proved using exhaustive singleton certificate pieces; shards are not GPU variants. These current-winner certificates supplement the original four-family G256/G340 proof. `SYMBOLIC/queue_roundtrip/` additionally compares the actual native MaterializedPlan.queue vectors against evaluation of the archived proved expressions: forty original cases plus fifteen current-winner cases. These include worker count, every dense slot, stage and logical task; earlier pi/sigma-only checks remain alongside them.

No worse template replaces a measured winner. The retained same-CG G256/G340 template versus fresh-solve comparison is CPU portability evidence over two finite grid branches, not cross-architecture performance or unbounded variable-divisor support. Materialized winners remain exact in the finite W2 interval carrier.

## 10. A1 — model sources, coverage and executable subset

| model | num_hidden_layers | hidden_size | intermediate_size | num_attention_heads | num_key_value_heads | head_dim | rope_theta | rope_scaling | rms_norm_eps | vocab_size | tie_word_embeddings | max_position_embeddings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 16 | 2048 | 8192 | 32 | 8 | 64 | 500000.0 | {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"} | 1e-05 | 128256 | true | 131072 |
| qwen | 28 | 2048 | 6144 | 16 | 8 | 128 | 1000000 | null | 1e-06 | 151936 | true | 40960 |

Sources: Qwen official public config; Meta official download returned 401, so Llama uses an explicitly identified public redistributor config cross-checked against Meta SKU/implementation. Exact URLs and SHA256 are in `MODELS/sources/manifest.json`. No pretrained weights or HF runtime dependency.

| Archived source | URL | SHA256 / retrieval result |
| --- | --- | --- |
| llama_config.json | [source](https://huggingface.co/meta-llama/Llama-3.2-1B/raw/main/config.json) | HTTP Error 401: Unauthorized |
| llama_config_public_copy.json | [source](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json) | 8f028e2cd88148fad38c7dece460947681adf6178e46e11ece9742aa49b378dd |
| qwen_config.json | [source](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) | 1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197 |
| meta_sku_list.py | [source](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/sku_list.py) | a71a8ca23c8daf3d1daec02c13e9916c36e315c5056b56ed20b18df98d10c486 |
| meta_model.py | [source](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/model.py) | ff135f051d66b03a9e32789966d2be84186a00b90cef398c05b10f7e9d2a7dc3 |

| model | operator | task_kind | status |
| --- | --- | --- | --- |
| llama | token embedding | missing | no indexed embedding TaskKind / decoder importer input is hidden states |
| llama | input pre-attention RMSNorm | kRMSNorm | configured epsilon=1e-05; backend epsilon=1e-06; parameter gap |
| llama | Q/K/V projections | kGemm | shape_supported; maximal supported-region exporter/importer now exercised (whole-graph numerical gate fails separately) |
| llama | RoPE frequency scaling | kRoPE | config-dependent frequency table can be an input buffer |
| llama | RoPE rotation | kRoPE | partial: backend rounds positions and angles to ModelElement before sin/cos; public implementation uses FP32 angles |
| llama | KV cache append | kKVAppend | shape_supported |
| llama | GQA score/mask/softmax/value | kAttention | shape_supported; exported composite checked at explicit RoPE/RMSNorm cuts; full residual-chain numerical tolerance not met |
| llama | output projection | kGemm | shape_supported |
| llama | attention residual | kAdd / kGemm epilogue | shape_supported |
| llama | post-attention RMSNorm | kRMSNorm | configured epsilon=1e-05; backend epsilon=1e-06; parameter gap |
| llama | gate/up projections | kGemm | shape_supported |
| llama | SiLU times gate | kElementwise | shape_supported |
| llama | down projection and residual | kGemm / kAdd | shape_supported |
| llama | final RMSNorm | kRMSNorm | partial: no standalone final-normalization stage in DecoderLayerPattern; configured epsilon=1e-05; backend epsilon=1e-06; parameter gap |
| llama | vocabulary projection | kGemm | covered-region importer now emits final LM head with existing kGemm; maximal-graph output head passes its CPU check; embedding/tied-weight full architecture remains cut |
| llama | reshape/transpose/broadcast | layout metadata | absorbed by semantic access maps; no standalone task |
| qwen | token embedding | missing | no indexed embedding TaskKind / decoder importer input is hidden states |
| qwen | input pre-attention RMSNorm | kRMSNorm | configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | Q/K/V projections | kGemm | shape_supported; maximal supported-region exporter/importer now exercised (whole-graph numerical gate fails separately) |
| qwen | per-head Q/K RMSNorm | missing | Qwen only; existing RMSNorm owns one token row, not one token/head |
| qwen | RoPE frequency scaling | kRoPE | config-dependent frequency table can be an input buffer |
| qwen | RoPE rotation | kRoPE | partial: backend rounds positions and angles to ModelElement before sin/cos; public implementation uses FP32 angles |
| qwen | KV cache append | kKVAppend | shape_supported |
| qwen | GQA score/mask/softmax/value | kAttention | shape_supported; exported composite checked at explicit RoPE/RMSNorm cuts; full residual-chain numerical tolerance not met |
| qwen | output projection | kGemm | shape_supported |
| qwen | attention residual | kAdd / kGemm epilogue | shape_supported |
| qwen | post-attention RMSNorm | kRMSNorm | configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | gate/up projections | kGemm | shape_supported |
| qwen | SiLU times gate | kElementwise | shape_supported |
| qwen | down projection and residual | kGemm / kAdd | shape_supported |
| qwen | final RMSNorm | kRMSNorm | partial: no standalone final-normalization stage in DecoderLayerPattern; configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | vocabulary projection | kGemm | covered-region importer now emits final LM head with existing kGemm; maximal-graph output head passes its CPU check; embedding/tied-weight full architecture remains cut |
| qwen | reshape/transpose/broadcast | layout metadata | absorbed by semantic access maps; no standalone task |

| index | site | path | condition |
| --- | --- | --- | --- |
| 1 | TaskKind enum | include/tilemega/Codegen/tasks/TaskBase.h | new distinct TaskKind |
| 2 | OwnershipOf | include/tilemega/Codegen/tasks/TaskBase.h | new ownership classification |
| 3 | TaskBody implementation | include/tilemega/Codegen/tasks/ | new backend body |
| 4 | resource traits and resource query | include/tilemega/Codegen/tasks/TaskResources.h | threads/shared source |
| 5 | backend scalar dataflow | include/tilemega/Codegen/tasks/ScalarDataflow.h | control-flow latency declaration |
| 6 | L1 dispatch | include/tilemega/Codegen/tasks/ModelHarness.cuh | RunStage |
| 7 | L2 dispatch | include/tilemega/Codegen/tasks/ModelHarness.cuh | RunTask |
| 8 | active ownership dispatch | include/tilemega/Codegen/tasks/ModelHarness.cuh | ActiveTaskCount |
| 9 | frontend plan role | include/tilemega/Frontend/SemanticLifting.h | plan role and operator semantics |
| 10 | frontend pattern | lib/Frontend/ModelPlan.cpp | recognize actual use-def subgraph |
| 11 | semantic access lifting | lib/Frontend/SemanticLifting.cpp | read/write/reduction relations |
| 12 | frontend runtime spelling | lib/Frontend/Frontend.cpp | PlanTaskKind to emitted TaskKind |
| 13 | model stage representation | include/tilemega/Solver/ModelDescription.h | new kind identity, no prices |
| 14 | model stage parser | lib/Solver/ModelDescription.cpp | new kind spelling, no prices |
| 15 | ownership adapter if new map | lib/Solver/ScalarTaskWork.cpp | conditional for a new physical ownership relation |

There are 11 current TaskKind values, not the prompt’s assumed 16. A distinct kind/ownership can touch 15 conditional sites; costs come from semantics/traits rather than new per-operator Solver formulas. The continuation imports all supported regions of the 16-layer Llama architecture, keeping residual edges connected and cutting only known missing embedding, exact normalization-parameter and RoPE semantics. It includes Q/K/V/O, KV append, attention, MLP and vocabulary head: 98 inputs, 66 outputs, 209 generated stages. See MODELS/subset.md for the complete one-command .pt2 invocation and exact cut contract.

Maximal-graph numerical admission is FAIL: actual final residual -0.41796875 versus CPU -0.44921875 at index 389, difference .03125 above tolerance .0231875014. All other 65 outputs pass; L0.5/L1/L2 agree exactly. Two solver-selected and three diagnostic geometries reproduce it. FP64 at first V[0,463] agrees with CPU; the GPU differs one BF16 unit near a rounding midpoint. Diagnostic CPU attention using GPU V exactly reproduces the GPU context. No reference or tolerance was changed. Accepted end-to-end timing for this maximal graph is therefore withheld. The older independent-MLP 50/50 remains a narrower result, never a substitute. Raw sources, commands and failures are in MODELS/covered_llama*, covered_geometry_probes and diagnostic/.

Maximal covered graph reproduction (the CUDA generation succeeds; numerical admission fails as reported above):

```sh
python3 docs/experiments/MODELS/export_covered.py --seq 4 --out docs/experiments/MODELS/covered_llama
build-portable/tools/tilemega-compile docs/experiments/MODELS/covered_llama/exported_program.pt2 docs/experiments/MODELS/covered_llama_admitted2/auto.cu --solve docs/experiments/COSTMODEL/event_fit/target.json --seq 4 --past 3 --search-capacity 3 --search-domain docs/experiments/COSTMODEL/event_fit/search_domain.json --dump-cg docs/experiments/MODELS/covered_llama_admitted2/auto.mlir --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv --numerical-rejections docs/experiments/MODELS/covered_llama/numerical_rejections.json
python3 docs/experiments/MODELS/run_covered.py --root docs/experiments/MODELS/covered_llama_admitted2
```

## 11. Deviations and reasons

The explicit degraded forms in §4 remain open. The requested conventional source filenames include header-only production JointSearch/placement passes in this checkout; symbol names were used. S5’s large-map parser was reused inside the evidence driver without changing the proof. The global timer has 1024 ns ticks; zero durations are preserved. Source calibration replay uses historical GPU traces exactly as requested for replay gates, while J/B performance controls are fresh. sm_120 runners were compile/guard self-checked here and were not run on sm_120. The report and SASS stamp use a parent/child commit manifest to avoid claiming that an earlier code revision was the final one.

The integrated upstream sm_120 account separately reports a real-s128 control mismatch and a still-unrepaired cluster-residency hazard (F-199/F-200). Those cross-architecture observations are retained as attributed upstream evidence. This machine did not rerun them; R6 target runners retain their correctness guards. No cluster protocol change was introduced here.

## 12. Excluded work confirmation

No EX-V1 full anchored decode sweep; no Fuse outer-search decision or fused implementation; no EX-E4 shared-memory pipeline or TaskSmem union-lifetime change; no A1 missing-operator implementation; no EX-E5, EX-S4 or L5 serving. Plan execution semantics, W=1, monotonic epochs, release rules and legality checks remain unchanged. Only the skeleton changelog and §4.4.2 were edited. User-owned edits in PLACE_EFT2/summary.md and SYNC_V2/sass_identity/meta.tsv were preserved.

## 13. Failed gates: causes and concrete next changes

The earlier C-c ownership/coordinate defects, combine whole-stage shortcut and zero outer bounds have been repaired. For any remaining J-d/J-e failure, inspect per-task prices under compiled residency and the deferred-bound candidate list; fit backend service dilation and widen the measured feasible shortlist without changing this round’s gates. Concrete model sites are CostModel::TaskInstanceNs (the task_body fixed/loop_body/loop_wait fits) and ScalarInstanceNs. AttentionChunkTaskBody.h has a thread-zero softmax loop whose active-lane service should be represented in backend phase traits; the audit establishes the price error, while this causal explanation still requires a phase-specific calibration. J-c remains limited by per-plan readiness and recurrence memory traffic; next optimize immutable prepared storage and allocation while retaining the full cold metric. Separately, the slow real128 outer search was sampled inside VisitFiniteRelation → isl_set_foreach_point under PreparePlacementProblem (closure/search_profile/); the sample lands in dense finite-CG edge enumeration before bound evaluation. Exact integer-box pieces and proven dense coordinate slices now bypass ISL per-point allocation after equality checks (F-209/F-210; complete point-multiset checks and 49 CTest pass). The next preparation change is to retain relation intervals/shared successor regions through the bound computation, rather than still materializing every dense edge, independently of the timed per-plan recurrence. This is a stack observation, not a measured causal percentage. J-b is a separate feasibility condition: minimizing the binding makespan does not mathematically enforce queue/semantic-CP≤1. Any explicit feasibility restriction must expose its latency tradeoff and be measured against the same controls. A1 requires accurate contraction accumulation near BF16 rounding midpoints, then revalidation of the connected residual chain.

## 14. R7 priorities and decisions still outside the solver

| Priority | Block | Dependency / action | Inferred effort |
| --- | --- | --- | --- |
| 0 | Finish price/ranking closure | Remaining calibrated serial/resource service, shortlist coverage and cold budget; preserve the frozen gates. | 3–7 developer-days + fresh campaigns |
| 1 | A1 semantic/operator gaps and maximal covered graph | Exact epsilon/RoPE, embedding, QK-norm ownership, and the exposed BF16 accumulation mismatch; supported attention/head import is already assembled. | 5–10 developer-days |
| 2 | EX-V1 full anchored main benchmark | Depends on corrected anchor and qualified search; decode seq 1/4/16/64, complete ablations. | 3–5 developer-days + 1–3 GPU-days |
| 3 | EX-E4 step 2 design and bounded prototype | FORK6 rule1 on GEMM scope; distinguish intra-K-loop waits from first-slot prefetch; explicitly redesign §8.6 before shared buffering. | 4–8 developer-days |
| defer | Fuse in outer search | Apply the recomputed FUSE6 threshold decision above; expand rejected ownership families before claiming a wider fusion result. | 2–5 developer-days for search integration or an expanded bound |
| later | EX-E5/S4 → L5 serving | After single-inference correctness/resource contracts and cost quality; separate serving-state/lifetime design. | 1–2 weeks for E5/S4, then 2–4 weeks for L5 |

Automatic decisions now include geometry, split-K, global κ, residency, placement and slot order for the evaluated point/candidate domain. Still outside complete solver control: fusion partitioning, cross-task shared-memory pipeline selection, per-stage κ, general interval geometry/variant optimization beyond the finite fixed-geometry carrier, search-domain/capacity policy, and missing architecture import/semantics. Therefore single-inference closure is partial even though the concrete import→solve→CG→CUDA channel now executes.
