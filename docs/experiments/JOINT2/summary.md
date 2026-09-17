# TileMega R6 — solver pipeline closure and real-model anchoring

The production point-solve/writeback path and the bounded symbolic templates are implemented. Performance acceptance is **not fully closed**: the verifier below retains every failed hard gate. These results do not establish that all single-inference decisions or all real-model operators are solved. The selected configurations remain opt-in; their reference regressions are not a delivered performance improvement.

## 1. Provenance and auditable order

Baseline: `bad8a0d9b17804b73afe00a6d545dcea72cc6cbb`. Branch: `tilemega`. External prompt: `/root/Prompt/TileMega_R6_prompt.md`; SHA256 `908c09131f8b395c6dfdf3e9329db5a684baf965822e528cfc4c5c2bf551793c`. Machine: RTX 4090 / sm_89. The final artifact-only child commit records its source parent in `sass_identity/manifest.json`; that parent is the final source/document revision for this report.

H4 is recomputed by the verifier: C1/C2 precede J1; the separately committed FORK6 precedes the Fuse/R7 decision; every frozen selection exists in its recorded commit before its B1 processes start; W2 precedes S5.

Commits already present when this report was rendered (the subsequent report commit and evidence-only stamp are resolved through the manifest):

```text
567cb81d solver: cost tasks from derived access relations
dbfb4051 solver: retire the per-operator stage formulas
e3ee9419 trace: split the mainloop into loop and fixed work
7fc201bd experiments: calibrate the K loop fixed term
6344daa5 solver: minimize the binding bound, not the path
294c146d solver: reuse prepared readiness across plans
2d388552 cg: carry parametric placement parameters
c8d58a18 cg: write the solved placement back to the graph
71ce688e solver: count worker ownership without sorting groups
80cb9be0 tools: solve before generating the megakernel
5bd862ec experiments: audit operator coverage for real models
b16e839e tools: query compiled occupancy before choosing the grid
39113b7a docs: correct the place channel status and objective
a88ec285 solver: prove placement legality over intervals
66aced8e solver: price loop service from measured backend work
4711e065 experiments: check legacy tables and the complete seqscan subset
fa6d208a solver: carry causal event prices into placement evaluation
d08e0a79 tools: retain the solved shortlist for direct measurement
adca051a solver: batch access cardinalities across a task wave
dc824af3 solver: reuse prices for identical derived task work
0bef657e frontend: lower the supported real model MLP regions
7ade63a2 solver: carry derived traffic into instance pricing
4772fba1 experiments: freeze the reference search choices
e6730cc9 solver: cache immutable task prices across resident plans
887e91d1 solver: exclude numerical failures within their model and theta
32549cbf experiments: freeze the admissible real width small seq choice
b67988f6 solver: prove bounded placement templates across resident grids
1a8d0f24 experiments: compare legacy tables using matching host archives
d61ee926 tools: import torch exports inside the solve command
98827739 experiments: freeze the real width long seq search choice
eaedcd14 experiments: bound fixed work and traffic removed by fusion
99c1a814 experiments: locate task price errors in selected configurations
b41aa9e0 experiments: rebase five completed cells at solved geometry
14347b32 experiments: verify symbolic plans across intervals and grids
1c01faae experiments: add local regeneration runners for sm_120
1ea48486 experiments: record five cells and selected plan sequence checks
265b5bc3 docs: distinguish verified closure from remaining price gaps
4fbb9fa0 experiments: rebuild price and attribution checks from raw data
42fe3eac experiments: complete the six cell search confirmation
1342ecff experiments: retain signed intervention pairs and audit controls
1408f0a9 experiments: render the round six closure report
2a5f2576 experiments: finish the solved configuration ablations
e5f7663d docs: record the round six closure results
d95e1e2b experiments: retain signed reference intervention pairs
3bea714a experiments: check default sass before rendering the report
2c948cef experiments: record the complete real width attribution
```

## 2. Gate results

| Gate | Result | Type | Measurement and raw evidence |
| --- | --- | --- | --- |
| C-a | PASS | hard | command=rg -n StageKind\|NonGemmStageNs lib/Solver/CostModel.cpp lib/Solver/ChainDP.cpp lib/Solver/CouplingInterfaceDP.cpp lib/Solver/TaskModel.cpp; output=''; whole-tree command=rg -n StageKind lib/Solver; reviewed non-price roles={'ModelDescription.cpp': 'model parsing', 'AlignmentPropagation.cpp': 'alignment constraints', 'RuntimeProjection.cpp': 'runtime ownership and dependency projection', 'ScalarTaskWork.cpp': 'CG-derived retained-prefix access region', 'AttentionWork.cpp': 'attention semantic/resource contract validation'}; full grep evidence=COSTMODEL/stagekind_audit.txt |
| C-b | PASS | hard | n=18 rho=0.884416924665 required=0.880288958; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation) |
| C-c | FAIL | hard | n=18 rho=0.766769865841 required=0.85; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation) |
| C-d | PASS | report | n=68 relative_error p50=0.045399639 p90=0.108475523 max=0.138277099; docs/experiments/COSTMODEL/calibrated_replay/replay.tsv |
| C-e | PASS | report | FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4; (iteration_ns,fixed_ns)=[(494.5739772318567, 84.15595062406445), (519.3551879204905, 219.34076673175684), (504.26682039442346, 52.23934537324856), (318.9449407298156, 67.03824809284495)]; scope=instrumented GEMM mainloops, SIMT wait unmeasured; COSTMODEL/raw_kloop/*/phase/selected/dump |
| C-f | PASS | hard | 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s128/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s128/correctness; models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity |
| J-a | PASS | research | real_s4 0.556566970 [0.556037286,0.577495619] docs/experiments/JOINT2/admissible/real_s4/measure; real_s128 0.927801799 [0.913513489,0.948331047] docs/experiments/JOINT2/prepared/real_s128/measure |
| J-b | FAIL | hard | real_s4 queue/semantic_CP=1.849117175 /root/TileMega/docs/experiments/JOINT2/admissible/real_s4/trace/top3/dump; real_s128 queue/semantic_CP=1.701968135 /root/TileMega/docs/experiments/JOINT2/prepared/real_s128/trace/top3/dump |
| J-c | FAIL | hard | reference full_max_us=2480.201 budget=1000 at mha4/s512/legacy_grid_stride; prepare_max_us=145561.384; real full_max_us=2873.300 budget=10000 at real/s128/legacy_grid_stride; prepare_max_us=242049.983; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv |
| J-d | FAIL | hard | 3/6 top1 in measured top2; gqa2_s4 predicted_top1=1/3; gqa2_s128 predicted_top1=3/3; mha4_s4 predicted_top1=1/3; mha4_s128 predicted_top1=1/3; real_s4 predicted_top1=3/3; real_s128 predicted_top1=3/3 |
| J-e | FAIL | hard | gqa2_s4 1.291205752 [1.285477178,1.292035398] docs/experiments/JOINT2/reduced/gqa2_s4/measure; gqa2_s128 1.253164557 [1.252869015,1.254237288] docs/experiments/JOINT2/reduced/gqa2_s128/measure; mha4_s4 1.345579955 [1.342342342,1.352684233] docs/experiments/JOINT2/reduced/mha4_s4/measure; mha4_s128 1.087647778 [1.085632000,1.089210291] docs/experiments/JOINT2/reduced/mha4_s128/measure |
| J-f | PASS | hard | 50/50 docs/experiments/JOINT2/reduced/gqa2_s4/correctness; 50/50 docs/experiments/JOINT2/reduced/gqa2_s128/correctness; 50/50 docs/experiments/JOINT2/reduced/mha4_s4/correctness; 50/50 docs/experiments/JOINT2/reduced/mha4_s128/correctness; 50/50 docs/experiments/JOINT2/admissible/real_s4/correctness; 50/50 docs/experiments/JOINT2/prepared/real_s128/correctness; selected SEQSCAN 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s4_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s4_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s128_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s128_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s4_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s4_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s128_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s128_p512/correctness |
| J-g | PASS | report | FUSE6 enter_r7=0 maximum_bound_share=0.012657 cells=6; gqa2_s4 supported=2 upper_ns=1263.712247 share=0.012593 docs/experiments/JOINT2/fuse_upper/gqa2_s4_selected.tsv; gqa2_s128 supported=2 upper_ns=1302.737889 share=0.004893 docs/experiments/JOINT2/fuse_upper/gqa2_s128_selected.tsv; mha4_s4 supported=4 upper_ns=2527.424495 share=0.012657 docs/experiments/JOINT2/fuse_upper/mha4_s4_selected.tsv; mha4_s128 supported=4 upper_ns=2761.578343 share=0.007023 docs/experiments/JOINT2/fuse_upper/mha4_s128_selected.tsv; real_s4 supported=4 upper_ns=2652.823643 share=0.001124 docs/experiments/JOINT2/fuse_upper/real_s4_selected.tsv; real_s128 supported=4 upper_ns=8975.721727 share=0.001609 docs/experiments/JOINT2/fuse_upper/real_s128_selected.tsv |
| W-a | PASS | hard | rg -n setAttr include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h => ['47:module->setAttr(kPlacementTableAttr,b.getDictionaryAttr({', '54:placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));', '55:placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));', '56:placement->setAttr("window",b.getI64IntegerAttr(1));', '57:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '58:placement->setAttr("resident_only",b.getBoolAttr(true));', '62:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));', '63:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));', '65:module->setAttr("tilemega.solved_placement",b.getStringAttr(selected.name));', '66:module->setAttr("tilemega.solved_kappa",b.getI64IntegerAttr(options.kappa));', '67:module->setAttr("tilemega.solved_residency",b.getI64IntegerAttr(options.residency));', '68:module->setAttr("tilemega.solved_seq",b.getI64IntegerAttr(options.dims.seq));', '69:module->setAttr("tilemega.solved_past",b.getI64IntegerAttr(options.dims.past));', '70:module->setAttr("tilemega.solved_grid",b.getI64IntegerAttr(grid));', '71:module->setAttr("tilemega.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));', '182:module->setAttr("tilemega.event_cost_calibrated",mlir::BoolAttr::get(module.getContext(),']; docs/experiments/WRITEBACK/gqa2_resident_auto.mlir |
| W-b | PASS | hard | command=['build-portable/tools/tilemega-compile', 'docs/experiments/MODELS/llama_mlp/exported_program.pt2', 'docs/experiments/MODELS/llama_mlp/direct_pt2.cu', '--solve', 'docs/experiments/COSTMODEL/event_fit/target.json', '--seq', '4', '--past', '3', '--search-capacity', '3', '--search-domain', 'docs/experiments/COSTMODEL/event_fit/search_domain.json', '--dump-cg', 'docs/experiments/MODELS/llama_mlp/direct_pt2.mlir', '--hop-curve', 'docs/experiments/SIMULATOR/hop_ns.tsv']; output=docs/experiments/MODELS/llama_mlp/direct_pt2.cu byte_equal_to_50_process_tested_source |
| W-c | PASS | hard | 12/12 byte-identical tables; WRITEBACK/legacy_matched/identity |
| W-d | PASS | hard | diff_bytes=0 nodes=2696; docs/experiments/WRITEBACK/roundtrip_gqa2_s4 |
| W-e | PASS | hard | CTest=49 docs/experiments/WRITEBACK/closure_ctest.log; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s2048_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s2048_p512 |
| B1 | PASS | report | gqa2_s4 1250/1250; gqa2_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7633479899497487, 0.051199999999999996, 0.02384, 0.024575999999999987, 0.08704, 0.8610394397346111]; selected_nonpositive_barrier_pairs=0; gqa2_s128 1250/1250; gqa2_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7391304347826086, 0.072512, 0.009024000000000004, 0.012288000000000021, 0.08396800000000004, 0.9664634146341459]; selected_nonpositive_barrier_pairs=0; mha4_s4 1250/1250; mha4_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.711779448621554, 0.11161599999999997, 0.05184, 0.05324799999999996, 0.142432, 1.1497659906396258]; selected_nonpositive_barrier_pairs=0; mha4_s128 1250/1250; mha4_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7347149367959411, 0.221184, 0.019071999999999978, 0.03180800000000006, 0.142528, 1.6800538478797387]; selected_nonpositive_barrier_pairs=0; real_s4 1250/1250; real_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9431479246829535, -0.053279999999999994, 0.061344000000000065, 0.037888000000000144, 0.166976, 0.05989144675276077]; selected_nonpositive_barrier_pairs=0; real_s128 1250/1250; real_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9549205996444158, 1.0341120000000004, 0.22540799999999983, 0.1227520000000002, 0.40748800000000074, 2.0336991406977942]; selected_nonpositive_barrier_pairs=6; REBASE/raw/*/measure |
| S-a | PASS | hard | legacy_grid_stride/G256 proved=128/128; legacy_grid_stride/G340 proved=128/128; rotate/G256 proved=128/128; rotate/G340 proved=128/128; band/G256 proved=128/128; band/G340 proved=128/128; wavefront/G256 proved=128/128; wavefront/G340 proved=128/128; gqa2_s4 binary_variants=1; gqa2_s128 binary_variants=1; mha4_s4 binary_variants=1; mha4_s128 binary_variants=1; real_s4 binary_variants=1; real_s128 binary_variants=1; SYMBOLIC/complete/proofs.tsv and JOINT2 selected raw trace resource records |
| S-b | PASS | hard | 40/40 endpoint/interior complete plan byte comparisons; SYMBOLIC/complete/ |
| S-c | PASS | hard | gqa2_s4=outside_these_four_template_families; gqa2_s128=outside_these_four_template_families; mha4_s4=outside_these_four_template_families; mha4_s128=rotate; real_s4=outside_these_four_template_families; real_s128=outside_these_four_template_families; SYMBOLIC/fit_native/*/fit.tsv and complete native/symbolic/selected tables; point witnesses do not assert impossibility in every quasi-affine family |
| S-d | PASS | report | 16 template/champion/fresh-solve comparisons recomputed from complete raw tables; 40 CG round trips; CPU only; SYMBOLIC/cross_grid/ |
| A1-subset | PASS | report | 50/50 docs/experiments/MODELS/llama_mlp/correctness; exact full architectures NOT covered: executable scope is documented independent normalized MLP components |
| H2 | PASS | hard | models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity |
| H4 | PASS | hard | C1 before J1; C2 before J1; FORK6 before Fuse R7 decision; W2 before symbolic proof implementation; gqa2_s4 selection 4772fba1 precedes all B1 measurements; gqa2_s128 selection 4772fba1 precedes all B1 measurements; mha4_s4 selection 4772fba1 precedes all B1 measurements; mha4_s128 selection 4772fba1 precedes all B1 measurements; real_s4 selection 32549cbf precedes all B1 measurements; real_s128 selection 98827739 precedes all B1 measurements |
| H7 | PASS | report | four local compile/guard checks; no sm_120 execution claim; docs/experiments/COSTMODEL/selfcheck_sm120_closure; docs/experiments/JOINT2/selfcheck_sm120_closure; docs/experiments/REBASE/selfcheck_sm120_closure; docs/experiments/MODELS/selfcheck_sm120_closure |

## 3. Complete verification output

```text
C-a PASS [hard] command=rg -n StageKind|NonGemmStageNs lib/Solver/CostModel.cpp lib/Solver/ChainDP.cpp lib/Solver/CouplingInterfaceDP.cpp lib/Solver/TaskModel.cpp; output=''; whole-tree command=rg -n StageKind lib/Solver; reviewed non-price roles={'ModelDescription.cpp': 'model parsing', 'AlignmentPropagation.cpp': 'alignment constraints', 'RuntimeProjection.cpp': 'runtime ownership and dependency projection', 'ScalarTaskWork.cpp': 'CG-derived retained-prefix access region', 'AttentionWork.cpp': 'attention semantic/resource contract validation'}; full grep evidence=COSTMODEL/stagekind_audit.txt
C-b PASS [hard] n=18 rho=0.884416924665 required=0.880288958; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation)
C-c FAIL [hard] n=18 rho=0.766769865841 required=0.85; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv + SIMULATOR/raw/time/l2.tsv (historical GPU calibration, fresh CPU evaluation)
C-d PASS [report] n=68 relative_error p50=0.045399639 p90=0.108475523 max=0.138277099; docs/experiments/COSTMODEL/calibrated_replay/replay.tsv
C-e PASS [report] FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4; (iteration_ns,fixed_ns)=[(494.5739772318567, 84.15595062406445), (519.3551879204905, 219.34076673175684), (504.26682039442346, 52.23934537324856), (318.9449407298156, 67.03824809284495)]; scope=instrumented GEMM mainloops, SIMT wait unmeasured; COSTMODEL/raw_kloop/*/phase/selected/dump
C-f PASS [hard] 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/gqa2_s128/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s4/correctness; 50/50 docs/experiments/COSTMODEL/raw_kloop/mha4_s128/correctness; models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity
J-a PASS [research] real_s4 0.556566970 [0.556037286,0.577495619] docs/experiments/JOINT2/admissible/real_s4/measure; real_s128 0.927801799 [0.913513489,0.948331047] docs/experiments/JOINT2/prepared/real_s128/measure
J-b FAIL [hard] real_s4 queue/semantic_CP=1.849117175 /root/TileMega/docs/experiments/JOINT2/admissible/real_s4/trace/top3/dump; real_s128 queue/semantic_CP=1.701968135 /root/TileMega/docs/experiments/JOINT2/prepared/real_s128/trace/top3/dump
J-c FAIL [hard] reference full_max_us=2480.201 budget=1000 at mha4/s512/legacy_grid_stride; prepare_max_us=145561.384; real full_max_us=2873.300 budget=10000 at real/s128/legacy_grid_stride; prepare_max_us=242049.983; docs/experiments/COSTMODEL/calibrated_replay/evaluations.tsv
J-d FAIL [hard] 3/6 top1 in measured top2; gqa2_s4 predicted_top1=1/3; gqa2_s128 predicted_top1=3/3; mha4_s4 predicted_top1=1/3; mha4_s128 predicted_top1=1/3; real_s4 predicted_top1=3/3; real_s128 predicted_top1=3/3
J-e FAIL [hard] gqa2_s4 1.291205752 [1.285477178,1.292035398] docs/experiments/JOINT2/reduced/gqa2_s4/measure; gqa2_s128 1.253164557 [1.252869015,1.254237288] docs/experiments/JOINT2/reduced/gqa2_s128/measure; mha4_s4 1.345579955 [1.342342342,1.352684233] docs/experiments/JOINT2/reduced/mha4_s4/measure; mha4_s128 1.087647778 [1.085632000,1.089210291] docs/experiments/JOINT2/reduced/mha4_s128/measure
J-f PASS [hard] 50/50 docs/experiments/JOINT2/reduced/gqa2_s4/correctness; 50/50 docs/experiments/JOINT2/reduced/gqa2_s128/correctness; 50/50 docs/experiments/JOINT2/reduced/mha4_s4/correctness; 50/50 docs/experiments/JOINT2/reduced/mha4_s128/correctness; 50/50 docs/experiments/JOINT2/admissible/real_s4/correctness; 50/50 docs/experiments/JOINT2/prepared/real_s128/correctness; selected SEQSCAN 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s4_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s4_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s128_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/gqa2_s128_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s4_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s4_p512/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s128_p0/correctness; 50/50 docs/experiments/JOINT2/selected_seqscan/mha4_s128_p512/correctness
J-g PASS [report] FUSE6 enter_r7=0 maximum_bound_share=0.012657 cells=6; gqa2_s4 supported=2 upper_ns=1263.712247 share=0.012593 docs/experiments/JOINT2/fuse_upper/gqa2_s4_selected.tsv; gqa2_s128 supported=2 upper_ns=1302.737889 share=0.004893 docs/experiments/JOINT2/fuse_upper/gqa2_s128_selected.tsv; mha4_s4 supported=4 upper_ns=2527.424495 share=0.012657 docs/experiments/JOINT2/fuse_upper/mha4_s4_selected.tsv; mha4_s128 supported=4 upper_ns=2761.578343 share=0.007023 docs/experiments/JOINT2/fuse_upper/mha4_s128_selected.tsv; real_s4 supported=4 upper_ns=2652.823643 share=0.001124 docs/experiments/JOINT2/fuse_upper/real_s4_selected.tsv; real_s128 supported=4 upper_ns=8975.721727 share=0.001609 docs/experiments/JOINT2/fuse_upper/real_s128_selected.tsv
W-a PASS [hard] rg -n setAttr include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h => ['47:module->setAttr(kPlacementTableAttr,b.getDictionaryAttr({', '54:placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));', '55:placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));', '56:placement->setAttr("window",b.getI64IntegerAttr(1));', '57:placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));', '58:placement->setAttr("resident_only",b.getBoolAttr(true));', '62:placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));', '63:placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));', '65:module->setAttr("tilemega.solved_placement",b.getStringAttr(selected.name));', '66:module->setAttr("tilemega.solved_kappa",b.getI64IntegerAttr(options.kappa));', '67:module->setAttr("tilemega.solved_residency",b.getI64IntegerAttr(options.residency));', '68:module->setAttr("tilemega.solved_seq",b.getI64IntegerAttr(options.dims.seq));', '69:module->setAttr("tilemega.solved_past",b.getI64IntegerAttr(options.dims.past));', '70:module->setAttr("tilemega.solved_grid",b.getI64IntegerAttr(grid));', '71:module->setAttr("tilemega.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));', '182:module->setAttr("tilemega.event_cost_calibrated",mlir::BoolAttr::get(module.getContext(),']; docs/experiments/WRITEBACK/gqa2_resident_auto.mlir
W-b PASS [hard] command=['build-portable/tools/tilemega-compile', 'docs/experiments/MODELS/llama_mlp/exported_program.pt2', 'docs/experiments/MODELS/llama_mlp/direct_pt2.cu', '--solve', 'docs/experiments/COSTMODEL/event_fit/target.json', '--seq', '4', '--past', '3', '--search-capacity', '3', '--search-domain', 'docs/experiments/COSTMODEL/event_fit/search_domain.json', '--dump-cg', 'docs/experiments/MODELS/llama_mlp/direct_pt2.mlir', '--hop-curve', 'docs/experiments/SIMULATOR/hop_ns.tsv']; output=docs/experiments/MODELS/llama_mlp/direct_pt2.cu byte_equal_to_50_process_tested_source
W-c PASS [hard] 12/12 byte-identical tables; WRITEBACK/legacy_matched/identity
W-d PASS [hard] diff_bytes=0 nodes=2696; docs/experiments/WRITEBACK/roundtrip_gqa2_s4
W-e PASS [hard] CTest=49 docs/experiments/WRITEBACK/closure_ctest.log; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/gqa2_s2048_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s4_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s4_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s128_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s128_p512; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s2048_p0; 50/50 docs/experiments/WRITEBACK/legacy/seqscan/mha4_s2048_p512
B1 PASS [report] gqa2_s4 1250/1250; gqa2_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7633479899497487, 0.051199999999999996, 0.02384, 0.024575999999999987, 0.08704, 0.8610394397346111]; selected_nonpositive_barrier_pairs=0; gqa2_s128 1250/1250; gqa2_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7391304347826086, 0.072512, 0.009024000000000004, 0.012288000000000021, 0.08396800000000004, 0.9664634146341459]; selected_nonpositive_barrier_pairs=0; mha4_s4 1250/1250; mha4_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.711779448621554, 0.11161599999999997, 0.05184, 0.05324799999999996, 0.142432, 1.1497659906396258]; selected_nonpositive_barrier_pairs=0; mha4_s128 1250/1250; mha4_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.7347149367959411, 0.221184, 0.019071999999999978, 0.03180800000000006, 0.142528, 1.6800538478797387]; selected_nonpositive_barrier_pairs=0; real_s4 1250/1250; real_s4 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9431479246829535, -0.053279999999999994, 0.061344000000000065, 0.037888000000000144, 0.166976, 0.05989144675276077]; selected_nonpositive_barrier_pairs=0; real_s128 1250/1250; real_s128 (rotate/legacy,selected_wait_ms,notify_ms,fence_ms,barrier_ms,protocol/barrier)=[0.9549205996444158, 1.0341120000000004, 0.22540799999999983, 0.1227520000000002, 0.40748800000000074, 2.0336991406977942]; selected_nonpositive_barrier_pairs=6; REBASE/raw/*/measure
S-a PASS [hard] legacy_grid_stride/G256 proved=128/128; legacy_grid_stride/G340 proved=128/128; rotate/G256 proved=128/128; rotate/G340 proved=128/128; band/G256 proved=128/128; band/G340 proved=128/128; wavefront/G256 proved=128/128; wavefront/G340 proved=128/128; gqa2_s4 binary_variants=1; gqa2_s128 binary_variants=1; mha4_s4 binary_variants=1; mha4_s128 binary_variants=1; real_s4 binary_variants=1; real_s128 binary_variants=1; SYMBOLIC/complete/proofs.tsv and JOINT2 selected raw trace resource records
S-b PASS [hard] 40/40 endpoint/interior complete plan byte comparisons; SYMBOLIC/complete/
S-c PASS [hard] gqa2_s4=outside_these_four_template_families; gqa2_s128=outside_these_four_template_families; mha4_s4=outside_these_four_template_families; mha4_s128=rotate; real_s4=outside_these_four_template_families; real_s128=outside_these_four_template_families; SYMBOLIC/fit_native/*/fit.tsv and complete native/symbolic/selected tables; point witnesses do not assert impossibility in every quasi-affine family
S-d PASS [report] 16 template/champion/fresh-solve comparisons recomputed from complete raw tables; 40 CG round trips; CPU only; SYMBOLIC/cross_grid/
A1-subset PASS [report] 50/50 docs/experiments/MODELS/llama_mlp/correctness; exact full architectures NOT covered: executable scope is documented independent normalized MLP components
H2 PASS [hard] models=2/2 bytes_identical source_hashes_match final_source_parent_stamp_valid; docs/experiments/JOINT2/sass_identity
H4 PASS [hard] C1 before J1; C2 before J1; FORK6 before Fuse R7 decision; W2 before symbolic proof implementation; gqa2_s4 selection 4772fba1 precedes all B1 measurements; gqa2_s128 selection 4772fba1 precedes all B1 measurements; mha4_s4 selection 4772fba1 precedes all B1 measurements; mha4_s128 selection 4772fba1 precedes all B1 measurements; real_s4 selection 32549cbf precedes all B1 measurements; real_s128 selection 98827739 precedes all B1 measurements
H7 PASS [report] four local compile/guard checks; no sm_120 execution claim; docs/experiments/COSTMODEL/selfcheck_sm120_closure; docs/experiments/JOINT2/selfcheck_sm120_closure; docs/experiments/REBASE/selfcheck_sm120_closure; docs/experiments/MODELS/selfcheck_sm120_closure
R6_VERIFY gates=27 hard_failures=5 exit=1
```

The nonzero result is intentional whenever hard gates remain failed. All gates execute before exit. The verifier reads raw processes, evaluated cost rows, proof logs and full Plan tables; it does not read this report or the derived comparison/attribution tables. A preliminary valid identity stamp permits capture of actual complete verifier output before rendering this report. After the report commit, the final SASS is regenerated and the full verifier is rerun; its output must match the embedded output byte for byte. No expected PASS line substitutes for execution.

## 4. Stalled items and explicit degraded forms

| Item | Affected scope | Cause / unlock | Inferred work |
| --- | --- | --- | --- |
| C-c/J-c | Unqualified pruning and unrestricted outer search | Coarse rho <0.85; reference full evaluation >1 ms. Repair weights and remaining queue/graph scans, then rerun the frozen 18 points. | 3–7 developer-days plus measurement |
| J-b/J-d/J-e | Claim of balanced and non-regressing selected optimum | Incorrect task service prices and shortlist ranking. Derive combine task traffic/iteration work, price serial scalar phases and compiled occupancy. | 3–7 developer-days |
| Unfit S5 winners | Automatic interval dispatch for five EFT winners | Outside four implemented template families; retain original materialized winners and expand templates only with equivalence/proof. | 2–5 developer-days per additional family |
| B1 causal separation | Real-s128 physical attribution | Unchanged control kernels have two timing bands; preserve signed pairs. Log process-level clocks/power/allocation and rotate five arms within each configuration rather than across all fifty. | 1–2 developer-days + fresh GPU campaign |
| A1 full/maximal subset | Full architecture EX-V1 anchor | Missing or mismatched embedding/head, epsilon/RoPE, QK-norm semantics; attention cuts not yet assembled. | 5–10 developer-days plus correctness runs |

No global correctness-stop was triggered. New real-s4 split8 candidates failed one CPU-golden value identically in L0.5/L1/L2 and were excluded with input-model SHA256 and theta checks; existing configurations were not relabeled. The admitted split16 selection was frozen after a separate pilot.

Degraded forms: finite outer capacity (9 geometry/kappa combinations, 12 for the rejected real-s4 search), five calibrated tile shapes, retained global κ, point-only production W2 solve rather than automatic interval optimization, finite S5 grid branches, GEMM-only FORK6 wait denominator, common-mode-contaminated real-s128 intervention attribution, and sixteen independent normalized MLP components rather than a maximal whole-model subset. B1 measures solver-selected diagnostic configurations even where J gates fail; it is not called a re-established optimal baseline.

## 5. C — unified costs, K-loop calibration and remaining errors

`DeriveTaskWork → TaskMemoryTraffic → TaskInstanceNs` replaces the three NonGemmStageNs call sites. Backend TaskBody traits/dataflow supply resources and phase structure. `StageKind` remains in parsing/ownership projection, not the four core cost-evaluation files.

```sh
rg -n "StageKind|NonGemmStageNs" lib/Solver/{CostModel,ChainDP,CouplingInterfaceDP,TaskModel}.cpp
# no matches
rg -n "StageKind|NonGemmStageNs" lib/Solver
# Complete output and reviewed non-price roles: COSTMODEL/stagekind_audit.txt
```

18-point full Spearman 0.884416925 (R5 threshold 0.880288958); coarse 0.766769866 (gate 0.85). Historical 68-dump replay absolute relative errors: p50 4.539964%, p90 10.847552%, max 13.827710%, versus R5 4.54/10.85/13.83%. These are fresh CPU reevaluations of historical calibration evidence, not fresh GPU speedup claims.

```text
FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4
```

The denominator is instrumented GEMM mainloops. SIMT exposed wait is unmeasured; the all-path-mainloop lower-bound denominator gives 0.161782. The prescribed rule is applied to the measured GEMM scope, with this deviation explicit. Per-cell iteration/fixed p50 ns: gqa2 s4 494.574/84.156, s128 519.355/219.341; mha4 s4 504.267/52.239, s128 318.945/67.038; real s4 266.040/52.492, s128 538.262/102.693. Raw: `../COSTMODEL/raw_kloop/`.

The task-zero audit (`../COSTMODEL/selected_prices/comparison.tsv`, reconstructed from price rows and fresh trace slots) finds combine measured/predicted medians 23.282/25.320/31.399/12.885 for gqa2 s4, mha4 s4, real s4/s128. This is the remaining `PlacementSolvePass.h::projected.combine` whole-stage/wave conversion. Non-combine medians are 1.449/1.902/1.449/1.313/2.213/2.007. Attention’s thread-zero softmax loop also needs backend-declared serial iteration work. The retired branches are unified; **every runtime-task price is not yet fully unified/calibrated**.

## 6. J — binding objective, fresh comparisons and Fuse bound

| Cell | L2 ms | / fresh R5 control [95% CI] | / fresh R5 champion [95% CI] | queue/semantic CP | L2/L1 | L2/floor | Predicted top1 measured rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 0.149504 | 0.529205 [0.528152,0.530387] | 1.291206 [1.285477,1.292035] | 0.816327 | 0.656250 | 1.489796 | 1/3 |
| gqa2_s128 | 0.304064 | 0.687053 [0.686110,0.687500] | 1.253165 [1.252869,1.254237] | 0.853846 | 0.824826 | 1.142067 | 3/3 |
| mha4_s4 | 0.305152 | 0.546332 [0.542867,0.547619] | 1.345580 [1.342342,1.352684] | 0.856410 | 0.726829 | 1.528205 | 1/3 |
| mha4_s128 | 0.523456 | 0.588412 [0.569412,0.603033] | 1.087648 [1.085632,1.089210] | 0.760417 | 0.770897 | 1.331217 | 1/3 |
| real_s4 | 2.626560 | 0.556567 [0.556037,0.577496] | 0.637556 [0.636956,0.638261] | 1.849117 | 0.926678 | 1.113281 | 3/3 |
| real_s128 | 6.368480 | 0.927802 [0.913513,0.948331] | 0.934261 [0.905859,0.972871] | 1.701968 | 0.954487 | 1.141560 | 3/3 |

| Cell | Selected geometry / split / κ / residency | Placement | CP µs | Queue µs | CP nodes | Mean node ns |
| --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 32x16x64s2split4kappa1r5 | eft | 100.352 | 81.920 | 28 | 3584.000 |
| gqa2_s128 | 32x16x64s2split1kappa4r5 | eft | 266.240 | 227.328 | 20 | 13312.000 |
| mha4_s4 | 32x16x64s2split4kappa1r4 | eft | 199.680 | 171.008 | 56 | 3565.714 |
| mha4_s128 | 32x16x64s2split1kappa1r4 | rotate | 393.216 | 299.008 | 36 | 10922.667 |
| real_s4 | 64x128x16s2split16kappa4r3 | eft | 1275.904 | 2359.296 | 52 | 24536.615 |
| real_s128 | 64x128x16s2split4kappa4r3 | eft | 3277.824 | 5578.752 | 56 | 58532.571 |

| Cell | Fresh R5 champion CP µs | Selected CP µs | Fresh R5 champion queue/CP | Selected queue/CP |
| --- | --- | --- | --- | --- |
| gqa2_s4 | 82.944 | 100.352 | 0.962963 | 0.816327 |
| gqa2_s128 | 204.800 | 266.240 | 0.895000 | 0.853846 |
| mha4_s4 | 162.816 | 199.680 | 0.949686 | 0.856410 |
| mha4_s128 | 372.736 | 393.216 | 0.920330 | 0.760417 |
| real_s4 | 2004.992 | 1275.904 | 1.751788 | 1.849117 |
| real_s128 | 4577.280 | 3277.824 | 1.428188 | 1.701968 |

Historical R5 real-width queue/CP was 1.7589/1.4174; current selected trace ratios are 1.849117/1.701968. These ratios remain above the frozen J-b line.

Controls and candidates are twenty-five rotated, same-session, fresh-process rounds; choice uses the separate five-round pilot and is frozen before confirmation. The zero-sync union bound used by search is distinguished from semantic CP in J-b; replacing J-b’s denominator with a queue-containing CP would make that gate tautological. Minimizing max(CP,queue) itself does not enforce queue/semantic-CP ≤1.

| Cell | Predicted six-placement order at selected geometry | Fresh B1 full-time order |
| --- | --- | --- |
| gqa2_s4 | eft < rotate < wavefront < chain < legacy_grid_stride < balanced | eft < wavefront < rotate < chain < legacy_grid_stride < balanced |
| gqa2_s128 | eft < rotate < wavefront < chain < legacy_grid_stride < balanced | rotate < wavefront < eft < chain < legacy_grid_stride < balanced |
| mha4_s4 | eft < wavefront < rotate < chain < legacy_grid_stride < balanced | chain < rotate < eft < wavefront < legacy_grid_stride < balanced |
| mha4_s128 | rotate < wavefront < eft < chain < legacy_grid_stride < balanced | rotate < wavefront < eft < chain < legacy_grid_stride < balanced |
| real_s4 | eft < wavefront < rotate < chain < legacy_grid_stride < balanced | wavefront < eft < rotate < chain < legacy_grid_stride < balanced |
| real_s128 | eft < wavefront < rotate < chain < legacy_grid_stride < balanced | wavefront < eft < rotate < legacy_grid_stride < chain < balanced |

The B1 order above uses marginal full-time medians. Paired ratio medians are reported separately and need not induce the same order; neither is used to replace the independently frozen selection after seeing confirmatory data.

Full simulator evaluation peaks at 2480.201 µs for a reference point and 2873.300 µs for real-width; graph preparation separately peaks at 145561.384/242049.983 µs. The reference 1 ms gate still fails; real-width meets 10 ms. Preparation is not hidden as free.

| Cell | Supported pairs | Removed nodes | Removed bytes | Fixed upper ns | Traffic upper ns | Total upper ns | / measured floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 2 | 16.0 | 8192.0 | 795.82491445376 | 467.88733279999997 | 1263.7122472537599 | 0.012592795831211734 |
| gqa2_s128 | 2 | 512.0 | 262144.0 | 809.20952995616 | 493.52835865999987 | 1302.73788861616 | 0.004893096035968149 |
| mha4_s4 | 4 | 64.0 | 32768.0 | 1591.64982890752 | 935.7746655999999 | 2527.4244945075197 | 0.012657374271371794 |
| mha4_s128 | 4 | 2048.0 | 1048576.0 | 1671.95752190336 | 1089.6208207199998 | 2761.5783426233597 | 0.0070230569016097 |
| real_s4 | 4 | 128.0 | 65536.0 | 1634.65794674848 | 1018.16569592 | 2652.82364266848 | 0.0011244132328747558 |
| real_s128 | 4 | 4096.0 | 2097152.0 | 5252.86011959264 | 3722.8616076 | 8975.72172719264 | 0.0016089121235704042 |

```text
FUSE6 enter_r7=0 maximum_bound_share=0.012657 cells=6
```

Only the existing legal adjacent RoPE→KVAppend family is bounded. Fixed upper=.232×separate envelope; traffic upper is the unified-path reduction with internal storage free, external traffic/arithmetic retained; total is capped at the separate envelope. No added barrier, copy, recomputation or occupancy loss is charged. This is an optimistic model bound under its wave assumptions, not an executed fusion or proof about all possible fusion families. The common .232 fixed fraction is an extrapolation, not per-pair phase data; even eliminating the entire modeled pair envelope caps the largest share at .034358, below .10. The existing Fuse direction gate remains unchanged.

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

Physical EFT worker/slot arrays are stored in CG module attributes, trading module size for self-contained round trips. `WRITEBACK/roundtrip_gqa2_s4/{cg_plan,host_plan}.tsv`: 2696 nodes, diff empty. Matched baseline/current host archives produce 12/12 identical legacy schedule/waits/events files. Default SEQSCAN is 600/600; selected reference subset adds 400/400. All 49 CTest cases pass. W2 production is a point solve; automatic interval winner selection remains partial.

## 8. B1 — attribution at solver-selected geometry

Six placements and four additional protocol variants, five arms each, twenty-five rotated fresh rounds per cell. C3(a) is intentionally inert at W=1. C1/C2/C3 are cumulative flags; no window is enabled. Unsafe probes may fail numerical comparison, but full arms must PASS.

| Cell | Config | Full L2 ms | Wait ms | Notify ms | Fence ms | Barrier ms | Protocol/barrier | / selected | / legacy | L2/L1 | Floor µs | L2/floor | Nonpositive barrier pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa2_s4 | eft | 0.149696 | 0.051200 | 0.023840 | 0.024576 | 0.087040 | 0.861039 | 1.000000 | 0.732412 | 0.656285 | 97.280 | 1.538816 | 0 |
| gqa2_s4 | rotate | 0.155648 | 0.115712 | 0.003264 | 0.023552 | 0.087040 | 1.365208 | 1.039353 | 0.763348 | 0.681901 | 101.376 | 1.535354 | 0 |
| gqa2_s4 | wavefront | 0.151552 | 0.052224 | 0.024544 | 0.022784 | 0.087008 | 0.882310 | 1.010940 | 0.742305 | 0.664083 | 98.304 | 1.541667 | 0 |
| gqa2_s4 | chain | 0.163552 | 0.121824 | 0.005120 | 0.021504 | 0.087040 | 1.461853 | 1.089921 | 0.798995 | 0.716889 | 106.496 | 1.535757 | 0 |
| gqa2_s4 | legacy_grid_stride | 0.204480 | 0.059392 | 0.037888 | 0.031744 | 0.087040 | 1.110373 | 1.365352 | 1.000000 | 0.896396 | 123.904 | 1.650310 | 0 |
| gqa2_s4 | balanced | 1.048576 | 0.828160 | 0.032448 | 0.174272 | 0.087136 | 9.881487 | 7.006849 | 5.140623 | 4.602925 | 165.888 | 6.320988 | 0 |
| gqa2_s4 | protocol_off | 0.165888 | 0.041984 | 0.044032 | 0.015360 | 0.088064 | 0.989091 | 1.111969 | 0.814070 | 0.726457 | 101.376 | 1.636364 | 0 |
| gqa2_s4 | c1 | 0.149440 | 0.049344 | 0.025312 | 0.023648 | 0.087040 | 0.858824 | 0.997849 | 0.730000 | 0.654800 | 100.352 | 1.489158 | 0 |
| gqa2_s4 | c2 | 0.149376 | 0.049344 | 0.025088 | 0.023648 | 0.086816 | 0.859455 | 0.993831 | 0.730000 | 0.654709 | 101.376 | 1.473485 | 0 |
| gqa2_s4 | c3 | 0.149504 | 0.050176 | 0.024608 | 0.024320 | 0.086816 | 0.865160 | 0.998494 | 0.731028 | 0.654892 | 100.352 | 1.489796 | 0 |
| gqa2_s128 | eft | 0.304128 | 0.072512 | 0.009024 | 0.012288 | 0.083968 | 0.966463 | 1.000000 | 0.806556 | 0.825000 | 265.216 | 1.146718 | 0 |
| gqa2_s128 | rotate | 0.278528 | 0.109248 | 0.006272 | 0.010080 | 0.085952 | 1.343006 | 0.916992 | 0.739130 | 0.754821 | 232.448 | 1.198238 | 0 |
| gqa2_s128 | wavefront | 0.283648 | 0.068640 | 0.014176 | 0.010240 | 0.084032 | 0.986672 | 0.934163 | 0.752044 | 0.769910 | 234.496 | 1.209607 | 0 |
| gqa2_s128 | chain | 0.355072 | 0.182080 | 0.004992 | 0.013312 | 0.084000 | 2.215619 | 1.167851 | 0.940217 | 0.963194 | 260.096 | 1.365157 | 0 |
| gqa2_s128 | legacy_grid_stride | 0.377856 | 0.092160 | 0.016256 | 0.016672 | 0.084768 | 1.280497 | 1.239840 | 1.000000 | 1.022692 | 266.240 | 1.419231 | 0 |
| gqa2_s128 | balanced | 0.818880 | 0.488384 | 0.016384 | 0.034912 | 0.084768 | 5.947866 | 2.689330 | 2.170732 | 2.214619 | 313.344 | 2.613358 | 0 |
| gqa2_s128 | protocol_off | 0.318144 | 0.070656 | 0.021792 | 0.010240 | 0.079872 | 1.154400 | 1.046423 | 0.843661 | 0.872595 | 243.712 | 1.305410 | 0 |
| gqa2_s128 | c1 | 0.304128 | 0.073728 | 0.007488 | 0.012288 | 0.084544 | 0.949510 | 0.999790 | 0.804878 | 0.824317 | 266.240 | 1.142308 | 0 |
| gqa2_s128 | c2 | 0.303104 | 0.072576 | 0.008352 | 0.011456 | 0.082880 | 0.975507 | 0.997054 | 0.802703 | 0.825123 | 268.288 | 1.129771 | 0 |
| gqa2_s128 | c3 | 0.304128 | 0.073024 | 0.008192 | 0.012288 | 0.083936 | 0.971571 | 1.000000 | 0.806014 | 0.825000 | 264.192 | 1.151163 | 0 |
| mha4_s4 | eft | 0.307200 | 0.111616 | 0.051840 | 0.053248 | 0.142432 | 1.149766 | 1.000000 | 0.752500 | 0.726667 | 180.224 | 1.704545 | 0 |
| mha4_s4 | wavefront | 0.311296 | 0.116736 | 0.051200 | 0.052256 | 0.141408 | 1.185551 | 1.015873 | 0.765000 | 0.739154 | 180.224 | 1.727273 | 0 |
| mha4_s4 | rotate | 0.290560 | 0.205824 | 0.006464 | 0.044032 | 0.141600 | 1.485383 | 0.946667 | 0.711779 | 0.690617 | 177.152 | 1.640173 | 0 |
| mha4_s4 | chain | 0.285696 | 0.210208 | 0.009216 | 0.045056 | 0.130400 | 1.690476 | 1.010806 | 0.761905 | 0.737864 | 172.032 | 1.660714 | 0 |
| mha4_s4 | legacy_grid_stride | 0.407552 | 0.116576 | 0.078848 | 0.067584 | 0.141312 | 1.388788 | 1.328904 | 1.000000 | 0.968370 | 245.760 | 1.658333 | 0 |
| mha4_s4 | balanced | 3.113824 | 2.677760 | 0.062464 | 0.604160 | 0.142112 | 19.404560 | 10.120482 | 7.627253 | 7.411192 | 299.008 | 10.413848 | 0 |
| mha4_s4 | protocol_off | 0.321536 | 0.085856 | 0.086016 | 0.031840 | 0.142336 | 1.164503 | 1.080000 | 0.814070 | 0.763990 | 199.680 | 1.610256 | 0 |
| mha4_s4 | c1 | 0.307200 | 0.112544 | 0.051200 | 0.053280 | 0.142336 | 1.152400 | 1.000000 | 0.753916 | 0.728487 | 198.656 | 1.546392 | 0 |
| mha4_s4 | c2 | 0.308224 | 0.113664 | 0.051136 | 0.055296 | 0.143360 | 1.143973 | 1.003344 | 0.756892 | 0.728979 | 199.680 | 1.543590 | 0 |
| mha4_s4 | c3 | 0.307200 | 0.111744 | 0.052128 | 0.053984 | 0.142336 | 1.151079 | 1.000521 | 0.753769 | 0.728558 | 179.200 | 1.714286 | 0 |
| mha4_s128 | rotate | 0.547840 | 0.221184 | 0.019072 | 0.031808 | 0.142528 | 1.680054 | 1.000000 | 0.734715 | 0.772006 | 438.272 | 1.250000 | 0 |
| mha4_s128 | wavefront | 0.571264 | 0.141312 | 0.029344 | 0.025600 | 0.142528 | 1.198698 | 1.043071 | 0.766939 | 0.805141 | 467.968 | 1.220733 | 0 |
| mha4_s128 | eft | 0.618496 | 0.158688 | 0.030912 | 0.032544 | 0.143136 | 1.325134 | 1.130491 | 0.829670 | 0.871730 | 518.144 | 1.193676 | 0 |
| mha4_s128 | chain | 0.673792 | 0.305152 | 0.021504 | 0.034048 | 0.143264 | 2.277047 | 1.229907 | 0.903950 | 0.950867 | 482.304 | 1.397028 | 0 |
| mha4_s128 | legacy_grid_stride | 0.745280 | 0.168928 | 0.034880 | 0.030464 | 0.142336 | 1.427870 | 1.361072 | 1.000000 | 1.050578 | 491.520 | 1.516276 | 0 |
| mha4_s128 | balanced | 1.140736 | 0.654144 | 0.033664 | 0.087040 | 0.142592 | 4.812950 | 2.084270 | 1.531068 | 1.608971 | 445.440 | 2.560920 | 0 |
| mha4_s128 | protocol_off | 0.551936 | 0.200704 | 0.038976 | 0.023424 | 0.157696 | 1.524732 | 1.007477 | 0.740130 | 0.762612 | 415.744 | 1.327586 | 0 |
| mha4_s128 | c1 | 0.540576 | 0.216064 | 0.017120 | 0.025504 | 0.142336 | 1.633116 | 0.987394 | 0.725119 | 0.762421 | 395.264 | 1.367633 | 0 |
| mha4_s128 | c2 | 0.541696 | 0.217088 | 0.016384 | 0.026848 | 0.142336 | 1.640063 | 0.989136 | 0.727136 | 0.764328 | 435.200 | 1.244706 | 0 |
| mha4_s128 | c3 | 0.541696 | 0.217024 | 0.017408 | 0.026688 | 0.142528 | 1.639929 | 0.990637 | 0.727023 | 0.763824 | 435.200 | 1.244706 | 0 |
| real_s4 | eft | 2.624512 | -0.053280 | 0.061344 | 0.037888 | 0.166976 | 0.059891 | 1.000000 | 0.935120 | 0.926447 | 2293.760 | 1.144196 | 0 |
| real_s4 | wavefront | 2.583552 | -0.047264 | 0.048256 | 0.038208 | 0.157696 | 0.022313 | 0.984545 | 0.920161 | 0.913656 | 2343.936 | 1.102228 | 0 |
| real_s4 | rotate | 2.650432 | 0.630048 | 0.014112 | 0.038976 | 0.164608 | 3.833333 | 1.009861 | 0.943148 | 0.936087 | 1800.192 | 1.472305 | 0 |
| real_s4 | chain | 2.673312 | 0.505792 | 0.022624 | 0.036224 | 0.165152 | 3.194210 | 1.017899 | 0.952554 | 0.944384 | 1777.664 | 1.503834 | 0 |
| real_s4 | legacy_grid_stride | 2.808864 | 0.006848 | 0.080896 | 0.056224 | 0.167616 | 0.521472 | 1.069382 | 1.000000 | 0.990979 | 2514.944 | 1.116869 | 0 |
| real_s4 | balanced | 24.119295 | 18.930592 | 0.172960 | 0.816128 | 0.164672 | 115.986203 | 9.196339 | 8.591859 | 8.526573 | 3986.432 | 6.050347 | 0 |
| real_s4 | protocol_off | 2.680832 | -0.043008 | 0.098304 | 0.044032 | 0.195616 | 0.286328 | 1.021764 | 0.954570 | 0.936861 | 2259.968 | 1.186226 | 0 |
| real_s4 | c1 | 2.628608 | -0.035712 | 0.044864 | 0.042176 | 0.165888 | 0.066886 | 1.001633 | 0.935837 | 0.928052 | 2342.912 | 1.121941 | 0 |
| real_s4 | c2 | 2.629664 | -0.031520 | 0.044992 | 0.042112 | 0.162752 | 0.085668 | 1.002379 | 0.936886 | 0.928494 | 2285.568 | 1.150552 | 0 |
| real_s4 | c3 | 2.629568 | -0.032992 | 0.048800 | 0.040704 | 0.166688 | 0.079738 | 1.001890 | 0.936178 | 0.928829 | 2359.296 | 1.114556 | 0 |
| real_s128 | eft | 6.371328 | 1.034112 | 0.225408 | 0.122752 | 0.407488 | 2.033699 | 1.000000 | 0.958887 | 0.956501 | 5579.776 | 1.141861 | 6 |
| real_s128 | wavefront | 6.240512 | 0.403456 | 0.126720 | 0.040192 | 0.271552 | 0.970964 | 0.990829 | 0.933417 | 0.950715 | 5839.872 | 1.068604 | 6 |
| real_s128 | rotate | 6.383616 | 1.512576 | 0.032032 | 0.118560 | 0.415744 | 3.641829 | 1.000078 | 0.954921 | 0.955994 | 4838.400 | 1.319365 | 3 |
| real_s128 | chain | 6.824864 | 1.442912 | 0.197728 | 0.125056 | 0.469248 | 2.205479 | 1.068431 | 1.018787 | 1.022154 | 5017.600 | 1.360185 | 3 |
| real_s128 | legacy_grid_stride | 6.686720 | 0.181408 | 0.069376 | 0.128000 | 0.271328 | 0.808844 | 1.042876 | 1.000000 | 0.999620 | 6227.968 | 1.073660 | 3 |
| real_s128 | balanced | 67.603455 | 50.837502 | 0.419041 | 1.272827 | 0.402784 | 89.383073 | 11.105409 | 10.649179 | 10.607531 | 14443.520 | 4.680539 | 2 |
| real_s128 | protocol_off | 6.439936 | 1.008416 | 0.267232 | 0.093184 | 0.407296 | 2.371978 | 1.010768 | 0.971706 | 0.971392 | 5525.504 | 1.165493 | 4 |
| real_s128 | c1 | 6.108032 | 0.928096 | 0.206848 | -0.027648 | 0.273408 | 0.899803 | 0.999155 | 0.958340 | 0.956499 | 5593.088 | 1.092068 | 8 |
| real_s128 | c2 | 6.362816 | 0.948224 | 0.216000 | 0.120096 | 0.341280 | 2.881288 | 1.000825 | 0.952785 | 0.954456 | 5592.064 | 1.137830 | 3 |
| real_s128 | c3 | 6.372352 | 1.172480 | 0.214848 | 0.221376 | 0.439040 | 1.315940 | 1.000670 | 0.952374 | 0.955409 | 5577.728 | 1.142464 | 6 |

| Cell | Rotate / legacy [95% CI] | R3 B on / off [95% CI] | +C1 / B | +C1+C2 / B | +C1+C2+C3(a) / B |
| --- | --- | --- | --- | --- | --- |
| gqa2_s4 | 0.763348 [0.759781,0.764687] | 0.899306 [0.896412,0.901265] | 0.997849 | 0.993831 | 0.998494 |
| gqa2_s128 | 0.739130 [0.734291,0.743233] | 0.955636 [0.954281,0.958065] | 0.999790 | 0.997054 | 1.000000 |
| mha4_s4 | 0.711779 [0.711055,0.713568] | 0.925926 [0.910370,0.967742] | 1.000000 | 1.003344 | 1.000521 |
| mha4_s128 | 0.734715 [0.733359,0.736485] | 0.992579 [0.990724,0.994424] | 0.987394 | 0.989136 | 0.990637 |
| real_s4 | 0.943148 [0.933309,0.948023] | 0.978700 [0.978103,0.983907] | 1.001633 | 1.002379 | 1.001890 |
| real_s128 | 0.954921 [0.952561,0.981870] | 0.989346 [0.940345,1.013014] | 0.999155 | 1.000825 | 1.000670 |

Placement ratios keep the R3 B protocol fixed. Protocol ratios keep the solver-selected placement and geometry fixed; the on/off row is the reciprocal of the retained paired off/on statistic, with interval endpoints reversed. These are conditional contributions, not multiplicative independent factors. R1’s historical pooled rotate/legacy was .6705; R3’s historical default-placement cumulative B reductions were 5.16/3.48/4.79/3.74%. R4 reduced the protocol/barrier diagnostic by 36% while full time increased. None of those historical percentages substitutes for the fresh configuration-specific rows here.

Each column is aggregated from its own twenty-five pairs: medians need not add, and a median of ratios need not equal a ratio of medians. All medians and paired 95% intervals are retained in `REBASE/attribution_*.tsv`; raw logs and trace dumps are under `REBASE/raw/`. Wait=full−nowait, notify=nowait−neither, fence=full−nofence, barrier=L1full−L1nosync. Signed differences are not clamped: an unsafe arm can change execution/overlap, so these are intervention differences rather than disjoint nonnegative physical service times. Real-s128 also exhibits two timing bands in the unchanged L0.5/L1 controls. Full/nofence L1 SASS is identical, so that control variation cannot be assigned to the removed L2 fence. Negative barrier pairs are retained with their original signs, including the ratio; a zero denominator makes the complete ratio statistic undefined instead of deleting a pair. This corrects an extra analyzer positivity assumption, not a frozen R6 gate or its measurement formula. Causal attribution is explicitly degraded; next diagnostics need per-process clocks/power/allocation traces and arm rotation within each configuration to keep paired controls close in time. The historical rotate/legacy 0.6705 is compared with the fresh `rotate` rows above, not reused as a current measurement.

## 9. S5 — proven domain and cross-grid comparison

Source graph and geometry are in `SYMBOLIC/complete/provenance.json`. Four families, integer seq [1,128], grids 256/340; 166 exhaustive ISL certificate pieces. Wavefront G340 uses singleton certificates for every integer in the interval, not inference from five samples. Forty endpoint/interior native/template tables and forty CG serialize/read evaluations agree. One carried CG contains the two constant-grid branches per family; these are not extra kernel variants.

Five EFT champions do not fit the four families; mha4 s128 fits rotate. No worse template replaces a measured winner. Fresh six-catalog solves on the same CG choose wavefront at G256 and EFT at G340 (`SYMBOLIC/cross_grid/`). Actual SM count/resident witness are unchanged. This is finite CPU portability evidence, not unbounded variable-divisor Presburger support or cross-architecture performance.

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
| llama | Q/K/V projections | kGemm | shape_supported; exact export path pending |
| llama | RoPE frequency scaling | kRoPE | config-dependent frequency table can be an input buffer |
| llama | RoPE rotation | kRoPE | partial: backend rounds positions and angles to ModelElement before sin/cos; public implementation uses FP32 angles |
| llama | KV cache append | kKVAppend | shape_supported |
| llama | GQA score/mask/softmax/value | kAttention | shape_supported; composite implementation, precise export check pending |
| llama | output projection | kGemm | shape_supported |
| llama | attention residual | kAdd / kGemm epilogue | shape_supported |
| llama | post-attention RMSNorm | kRMSNorm | configured epsilon=1e-05; backend epsilon=1e-06; parameter gap |
| llama | gate/up projections | kGemm | shape_supported |
| llama | SiLU times gate | kElementwise | shape_supported |
| llama | down projection and residual | kGemm / kAdd | shape_supported |
| llama | final RMSNorm | kRMSNorm | partial: no standalone final-normalization stage in DecoderLayerPattern; configured epsilon=1e-05; backend epsilon=1e-06; parameter gap |
| llama | vocabulary projection | kGemm | partial: collective can implement dimensions; decoder-only importer does not emit final LM head |
| llama | reshape/transpose/broadcast | layout metadata | absorbed by semantic access maps; no standalone task |
| qwen | token embedding | missing | no indexed embedding TaskKind / decoder importer input is hidden states |
| qwen | input pre-attention RMSNorm | kRMSNorm | configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | Q/K/V projections | kGemm | shape_supported; exact export path pending |
| qwen | per-head Q/K RMSNorm | missing | Qwen only; existing RMSNorm owns one token row, not one token/head |
| qwen | RoPE frequency scaling | kRoPE | config-dependent frequency table can be an input buffer |
| qwen | RoPE rotation | kRoPE | partial: backend rounds positions and angles to ModelElement before sin/cos; public implementation uses FP32 angles |
| qwen | KV cache append | kKVAppend | shape_supported |
| qwen | GQA score/mask/softmax/value | kAttention | shape_supported; composite implementation, precise export check pending |
| qwen | output projection | kGemm | shape_supported |
| qwen | attention residual | kAdd / kGemm epilogue | shape_supported |
| qwen | post-attention RMSNorm | kRMSNorm | configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | gate/up projections | kGemm | shape_supported |
| qwen | SiLU times gate | kElementwise | shape_supported |
| qwen | down projection and residual | kGemm / kAdd | shape_supported |
| qwen | final RMSNorm | kRMSNorm | partial: no standalone final-normalization stage in DecoderLayerPattern; configured epsilon=1e-06; backend epsilon=1e-06; matches |
| qwen | vocabulary projection | kGemm | partial: collective can implement dimensions; decoder-only importer does not emit final LM head |
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

There are 11 current TaskKind values, not 16. A new distinct kind/ownership can touch 15 conditional sites; new prices come through semantics/traits rather than per-operator Solver formulas. The executed subset is sixteen independent, normalized Llama-width MLP components with distinct random weights and 32 boundary inputs, not a sequential decoder or maximal covered whole graph. Correctness 50/50, sixteen output checks each. Warmup=0/repeat=1 L2 median 2.016880 ms (range 1.973248–2.034688), L1 median 3.050624 ms; these are single-launch diagnostic observations, not full-model steady-state gains.

## 11. Deviations and reasons

The explicit degraded forms in §4 remain open. The requested conventional source filenames include header-only production JointSearch/placement passes in this checkout; symbol names were used. S5’s large-map parser was reused inside the evidence driver without changing the proof. The global timer has 1024 ns ticks; zero durations are preserved. Source calibration replay uses historical GPU traces exactly as requested for replay gates, while J/B performance controls are fresh. sm_120 runners were compile/guard self-checked here and were not run on sm_120. The report and SASS stamp use a parent/child commit manifest to avoid claiming that an earlier code revision was the final one.

## 12. Excluded work confirmation

No EX-V1 full anchored decode sweep; no Fuse outer-search decision or fused implementation; no EX-E4 shared-memory pipeline or TaskSmem union-lifetime change; no A1 missing-operator implementation; no EX-E5, EX-S4 or L5 serving. Plan execution semantics, W=1, monotonic epochs, release rules and legality checks remain unchanged. Only the skeleton changelog and §4.4.2 were edited. User-owned edits in PLACE_EFT2/summary.md and SYNC_V2/sass_identity/meta.tsv were preserved.

## 13. Failed gates: causes and concrete next changes

C-c/J-d: bound weights and shortlist fine ordering do not resolve task duration correctly. Fix the combine `whole_stage / waves` conversion in PlacementSolvePass.h, and add serial iteration/active-lane terms to backend ScalarDataflow for AttentionChunkTaskBody’s thread-zero loops. PriceTaskInstances currently passes active_ctas=1; fit service dilation under actual compiled residency. J-c: retain compressed/grouped graph readiness and replace remaining per-plan initialization/queue rebuild scans; cache only immutable shape/target/theta inputs, measuring preparation separately. J-b: the binding objective does not mathematically enforce queue/semantic-CP≤1; if retained as an admissibility constraint, explicitly search its feasible set and remeasure the latency tradeoff, without changing this round’s gate. J-e: the same weight defects select extra split nodes and poorer geometry/placement; verify repair against the unchanged fresh-control protocol. Do not call an R5 fallback a completed new solution.

## 14. R7 priorities and decisions still outside the solver

| Priority | Block | Dependency / action | Inferred effort |
| --- | --- | --- | --- |
| 0 | Finish price/ranking closure | Access-derived combine work, serial scalar phases, residency service and budget; preserve the frozen gates. | 3–7 developer-days + fresh campaigns |
| 1 | A1 semantic/operator gaps and maximal covered graph | Exact epsilon/RoPE, embedding/head, QK-norm ownership, compose covered attention boundaries. | 5–10 developer-days |
| 2 | EX-V1 full anchored main benchmark | Depends on corrected anchor and qualified search; decode seq 1/4/16/64, complete ablations. | 3–5 developer-days + 1–3 GPU-days |
| 3 | EX-E4 step 2 design and bounded prototype | FORK6 rule1 on GEMM scope; distinguish intra-K-loop waits from first-slot prefetch; explicitly redesign §8.6 before shared buffering. | 4–8 developer-days |
| defer | Fuse in outer search | Current legal-family model upper <10%; expand rejected ownership families and reprice before changing the decision. | 2–5 developer-days for an expanded bound |
| later | EX-E5/S4 → L5 serving | After single-inference correctness/resource contracts and cost quality; separate serving-state/lifetime design. | 1–2 weeks for E5/S4, then 2–4 weeks for L5 |

Automatic decisions now include geometry, split-K, global κ, residency, placement and slot order for the evaluated point/candidate domain. Still outside complete solver control: fusion partitioning, cross-task shared-memory pipeline selection, per-stage κ, interval winner/variant selection for unfit materialized schedules, search-domain/capacity policy, and missing architecture import/semantics. Therefore single-inference closure is partial even though the concrete import→solve→CG→CUDA channel now executes.
