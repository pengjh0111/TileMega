# TileMega R9 — solver reconstruction

**IN PROGRESS — 8/40 measured arms. This is not the final R9 delivery.**

**2026-09-25 阶段性代码审查推送：完整测试中。** 当前实现导览、已完成验证、排队情况及配置数定义见 [review_status.md](review_status.md)。本次推送不代表 R9 最终验收；所有现有测试进程保持不变。

## 1. Provenance and commits

Generated UTC: 2026-09-25T04:13:03.185984+00:00.

Baseline: `197cd66da36c4fe8929698135d62dcb032b4e91f`. Report source HEAD: `c5b62e5364d0f94f1b3b1298d0891187a9902a51`.

Prompt: `/root/Prompt/TileMega_R9_prompt.md`; SHA256 `6779512fd12cde3dd13800a52a330d8843e457cdfa6e766a505296750eb5ece1`. TODO update was already applied in baseline commit `197cd66da36c4fe8929698135d62dcb032b4e91f`.

````text
9a6353661 docs: restructure the plan around the solver rework
92f07b2b3 experiments: time the legacy solver on the anchored models
b6d9a27fc solver: import once and cache couplings by semantics
b64d6a388 solver: choose a tile per operator class
3f4a7a11c solver: probe resources per variant before placement
127a184d6 solver: answer predecessors from the symbolic relation
2f56d550d solver: build the plan skeleton after residency
b8362dff1 solver: place ready tiles by earliest start
ef560951b frontend: count fp32 rotary weights in model storage units
1d054b6ae solver: compose exact symbolic runtime predecessors
26d5c545c tools: serialize shared variant resource probes
be55ba748 solver: reuse symbolic work across resident levels
ff3f54c23 solver: search operator tiles by coordinate descent
3f87a25a7 experiments: run the solver arms with internal equality gates
ffe67f528 analysis: count translated unsplit reductions per output
d05b74a3f frontend: cache reachability during semantic grouping
25386a586 solver: reuse candidate buffers in the ready frontier
1223018d8 experiments: dump the control EFT queue for comparison
87d3c68ca frontend: preserve exported projection and norm ordering
9fdbeddc2 solver: enumerate proven local fibers without point scans
d562b8a76 solver: reuse verified geometry across resident grids
23c772dc3 experiments: audit solver structure and raw plan statistics
b2616c2b5 solver: memoize exact calculations within skeleton search
2d570a92c analysis: evaluate unique oracle coordinates directly
cdbc019be analysis: reuse proven oracle bounds for a bound theta
4012aa9d4 frontend: preserve dimension roles in static semantic imports
93c3a8bca analysis: evaluate exact oracle expressions without ISL allocations
925990a65 solver: honor executor ordering through symbolic relations
2e68767cf analysis: cache exact predicates for local oracle enumeration
69bb7ba2c experiments: require anchored oracle audits and cache key checks
fecae73f0 solver: reuse symbolic preparation across resident grids
65c5fae46 solver: evaluate coordinate candidates in isolated processes
6304fff5d experiments: record isolated search equivalence and concurrency
10ba48b47 docs: record verified solver structure and pending model gates
4edd41b19 analysis: union symbolic edge relations without repeated parsing
043097b34 solver: retain ready candidate sets across lazy retries
5431e6972 experiments: bound concurrent solves and retain legacy metadata
b6796079c experiments: retain all reference internal equality evidence
e4fd19a97 docs: record reference correctness and solver preparation checks
e60a4fae2 experiments: retain completed anchored control measurements
30422d55d experiments: retain raw control process and probe logs
fc2ed94c5 experiments: rebuild report tables from raw solver evidence
469a4b815 experiments: archive variant resource probe inputs and logs
511cbc6ee analysis: memoize exact relation parsing and composition
eda990df3 docs: enumerate solver deviations and measurement boundaries
876ac0d5b experiments: isolate coupling cache derivation cost
5050c268d docs: record anchored coupling cache timings
ddaa0eb42 experiments: audit the measured winner and semantic edge census
0b7fb1d27 experiments: retain the completed Qwen seq four control
601872c56 docs: track anchored cache evidence and winner auditing
596e21c5f experiments: recover resource failures without masking numeric errors
037dd936d experiments: report interleaving for every worker queue
fda13924b experiments: preserve and remeasure the Qwen seq16 control
30d26ad09 docs: record the recovered Qwen control measurement
3a97410fe experiments: assemble the raw evidence report without hiding gaps
f64edfcac experiments: recheck controls with recorded device contention
7f5b8ca9f experiments: validate higher parallelism for queued solver arms
d51097d28 experiments: retain idle-admitted control remeasurements
e02d71ff1 docs: record concurrency and control measurement conditions
9692f21dd experiments: complete the eight anchored legacy controls
0340740c1 docs: close the anchored baseline measurement item
c2cc952fa experiments: reuse a dedicated lane for queued full searches
bbbbae896 docs: preserve the complete baseline and remaining solver gates
e376b5eda experiments: hand off the queued round nine validation
c5b62e536 experiments: snapshot ongoing solver validation for review
````

## 2. Complete verifier output

Fresh raw-evidence verification exited 1. The verifier reads source, logs and dumps, not this report or report tables. Missing evidence produces FAIL; a pending experiment is not a measured negative result.

````text
C-1 PASS ['lib/Solver/SkeletonSearch.cpp:21:auto options=ClassGranularity(imported,classes,config);auto lifted=imported.lifted;', 'lib/Solver/SkeletonSearch.cpp:50:auto module=importer.InstantiateForGranularity(imported,context,ClassGranularity(imported,classes,config),&cache,nullptr,timing);']
C-2 FAIL completed skeleton cells=0/32; import counts=[]
C-3 PASS no flat adjacency in placement/Oracle source or headers
C-4 PASS ['lib/Solver/SkeletonPlacement.cpp:109:for(int w:candidates) {']
C-5 FAIL ['lib/Solver/SkeletonSearch.cpp:20:std::vector<GemmConfig> const& config,std::size_t changed,analysis::CouplingCache& cache) {', 'lib/Solver/SkeletonSearch.cpp:26:auto edges=cache.Derive(lifted.sem,graph,g,known);', 'lib/Solver/SkeletonSearch.cpp:36:analysis::CouplingCache cache;', 'lib/Solver/SkeletonSearch.cpp:80:OracleCensus Census(analysis::CouplingCache const& cache) {']
C-6 PASS ['lib/Analysis/CouplingCache.cpp:7:std::string SemanticSignature(SemanticOp const& original) {', 'lib/Analysis/CouplingCache.cpp:24:for(auto& read:op.element_reads)tensor(read.tensor);', 'lib/Analysis/CouplingCache.cpp:27:return op.Serialize();', 'lib/Analysis/CouplingCache.cpp:58:Key key{SemanticSignature(*ps)+pr,SemanticSignature(*cs)+cr,k,']
C-7 PASS ['lib/Analysis/SymbolicOracle.cpp:43:auto bijective=isl_map_is_bijective(map),injective=isl_map_is_injective(map),single=isl_map_is_single_valued(map);', 'lib/Analysis/SymbolicOracle.cpp:76:if(isl_map_is_single_valued(d.map)==isl_bool_true) {']
C-8 PASS ['lib/Analysis/SymbolicOracle.cpp:77:d.unique=isl_pw_multi_aff_from_map(isl_map_copy(d.map));', 'lib/Analysis/SymbolicOracle.cpp:80:auto* affine=isl_pw_multi_aff_from_map(parameterized_unique);parameterized_unique=nullptr;']
C-9 PASS ['lib/Analysis/SymbolicOracle.cpp:85:} else if(isl_set_is_box(d.image)==isl_bool_true) {', 'lib/Analysis/SymbolicOracle.cpp:90:d.lower.push_back(isl_set_dim_min(isl_set_copy(d.image),i));', 'lib/Analysis/SymbolicOracle.cpp:91:d.upper.push_back(isl_set_dim_max(isl_set_copy(d.image),i));', 'lib/Analysis/SymbolicOracle.cpp:144:if(d.kind==OracleKind::General && isl_set_is_box(d.theta_image)==isl_bool_true) {', 'lib/Analysis/SymbolicOracle.cpp:146:d.theta_lower.push_back(isl_set_dim_min(isl_set_copy(d.theta_image),i));', 'lib/Analysis/SymbolicOracle.cpp:147:d.theta_upper.push_back(isl_set_dim_max(isl_set_copy(d.theta_image),i));', 'lib/Analysis/SymbolicOracle.cpp:162:for(int i=0;i<d.output;++i){bounds.push_back(isl_set_dim_min(isl_set_copy(d.theta_image),i));bounds.push_back(isl_set_dim_max(isl_set_copy(d.theta_image),i));}', 'lib/Analysis/SymbolicOracle.cpp:237:if(isl_set_is_box(fiber)==isl_bool_true) {', 'lib/Analysis/SymbolicOracle.cpp:240:integer(isl_pw_aff_eval(isl_set_dim_min(isl_set_copy(fiber),i),isl_point_copy(point))),', 'lib/Analysis/SymbolicOracle.cpp:241:integer(isl_pw_aff_eval(isl_set_dim_max(isl_set_copy(fiber),i),isl_point_copy(point))));']
C-10 PASS Oracle has no lexicographic range construction
C-11 PASS ['lib/Solver/SkeletonPlacement.cpp:38:auto key() const{return std::tie(est,negative_rank,stage_order,tile);}', 'lib/Solver/SkeletonPlacement.cpp:74:std::priority_queue<Ready> ready;']
C-12 FAIL cells=0/32 invalid=[]
C-13 FAIL []
C-14 PASS no ISL scheduler calls in skeleton search chain
C-15 FAIL ['include/tilemega/Dialect/CouplingGraph/ExecOps.td:96:def Exec_SkeletonOp : Exec_Op<"skeleton", []> {']
C-16 PASS unchanged control sources=['include/tilemega/Solver/JointPlacement.h', 'lib/Solver/EftPlacement.cpp', 'lib/Solver/ChainPlacement.cpp', 'lib/Solver/BalancedPlacement.cpp', 'lib/Solver/WavefrontPlacement.cpp', 'lib/Solver/PlanMaterialize.cpp']; changed intersection=set()
C-17 FAIL []
C-18 PASS ['lib/Solver/SkeletonSearch.cpp:118:std::vector<SkeletonCandidate> CoordinateDescent(SearchContext& search,int& rounds,std::ostream& out) {', 'lib/Solver/SkeletonSearch.cpp:203:evidence<<std::setprecision(17);result.evaluated=CoordinateDescent(search,result.rounds,evidence);', 'lib/Solver/SkeletonFinalize.cpp:45:{SolverPhase phase(options.common.timing,"simulate");if(!SimulateExecution(input,selected.plan,sim,options.common.placement.hop,&simulated,&error))throw std::runtime_error(error);}']
C-19 PASS command files=900 forbidden=[] nvcc_missing_explicit_zero=[]
C-20 PASS git diff 197cd66da36c4fe8929698135d62dcb032b4e91f --name-only: TaskBody changes=[]
G-1 FAIL all twenty code and dynamic structure checks
G-2 FAIL internal 10/10 and three timed shortlist candidates per skeleton arm; missing/failed=[('llama', 1, 'skeleton-k4', []), ('llama', 1, 'skeleton-k8', []), ('llama', 1, 'skeleton-k16', []), ('llama', 1, 'skeleton-kW', []), ('llama', 4, 'skeleton-k4', []), ('llama', 4, 'skeleton-k8', []), ('llama', 4, 'skeleton-k16', []), ('llama', 4, 'skeleton-kW', []), ('llama', 16, 'skeleton-k4', []), ('llama', 16, 'skeleton-k8', []), ('llama', 16, 'skeleton-k16', []), ('llama', 16, 'skeleton-kW', []), ('llama', 64, 'skeleton-k4', []), ('llama', 64, 'skeleton-k8', []), ('llama', 64, 'skeleton-k16', []), ('llama', 64, 'skeleton-kW', []), ('qwen3', 1, 'skeleton-k4', []), ('qwen3', 1, 'skeleton-k8', []), ('qwen3', 1, 'skeleton-k16', []), ('qwen3', 1, 'skeleton-kW', []), ('qwen3', 4, 'skeleton-k4', []), ('qwen3', 4, 'skeleton-k8', []), ('qwen3', 4, 'skeleton-k16', []), ('qwen3', 4, 'skeleton-kW', []), ('qwen3', 16, 'skeleton-k4', []), ('qwen3', 16, 'skeleton-k8', []), ('qwen3', 16, 'skeleton-k16', []), ('qwen3', 16, 'skeleton-kW', []), ('qwen3', 64, 'skeleton-k4', []), ('qwen3', 64, 'skeleton-k8', []), ('qwen3', 64, 'skeleton-k16', []), ('qwen3', 64, 'skeleton-kW', [])]
G-3 PASS [('gqa2', 4, True, '/root/TileMega/docs/experiments/SOLVER_V2/reference/gqa2_s4'), ('gqa2', 128, True, '/root/TileMega/docs/experiments/SOLVER_V2/reference/gqa2_s128'), ('mha4', 4, True, '/root/TileMega/docs/experiments/SOLVER_V2/reference/mha4_s4'), ('mha4', 128, True, '/root/TileMega/docs/experiments/SOLVER_V2/reference/mha4_s128')]
G-4 PASS /root/TileMega/docs/experiments/SOLVER_V2/cache_test.log error: 'tmcg.coupling' op wait { 999999 } does not match the relation's fiber cardinality [s11] -> { [m, n] -> 32 : m >= 0 and 32m <= -33 + s11 and 0 <= n <= 31; [m, n] -> (s11 - 32 * m) : m >= 0 and -32 + s11 <= 32m < s11 and 0 <= n <= 31 }
error: 'tmcg.coupling' op wait { 999999 } does not match the relation's fiber cardinality [s11] -> { [m, n, j] -> 32 : m >= 0 and 32m <= -33 + s11 and 0 <= n <= 31 and 0 <= j <= 1; [m, n, j] -> (s11 - 32 * m) : m >= 0 and -32 + s11 <= 32m < s11 and 0 <= n <= 31 and 0 <= j <= 1 }
EXACT_MEMO hit=2270 miss=221 mutation_rejected=PASS
CACHE_EQ split=1 PASS
EXACT_MEMO hit=3016 miss=258 mutation_rejected=PASS
CACHE_EQ split=2 PASS
SEMSIG_COLLISIONS pairs=2 distinct_keys=2 PASS
CACHE hit=124 miss=72 PASS
G-5 FAIL unit=/root/TileMega/docs/experiments/SOLVER_V2/oracle_membership_test.log: ORACLE_EXPRESSION signed_floor_domain_holes=45 PASS
ORACLE Rectangular reverse=Unique structure=1:N PASS
ORACLE Unique reverse=Rectangular structure=N:1 PASS
ORACLE Unique reverse=Unique structure=1:1 PASS
ORACLE Unique reverse=Rectangular structure=N:1 PASS
ORACLE Rectangular reverse=Rectangular structure=M:N PASS
ORACLE General reverse=General structure=M:N PASS
ORACLE General reverse=General structure=M:N PASS
ORACLE General reverse=General structure=M:N PASS
ORACLE General reverse=General structure=M:N PASS
ORACLE Unique reverse=Unique structure=1:1 PASS
ORACLE_SET_EQUAL comparisons=384 PASS; anchored exact-set audits=[]; missing/failed=['/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s1/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s1/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s1/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s1/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s4/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s4/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s4/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s4/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s16/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s16/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s16/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s16/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s64/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s64/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s64/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/llama_s64/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s1/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s1/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s1/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s1/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s4/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s4/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s4/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s4/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s16/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s16/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s16/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s16/skeleton-kW/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s64/skeleton-k4/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s64/skeleton-k8/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s64/skeleton-k16/winner_oracle_audit.log', '/root/TileMega/docs/experiments/SOLVER_V2/matrix/qwen3_s64/skeleton-kW/winner_oracle_audit.log']
G-6 PASS build command audit C-19; default header=['include/tilemega/Codegen/tasks/ModelRuntime.h:85:#define TILEMEGA_MIDPOINT_REFINE 0']
G-7 FAIL completed=0/32 raw probe violations=[]
G-8 FAIL wins=0/8; []
G-9 FAIL phase ledgers=8/40; [('llama', 1, 'legacy', {'solver': 'legacy', 'model': 'auto.cu.export', 'seq': '1', 'phase': 'total', 'count': '1', 'total_ms': '4.71891e+06'}), ('llama', 4, 'legacy', {'solver': 'legacy', 'model': 'auto.cu.export', 'seq': '4', 'phase': 'total', 'count': '1', 'total_ms': '5.41758e+06'}), ('llama', 16, 'legacy', {'solver': 'legacy', 'model': 'auto.cu.export', 'seq': '16', 'phase': 'total', 'count': '1', 'total_ms': '6.75097e+06'}), ('llama', 64, 'legacy', {'solver': 'legacy', 'model': 'auto.cu.export', 'seq': '64', 'phase': 'total', 'count': '1', 'total_ms': '1.14952e+07'}), ('qwen3', 1, 'legacy', {'solver': 'legacy', 'model': 'selected.cu.export', 'seq': '1', 'phase': 'total', 'count': '1', 'total_ms': '5.59484e+06'}), ('qwen3', 4, 'legacy', {'solver': 'legacy', 'model': 'auto.cu.export', 'seq': '4', 'phase': 'total', 'count': '1', 'total_ms': '7.02465e+06'}), ('qwen3', 16, 'legacy', {'solver': 'legacy', 'model': 'selected.cu.export', 'seq': '16', 'phase': 'total', 'count': '1', 'total_ms': '8.96673e+06'}), ('qwen3', 64, 'legacy', {'solver': 'legacy', 'model': 'selected.cu.export', 'seq': '64', 'phase': 'total', 'count': '1', 'total_ms': '1.97511e+07'})]
G-10 FAIL narrow/wide ratios=[]
G-11 FAIL selected queue transitions versus control EFT=[]; missing/inconsistent=[('llama', 1, 'skeleton-k4'), ('llama', 1, 'skeleton-k8'), ('llama', 1, 'skeleton-k16'), ('llama', 1, 'skeleton-kW'), ('llama', 4, 'skeleton-k4'), ('llama', 4, 'skeleton-k8'), ('llama', 4, 'skeleton-k16'), ('llama', 4, 'skeleton-kW'), ('llama', 16, 'skeleton-k4'), ('llama', 16, 'skeleton-k8'), ('llama', 16, 'skeleton-k16'), ('llama', 16, 'skeleton-kW'), ('llama', 64, 'skeleton-k4'), ('llama', 64, 'skeleton-k8'), ('llama', 64, 'skeleton-k16'), ('llama', 64, 'skeleton-kW'), ('qwen3', 1, 'skeleton-k4'), ('qwen3', 1, 'skeleton-k8'), ('qwen3', 1, 'skeleton-k16'), ('qwen3', 1, 'skeleton-kW'), ('qwen3', 4, 'skeleton-k4'), ('qwen3', 4, 'skeleton-k8'), ('qwen3', 4, 'skeleton-k16'), ('qwen3', 4, 'skeleton-kW'), ('qwen3', 16, 'skeleton-k4'), ('qwen3', 16, 'skeleton-k8'), ('qwen3', 16, 'skeleton-k16'), ('qwen3', 16, 'skeleton-kW'), ('qwen3', 64, 'skeleton-k4'), ('qwen3', 64, 'skeleton-k8'), ('qwen3', 64, 'skeleton-k16'), ('qwen3', 64, 'skeleton-kW')]
G-12 FAIL simulated dependency CP and raw timing ratios=[]; missing/inconsistent=[('llama', 16, 'skeleton-k4'), ('llama', 16, 'skeleton-k8'), ('llama', 16, 'skeleton-k16'), ('llama', 16, 'skeleton-kW'), ('llama', 64, 'skeleton-k4'), ('llama', 64, 'skeleton-k8'), ('llama', 64, 'skeleton-k16'), ('llama', 64, 'skeleton-kW'), ('qwen3', 16, 'skeleton-k4'), ('qwen3', 16, 'skeleton-k8'), ('qwen3', 16, 'skeleton-k16'), ('qwen3', 16, 'skeleton-kW'), ('qwen3', 64, 'skeleton-k4'), ('qwen3', 64, 'skeleton-k8'), ('qwen3', 64, 'skeleton-k16'), ('qwen3', 64, 'skeleton-kW')]
G-13 FAIL measured winner classes and codegen mapping=[]; violations=[]
VERIFY hard_failed=G-1,G-2,G-5,G-7 research_pass=0
````

## 3. Gates

| gate | type | result | evidence |
| --- | --- | --- | --- |
| G-1 | hard | FAIL | verify_report.log: G-1 |
| G-2 | hard | FAIL | verify_report.log: G-2 |
| G-3 | hard | PASS | verify_report.log: G-3 |
| G-4 | hard | PASS | verify_report.log: G-4 |
| G-5 | hard | FAIL | verify_report.log: G-5 |
| G-6 | hard | PASS | verify_report.log: G-6 |
| G-7 | hard | FAIL | verify_report.log: G-7 |
| G-8 | research | FAIL | verify_report.log: G-8 |
| G-9 | report | FAIL | verify_report.log: G-9 |
| G-10 | report | FAIL | verify_report.log: G-10 |
| G-11 | report | FAIL | verify_report.log: G-11 |
| G-12 | report | FAIL | verify_report.log: G-12 |
| G-13 | report | FAIL | verify_report.log: G-13 |


## 4. Stopped items and degraded forms

No active control downstream-block marker at render time.

Searches still running or awaiting admission are unfinished work, not stopped items. The Qwen seq16 pre-output OOM attempt is archived separately; recovery reran all ten processes.

| model | seq | arm | state | evidence |
| --- | --- | --- | --- | --- |
| llama | 1 | skeleton-k4 | pending_or_failed | matrix/llama_s1/skeleton-k4 |
| llama | 1 | skeleton-k8 | pending_or_failed | matrix/llama_s1/skeleton-k8 |
| llama | 1 | skeleton-k16 | pending_or_failed | matrix/llama_s1/skeleton-k16 |
| llama | 1 | skeleton-kW | pending_or_failed | matrix/llama_s1/skeleton-kW |
| llama | 4 | skeleton-k4 | pending_or_failed | matrix/llama_s4/skeleton-k4 |
| llama | 4 | skeleton-k8 | pending_or_failed | matrix/llama_s4/skeleton-k8 |
| llama | 4 | skeleton-k16 | pending_or_failed | matrix/llama_s4/skeleton-k16 |
| llama | 4 | skeleton-kW | pending_or_failed | matrix/llama_s4/skeleton-kW |
| llama | 16 | skeleton-k4 | pending_or_failed | matrix/llama_s16/skeleton-k4 |
| llama | 16 | skeleton-k8 | pending_or_failed | matrix/llama_s16/skeleton-k8 |
| llama | 16 | skeleton-k16 | pending_or_failed | matrix/llama_s16/skeleton-k16 |
| llama | 16 | skeleton-kW | pending_or_failed | matrix/llama_s16/skeleton-kW |
| llama | 64 | skeleton-k4 | pending_or_failed | matrix/llama_s64/skeleton-k4 |
| llama | 64 | skeleton-k8 | pending_or_failed | matrix/llama_s64/skeleton-k8 |
| llama | 64 | skeleton-k16 | pending_or_failed | matrix/llama_s64/skeleton-k16 |
| llama | 64 | skeleton-kW | pending_or_failed | matrix/llama_s64/skeleton-kW |
| qwen3 | 1 | skeleton-k4 | pending_or_failed | matrix/qwen3_s1/skeleton-k4 |
| qwen3 | 1 | skeleton-k8 | pending_or_failed | matrix/qwen3_s1/skeleton-k8 |
| qwen3 | 1 | skeleton-k16 | pending_or_failed | matrix/qwen3_s1/skeleton-k16 |
| qwen3 | 1 | skeleton-kW | pending_or_failed | matrix/qwen3_s1/skeleton-kW |
| qwen3 | 4 | skeleton-k4 | pending_or_failed | matrix/qwen3_s4/skeleton-k4 |
| qwen3 | 4 | skeleton-k8 | pending_or_failed | matrix/qwen3_s4/skeleton-k8 |
| qwen3 | 4 | skeleton-k16 | pending_or_failed | matrix/qwen3_s4/skeleton-k16 |
| qwen3 | 4 | skeleton-kW | pending_or_failed | matrix/qwen3_s4/skeleton-kW |
| qwen3 | 16 | skeleton-k4 | pending_or_failed | matrix/qwen3_s16/skeleton-k4 |
| qwen3 | 16 | skeleton-k8 | pending_or_failed | matrix/qwen3_s16/skeleton-k8 |
| qwen3 | 16 | skeleton-k16 | pending_or_failed | matrix/qwen3_s16/skeleton-k16 |
| qwen3 | 16 | skeleton-kW | pending_or_failed | matrix/qwen3_s16/skeleton-kW |
| qwen3 | 64 | skeleton-k4 | pending_or_failed | matrix/qwen3_s64/skeleton-k4 |
| qwen3 | 64 | skeleton-k8 | pending_or_failed | matrix/qwen3_s64/skeleton-k8 |
| qwen3 | 64 | skeleton-k16 | pending_or_failed | matrix/qwen3_s64/skeleton-k16 |
| qwen3 | 64 | skeleton-kW | pending_or_failed | matrix/qwen3_s64/skeleton-kW |

Declared forms: GEMM-only coordinate search; diagnostic reference-domain regression; legacy R8 five-shape control; parallel child evaluation; warm physical resource cache. None substitutes for anchored coverage. Full details follow.

## 5. Deviation declarations

# R9 specification differences and implementation choices

This ledger is provisional while the anchored matrix runs. It does not waive
missing measurements or turn an incomplete gate into PASS. The final report
must incorporate this ledger and any additional differences found by verification.

## Declared specification extension: executor ordering

- **Specification (§4.4/§4.6):** predecessor/successor queries come from the
  exact producer-to-consumer CG coupling; `ready(t,w)` considers those producers.
- **Actual:** `PlanSkeleton.cpp::ExecutionOrdering` additionally composes the
  unchanged executor's `requested_events` with event-group membership. The
  search Oracle uses the union of exact CG dependencies and these executor
  ordering prerequisites, with both relations stored separately.
- **Reason:** the existing wait-window projection can require more ordering
  than exact CG data dependence. The first gqa2 seq=128 top-K materialization
  failed L-c because such a producer appeared later in the consumer's queue.
  Altering the executor, wait semantics, or legality checks is prohibited by R9.
- **Evidence:** `native_union_test.log` independently expands requested events
  at κ=1/2/4 (13,760/13,952/14,336 pairs) and checks set equality and order.
  `reference/` contains the subsequent 120/120 internally equal fresh runs.
  This extension adds ordering; it does not claim those pairs are data edges.

## Disclosed search and measurement forms

| Specification | Actual form | Reason and effect |
|---|---|---|
| §4.7 searches operator classes; §7.3 explicitly permits GEMM-only search | Discrete coordinate descent searches GEMM SemSig classes. Scalar ownership remains the current TaskBody ABI. | No TaskBody changes are allowed. This is the permitted GEMM-only form, not a claim of searching all scalar tile shapes. |
| §4.8 reference-model regression | Reference G-3 uses one disclosed tile shape, all split values, and P=1. Each of three shortlisted candidates has ten fresh internally equal runs in all four cells. | Establishes the reference correctness gate; does not establish full-domain search quality. Real-model arms retain the full CandidateGenerator domain and P=3. |
| §4.0 legacy control is the current solver | Legacy measurements use the deployed R8 five-shape search domain, capacity 12, and the unchanged six heuristics. | The uncontrolled unrestricted audit did not finish and is excluded. This domain must be stated when comparing solve latency; skeleton's larger domain is not a matched search-work comparison. |
| §4.3 each semantic variant has cached resources | Logical cache keys include SemSig/tile/stages/split. The physical compiler cache shares identical template specializations across classes and splits. | Split-K is a runtime field. Recompiling identical source would not add resource information; actual compile counts, command logs, and exact source are preserved. |

## Details not fixed by the specification

- Optional `--search-jobs` evaluates a fixed coordinate's candidates in isolated
  processes. Defaults to 1; current primary runs use 3. Acceptance retains the
  original candidate order and ties. The import occurs once in the parent;
  children inherit immutable semantic state and isolated copies of caches.
  `search_native_cache_test.log` and `search_relation_memo_test.log` compare
  all 26 candidate records and five final schedules with serial evaluation.
- Four whole-solve admission slots bound CPU contention. Waiting before
  compiler launch is recorded separately in `solver_admission.json`; it is not
  solver execution time. Phase durations sum child work, so their sum can
  exceed wall-clock total. The primary physical variant cache is warm across
  diagnostics and arms, and compile counts report actual new compilations.
- Exact expression and relation memoization stores complete input keys and
  immutable values within a bounded search scope. It does not cache successful
  verification or skip L-a/L-b/L-c/L-e checks. The short relation-memo profile
  is diagnostic only; it is not an anchored performance observation.
- General Oracle queries may locally enumerate bounded candidate coordinates
  only after exact membership testing of every emitted point. A bounding box
  is never substituted for the dependency set. Large sparse fibers retain ISL
  enumeration. This is the exact local enumeration permitted by §4.4, not a
  claim that a General edge has been proven Rectangular.
- Three pre-existing frontend defects were repaired: F32 RoPE constant storage,
  translated head-local reductions, and alternative valid Qwen FX emission
  order. Static-export default dimension-role symbols were also preserved in
  the split importer. No TaskBody, event, barrier, old heuristic, simulator
  semantics, or CUTLASS implementation was changed.

## Excluded attempts

`before_execution_order/`, `before_isolation/`, `before_bulk_union/`, and the
named `*_profile/` directories are diagnostic or interrupted attempts. They
must not supply completed real-model latency or performance claims. Admissions
that failed before code generation remain as debugging evidence. Invalid new
shortlist candidates remain recorded and cannot become an arm's selected
result; an entirely invalid shortlist fails that arm. A regression in an
existing reference configuration remains the prompt's global-stop condition.


## GPU resource admission and a rejected control attempt

Verified: the first Qwen seq16 control attempt had four pre-output CUDA
allocation failures at `ModelHarness.cuh:2585`, followed by six internally
bit-exact runs. The ten-process gate failed; its six timings are not a control
result. The entire attempt is retained under that cell's `failed_attempts/`.
Recovery repeats all ten fresh processes, not just the failed four.

The measurement runner now waits for three consecutive device observations
with utilization at most 5% and available memory at least fixture bytes plus
2048 MiB. This headroom is an admission estimate, not a measured peak-memory
claim or a changed acceptance gate. Observations are saved in
`gpu_admission.jsonl`. External GPU users can still race admission; any new
failure remains in the raw logs. Numeric/hash failures and unexplained exits
are not automatically retried. `gpu_admission_test.log` exercises these guards.


## Additional admission lane after concurrency equivalence

Verified CPU fixture: serial and `--search-jobs=36` evaluation produced identical
116 candidate records, top-five keys, and final worker/slot/start/end tables;
each imported once (`search_isolation_36_test.log`). On 2026-09-24 the still
queued Llama seq1/k8 arm was moved to a fifth admission slot with 36 workers.
The four admitted seq4 arms were not interrupted and retain three workers.
The compiler snapshot, full 1218-candidate class domains, P=3, residency
sweeps and top-K procedure are unchanged. Original queued launch metadata is
retained in that arm's `queued_launch_history/`. Compare wall times with the
recorded per-arm concurrency and warm-cache state, not as equal-CPU budgets.
No anchored speedup is asserted from the CPU fixture.

The two control cells Llama seq64 and Qwen seq4 had busy-GPU pre-measurement
snapshots (93%/100% utilization, 5157/43962 MiB used). These snapshots alone
do not prove interference throughout the full sample. Their original valid
internal-equality samples are retained under `excluded_attempts/`; complete
fresh ten-process timings are remeasured with resource/idle admission. This
is not selective removal of slow rounds, and numerical failures cannot be
retried by this route.


The fifth slot is now a dedicated continuation lane: after one promoted arm
releases it, `promote_queued_arm.py --drain-queue` can promote another still
unadmitted arm with the same 36-worker setting. A singleton supervisor lock
and an explicit `--solver-slot=4` keep at most one such high-parallelism search
admitted. Active searches are checked while their queue parent is frozen and
are resumed rather than cancelled if admission raced the check. Original
queued launch metadata is archived; search-domain and compiler bytes are
unchanged. `dedicated_slot_test.log` checks exclusivity, release and range
validation. This is CPU admission policy, not a new placement/search policy.


## 6. Solver latency and cache accounting

The full phase decomposition is [phases.tsv](report_tables/phases.tsv). Parallel phase sums can exceed wall-clock total; admission queue time is separate. Legacy and skeleton search domains differ as declared above.

| model | seq | arm | phase | count | total_ms |
| --- | --- | --- | --- | --- | --- |
| llama | 1 | legacy | import | 43 | 192176 |
| llama | 1 | legacy | total | 1 | 4.71891e+06 |
| llama | 4 | legacy | import | 43 | 198346 |
| llama | 4 | legacy | total | 1 | 5.41758e+06 |
| llama | 16 | legacy | import | 43 | 192406 |
| llama | 16 | legacy | total | 1 | 6.75097e+06 |
| llama | 64 | legacy | import | 43 | 187576 |
| llama | 64 | legacy | total | 1 | 1.14952e+07 |
| qwen3 | 1 | legacy | import | 43 | 368530 |
| qwen3 | 1 | legacy | total | 1 | 5.59484e+06 |
| qwen3 | 4 | legacy | import | 43 | 377922 |
| qwen3 | 4 | legacy | total | 1 | 7.02465e+06 |
| qwen3 | 16 | legacy | import | 43 | 314577 |
| qwen3 | 16 | legacy | total | 1 | 8.96673e+06 |
| qwen3 | 64 | legacy | import | 43 | 302462 |
| qwen3 | 64 | legacy | total | 1 | 1.97511e+07 |

| model | split | mode | count | total_ms | cache_hit | cache_miss | bytes_equal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 1 | uncached | 1 | 137.831 | 0 | 0 | 1 |
| reference | 1 | cold | 1 | 194.285 | 6 | 36 | 1 |
| reference | 1 | warm | 1 | 0.517806 | 42 | 0 | 1 |
| reference | 2 | uncached | 1 | 136.316 | 0 | 0 | 1 |
| reference | 2 | cold | 1 | 262.411 | 10 | 46 | 1 |
| reference | 2 | warm | 1 | 0.714498 | 56 | 0 | 1 |
| llama | 1 | uncached | 1 | 1245.38 | 0 | 0 | 1 |
| llama | 1 | cold | 1 | 233.508 | 309 | 45 | 1 |
| llama | 1 | warm | 1 | 5.77378 | 354 | 0 | 1 |
| llama | 2 | uncached | 1 | 1148.18 | 0 | 0 | 1 |
| llama | 2 | cold | 1 | 219.15 | 411 | 56 | 1 |
| llama | 2 | warm | 1 | 5.28773 | 467 | 0 | 1 |
| qwen3 | 1 | uncached | 1 | 2058.12 | 0 | 0 | 1 |
| qwen3 | 1 | cold | 1 | 241.344 | 627 | 47 | 1 |
| qwen3 | 1 | warm | 1 | 11.4686 | 674 | 0 | 1 |
| qwen3 | 2 | uncached | 1 | 2009.18 | 0 | 0 | 1 |
| qwen3 | 2 | cold | 1 | 218.52 | 813 | 58 | 1 |
| qwen3 | 2 | warm | 1 | 10.5149 | 871 | 0 | 1 |

Full-search cache statistics:

**Pending: no complete evidence rows.**


## 7. Eight cells by five arms

| model | seq | arm | state | l05_ms | l1_ms | l2_ms | l2_l1 | l2_l05 | l2_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 1 | legacy | measured | 5.627304 | 5.706632000000001 | 5.422352 | 0.9501842768203731 | 0.9635790069276514 | 1 |
| llama | 1 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 1 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 1 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 1 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 4 | legacy | measured | 6.3168 | 6.460928 | 6.451456 | 0.9985339567319123 | 1.0213171225937183 | 1 |
| llama | 4 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 4 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 4 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 4 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 16 | legacy | measured | 6.57912 | 6.6780159999999995 | 7.122888 | 1.0666173905543204 | 1.0826505672491153 | 1 |
| llama | 16 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 16 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 16 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 16 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 64 | legacy | measured | 11.785504 | 11.6687035 | 12.5220235 | 1.0731289470162646 | 1.062493678675091 | 1 |
| llama | 64 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 64 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 64 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| llama | 64 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 1 | legacy | measured | 8.801936 | 8.956728 | 8.266511999999999 | 0.922938823195256 | 0.9391697462921793 | 1 |
| qwen3 | 1 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 1 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 1 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 1 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 4 | legacy | measured | 9.111864 | 9.5751525 | 8.949024 | 0.9346090310310985 | 0.9821287938450354 | 1 |
| qwen3 | 4 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 4 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 4 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 4 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 16 | legacy | measured | 8.8882315 | 9.112672 | 9.394016 | 1.0308739302808223 | 1.0569049647277977 | 1 |
| qwen3 | 16 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 16 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 16 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 16 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 64 | legacy | measured | 15.561152 | 15.725656 | 16.576512 | 1.0541062325158328 | 1.065249667890912 | 1 |
| qwen3 | 64 | skeleton-k4 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 64 | skeleton-k8 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 64 | skeleton-k16 | pending_or_failed | pending | pending | pending | pending | pending | pending |
| qwen3 | 64 | skeleton-kW | pending_or_failed | pending | pending | pending | pending | pending | pending |


## 8. Per-class geometry, uniform seed and residency

**Pending: no complete evidence rows.**

All GEMM-to-class mappings: [classes.tsv](report_tables/classes.tsv).

**Pending: no complete evidence rows.**


## 9. Top-five resource estimates and driver results

**Pending: no complete evidence rows.**

Candidate keys and raw evidence paths: [resources.tsv](report_tables/resources.tsv).

## 10. Symbolic edge census and exact Oracle validation

Semantic CG edges and physical executor-order edges are separate populations. G-5 audits the GPU-measured winner, including CG hash and actual grid/residency/κ.

**Pending: no complete evidence rows.**

SemSig collision and byte-equality evidence:

````text
error: 'tmcg.coupling' op wait { 999999 } does not match the relation's fiber cardinality [s11] -> { [m, n] -> 32 : m >= 0 and 32m <= -33 + s11 and 0 <= n <= 31; [m, n] -> (s11 - 32 * m) : m >= 0 and -32 + s11 <= 32m < s11 and 0 <= n <= 31 }
error: 'tmcg.coupling' op wait { 999999 } does not match the relation's fiber cardinality [s11] -> { [m, n, j] -> 32 : m >= 0 and 32m <= -33 + s11 and 0 <= n <= 31 and 0 <= j <= 1; [m, n, j] -> (s11 - 32 * m) : m >= 0 and -32 + s11 <= 32m < s11 and 0 <= n <= 31 and 0 <= j <= 1 }
EXACT_MEMO hit=2270 miss=221 mutation_rejected=PASS
CACHE_EQ split=1 PASS
EXACT_MEMO hit=3016 miss=258 mutation_rejected=PASS
CACHE_EQ split=2 PASS
SEMSIG_COLLISIONS pairs=2 distinct_keys=2 PASS
CACHE hit=124 miss=72 PASS
````

**Pending: no complete evidence rows.**


## 11. Placement destinations and interleaving

**Pending: no complete evidence rows.**

Every worker, including idle workers where grid is known: [worker_queues.tsv](report_tables/worker_queues.tsv). Aggregate ratios use total transitions / total adjacent slots, not the mean of worker ratios.

## 12. Loss from narrowing

**Pending: no complete evidence rows.**


## 13. Level-2 versus simulator ranking

**Pending: no complete evidence rows.**

These scores use each top-five candidate after real occupancy re-solving when needed. The outer-loop pre-recheck score is retained in each arm’s raw search.tsv.

## 14. Attention attribution at seq16/64

The simulator field reconstructed here is dependency-only node-duration CP. Its attention fraction is not a measured arithmetic-utilization fraction.

**Pending: no complete evidence rows.**

Corresponding raw timing ratios:

| model | seq | arm | l2_l1 | l2_l05 | l2_legacy |
| --- | --- | --- | --- | --- | --- |
| llama | 16 | legacy | 1.0666173905543204 | 1.0826505672491153 | 1 |
| llama | 16 | skeleton-k4 | pending | pending | pending |
| llama | 16 | skeleton-k8 | pending | pending | pending |
| llama | 16 | skeleton-k16 | pending | pending | pending |
| llama | 16 | skeleton-kW | pending | pending | pending |
| llama | 64 | legacy | 1.0731289470162646 | 1.062493678675091 | 1 |
| llama | 64 | skeleton-k4 | pending | pending | pending |
| llama | 64 | skeleton-k8 | pending | pending | pending |
| llama | 64 | skeleton-k16 | pending | pending | pending |
| llama | 64 | skeleton-kW | pending | pending | pending |
| qwen3 | 16 | legacy | 1.0308739302808223 | 1.0569049647277977 | 1 |
| qwen3 | 16 | skeleton-k4 | pending | pending | pending |
| qwen3 | 16 | skeleton-k8 | pending | pending | pending |
| qwen3 | 16 | skeleton-k16 | pending | pending | pending |
| qwen3 | 16 | skeleton-kW | pending | pending | pending |
| qwen3 | 64 | legacy | 1.0541062325158328 | 1.065249667890912 | 1 |
| qwen3 | 64 | skeleton-k4 | pending | pending | pending |
| qwen3 | 64 | skeleton-k8 | pending | pending | pending |
| qwen3 | 64 | skeleton-k16 | pending | pending | pending |
| qwen3 | 64 | skeleton-kW | pending | pending | pending |


## 15. Unmet gates and concrete next work

Anchored searches/measurements remain unfinished; G-8 has no complete eight-cell verdict.

The current long-running CPU path consists of `SearchContext::Evaluate` residency sweeps, `PrepareSymbolicProblem` task pricing and `ScheduleBySkeleton` lazy EST requeues. Candidate `.outer/*.result` files retain those phase costs. Further speed changes must preserve exact dependency sets, worker/slot/times, candidate coverage and acceptance order; the primary runs have not been restarted for unproven memoization speedups.

Remaining acceptance work: finish all 32 anchored skeleton arms; driver-check each top-five; measure all shortlisted triples; audit each measured winner; recompute all gates; record measured failures with specific causes; finalize FINDINGS/TODO/STATUS and push.

## 16. Excluded scope

| item | changed |
| --- | --- |
| TaskBody changes | none |
| CUTLASS submodule | none |
| Old six placement heuristics | none |
| Simulator implementation | none |

No attention TaskBody rewrite, sync/barrier/event redesign, paging, static batch, tile-transfer fusion or correctness-criterion redesign is part of R9. `MIDPOINT_REFINE=0` is enforced by the raw build-command audit C-19/G-6. Compiler/front-end repairs and executor-order constraints are explicitly declared in section 5.
