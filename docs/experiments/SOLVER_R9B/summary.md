# TileMega R9b：物理模型、评估与放置

状态：**进行中；不是收尾报告**。生成时间：2026-09-25T11:53:31.197229+00:00。

仍缺以下证据；不以部分矩阵判定研究门：

- matrix/llama_s16/selected.cu.top1.cu.measurement: 6/10 processes
- [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv'
- matrix/qwen3_s4/selected.cu.top1.cu.measurement: 0/10 processes
- [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv'
- [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv'
- llama_s64: missing top-M materializations
- qwen3_s16: missing top-M materializations
- qwen3_s64: missing top-M materializations
- llama_s16: shortlist prediction comparison incomplete: matrix/llama_s16/selected.cu.top1.cu.measurement: 6/10 processes
- llama_s64: shortlist prediction comparison incomplete: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv'
- qwen3_s4: shortlist prediction comparison incomplete: matrix/qwen3_s4/selected.cu.top1.cu.measurement: 0/10 processes
- qwen3_s16: shortlist prediction comparison incomplete: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv'
- qwen3_s64: shortlist prediction comparison incomplete: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv'
- theta/llama_s16: incomplete
- theta/llama_s32: incomplete
- theta/llama_s64: incomplete
- theta/qwen3_s4: incomplete
- theta/qwen3_s8: incomplete
- theta/qwen3_s16: incomplete
- theta/qwen3_s32: incomplete
- theta/qwen3_s64: incomplete
- ablations/llama_s4/template: 2/10 processes
- ablations/llama_s4/eft: 0/10 processes
- ablations/llama_s4/wide: 0/10 processes
- ablations/qwen3_s4/template: 0/10 processes
- ablations/qwen3_s4/eft: 0/10 processes
- ablations/qwen3_s4/wide: 0/10 processes
- stages/llama_s4_S2: 0/10 processes
- stages/llama_s4_S3: 0/10 processes
- stages/llama_s4_S4: 0/10 processes
- traffic/skeleton/llama_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/llama_s16/exit.json'
- traffic/skeleton/llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/llama_s64/exit.json'
- traffic/skeleton/qwen3_s4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s4/exit.json'
- traffic/skeleton/qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s16/exit.json'
- traffic/skeleton/qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s64/exit.json'
- colocation/llama: 0/50
- validation_colocated/qwen3 incomplete
- colocation/qwen3: 0/50
- reference/mha4_s4 incomplete
- reference/mha4_s128 incomplete

发布说明：为满足 GitHub 100 MiB 限制，一份原始 trace 无损压缩；未发布提交的原哈希到发布哈希见 [publication_commit_map.tsv](publication_commit_map.tsv)。源码内容与提交信息保持不变，运行中测试保留原始文件。以下为报告生成时的提交身份，基线不变。

## 1. 基线、规格与提交

基线 `ce6296f0d6e7fa14d7d0dc3800caed184adf9c72`；报告所见 HEAD `333723932ccc4311b6b4e152fd5bb81c98a20f9a`。规格 `/root/Prompt/TileMega_R9b_prompt.md`，SHA256 `fec7aae202feb7864a9011aa70edc8091988d9259a15c5e578834cf905d4a616`。TODO 更新与冻结来自 `ce6296f0d`、`ed12491fe`。

```text
eb40eec15 docs: supersede the round nine outer search
250f9b0c1 analysis: derive the dram floor from access relations
fc0ca2610 experiments: audit the access-derived floor on eight cells
e18580d81 solver: derive physical traffic by operand provenance
5d94d95d0 solver: calibrate stage latency and physical fixed work
4efed9c31 solver: split task prices into fixed compute and dram parts
52625afb3 solver: price task spaces per boundary piece
b995f0f3e solver: evaluate configurations with a task space flow model
9289c537a solver: add a dram fluid mode to the simulator
6bdac05c4 solver: derive colocated homes from unique symbolic maps
fb7ca9007 solver: preserve physical scalar and wave price classes
e827cd628 solver: connect symbolic flow inputs to template simulation
a3a64f912 solver: retain the validated fixed term and reuse equal prices
aace1bf5d solver: search configurations on the flow model
15bad9f73 solver: batch causal prices and share fluid cap allocation
df3e8d4e4 experiments: replay the oracle after the cost fixes
1262afcd5 solver: advance dram service by cap class
e7fc27e83 analysis: cache scalar release maxima for general fibers
3c4e80734 solver: reuse cohort task storage across flow evaluations
5c7983129 solver: reuse closed task counts for unchanged theta bindings
8a9f87449 solver: reuse imported semantics across theta searches
6001736e2 experiments: expose flow validation and fixed point ablations
16859c260 experiments: reconstruct large traces without dense dependencies
a3198ef55 experiments: compare unscaled binary64 replay predictions
0bfa0d4e0 experiments: queue the complete flow solver validation matrix
cc6f2d3ea experiments: record fresh controls and decode chain traces
b4b285d4e solver: include kernel union resources in the price cache key
5e4f18e37 experiments: verify price isolation and complete boundary sums
6fc316b3d experiments: retain the superseded exhaustive audit prefix
88a51bba3 experiments: audit runtime horizons and serial search contracts
1a9f494c2 analysis: bound releases from the exact local oracle fiber
f82857765 experiments: verify r9b gates from complete raw evidence
5f39f8307 experiments: prove the legacy simulator remains bit identical
b8c8dfc53 solver: count template displacement independently of affinity
8987672b2 analysis: hoist invariant extents out of window fitting
5f503b53f experiments: record the complete llama seq1 shortlist measurements
0639e4268 experiments: assemble the r9b report without hiding missing evidence
101241bc9 docs: track r9b results and open acceptance gates
89f7330ec experiments: finish the replay and trace correctness checks
2e18080cd experiments: sum complete model traffic from boundary pieces
3b1360ea2 experiments: retain the complete llama flow concordance sample
ebe3027a2 experiments: record both seq1 shortlists and their absolute gaps
68f621d43 experiments: prepare queued cells without duplicating searches
8d213f5ad experiments: complete the llama colocation concordance audit
360a2b10d experiments: compare trace gaps and batch traffic diagnostics
061859ec2 experiments: pass the complete flow ordering comparison
c506c7a2c experiments: check random coverage and preserved resume prefixes
fff18c810 analysis: optimize scalar releases on exact local fibers
165d51eab experiments: retain full graph nominal and physical byte totals
38c85e73a solver: report actual displacement in materialization tables
333723932 experiments: audit every template and refinement pair
```

## 2. verify.py 完整输出

命令：`python3 docs/experiments/SOLVER_R9B/verify.py`；退出码 `1`。每个门均执行，缺证据也输出 FAIL。

```text
K-1 PASS lib/Solver/SkeletonSearch.cpp:57: EvaluateFlow in Evaluate; CoordinateDescent checked; EvaluateFlow=1, forbidden=False
K-2 PASS lib/Solver/SkeletonSearch.cpp: prohibited tokens absent
K-3 PASS lib/Solver/PiecePricing.cpp:13: std::ostringstream key;key<<analysis::SemanticSignature(semantic.op)<<std::hexfloat;; lib/Solver/PiecePricing.cpp:10: ModelDescription const& model,int chunks,PiecePriceCache* cache,int kernel_shared_bytes) {; lib/Solver/FlowPreparation.cpp:107: std::vector<std::string> signatures,geometry_keys;
K-4 PASS lib/Solver/PlanSkeleton.cpp:230: space.width=all_workers ? grid:std::min(grid,k_base);
K-5 PASS lib/Solver/PlanSkeleton.cpp:201: auto image=s.colocation->reverse.Query({tile},theta);; lib/Solver/PlanSkeleton.cpp:262: if(colocation->structure==analysis::EdgeStructure::OneToOne && colocation->forward.kind()==analysis::OracleKind::Unique && colocation->reverse.kind()==analysis::OracleKind::Unique && (previous<0 || result.spaces[p].order>result.spaces[previous].order)) {; lib/Solver/PlanSkeleton.cpp:290: b.getNamedAttr("unique_home_map",b.getStringAttr(t.colocation?t.colocation->reverse.UniqueMapText():""))}));}
K-6 PASS lib/Analysis/DramFloor.cpp:17: isl_util::Set set(isl_map_range(map.release()));; lib/Analysis/DramFloor.cpp:18: isl_util::PwQPolynomial count(isl_set_card(set.release()));; lib/Analysis/DramFloor.cpp:87: result.compute_ns=result.matmul_flops.ScaleRational(Reciprocal(options.tc_gflops));; lib/Analysis/DramFloor.cpp:78: t.no_producer=t.reads.Subtract(t.writes);
K-7 PASS lib/Solver/TaskPriceParts.cpp:14: if(!options_.regime_a || dtype_!=ScalarType::kBF16); lib/Solver/TaskPriceParts.cpp:25: df_np=1-curve.HitFraction(stream,calib_->l2_gbps,calib_->dram_gbps);; lib/Solver/TaskPriceParts.cpp:20: double stream=input.no_producer_read_bytes?input.stream_bytes:model.LiveFootprintBytes();; lib/Solver/TaskPriceParts.cpp:32: result.dram_bytes=result.no_producer_dram_bytes+traffic.produced_read_bytes*df_p+traffic.external_write_bytes;; lib/Solver/TaskPriceParts.cpp:18: auto domain=traits.stages<=0 || options_.physical_traffic?analysis::AccessDomain::kPhysicalTensor:analysis::AccessDomain::kNominalTile;; include/tilemega/Solver/CostModel.h:62: double no_producer_read_bytes=0, produced_read_bytes=0;; include/tilemega/Solver/CostModel.h:62: double no_producer_read_bytes=0, produced_read_bytes=0;
K-8 PASS lib/Solver/TaskPriceParts.cpp:11: TaskPriceParts CostModel::PriceParts(DerivedTaskInput const& input,BackendTraits const& traits,; lib/Solver/TaskPriceParts.cpp:78: iteration=std::max({u.Bottleneck(),body,(fit.latency_scale*latency+o*bytes/fit.stage_rate_bytes_per_ns)/(traits.stages-1)});; lib/Solver/TaskPriceParts.cpp:75: if(!(fit.stage_rate_bytes_per_ns>0))throw std::invalid_argument("regime-A stages require a calibrated transfer rate");; lib/Solver/TaskPriceParts.cpp:78: iteration=std::max({u.Bottleneck(),body,(fit.latency_scale*latency+o*bytes/fit.stage_rate_bytes_per_ns)/(traits.stages-1)});; lib/Solver/CostModel.cpp:530: if(coordinates)return IsolatedNs(PriceParts(input,traits,residency,model,chunks,; docs/experiments/SOLVER_R9B/unit/prices.log:7: REGIME_A_PRICE bit_exact=18 stage_monotonic=PASS external_df_2GiB=1
K-9 PASS lib/Solver/StageFlowModel.cpp:11: int CoarsenRelease(int maximum,int n,int kappa){if(maximum<0 || maximum>=n || kappa<1)throw std::invalid_argument("invalid release domain");return std::min(n-1,kappa*(maximum/kappa)+kappa-1);}; lib/Solver/StageFlowModel.cpp:13: auto name=dtype==ScalarType::kBF16?"bf16":"f32";auto const& event=target.EventCalibrationFor(name);; lib/Solver/StageFlowModel.cpp:12: void SetFlowCalibration(FlowProblem& p,TargetSpec const& target,ScalarType dtype,HopCurve const& hop) {; lib/Solver/StageFlowModel.cpp:52: std::vector<int> fluid_owner;DramFluidServer fluid(p.dram_gbps);double now=0;; lib/Solver/DramFluid.cpp:9: int DramFluidServer::Add(double bytes,double cap,int count) {; lib/Solver/DramFluid.cpp:8: DramFluidServer::DramFluidServer(double rate):bandwidth_(rate){if(!(rate>0))throw std::invalid_argument("invalid DRAM bandwidth");}
K-10 PASS lib/Solver/SkeletonSearch.cpp: prohibited tokens absent; lib/Solver/FlowPreparation.cpp: prohibited tokens absent; lib/Solver/StageFlowModel.cpp: prohibited tokens absent
K-11 PASS 39 changed implementation files; 
K-12 FAIL 775 raw command files (including occupancy build_command.txt); refine1=[]; nvcc_without_explicit_zero=[]; examples=['docs/experiments/SOLVER_R9B/ablations/llama_s4/template/build.command.json', 'docs/experiments/SOLVER_R9B/ablations/llama_s4/template/materialized/command.json', 'docs/experiments/SOLVER_R9B/ablations/llama_s4/template/occupancy/build.command.json']; measured_binaries=36 missing_sass=[] kernel_functions=76 FP64=1368 first=[('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*1380*/                   I2F.F64.S64 R10, R18 ;                                          /* 0x00000012000a7312 */'), ('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*13b0*/                   DMUL R10, R10, c[0x2][0x70] ;                                   /* 0x00801c000a0a7a28 */')]
K-13 PASS git diff ce6296f0d6e7fa14d7d0dc3800caed184adf9c72: protected=none; simulator guarded additions=True; CostModel removing BF16 regime_a branch equals baseline=True; PriceTaskInstances with regime_a=false equals baseline body=True; new traffic metadata audited by G-5/G-6
K-14 PASS measured sources=36; CG versus emitted shapes: []; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s128/selected.cu.top2.cu:50: #define TILEMEGA_GEMM_VARIANT_COUNT 2; CG distinct=2 emitted=2; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s128/selected.cu.top1.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s128/selected.cu.top3.cu:50: #define TILEMEGA_GEMM_VARIANT_COUNT 2; CG distinct=2 emitted=2; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s4/selected.cu.top2.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s4/selected.cu.top1.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/reference/gqa2_s4/selected.cu.top3.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s4/selected.cu.top2.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s4/selected.cu.top1.cu:46: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s1/selected.cu.top2.cu:77: #define TILEMEGA_GEMM_VARIANT_COUNT 5; CG distinct=5 emitted=5; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s1/selected.cu.top1.cu:77: #define TILEMEGA_GEMM_VARIANT_COUNT 5; CG distinct=5 emitted=5; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s1/selected.cu.top3.cu:77: #define TILEMEGA_GEMM_VARIANT_COUNT 5; CG distinct=5 emitted=5; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16/selected.cu.top1.cu:70: #define TILEMEGA_GEMM_VARIANT_COUNT 4; CG distinct=4 emitted=4; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s1/selected.cu.top2.cu:78: #define TILEMEGA_GEMM_VARIANT_COUNT 6; CG distinct=6 emitted=6; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s1/selected.cu.top1.cu:74: #define TILEMEGA_GEMM_VARIANT_COUNT 5; CG distinct=5 emitted=5; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s1/selected.cu.top3.cu:78: #define TILEMEGA_GEMM_VARIANT_COUNT 6; CG distinct=6 emitted=6; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s4/selected.cu.top2.cu:70: #define TILEMEGA_GEMM_VARIANT_COUNT 4; CG distinct=4 emitted=4; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s4/selected.cu.top1.cu:70: #define TILEMEGA_GEMM_VARIANT_COUNT 4; CG distinct=4 emitted=4; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s4/selected.cu.top3.cu:74: #define TILEMEGA_GEMM_VARIANT_COUNT 5; CG distinct=5 emitted=5; /root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/template/materialized/selected.cu:70: #define TILEMEGA_GEMM_VARIANT_COUNT 4; CG distinct=4 emitted=4; /root/TileMega/docs/experiments/SOLVER_R9B/early_resident/llama_s1.cu:66: #define TILEMEGA_GEMM_VARIANT_COUNT 3; CG distinct=3 emitted=3; /root/TileMega/docs/experiments/SOLVER_R9B/early_resident/llama_s4.cu:66: #define TILEMEGA_GEMM_VARIANT_COUNT 3; CG distinct=3 emitted=3; /root/TileMega/docs/experiments/SOLVER_R9B/materialize_pilot/llama_s1/selected.cu.top2.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/materialize_pilot/llama_s1/selected.cu.top1.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_R9B/materialize_pilot/llama_s1/selected.cu.top3.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s1/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s64/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s1/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s64/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s1/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s64/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s16/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s16/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s1/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s64/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/llama_s4/selected.cu:58: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1; /root/TileMega/docs/experiments/SOLVER_V2/legacy_r8_domain/qwen3_s4/selected.cu:61: #define TILEMEGA_GEMM_VARIANT_COUNT 1; CG distinct=1 emitted=1
K-15 PASS lib/Solver/TaskPriceParts.cpp:14: if(!options_.regime_a || dtype_!=ScalarType::kBF16); lib/Solver/TaskPriceParts.cpp:65: auto const& fixed=options_.physical_fixed && !fit.fixed_physical.empty()?fit.fixed_physical:fit.fixed;; lib/Solver/TaskPriceParts.cpp:65: auto const& fixed=options_.physical_fixed && !fit.fixed_physical.empty()?fit.fixed_physical:fit.fixed;; old task_body fields preserved
K-16 PASS lib/Solver/StageFlowModel.cpp:99: if(p.all_external_miss && !options.no_external && now+1e-6<p.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in StageFlowModel");; lib/Solver/StageFlowModel.cpp:99: if(p.all_external_miss && !options.no_external && now+1e-6<p.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in StageFlowModel");; lib/Solver/FluidExecutionSimulator.cpp:60: if(options.all_external_miss && !options.no_external_dram && now+1e-6<options.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in fluid simulator");; lib/Solver/FluidExecutionSimulator.cpp:60: if(options.all_external_miss && !options.no_external_dram && now+1e-6<options.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in fluid simulator");
G-1 PASS K-12 FP64 conflict declared in deviations.md; all other structural checks required
G-2 FAIL required samples passed=130; measured binaries=36; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv', /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s4: top-3 incomplete (0/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv', [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv', /root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s4: top-3 incomplete (1/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s128/selected.cu.top3.tsv', ablations/llama_s4/template=2/2 required=10, matrix/llama_s16/selected.cu.top1.cu.measurement=6/6 required=10, reference/mha4_s4/selected.cu.top2.cu.measurement=2/2 required=10, colocation/llama=0/0, colocation/qwen3=0/0
G-3 FAIL measured_binaries=36 missing_sass=[] kernel_functions=76 FP64=1368 first=[('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*1380*/                   I2F.F64.S64 R10, R18 ;                                          /* 0x00000012000a7312 */'), ('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*13b0*/                   DMUL R10, R10, c[0x2][0x70] ;                                   /* 0x00801c000a0a7a28 */')]
G-4 PASS llama_s1: config_bytes=2471632960 CG_bytes=2471632960.0 rel=0 T_dram=2518402.1584694986 T_compute=13731.520267284712 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s4: config_bytes=2471645248 CG_bytes=2471645248.0 rel=0 T_dram=2519298.8244356406 T_compute=54926.081069138847 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s16: config_bytes=2471694400 CG_bytes=2471694400.0 rel=0 T_dram=2522885.4883002075 T_compute=219704.32427655539 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s64: config_bytes=2471891008 CG_bytes=2471891008.0 rel=0 T_dram=2537232.1437584748 T_compute=878817.29710622155 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; qwen3_s1: config_bytes=3441154176 CG_bytes=3441154176.0 rel=0 T_dram=3506496.2188630085 T_compute=19117.515382730071 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s4: config_bytes=3441166464 CG_bytes=3441166464.0 rel=0 T_dram=3507788.0017274912 T_compute=76470.061530920284 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s16: config_bytes=3441215616 CG_bytes=3441215616.0 rel=0 T_dram=3512955.1331854211 T_compute=305880.24612368114 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s64: config_bytes=3441412224 CG_bytes=3441412224.0 rel=0 T_dram=3533623.65901714 T_compute=1223520.9844947245 source=docs/experiments/MODELS/sources/qwen_config.json; raw timed anchored samples=340; allowance_ns=76914.00311779874; below_floor=[]
G-5 FAIL gqa2/bits_off_bf16 n=770 bit_text_equal=True; mha4/bits_off_bf16 n=462 bit_text_equal=True; gqa2/bits_off_f32 n=1077 bit_text_equal=True; gqa2/bits_on_f32 n=1077 bit_text_equal=True; mha4/bits_off_f32 n=1077 bit_text_equal=True; mha4/bits_on_f32 n=1077 bit_text_equal=True; gqa2 selected rho=0.837562128 required=0.9039; mha4 selected rho=0.842615454 required=0.8911
G-6 PASS old exhaustive prefix agrees=252 spaces; each space fully summed; bit identity sampled at first/middle/last price pieces; piece_validation_boundary_identity/llama/price_checks.tsv configs=3 spaces=1020 max_rel=8.781864124784988e-14 completed=True; piece_validation/qwen3/price_checks.tsv configs=3 spaces=1944 max_rel=1.8740564655672642e-13 completed=True
G-7 FAIL flow_max_ms=25.664808 pure_fluid_max_ms=665.42 solve_excluding_final_compile_max_s=4712.027763629027 missing=['solve/llama_s64', 'solve/qwen3_s16', 'solve/qwen3_s64']
G-8 PASS llama n=100 completed=True configuration_coverage=True legal_residency_kappa=True resume_prefix_intact=True rho=0.9942874287428742 ratio_p50=1.0027600020115222; qwen3 n=100 completed=True configuration_coverage=True legal_residency_kappa=True resume_prefix_intact=True rho=0.9907350735073507 ratio_p50=1.0066082294575116
A-runtime-release FAIL docs/experiments/SOLVER_R9B/unit/runtime_release.log: missing /RUNTIME_RELEASE checks=2064 mismatches=0(?:\s|$)/
G-9 FAIL ValueError: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner
G-10 FAIL ValueError: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner
G-11 FAIL matrix/llama_s1/selected.cu.m1B.flow.tsv closure_ns=0.0 sync=554821.1226004139 fixed=301966.37484181393 contention=0.0 chain=34827.300191765185 PG=891614.797633993 chain_protocol_ns=553073.8370557487 attention_fraction=0.019274255366057057 bubble_ns=5470.02943333738; matrix/llama_s1/selected.cu.m1A.flow.tsv closure_ns=0.0 sync=554821.1226004139 fixed=301966.37484181393 contention=0.0 chain=34827.300191765185 PG=891614.797633993 chain_protocol_ns=553073.8370557487 attention_fraction=0.019274255366057057 bubble_ns=5470.02943333738; matrix/llama_s4/selected.cu.m1B.flow.tsv closure_ns=0.0 sync=554695.9370484632 fixed=301643.8502343544 contention=0.0 chain=37856.98347884137 PG=894196.7707616589 chain_protocol_ns=553073.8370557487 attention_fraction=0.019573803539178446 bubble_ns=5485.869759273981; matrix/llama_s4/selected.cu.m1A.flow.tsv closure_ns=0.0 sync=554695.9370484632 fixed=301643.8502343544 contention=0.0 chain=37856.98347884137 PG=894196.7707616589 chain_protocol_ns=553073.8370557487 attention_fraction=0.019573803539178446 bubble_ns=5485.869759273981; llama_s16: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner; llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv'; matrix/qwen3_s1/selected.cu.m3B.flow.tsv closure_ns=0.0 sync=967105.8316037892 fixed=545912.704569947 contention=0.0 chain=106640.8787466758 PG=1619659.414920412 chain_protocol_ns=965341.4633927945 attention_fraction=0.022980523914461454 bubble_ns=5207.90808656081; matrix/qwen3_s1/selected.cu.m3A.flow.tsv closure_ns=0.0 sync=967105.8316037892 fixed=545912.704569947 contention=0.0 chain=106640.8787466758 PG=1619659.414920412 chain_protocol_ns=965341.4633927945 attention_fraction=0.022980523914461454 bubble_ns=5207.90808656081; qwen3_s4: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s4: top-3 incomplete (0/3); cannot select winner; qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv'; qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv'; trace/llama_s1/dump spaces=356 links=314 wall_ns=5562368 partition_closed=True; trace/llama_s64/dump spaces=356 links=544 wall_ns=12939264 partition_closed=True; trace/qwen3_s1/dump spaces=676 links=441 wall_ns=8472576 partition_closed=True; trace/qwen3_s64/dump spaces=676 links=692 wall_ns=16784384 partition_closed=True
G-12 FAIL llama import_count=1; llama_s1 complete T=3410016.9561034916 T/floor=1.3540398798640847 config=32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=1 changed_from_previous=False; llama_s2 complete T=3410753.863063284 T/floor=1.354171773167241 config=32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=4 changed_from_previous=False; llama_s4 complete T=3413495.5951972995 T/floor=1.354938748070893 config=32x32x64s3k1;32x16x64s3k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=4 changed_from_previous=True; llama_s8 complete T=3427307.9237988917 T/floor=1.3597760630916143 config=32x32x64s5k1;32x16x64s3k1;32x16x64s3k1;32x64x16s2k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=16 changed_from_previous=True; llama_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s16/completed.tsv'; llama_s32: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s32/completed.tsv'; llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s64/completed.tsv'; qwen3 import_count=1; qwen3_s1 complete T=5126155.633783421 T/floor=1.4619025128866647 config=32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=1 changed_from_previous=False; qwen3_s2 complete T=5127183.15594539 T/floor=1.4620160126291115 config=32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=4 changed_from_previous=False; qwen3_s4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s4/completed.tsv'; qwen3_s8: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s8/completed.tsv'; qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s16/completed.tsv'; qwen3_s32: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s32/completed.tsv'; qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s64/completed.tsv'
G-13 FAIL ablations/llama_s4/template n=3 L2=4.269568; ablations/llama_s4/template grid=128 residency=1 measured_limit=1 occupancy_ok=True moved=0/14525 flow_ns=3413495.5951972995 fluid_ns=3451745.5497145853; ablations/llama_s4/eft n=0 L2=nan; ablations/llama_s4/eft: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/eft/materialized/selected.metrics.tsv'; ablations/llama_s4/wide n=0 L2=nan; ablations/llama_s4/wide: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/wide/materialized/selected.metrics.tsv'; ablations/qwen3_s4/template n=0 L2=nan; ablations/qwen3_s4/template: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/template/materialized/selected.metrics.tsv'; ablations/qwen3_s4/eft n=0 L2=nan; ablations/qwen3_s4/eft: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/eft/materialized/selected.metrics.tsv'; ablations/qwen3_s4/wide n=0 L2=nan; ablations/qwen3_s4/wide: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/wide/materialized/selected.metrics.tsv'; stages/llama_s4_S2 n=0 L2=nan; stages/llama_s4_S2: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S2/materialized/selected.metrics.tsv'; stages/llama_s4_S3 n=0 L2=nan; stages/llama_s4_S3: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S3/materialized/selected.metrics.tsv'; stages/llama_s4_S4 n=0 L2=nan; stages/llama_s4_S4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S4/materialized/selected.metrics.tsv'
G-14 PASS early_resident/llama_s1.measurement internal=10/10 L2/control=1.1185298404848596; early_resident/llama_s4.measurement internal=10/10 L2/control=1.1338098562399996
G-15 PASS raw COSTMODEL/body_fit/observations.tsv + raw specs/meta + target parameters; S2 new_fixed n=350 p50=0.18465016 mean=0.22081826; S2 new_loop n=350 p50=0.11432774 mean=0.20137528; S2 old_fixed n=350 p50=0.16556912 mean=0.18567434; S2 old_loop n=350 p50=0.20525206 mean=0.22850406; S3 new_fixed n=140 p50=0.37013651 mean=0.4226047; S3 new_loop n=140 p50=0.050411865 mean=0.09074614; S3 old_fixed n=140 p50=0.38862564 mean=0.64860727; S3 old_loop n=140 p50=0.21825557 mean=0.22925597
A-simulator-legacy-identity PASS docs/experiments/SOLVER_R9B/simulator_identity: four cells x three placements x two sync settings; summaries=24 rows=102312 bitwise_equal=True inputs_intact=True sha256=552b95655ba9dcfeacafaa393b36d091ace3db0e300863aa8b542db805589c0c; historical trace durations are fixed replay inputs, not GPU timing claims
A-wait-window-identity PASS docs/experiments/SOLVER_R9B/wait_window_identity: cases=252 byte_equal=True sha256=211dcc83d327b27499c8a787f44335080f5236e6507c4d09b19040ad0fa3bc8a; loop-invariant extent hoisting preserves fitted event windows
A-model-traffic FAIL legacy/llama_s1 spaces=356 provenance_exact=True; skeleton/llama_s1 spaces=243 provenance_exact=True; legacy/llama_s4 spaces=356 provenance_exact=True; skeleton/llama_s4 spaces=243 provenance_exact=True; legacy/llama_s16 spaces=356 provenance_exact=True; traffic/skeleton/llama_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/llama_s16/spaces.tsv'; legacy/llama_s64 spaces=356 provenance_exact=True; traffic/skeleton/llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/llama_s64/spaces.tsv'; legacy/qwen3_s1 spaces=676 provenance_exact=True; skeleton/qwen3_s1 spaces=479 provenance_exact=True; legacy/qwen3_s4 spaces=676 provenance_exact=True; traffic/skeleton/qwen3_s4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s4/spaces.tsv'; legacy/qwen3_s16 spaces=676 provenance_exact=True; traffic/skeleton/qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s16/spaces.tsv'; legacy/qwen3_s64 spaces=676 provenance_exact=True; traffic/skeleton/qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/traffic/skeleton/qwen3_s64/spaces.tsv'
A-local-release-identity PASS llama_s1 six_predictions_bit_equal=True; llama_s4 six_predictions_bit_equal=True; llama_s16 six_predictions_bit_equal=True; llama_s64 six_predictions_bit_equal=True; qwen3_s1 six_predictions_bit_equal=True; qwen3_s4 six_predictions_bit_equal=True; qwen3_s16 six_predictions_bit_equal=True; qwen3_s64 six_predictions_bit_equal=True
A-materialization-displacement FAIL matrix/llama_s1 pairs=8/8 exact_worker_checks=True historical_fraction_differences=8; exact A/B worker comparison; matrix/llama_s4 pairs=8/8 exact_worker_checks=True historical_fraction_differences=8; exact A/B worker comparison; matrix/llama_s16 pairs=8/8 exact_worker_checks=True historical_fraction_differences=8; exact A/B worker comparison; matrix/llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.materializations.tsv'; matrix/qwen3_s1 pairs=8/8 exact_worker_checks=True historical_fraction_differences=8; exact A/B worker comparison; matrix/qwen3_s4 pairs=8/8 exact_worker_checks=True historical_fraction_differences=8; exact A/B worker comparison; matrix/qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.materializations.tsv'; matrix/qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.materializations.tsv'
R9B_VERIFY failures=K-12,G-2,G-3,G-5,G-7,A-runtime-release,G-9,G-10,G-11,G-12,G-13,A-model-traffic,A-materialization-displacement
```

## 3. 逐门结果

| 门 | 结果 | 实测与证据 |
| --- | --- | --- |
| G-1 | PASS | K-12 FP64 conflict declared in deviations.md; all other structural checks required |
| G-2 | FAIL | required samples passed=130; measured binaries=36; /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv', /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s4: top-3 incomplete (0/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv', [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv', /root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s4: top-3 incomplete (1/3); cannot select winner, [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/reference/mha4_s128/selected.cu.top3.tsv', ablations/llama_s4/template=2/2 required=10, matrix/llama_s16/selected.cu.top1.cu.measurement=6/6 required=10, reference/mha4_s4/selected.cu.top2.cu.measurement=2/2 required=10, colocation/llama=0/0, colocation/qwen3=0/0 |
| G-3 | FAIL | measured_binaries=36 missing_sass=[] kernel_functions=76 FP64=1368 first=[('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*1380*/                   I2F.F64.S64 R10, R18 ;                                          /* 0x00000012000a7312 */'), ('docs/experiments/SOLVER_R9B/ablations/llama_s4/template/sass.log', '/*13b0*/                   DMUL R10, R10, c[0x2][0x70] ;                                   /* 0x00801c000a0a7a28 */')] |
| G-4 | PASS | llama_s1: config_bytes=2471632960 CG_bytes=2471632960.0 rel=0 T_dram=2518402.1584694986 T_compute=13731.520267284712 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s4: config_bytes=2471645248 CG_bytes=2471645248.0 rel=0 T_dram=2519298.8244356406 T_compute=54926.081069138847 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s16: config_bytes=2471694400 CG_bytes=2471694400.0 rel=0 T_dram=2522885.4883002075 T_compute=219704.32427655539 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; llama_s64: config_bytes=2471891008 CG_bytes=2471891008.0 rel=0 T_dram=2537232.1437584748 T_compute=878817.29710622155 source=docs/experiments/MODELS/sources/llama_config_public_copy.json; qwen3_s1: config_bytes=3441154176 CG_bytes=3441154176.0 rel=0 T_dram=3506496.2188630085 T_compute=19117.515382730071 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s4: config_bytes=3441166464 CG_bytes=3441166464.0 rel=0 T_dram=3507788.0017274912 T_compute=76470.061530920284 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s16: config_bytes=3441215616 CG_bytes=3441215616.0 rel=0 T_dram=3512955.1331854211 T_compute=305880.24612368114 source=docs/experiments/MODELS/sources/qwen_config.json; qwen3_s64: config_bytes=3441412224 CG_bytes=3441412224.0 rel=0 T_dram=3533623.65901714 T_compute=1223520.9844947245 source=docs/experiments/MODELS/sources/qwen_config.json; raw timed anchored samples=340; allowance_ns=76914.00311779874; below_floor=[] |
| G-5 | FAIL | gqa2/bits_off_bf16 n=770 bit_text_equal=True; mha4/bits_off_bf16 n=462 bit_text_equal=True; gqa2/bits_off_f32 n=1077 bit_text_equal=True; gqa2/bits_on_f32 n=1077 bit_text_equal=True; mha4/bits_off_f32 n=1077 bit_text_equal=True; mha4/bits_on_f32 n=1077 bit_text_equal=True; gqa2 selected rho=0.837562128 required=0.9039; mha4 selected rho=0.842615454 required=0.8911 |
| G-6 | PASS | old exhaustive prefix agrees=252 spaces; each space fully summed; bit identity sampled at first/middle/last price pieces; piece_validation_boundary_identity/llama/price_checks.tsv configs=3 spaces=1020 max_rel=8.781864124784988e-14 completed=True; piece_validation/qwen3/price_checks.tsv configs=3 spaces=1944 max_rel=1.8740564655672642e-13 completed=True |
| G-7 | FAIL | flow_max_ms=25.664808 pure_fluid_max_ms=665.42 solve_excluding_final_compile_max_s=4712.027763629027 missing=['solve/llama_s64', 'solve/qwen3_s16', 'solve/qwen3_s64'] |
| G-8 | PASS | llama n=100 completed=True configuration_coverage=True legal_residency_kappa=True resume_prefix_intact=True rho=0.9942874287428742 ratio_p50=1.0027600020115222; qwen3 n=100 completed=True configuration_coverage=True legal_residency_kappa=True resume_prefix_intact=True rho=0.9907350735073507 ratio_p50=1.0066082294575116 |
| G-9 | FAIL | ValueError: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner |
| G-10 | FAIL | ValueError: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner |
| G-11 | FAIL | matrix/llama_s1/selected.cu.m1B.flow.tsv closure_ns=0.0 sync=554821.1226004139 fixed=301966.37484181393 contention=0.0 chain=34827.300191765185 PG=891614.797633993 chain_protocol_ns=553073.8370557487 attention_fraction=0.019274255366057057 bubble_ns=5470.02943333738; matrix/llama_s1/selected.cu.m1A.flow.tsv closure_ns=0.0 sync=554821.1226004139 fixed=301966.37484181393 contention=0.0 chain=34827.300191765185 PG=891614.797633993 chain_protocol_ns=553073.8370557487 attention_fraction=0.019274255366057057 bubble_ns=5470.02943333738; matrix/llama_s4/selected.cu.m1B.flow.tsv closure_ns=0.0 sync=554695.9370484632 fixed=301643.8502343544 contention=0.0 chain=37856.98347884137 PG=894196.7707616589 chain_protocol_ns=553073.8370557487 attention_fraction=0.019573803539178446 bubble_ns=5485.869759273981; matrix/llama_s4/selected.cu.m1A.flow.tsv closure_ns=0.0 sync=554695.9370484632 fixed=301643.8502343544 contention=0.0 chain=37856.98347884137 PG=894196.7707616589 chain_protocol_ns=553073.8370557487 attention_fraction=0.019573803539178446 bubble_ns=5485.869759273981; llama_s16: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s16: top-3 incomplete (0/3); cannot select winner; llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/llama_s64/selected.cu.top3.tsv'; matrix/qwen3_s1/selected.cu.m3B.flow.tsv closure_ns=0.0 sync=967105.8316037892 fixed=545912.704569947 contention=0.0 chain=106640.8787466758 PG=1619659.414920412 chain_protocol_ns=965341.4633927945 attention_fraction=0.022980523914461454 bubble_ns=5207.90808656081; matrix/qwen3_s1/selected.cu.m3A.flow.tsv closure_ns=0.0 sync=967105.8316037892 fixed=545912.704569947 contention=0.0 chain=106640.8787466758 PG=1619659.414920412 chain_protocol_ns=965341.4633927945 attention_fraction=0.022980523914461454 bubble_ns=5207.90808656081; qwen3_s4: /root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s4: top-3 incomplete (0/3); cannot select winner; qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s16/selected.cu.top3.tsv'; qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/matrix/qwen3_s64/selected.cu.top3.tsv'; trace/llama_s1/dump spaces=356 links=314 wall_ns=5562368 partition_closed=True; trace/llama_s64/dump spaces=356 links=544 wall_ns=12939264 partition_closed=True; trace/qwen3_s1/dump spaces=676 links=441 wall_ns=8472576 partition_closed=True; trace/qwen3_s64/dump spaces=676 links=692 wall_ns=16784384 partition_closed=True |
| G-12 | FAIL | llama import_count=1; llama_s1 complete T=3410016.9561034916 T/floor=1.3540398798640847 config=32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=1 changed_from_previous=False; llama_s2 complete T=3410753.863063284 T/floor=1.354171773167241 config=32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=4 changed_from_previous=False; llama_s4 complete T=3413495.5951972995 T/floor=1.354938748070893 config=32x32x64s3k1;32x16x64s3k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 seed_seq=4 changed_from_previous=True; llama_s8 complete T=3427307.9237988917 T/floor=1.3597760630916143 config=32x32x64s5k1;32x16x64s3k1;32x16x64s3k1;32x64x16s2k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=16 changed_from_previous=True; llama_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s16/completed.tsv'; llama_s32: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s32/completed.tsv'; llama_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/llama_s64/completed.tsv'; qwen3 import_count=1; qwen3_s1 complete T=5126155.633783421 T/floor=1.4619025128866647 config=32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=1 changed_from_previous=False; qwen3_s2 complete T=5127183.15594539 T/floor=1.4620160126291115 config=32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 seed_seq=4 changed_from_previous=False; qwen3_s4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s4/completed.tsv'; qwen3_s8: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s8/completed.tsv'; qwen3_s16: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s16/completed.tsv'; qwen3_s32: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s32/completed.tsv'; qwen3_s64: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/theta/qwen3_s64/completed.tsv' |
| G-13 | FAIL | ablations/llama_s4/template n=3 L2=4.269568; ablations/llama_s4/template grid=128 residency=1 measured_limit=1 occupancy_ok=True moved=0/14525 flow_ns=3413495.5951972995 fluid_ns=3451745.5497145853; ablations/llama_s4/eft n=0 L2=nan; ablations/llama_s4/eft: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/eft/materialized/selected.metrics.tsv'; ablations/llama_s4/wide n=0 L2=nan; ablations/llama_s4/wide: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/llama_s4/wide/materialized/selected.metrics.tsv'; ablations/qwen3_s4/template n=0 L2=nan; ablations/qwen3_s4/template: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/template/materialized/selected.metrics.tsv'; ablations/qwen3_s4/eft n=0 L2=nan; ablations/qwen3_s4/eft: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/eft/materialized/selected.metrics.tsv'; ablations/qwen3_s4/wide n=0 L2=nan; ablations/qwen3_s4/wide: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/ablations/qwen3_s4/wide/materialized/selected.metrics.tsv'; stages/llama_s4_S2 n=0 L2=nan; stages/llama_s4_S2: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S2/materialized/selected.metrics.tsv'; stages/llama_s4_S3 n=0 L2=nan; stages/llama_s4_S3: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S3/materialized/selected.metrics.tsv'; stages/llama_s4_S4 n=0 L2=nan; stages/llama_s4_S4: [Errno 2] No such file or directory: '/root/TileMega/docs/experiments/SOLVER_R9B/stages/llama_s4_S4/materialized/selected.metrics.tsv' |
| G-14 | PASS | early_resident/llama_s1.measurement internal=10/10 L2/control=1.1185298404848596; early_resident/llama_s4.measurement internal=10/10 L2/control=1.1338098562399996 |
| G-15 | PASS | raw COSTMODEL/body_fit/observations.tsv + raw specs/meta + target parameters; S2 new_fixed n=350 p50=0.18465016 mean=0.22081826; S2 new_loop n=350 p50=0.11432774 mean=0.20137528; S2 old_fixed n=350 p50=0.16556912 mean=0.18567434; S2 old_loop n=350 p50=0.20525206 mean=0.22850406; S3 new_fixed n=140 p50=0.37013651 mean=0.4226047; S3 new_loop n=140 p50=0.050411865 mean=0.09074614; S3 old_fixed n=140 p50=0.38862564 mean=0.64860727; S3 old_loop n=140 p50=0.21825557 mean=0.22925597 |

## 4. 停摆、降级与偏离

| 项目 | 影响与现状 | 解锁或下一步 | 估计工作量（推断） |
| --- | --- | --- | --- |
| SV-9(e), G-3 | 静态 FP64 为零的门失败；不停止独立求解与计时。既有 RoPE `sinf/cosf` 的 CUDA 慢路径含 FP64，MIDPOINT_REFINE=0 | 需另行授权修改 TaskBody/math 语义，证明 RoPE 参数范围并验证替代范围约减；本轮禁改 | 定位已完成；实现及数值覆盖约 1–3 天 |
| SV-11(d) | 物理固定段均值误差下降、中位误差与回放排序变差，按 §7.3 保留旧固定段；新拟合仍独立输出 | 按有效行数、tile 与 split 分组校准固定段，保持留出集独立 | 约 1–2 天 |
| G-5 | BF16 排序硬门仍失败；关闭选项、FP32 开启的 binary64 不变门通过。开启物理流量与 stages，关闭劣化的 fixed 分量，降级支持下游实测 | 先重建 F-116 与当前目标文件/TaskBody 回放之间的差异，再标定 `TaskPriceParts.cpp` 固定/循环权重；不得通过换门线宣称通过 | 约 1–3 天 |
| SV-12(a), P-9 | CG 的 Coarsen 释放端点与执行器保守窗口并不总相同：参考抽样 472/2064 不同。Level 1 保留规格的 CG 释放，最终物化和 FIFO 模拟验证真实事件窗 | 下一版 Level 1 同时计 `max(CG endpoint, runtime event endpoint)` 与真实 masks，或另轮收紧执行窗口；前者需新搜索与 V2 | 约 1–2 天及重测 |
| G-7 | seq64 预热求值超过 10 ms；完整域搜索已有格超过 30 min。保留完整域继续计时，不宣称预算通过 | `FlowPreparation.cpp` 的逐边排序表准备、`PrepareSymbolicProblem` 精确关系准备占主导；缓存实际改变的 space/边并减少整模型重建；`EvaluateFlow` 的 pending 初始化与事件数另计 | 约 2–4 天及全矩阵复测 |

临时停止的旧 General Oracle 审计已解锁：全参数 `lexmax` 的分片爆炸改为精确当前纤维的最大线性序号，集合测试通过；已完成样本前缀保留，RNG 按原序续跑。这不是用包围盒近似。

其余明确偏离：embedding 唯一行取决于 token 值；trace 的 fixed/mainloop 合并；旧 prompt 的链深与真实 CG 不符，使用所测 CG 的 D；theta 2/8/32 的第一起点取下一已有 legacy 点，第二起点仍为当前 theta 的完整 uniform 全扫。详见 [逐项规格、实际做法与理由](deviations.md)。

## 5. 绝对位置

所有时间 ms；b 为 ns/链节。legacy: n=6/6，几何平均=2.58201267；skeleton: n=3/6，未齐，不计算整门结果。

| cell | arm | T_dram_ms | T_compute_ms | T_floor_ms | l05_ms | l1_ms | l2_ms | L2_over_floor | L2_over_legacy | D | bubble_ns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | legacy | 2.518402 | 0.01373152 | 2.518402 | 5.747704 | 5.788512 | 5.422432 | 2.153124 | 1 | 228 | 12736.97 |
| llama_s1 | skeleton | 2.518402 | 0.01373152 | 2.518402 | 5.51936 | 5.4712 | 4.446912 | 1.765767 | 0.8200955 | 163 | 11831.35 |
| llama_s4 | legacy | 2.519299 | 0.05492608 | 2.519299 | 6.308352 | 6.45808 | 6.47468 | 2.570033 | 1 | 228 | 17348.16 |
| llama_s4 | skeleton | 2.519299 | 0.05492608 | 2.519299 | 5.32204 | 5.166352 | 4.446888 | 1.765129 | 0.686812 | 163 | 11825.7 |
| llama_s16 | legacy | 2.522885 | 0.2197043 | 2.522885 | 6.583928 | 6.671104 | 7.11624 | 2.820675 | 1 | 228 | 20146.29 |
| llama_s64 | legacy | 2.537232 | 0.8788173 | 2.537232 | 11.81832 | 11.85805 | 12.67735 | 4.996528 | 1 | 228 | 44474.21 |
| qwen3_s1 | legacy | 3.506496 | 0.01911752 | 3.506496 | 8.802328 | 8.939392 | 8.24576 | 2.351567 | 1 | 424 | 11177.51 |
| qwen3_s1 | skeleton | 3.506496 | 0.01911752 | 3.506496 | 7.561456 | 7.501088 | 8.041976 | 2.293451 | 0.9752862 | 311 | 14583.54 |
| qwen3_s4 | legacy | 3.507788 | 0.07647006 | 3.507788 | 9.31388 | 9.58336 | 9.314824 | 2.655469 | 1 | 424 | 13695.84 |
| qwen3_s16 | legacy | 3.512955 | 0.3058802 | 3.512955 | 9.611 | 9.862832 | 10.67981 | 3.040121 | 1 | 424 | 16902.96 |
| qwen3_s64 | legacy | 3.533624 | 1.223521 | 3.533624 | 17.04294 | 15.97014 | 17.0185 | 4.816162 | 1 | 424 | 31803.96 |

完整字段与证据路径：[原始数据重算表](report_tables/performance.tsv)。

## 6. 缺口分解与 trace

以下四项来自同一物理模型的反事实求值，和为 **预测 T − T_floor**，并非实测 L2 − T_floor 的直接测量分解。`flow_over_measured` 显示仍未被模型解释的部分，不能把它藏入某个已测相位。负争用项原样保留。

| cell | sync_ns | fixed_ns | contention_ns | chain_ns | closure_ns | flow_over_measured | fluid_over_measured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 554821.1 | 301966.4 | 0 | 34827.3 | 0 | 0.7668281 | 0.7701696 |
| llama_s4 | 554695.9 | 301643.9 | 0 | 37856.98 | 0 | 0.7676145 | 0.776216 |
| qwen3_s1 | 967105.8 | 545912.7 | 0 | 106640.9 | 0 | 0.6374249 | 0.6371477 |

完整字段与证据路径：[原始数据重算表](report_tables/decomposition.tsv)。

同一 legacy 几何的并列诊断如下；模型采用全局池/纯 home，trace 来自实际 legacy 放置，不能把两列差值直接归因于某一个物理项。反事实分解与已实现路径区间也不是同一量。

| cell | actual_legacy_ms | trace_chain_ns | trace_wait_hop_ns | trace_task_combined_ns | trace_publication_ns | flow_ns | flow_synchronization_ns | flow_fixed_ns | flow_contention_ns | flow_chain_delay_ns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 5.422432 | 5562368 | 2236416 | 2262016 | 768000 | 5570757 | 1072114 | 1797094 | 9433.736 | 173713.8 |
| llama_s64 | 12.67735 | 1.293926e+07 | 2300928 | 9062400 | 1087488 | 8036011 | 2009223 | 794108.5 | 17398.2 | 2678049 |
| qwen3_s1 | 8.24576 | 8472576 | 3514368 | 3644416 | 1074176 | 8490983 | 2041583 | 2650249 | 4169.26 | 288486.1 |
| qwen3_s64 | 17.0185 | 1.678438e+07 | 2850816 | 1.054515e+07 | 1375232 | 1.187837e+07 | 2926805 | 1454296 | 256098.4 | 3707545 |

完整字段与证据路径：[原始数据重算表](report_tables/trace_vs_flow.tsv)。

四格 trace 的固定段与主循环合并报告；等待/hop、发布、屏障、空闲保持独立。逐节非重叠墙钟路径可求和；各 space 跨度相互重叠，不可相加当作关键路径。


llama_s1：[逐节](trace_analysis/llama_s1/chain_links.tsv)、[逐 space](trace_analysis/llama_s1/task_spaces.tsv)、[按类别汇总](trace_analysis/llama_s1/chain_categories.tsv)。

| category | links | wall_ns | task_fixed_plus_mainloop_ns | publish_ns | wait_hop_ns | barrier_ns | idle_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combine | 1 | 179200 | 8192 | 3072 | 1024 | 0 | 166912 |
| embed | 1 | 5120 | 5120 | 0 | 0 | 0 | 0 |
| lm_head | 72 | 741376 | 400384 | 217088 | 95232 | 7168 | 21504 |
| norm | 1 | 7168 | 5120 | 1024 | 1024 | 0 | 0 |
| proj | 239 | 4629504 | 1843200 | 546816 | 2139136 | 23552 | 76800 |


llama_s64：[逐节](trace_analysis/llama_s64/chain_links.tsv)、[逐 space](trace_analysis/llama_s64/task_spaces.tsv)、[按类别汇总](trace_analysis/llama_s64/chain_categories.tsv)。

| category | links | wall_ns | task_fixed_plus_mainloop_ns | publish_ns | wait_hop_ns | barrier_ns | idle_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| append | 63 | 184320 | 46080 | 73728 | 53248 | 3072 | 8192 |
| attn | 64 | 488448 | 307200 | 109568 | 55296 | 5120 | 11264 |
| combine | 81 | 681984 | 204800 | 147456 | 66560 | 9216 | 253952 |
| embed | 1 | 6144 | 6144 | 0 | 0 | 0 | 0 |
| lm_head | 63 | 2279424 | 1944576 | 204800 | 82944 | 18432 | 28672 |
| norm | 1 | 9216 | 7168 | 2048 | 0 | 0 | 0 |
| proj | 175 | 8972288 | 6462464 | 413696 | 1968128 | 59392 | 68608 |
| rope | 96 | 317440 | 83968 | 136192 | 74752 | 8192 | 14336 |


qwen3_s1：[逐节](trace_analysis/qwen3_s1/chain_links.tsv)、[逐 space](trace_analysis/qwen3_s1/task_spaces.tsv)、[按类别汇总](trace_analysis/qwen3_s1/chain_categories.tsv)。

| category | links | wall_ns | task_fixed_plus_mainloop_ns | publish_ns | wait_hop_ns | barrier_ns | idle_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combine | 5 | 49152 | 37888 | 3072 | 5120 | 2048 | 1024 |
| embed | 1 | 5120 | 5120 | 0 | 0 | 0 | 0 |
| lm_head | 99 | 994304 | 631808 | 226304 | 82944 | 8192 | 45056 |
| norm | 1 | 7168 | 5120 | 1024 | 0 | 0 | 1024 |
| proj | 335 | 7416832 | 2964480 | 843776 | 3426304 | 29696 | 152576 |


qwen3_s64：[逐节](trace_analysis/qwen3_s64/chain_links.tsv)、[逐 space](trace_analysis/qwen3_s64/task_spaces.tsv)、[按类别汇总](trace_analysis/qwen3_s64/chain_categories.tsv)。

| category | links | wall_ns | task_fixed_plus_mainloop_ns | publish_ns | wait_hop_ns | barrier_ns | idle_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| append | 56 | 164864 | 37888 | 66560 | 40960 | 4096 | 15360 |
| attn | 56 | 693248 | 481280 | 73728 | 84992 | 31744 | 21504 |
| combine | 142 | 2326528 | 424960 | 173056 | 88064 | 18432 | 1622016 |
| embed | 1 | 7168 | 7168 | 0 | 0 | 0 | 0 |
| lm_head | 17 | 890880 | 749568 | 73728 | 47104 | 10240 | 10240 |
| norm | 85 | 330752 | 88064 | 123904 | 84992 | 7168 | 26624 |
| proj | 279 | 1.220608e+07 | 8713216 | 788480 | 2475008 | 102400 | 126976 |
| rope | 56 | 164864 | 43008 | 75776 | 29696 | 6144 | 10240 |

§1.2 的强结论不能完整证实：seq1 的等待与发布确实占据很大一部分已实现路径，但 trace 未区分 task 固定与主循环，不能把合并段全部归为固定开销。

## 7. 代价模型

GB 级无生产者权重流量从 SDCM 约 0.5 的 DRAM 比例改为实测服务曲线给出的 1；有生产者中间量独立按 live footprint 定价。回放参考模型位于 L2 knee 以下，不能用其排序证明 GB 侧缓存规则。单元测试独立检查 2 GiB 的 df_np=1。

490 条观测拟合：λ=1.1593900607537728，r_sm=42.488827019004475 bytes/ns；保留旧 loop_wait/fixed 字段，生产路径不启用劣化的 physical_fixed。下表 nominal/physical 字节仅指注明的观测样本总体，不冒充完整真实模型总流量。

| stages | component | version | n | median | mean |
| --- | --- | --- | --- | --- | --- |
| all | loop | old | 490 | 0.213401 | 0.2287189 |
| all | loop | new | 490 | 0.09234741 | 0.169767 |
| all | fixed | old | 490 | 0.2155588 | 0.3179409 |
| all | fixed | new | 490 | 0.2349191 | 0.2784715 |
| 2 | loop | old | 350 | 0.2052521 | 0.2285041 |
| 2 | loop | new | 350 | 0.1143277 | 0.2013753 |
| 2 | fixed | old | 350 | 0.1655691 | 0.1856743 |
| 2 | fixed | new | 350 | 0.1846502 | 0.2208183 |
| 3 | loop | old | 140 | 0.2182556 | 0.229256 |
| 3 | loop | new | 140 | 0.05041187 | 0.09074614 |
| 3 | fixed | old | 140 | 0.3886256 | 0.6486073 |
| 3 | fixed | new | 140 | 0.3701365 | 0.4226047 |

完整字段与证据路径：[原始数据重算表](report_tables/fit_errors.tsv)。
| cell | population | nominal_operand_bytes | physical_operand_bytes | physical_over_nominal |
| --- | --- | --- | --- | --- |
| gqa2_s128 | 490 phase observations, weighted by raw sample count | 3.58613e+08 | 3.58613e+08 | 1 |
| gqa2_s4 | 490 phase observations, weighted by raw sample count | 1.321206e+08 | 5.691802e+07 | 0.4308036 |
| mha4_s128 | 490 phase observations, weighted by raw sample count | 7.969178e+08 | 7.969178e+08 | 1 |
| mha4_s4 | 490 phase observations, weighted by raw sample count | 2.936013e+08 | 1.264845e+08 | 0.4308036 |
| real_s128 | 490 phase observations, weighted by raw sample count | 8.724152e+09 | 8.724152e+09 | 1 |
| real_s4 | 490 phase observations, weighted by raw sample count | 8.724152e+09 | 3.980394e+09 | 0.45625 |

完整字段与证据路径：[原始数据重算表](report_tables/observed_traffic.tsv)。

整图 task 访存量（按分片计数，包含 task 间重复读取；不是唯一 DRAM 字节或设备实测流量；标量路径原先已用物理域，两列保持相同）：

| cell | arm | spaces | tasks | nominal_read_bytes | nominal_write_bytes | physical_read_bytes | physical_write_bytes | physical_over_nominal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | legacy | 356 | 131676 | 3.773567e+09 | 4.139273e+09 | 2.557129e+09 | 6.624307e+07 | 0.3315335 |
| llama_s1 | skeleton | 243 | 10151 | 4.417753e+09 | 3.290726e+07 | 2.53381e+09 | 1591808 | 0.5696687 |
| llama_s4 | legacy | 356 | 163672 | 7.454687e+09 | 2.649723e+08 | 3.129573e+09 | 3.869286e+07 | 0.4104153 |
| llama_s4 | skeleton | 243 | 14525 | 4.423565e+09 | 3.465216e+07 | 2.721939e+09 | 6367232 | 0.6119725 |
| llama_s16 | legacy | 356 | 181168 | 7.594181e+09 | 2.84074e+08 | 5.122687e+09 | 1.547715e+08 | 0.6698766 |
| llama_s64 | legacy | 356 | 282720 | 1.559115e+10 | 3.604808e+08 | 1.559115e+10 | 3.604808e+08 | 1 |
| qwen3_s1 | legacy | 676 | 189337 | 5.259461e+09 | 5.944886e+09 | 3.565892e+09 | 9.545395e+07 | 0.326779 |
| qwen3_s1 | skeleton | 479 | 20840 | 6.812647e+09 | 4.758733e+07 | 3.551126e+09 | 2605824 | 0.5180189 |
| qwen3_s4 | legacy | 676 | 235984 | 1.039223e+10 | 3.818158e+08 | 4.370653e+09 | 5.684736e+07 | 0.4109412 |
| qwen3_s16 | legacy | 676 | 263896 | 1.063384e+10 | 4.130857e+08 | 7.192937e+09 | 2.273894e+08 | 0.6717097 |
| qwen3_s64 | legacy | 676 | 420880 | 2.204703e+10 | 5.381652e+08 | 2.204703e+10 | 5.381652e+08 | 1 |

完整字段与证据路径：[原始数据重算表](report_tables/model_traffic.tsv)。
| arm | model | n | spearman | MAPE | best_actual_rank_in_predicted_top1 | best_actual_rank_in_predicted_top3 | best_actual_rank_in_predicted_top10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_bf16 | gqa2 | 770 | 0.8003551 | 0.2854757 | 79.5 | 79.5 | 45 |
| baseline_bf16 | mha4 | 462 | 0.8069801 | 0.2611139 | 62.5 | 62.5 | 29 |
| physical_bf16 | gqa2 | 770 | 0.8032988 | 0.2877332 | 79.5 | 79.5 | 42.5 |
| physical_bf16 | mha4 | 462 | 0.8085892 | 0.2632737 | 29 | 29 | 29 |
| stages_bf16 | gqa2 | 770 | 0.8348457 | 0.3153684 | 79.5 | 79.5 | 42.5 |
| stages_bf16 | mha4 | 462 | 0.8405346 | 0.2889472 | 29 | 29 | 29 |
| fixed_bf16 | gqa2 | 770 | 0.6340461 | 0.2920276 | 45 | 45 | 45 |
| fixed_bf16 | mha4 | 462 | 0.6910868 | 0.2669333 | 30 | 30 | 30 |
| all_complete_bf16 | gqa2 | 770 | 0.6771205 | 0.3218884 | 45 | 45 | 45 |
| all_complete_bf16 | mha4 | 462 | 0.7333179 | 0.2937858 | 30 | 30 | 29 |
| selected_bf16 | gqa2 | 770 | 0.8375621 | 0.3184948 | 79.5 | 79.5 | 42.5 |
| selected_bf16 | mha4 | 462 | 0.8426155 | 0.2920654 | 29 | 29 | 29 |
| historical_target_bf16 | gqa2 | 770 | 0.8976355 | 0.4853058 | 232 | 60 | 45 |
| historical_target_bf16 | mha4 | 462 | 0.8833542 | 0.4686821 | 145 | 33.5 | 29 |

完整字段与证据路径：[原始数据重算表](report_tables/replay.tsv)。

## 8. Level 1 与逐 tile 模拟

预热速度原始证据 `flow_final/`、`flow_arena/`；完整矩阵的求解阶段见 §11。随机样本只有完成 100 组且进程正常结束才标 complete。关闭流体模式的参考 24 计划对照共 102,312 行 binary64 输出与基线逐字节一致，见 `simulator_identity/`。

| family | model | n | complete | spearman | ratio_p10 | ratio_p50 | ratio_p90 | flow_ms_max | fluid_ms_max | nonprefix_edges_max | varying_spaces_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | llama | 100 | True | 0.9942874 | 0.957792 | 1.00276 | 1.023416 | 12.63802 | 530.139 | 273 | 16 |
| validation | qwen3 | 100 | True | 0.9907351 | 0.9558294 | 1.006608 | 1.031249 | 17.221 | 665.42 | 477 | 29 |
| validation_colocated | llama | 100 | True | 0.9946715 | 0.9577145 | 1.002709 | 1.023642 | 9.494698 | 155.811 | 273 | 16 |
| validation_colocated | qwen3 | 80 | False | 0.9882325 | 0.9429712 | 1.003656 | 1.030473 | 13.95459 | 231.87 | 477 | 29 |

完整字段与证据路径：[原始数据重算表](report_tables/consistency.tsv)。

V3 的完整 top-3 对照（模型相等的预测保留平均秩，不把名单顺序解释为模型分辨力）：

| cell | shortlist_rank | l2_ms | L2_over_floor | flow_over_measured | fluid_over_measured | flow_rank | fluid_rank | actual_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 1 | 4.446912 | 1.765767 | 0.7668281 | 0.7701696 | 2 | 2 | 1 |
| llama_s1 | 2 | 4.450816 | 1.767317 | 0.7661555 | 0.769494 | 2 | 2 | 2 |
| llama_s1 | 3 | 4.462032 | 1.771771 | 0.7642296 | 0.7675598 | 2 | 2 | 3 |
| llama_s4 | 1 | 4.446888 | 1.765129 | 0.7676145 | 0.776216 | 2 | 2 | 1 |
| llama_s4 | 2 | 4.470472 | 1.774491 | 0.7635649 | 0.7721211 | 2 | 2 | 2 |
| llama_s4 | 3 | 4.92416 | 1.954576 | 0.6932138 | 0.7009816 | 2 | 2 | 3 |
| qwen3_s1 | 1 | 8.33884 | 2.378112 | 0.6147325 | 0.6144652 | 2 | 2 | 3 |
| qwen3_s1 | 2 | 8.041976 | 2.293451 | 0.6374249 | 0.6371477 | 2 | 2 | 1 |
| qwen3_s1 | 3 | 8.057424 | 2.297856 | 0.6362028 | 0.6359261 | 2 | 2 | 2 |

完整字段与证据路径：[原始数据重算表](report_tables/shortlist_predictions.tsv)。
| cell | interpretation | T | T_floor | actual_legacy_ms | flow_over_actual_legacy |
| --- | --- | --- | --- | --- | --- |
| llama_s1 | legacy uniform geometry; pure home, not actual legacy placement | 5570757 | 2518402 | 5.422432 | 1.027354 |
| llama_s4 | legacy uniform geometry; pure home, not actual legacy placement | 4578558 | 2519299 | 6.47468 | 0.7071481 |
| llama_s16 | legacy uniform geometry; pure home, not actual legacy placement | 4624557 | 2522885 | 7.11624 | 0.6498596 |
| llama_s64 | legacy uniform geometry; pure home, not actual legacy placement | 8036011 | 2537232 | 12.67735 | 0.6338872 |
| qwen3_s1 | legacy uniform geometry; pure home, not actual legacy placement | 8490983 | 3506496 | 8.24576 | 1.029739 |
| qwen3_s4 | legacy uniform geometry; pure home, not actual legacy placement | 7038380 | 3507788 | 9.314824 | 0.7556107 |
| qwen3_s16 | legacy uniform geometry; pure home, not actual legacy placement | 7114441 | 3512955 | 10.67981 | 0.6661581 |
| qwen3_s64 | legacy uniform geometry; pure home, not actual legacy placement | 1.187837e+07 | 3533624 | 17.0185 | 0.6979678 |

完整字段与证据路径：[原始数据重算表](report_tables/legacy_flow.tsv)。

## 9. 所选配置

算子名到类的完整映射见表的 operators 字段。原始 `top3.tsv` 与 `metrics.tsv` 的 `floor_ns` 是历史放置下界字段，**不是**本轮物理 T_floor；本报告仅使用 `tmexec.dram_floor`/`*.flow.tsv` 的物理下界。

| cell | operator_class | tile_m | tile_n | tile_k | stages | split_k | kappa | residency | distinct_variants | legacy_uniform |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 0 | 32 | 32 | 64 | 3 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s1 | 1 | 32 | 16 | 64 | 2 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s1 | 2 | 32 | 16 | 64 | 3 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s1 | 3 | 32 | 128 | 64 | 3 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s1 | 4 | 32 | 16 | 64 | 3 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s1 | 5 | 32 | 256 | 16 | 2 | 1 | 1 | 1 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| llama_s4 | 0 | 32 | 32 | 64 | 3 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| llama_s4 | 1 | 32 | 16 | 64 | 3 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| llama_s4 | 2 | 32 | 16 | 64 | 3 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| llama_s4 | 3 | 32 | 128 | 64 | 3 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| llama_s4 | 4 | 32 | 16 | 64 | 3 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| llama_s4 | 5 | 32 | 256 | 16 | 2 | 1 | 1 | 1 | 4 | {"split_k": "4", "stages": "2", "tile_k": "64", "tile_m": "32", "tile_n": "16"} |
| qwen3_s1 | 0 | 32 | 32 | 64 | 3 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| qwen3_s1 | 1 | 32 | 16 | 32 | 3 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| qwen3_s1 | 2 | 32 | 16 | 64 | 3 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| qwen3_s1 | 3 | 32 | 64 | 64 | 3 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| qwen3_s1 | 4 | 32 | 16 | 64 | 3 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |
| qwen3_s1 | 5 | 32 | 256 | 16 | 4 | 1 | 1 | 2 | 5 | {"split_k": "32", "stages": "2", "tile_k": "16", "tile_m": "64", "tile_n": "128"} |

完整字段与证据路径：[原始数据重算表](report_tables/configurations.tsv)。

## 10. 放置、消融与同步检验

原有 affinity/home/spread_other 是互斥分类，affinity 与真实 home 可重叠。离开 home 的比例用独立计数，旧产物用同配置的纯模板 A worker 表逐 tile 重算；不能使用 1−home/placed。

| cell | grid | residency | flow_ns | simulated_ns | placed | moved_count | moved_fraction | interleaving | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 128 | 1 | 3410017 | 4419060 | 10151 | 9464 | 0.9323219 | 0.9564003 | matrix/llama_s1/selected.cu.m1B.metrics.tsv |
| llama_s1 | 128 | 1 | 3410017 | 3424876 | 10151 | 0 | 0 | 0.9627856 | matrix/llama_s1/selected.cu.m1A.metrics.tsv |
| llama_s4 | 128 | 1 | 3413496 | 4292149 | 14525 | 13191 | 0.9081583 | 0.9696465 | matrix/llama_s4/selected.cu.m1B.metrics.tsv |
| llama_s4 | 128 | 1 | 3413496 | 3451746 | 14525 | 0 | 0 | 0.9740918 | matrix/llama_s4/selected.cu.m1A.metrics.tsv |
| qwen3_s1 | 256 | 2 | 5126156 | 5123927 | 20840 | 15918 | 0.7638196 | 0.9835795 | matrix/qwen3_s1/selected.cu.m3B.metrics.tsv |
| qwen3_s1 | 256 | 2 | 5126156 | 5125636 | 20840 | 0 | 0 | 0.9835795 | matrix/qwen3_s1/selected.cu.m3A.metrics.tsv |

完整字段与证据路径：[原始数据重算表](report_tables/placements.tsv)。
| cell | arm | l05_ms | l1_ms | l2_ms | L2_over_floor | flow_ns | simulated_ns | residency | actual_limit | moved_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | early_R9 | 8.484096 | 6.254848 | 6.065152 | 2.408333 | 5460053 |  |  |  |  |
| llama_s4 | early_R9 | 6.971112 | 6.534152 | 7.341056 | 2.913928 | 4972549 |  |  |  |  |

完整字段与证据路径：[原始数据重算表](report_tables/additional_measurements.tsv)。

全部 top-M 的 A/B 物化对照；旧表中移离 home 的字段保留为 historical_reported_moved_fraction，下面按独立计数或逐 task A/B worker 表重算：

| cell | rank | pure_ns | eft_ns | eft_over_pure | pure_selected | moved_count | placed | moved_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 1 | 3424876 | 4419060 | 1.290283 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 2 | 3424876 | 4419060 | 1.290283 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 3 | 3424876 | 4419060 | 1.290283 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 4 | 3425627 | 4842917 | 1.413732 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 5 | 3425496 | 4701788 | 1.372586 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 6 | 3425496 | 4701788 | 1.372586 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 7 | 3425496 | 4701788 | 1.372586 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s1 | 8 | 3425308 | 4573567 | 1.335228 | 1 | 9464 | 10151 | 0.9323219 |
| llama_s4 | 1 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 2 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 3 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 4 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 5 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 6 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 7 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s4 | 8 | 3451746 | 4292149 | 1.243472 | 1 | 13191 | 14525 | 0.9081583 |
| llama_s16 | 1 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 2 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 3 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 4 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 5 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 6 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 7 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| llama_s16 | 8 | 3526937 | 3961428 | 1.123192 | 1 | 30372 | 34069 | 0.8914849 |
| qwen3_s1 | 1 | 5125691 | 5124016 | 0.9996732 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 2 | 5125636 | 5123927 | 0.9996666 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 3 | 5125636 | 5123927 | 0.9996666 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 4 | 5125636 | 5123927 | 0.9996666 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 5 | 5125667 | 5123977 | 0.9996703 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 6 | 5125691 | 5124016 | 0.9996732 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 7 | 5125691 | 5124016 | 0.9996732 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s1 | 8 | 5125691 | 5124016 | 0.9996732 | 0 | 15918 | 20840 | 0.7638196 |
| qwen3_s4 | 1 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 2 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 3 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 4 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 5 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 6 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 7 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |
| qwen3_s4 | 8 | 5129238 | 5129238 | 1 | 1 | 20318 | 27818 | 0.7303904 |

完整字段与证据路径：[原始数据重算表](report_tables/materializations.tsv)。
每模型 ≥50 个新进程与实际事件等待省略计数由 G-2 核查；不能以结构共置边数代替执行器真正省去的等待数。
| cell | rank | estimated | actual | re_solved | residency | flow_ns | simulated_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 1 | 1 | 1 | 0 | 1 | 3410020 | 3424880 |
| llama_s1 | 2 | 1 | 1 | 0 | 1 | 3410020 | 3424880 |
| llama_s1 | 3 | 1 | 1 | 0 | 1 | 3410020 | 3424880 |
| llama_s4 | 1 | 1 | 1 | 0 | 1 | 3413500 | 3451750 |
| llama_s4 | 2 | 1 | 1 | 0 | 1 | 3413500 | 3451750 |
| llama_s4 | 3 | 1 | 1 | 0 | 1 | 3413500 | 3451750 |
| llama_s16 | 1 | 2 | 2 | 0 | 2 | 3565490 | 3526940 |
| llama_s16 | 2 | 2 | 2 | 0 | 2 | 3565490 | 3526940 |
| llama_s16 | 3 | 2 | 2 | 0 | 2 | 3565490 | 3526940 |
| qwen3_s1 | 1 | 2 | 2 | 0 | 2 | 5126160 | 5123930 |
| qwen3_s1 | 2 | 2 | 2 | 0 | 2 | 5126160 | 5123930 |
| qwen3_s1 | 3 | 2 | 2 | 0 | 2 | 5126160 | 5123930 |
| qwen3_s4 | 1 | 2 | 2 | 0 | 2 | 5129240 | 5129240 |
| qwen3_s4 | 2 | 2 | 2 | 0 | 2 | 5129240 | 5129240 |
| qwen3_s4 | 3 | 2 | 2 | 0 | 2 | 5129240 | 5129240 |

完整字段与证据路径：[原始数据重算表](report_tables/resources.tsv)。

## 11. 求解耗时

阶段单位 ms；部分诊断计时嵌套，不能把所有行直接求和当作 total。每格 budget 按完整求解墙钟减最终 megakernel 编译核算；变体编译仍计入。R9 30–45 h 与旧 legacy 1.3–5.5 h 是历史对照，不是本轮重新求解时间。

| cell | phase | count | total_ms |
| --- | --- | --- | --- |
| llama_s1 | bridge_and_plan | 1 | 1284.26 |
| llama_s1 | cache_hit | 1.507238e+07 | 0 |
| llama_s1 | cache_miss | 2898 | 0 |
| llama_s1 | derive | 17 | 344.212 |
| llama_s1 | flow | 48593 | 97981.8 |
| llama_s1 | flow_report | 16 | 40.1164 |
| llama_s1 | import | 1 | 31.4547 |
| llama_s1 | instantiate | 17 | 15.4684 |
| llama_s1 | instantiate_and_derive | 48592 | 869424 |
| llama_s1 | materialize | 16 | 753.441 |
| llama_s1 | megakernel_compile | 3 | 159937 |
| llama_s1 | piece_pricing_and_release | 48609 | 1269680 |
| llama_s1 | prepare_relations | 17 | 73194.8 |
| llama_s1 | price_cache_hit | 15600 | 0 |
| llama_s1 | release_cache_hit | 1.777977e+07 | 0 |
| llama_s1 | resource_probe | 49827 | 21172.9 |
| llama_s1 | search_evaluations | 48592 | 0 |
| llama_s1 | search_rounds | 6 | 0 |
| llama_s1 | simulate | 16 | 101.951 |
| llama_s1 | skeleton | 16 | 26054.5 |
| llama_s1 | total | 1 | 2635870 |
| llama_s4 | bridge_and_plan | 1 | 1299.25 |
| llama_s4 | cache_hit | 1.642768e+07 | 0 |
| llama_s4 | cache_miss | 2836 | 0 |
| llama_s4 | derive | 17 | 322.933 |
| llama_s4 | flow | 51945 | 120483 |
| llama_s4 | flow_report | 16 | 62.0438 |
| llama_s4 | import | 1 | 17.5818 |
| llama_s4 | instantiate | 17 | 14.8995 |
| llama_s4 | instantiate_and_derive | 51944 | 460837 |
| llama_s4 | materialize | 16 | 694.434 |
| llama_s4 | megakernel_compile | 3 | 148753 |
| llama_s4 | piece_pricing_and_release | 51961 | 1558910 |
| llama_s4 | prepare_relations | 17 | 87715.1 |
| llama_s4 | price_cache_hit | 15600 | 0 |
| llama_s4 | release_cache_hit | 1.902963e+07 | 0 |
| llama_s4 | resource_probe | 53179 | 21414 |
| llama_s4 | search_evaluations | 51945 | 0 |
| llama_s4 | search_rounds | 6 | 0 |
| llama_s4 | simulate | 16 | 148.961 |
| llama_s4 | skeleton | 16 | 24899.5 |
| llama_s4 | total | 1 | 2533990 |
| llama_s16 | bridge_and_plan | 1 | 914.615 |
| llama_s16 | cache_hit | 1.086669e+07 | 0 |
| llama_s16 | cache_miss | 2836 | 0 |
| llama_s16 | derive | 17 | 321.831 |
| llama_s16 | flow | 37637 | 182465 |
| llama_s16 | flow_report | 16 | 187.769 |
| llama_s16 | import | 1 | 11.1034 |
| llama_s16 | instantiate | 17 | 12.5127 |
| llama_s16 | instantiate_and_derive | 37636 | 319681 |
| llama_s16 | materialize | 16 | 898.329 |
| llama_s16 | megakernel_compile | 3 | 133199 |
| llama_s16 | piece_pricing_and_release | 37653 | 2287370 |
| llama_s16 | prepare_relations | 17 | 85779.6 |
| llama_s16 | price_cache_hit | 12123 | 0 |
| llama_s16 | release_cache_hit | 1.392419e+07 | 0 |
| llama_s16 | resource_probe | 38871 | 21317 |
| llama_s16 | search_evaluations | 37637 | 0 |
| llama_s16 | search_rounds | 4 | 0 |
| llama_s16 | simulate | 16 | 416.589 |
| llama_s16 | skeleton | 16 | 23064.4 |
| llama_s16 | total | 1 | 3220440 |
| qwen3_s1 | bridge_and_plan | 1 | 17556.9 |
| qwen3_s1 | cache_hit | 2.665394e+07 | 0 |
| qwen3_s1 | cache_miss | 2902 | 0 |
| qwen3_s1 | derive | 17 | 393.539 |
| qwen3_s1 | flow | 46158 | 162462 |
| qwen3_s1 | flow_report | 16 | 69.4279 |
| qwen3_s1 | import | 1 | 35.0466 |
| qwen3_s1 | instantiate | 17 | 32.4442 |
| qwen3_s1 | instantiate_and_derive | 46157 | 1539620 |
| qwen3_s1 | materialize | 16 | 1750.54 |
| qwen3_s1 | megakernel_compile | 3 | 149322 |
| qwen3_s1 | piece_pricing_and_release | 46174 | 1548130 |
| qwen3_s1 | prepare_relations | 17 | 1140230 |
| qwen3_s1 | price_cache_hit | 15600 | 0 |
| qwen3_s1 | release_cache_hit | 3.20851e+07 | 0 |
| qwen3_s1 | resource_probe | 47392 | 21498.1 |
| qwen3_s1 | search_evaluations | 46158 | 0 |
| qwen3_s1 | search_rounds | 5 | 0 |
| qwen3_s1 | simulate | 16 | 146.397 |
| qwen3_s1 | skeleton | 16 | 54842.5 |
| qwen3_s1 | total | 1 | 4820870 |
| qwen3_s4 | bridge_and_plan | 1 | 17205.7 |
| qwen3_s4 | cache_hit | 2.515309e+07 | 0 |
| qwen3_s4 | cache_miss | 2840 | 0 |
| qwen3_s4 | derive | 17 | 382.976 |
| qwen3_s4 | flow | 43724 | 165430 |
| qwen3_s4 | flow_report | 16 | 90.0592 |
| qwen3_s4 | import | 1 | 39.7285 |
| qwen3_s4 | instantiate | 17 | 29.9771 |
| qwen3_s4 | instantiate_and_derive | 43723 | 753027 |
| qwen3_s4 | materialize | 16 | 1762.43 |
| qwen3_s4 | megakernel_compile | 3 | 145767 |
| qwen3_s4 | piece_pricing_and_release | 43740 | 2137310 |
| qwen3_s4 | prepare_relations | 17 | 1096900 |
| qwen3_s4 | price_cache_hit | 14013 | 0 |
| qwen3_s4 | release_cache_hit | 3.072209e+07 | 0 |
| qwen3_s4 | resource_probe | 44958 | 21337.9 |
| qwen3_s4 | search_evaluations | 43724 | 0 |
| qwen3_s4 | search_rounds | 5 | 0 |
| qwen3_s4 | simulate | 16 | 258.364 |
| qwen3_s4 | skeleton | 16 | 55744.1 |
| qwen3_s4 | total | 1 | 4582650 |

完整字段与证据路径：[原始数据重算表](report_tables/phases.tsv)。

## 12. θ 网格

同一模型只导入一次，改变 θ 绑定；两起点、完整域。只展示有 completed.tsv 的点；不能把尚未结束的当前最小值记成最终最优。变化区间只由相邻已完成点限定，不宣称区间内的全局最优证明。

| model | seq | seed_seq | evaluations | rounds | wall_seconds | flow_ns | key | dram_ns | compute_ns | floor_ns | flow_over_floor | changed_since_previous | previous_seq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 1 | 1 | 48592 | 6 | 2125.594 | 3410017 | 32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 | 2518402 | 13731.52 | 2518402 | 1.35404 | False |  |
| llama | 2 | 4 | 54686 | 6 | 2307.076 | 3410754 | 32x32x64s3k1;32x16x64s2k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 | 2518701 | 27463.04 | 2518701 | 1.354172 | False | 1 |
| llama | 4 | 4 | 51945 | 6 | 2247.236 | 3413496 | 32x32x64s3k1;32x16x64s3k1;32x16x64s3k1;32x128x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=1 | 2519299 | 54926.08 | 2519299 | 1.354939 | True | 2 |
| llama | 8 | 16 | 47935 | 6 | 2244.738 | 3427308 | 32x32x64s5k1;32x16x64s3k1;32x16x64s3k1;32x64x16s2k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 | 2520494 | 109852.2 | 2520494 | 1.359776 | True | 4 |
| qwen3 | 1 | 1 | 46158 | 5 | 3996.835 | 5126156 | 32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 | 3506496 | 19117.52 | 3506496 | 1.461903 | False |  |
| qwen3 | 2 | 4 | 35752 | 5 | 3300.705 | 5127183 | 32x32x64s3k1;32x16x32s3k1;32x16x64s3k1;32x64x64s3k1;32x16x64s3k1;32x256x16s2k1;kappa=1;residency=2 | 3506927 | 38235.03 | 3506927 | 1.462016 | False | 1 |

完整字段与证据路径：[原始数据重算表](report_tables/theta.tsv)。

## 13. R9 早期信号

使用冻结 R9 库重解两个已选几何，实际驻留度 3；512-worker 原分数不归给 384-worker 二进制。两格并未胜过重新计时的 legacy，不能用 per-class 的潜力解释成已得收益。

| cell | arm | l05_ms | l1_ms | l2_ms | L2_over_floor |
| --- | --- | --- | --- | --- | --- |
| llama_s1 | early_R9 | 8.484096 | 6.254848 | 6.065152 | 2.408333 |
| llama_s4 | early_R9 | 6.971112 | 6.534152 | 7.341056 | 2.913928 |

完整字段与证据路径：[原始数据重算表](report_tables/additional_measurements.tsv)。

## 14. 未达门的定位与下一步

G-3、G-5、G-7 和额外 runtime-release 审计的原因及处理见 §4。研究门按完整八格/六格固定门线判定，未齐时不判成功。

模型低估实测时，先核对 `FlowPreparation.cpp` 的释放律与 `SkeletonFinalize.cpp` 的 requested-event 掩码差异，再对关键链 GEMM/标量的固定项做分组回放。即使 V2 排序通过，也只能证明两种预测相互一致，不能证明其绝对价格正确。`StageFlowModel.cpp` 的 T_floor 断言证明流量守恒下界，不证明上层模型贴近实测。

TaskBody、同步协议、W=1、分页、预取、融合、静态 batch、element_chunk、legacy 六启发式、CUTLASS 均未为本轮优化而修改；K-13 核实受保护源码。新流体模拟与 regime_a 均有关闭路径的逐位对照。无通过缩小矩阵、降低正确性次数或改变门线取得的通过项。

## 15. R10 输入与 θ 通用性的边界

PG 上界按每格 `T − max(T_floor,T_np0)` 计算，属于模型上界。协议常数一栏是已识别关键链上的 wait+publication+hop 之和；与 T−T_s 的反事实差值可能不同，因为去掉同步会改变调度和关键链，不把二者等同。

| cell | pg_upper_ns | protocol_on_chain_ns | attention_chain_ns | attention_over_flow | sync_ns | fixed_ns | contention_ns | chain_ns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | 891614.8 | 553073.8 | 65725.54 | 0.01927426 | 554821.1 | 301966.4 | 0 | 34827.3 |
| llama_s4 | 894196.8 | 553073.8 | 66815.09 | 0.0195738 | 554695.9 | 301643.9 | 0 | 37856.98 |
| qwen3_s1 | 1619659 | 965341.5 | 117801.7 | 0.02298052 | 967105.8 | 545912.7 | 0 | 106640.9 |

完整字段与证据路径：[原始数据重算表](report_tables/decomposition.tsv)。
| cell | category | links | potentially_fusible | wait_ns | publication_ns | hop_ns | fixed_ns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama_s1 | attention | 16 | False | 28229.89 | 17186.05 | 0 | 19796.77 |
| llama_s1 | embedding | 1 | False | 0 | 1074.128 | 0 | 1046.657 |
| llama_s1 | gemm | 64 | False | 112919.6 | 68744.22 | 79066.36 | 157865.1 |
| llama_s1 | lm_head | 1 | False | 0 | 0 | 1235.412 | 4501.442 |
| llama_s1 | rmsnorm | 33 | True | 56459.78 | 35446.24 | 39533.18 | 72819.69 |
| llama_s1 | rope | 16 | True | 28229.89 | 0 | 19766.59 | 13439.07 |
| llama_s1 | swiglu | 16 | True | 28229.89 | 17186.05 | 19766.59 | 19951.89 |
| llama_s4 | attention | 16 | False | 28229.89 | 17186.05 | 0 | 19796.77 |
| llama_s4 | embedding | 1 | False | 0 | 1074.128 | 0 | 1046.657 |
| llama_s4 | gemm | 64 | False | 112919.6 | 68744.22 | 79066.36 | 157865.1 |
| llama_s4 | lm_head | 1 | False | 0 | 0 | 1235.412 | 4501.442 |
| llama_s4 | rmsnorm | 33 | True | 56459.78 | 35446.24 | 39533.18 | 72819.69 |
| llama_s4 | rope | 16 | True | 28229.89 | 0 | 19766.59 | 13439.07 |
| llama_s4 | swiglu | 16 | True | 28229.89 | 17186.05 | 19766.59 | 19951.89 |
| qwen3_s1 | attention | 28 | False | 49402.31 | 30075.6 | 0 | 34734.09 |
| qwen3_s1 | embedding | 1 | False | 0 | 1074.128 | 0 | 1046.657 |
| qwen3_s1 | gemm | 112 | False | 197609.2 | 120302.4 | 138366.1 | 258733.4 |
| qwen3_s1 | lm_head | 1 | False | 0 | 0 | 1235.412 | 4501.442 |
| qwen3_s1 | rmsnorm | 85 | True | 148206.9 | 61225.32 | 103774.6 | 184873.6 |
| qwen3_s1 | rope | 28 | True | 0 | 0 | 0 | 23608.11 |
| qwen3_s1 | swiglu | 28 | True | 49402.31 | 30075.6 | 34591.53 | 32044.01 |

完整字段与证据路径：[原始数据重算表](report_tables/fusion_chain.tsv)。

可融合节数是所列类别的候选机会，不是融合合法性证明；已折叠的 residual add 不重复计入。attention 的已实现路径份额不包括它可能造成的所有队列后果，不能仅凭这一个比例排除 AT-1。

下一步顺序：先补 Level 1 的真实事件窗口及绝对误差，随后按完成矩阵的 PG/Fuse/协议项上界分配 R10 工程投入；AT-1、静态 batch 与完整请求执行仍按用户规定范围推进。硬件后端或执行器变化需要独立正确性与性能门。

闭式读写计数、分片与 Oracle 接收 θ 绑定，未发现新求解价格/放置按某个 seq 数值或模型名挑分支。仍有两个接入 batch 前必须处理的边界：`ModelDims`（`include/tilemega/Solver/ModelDescription.h`）的生产接口目前显式承载 seq/past/total，没有 batch 字段；embedding 的精确唯一读像依赖 token 值，需要额外 distinct-token/间接像契约，不能称为仅 θ 的闭式量。实验脚本的 seq2/8/32 legacy 起点选择是唯一已声明的 seq 特判，未进入生产定价或放置代码。
