# MX-5 local LDS synchronization checkpoint

Authority: original MX5 prompt SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
User amendment: permit already-probed LDS-only primitives for pure local LDS
synchronization when pursuing TPOT, with the difference recorded. Global DT-P
remains P-B. Other original protocols, numerical rules and gates remain.

## Purpose and boundary

Inferred purpose: avoid unnecessarily retiring global prefetch loads when a
CTA only needs to publish/reuse LDS. The former restriction was the strict
implementation rule in section0.2 combined with P-B's synchronization scope;
it was not a hardware prohibition. This amendment does not reclassify DT-P
as P-A or remove global publication boundaries.

- `0b74c686c`: replace attention wave-result LDS publication with compiler
  barriers, `arrive_bsmcnt(0)` and `barrier_inst`, joined by every wave.
- `7c9658ac5`: additionally use that primitive for segment-tail LDS workspace
  reuse. Run-tail stdCTA, StageBarrier and all LA/global release/acquire remain.
- Source-level HB review: next segments use different heads; same-head global
  consumers remain after LA release/acquire or retained Run/grid boundaries.
  In particular, the earlier LA release does not publish later ctx writes.

## Actual validation (verified)

The first candidate passes5fresh×56 D128/Q2 P0 attention cases and the actual
helper litmus5fresh×6 cases(512/576×N0/2/8KiB)×2000iterations×104CTAs,0errors.
The second passes the same pilots/litmus independently. Its full required
attention matrix also passes5 distinct fresh processes×336 cases,0skips, plus
15 MX4 transform bitwise cases per process. Each process first checks actual
336 same-source P0 queries,24 selected unit kernels and6 helper kernels;
selected kernels have API0,frame0,static-shared0 andresident1.

Both candidates pass the manual Qwen0.6B B2 C-1/C2 checks at512 decode steps;
maxgap=.03125 andC2mismatches=0. These do not replace final top4 selections.

| Same-card7 P-PAIR | Old median ms | New median ms | Change | Distinguishable |
|---|---:|---:|---:|---|
| Row2678 → initial LDS publication |2.304253|2.307628|+0.146%|No|
| Initial publication → segment-tail reuse |2.309219|2.302293|−0.300%|No|

Each arm has2 fresh processes,warm1/timed2,ISL64/OSL1024,BF16 greedy.
No TPOT gain or default promotion is claimed. Both comparisons retain raw
rounds and their own old arm; do not compare their absolute medians as one pair.
The original MX4 B2 TPOT remains2.095492ms; the manual new result still exceeds it.

Normal source88327 producer is freshly built160/160 and59delivery files are
rehashed. New A19 isdd8abe75a9d5037649d4b7bfa61f9570a4c41c4d8cff8d3e3a467aa19d7fb3dd.
G/H transitive source bytes remain identical to the archived qualified pool.
Native B2 512 WD0/WD1 each has231MT/frame0 andactualresident1. Its576floor
control has168MT/frame360 andis queried only,never numerically executed.
Stagebench rejection does not authorize blanket whole-program512 fallback.
The independent28-kernel comparison confirms both local sites in MIR; steady
MMA loops have no4160/generic access. Some whole-kernel4160 counts remain.

## Cost lineage and remaining work

Verified old-source Row2678/A19cc122 attention400/400 has3fresh measurements,
with25 scoped512 consumer replays from4545G/H/A keys. These costs were never
blocked by local-LDS permission; the earlier192/400 checkpoint reflected guard75.
New-source400 P0/primary resource queries pass independently. New cold costs
complete400/400,3fresh per key and256MiB active rotation. All25 cost scopes
replay successfully from the new4545-key G/H/A pool. Old A rows are
not relabelled as new A. Neither25 cost scopes nor336 unit passes approve all
whole programs,final solver choices or defaults.

Raw roots: `runs/mx5/local_lds_publication_candidate_v1/`,
`local_lds_reuse_tail_candidate_v2/`, `local_lds_tail_required336_root_t512_v1/`,
`local_lds_tail_costs400_v1/`, `row2678_complete400_qualified25_t512_v3/`.

Verified storage maintenance:40 exact duplicate sealed graph exports now
share physical storage after SHA256 checks; all paths/content remain. About
20.687GiB reclaimed. No measurement,failed attempt or source archive removed.

Dense final25,MoE native5,LLM30 acceptance,andDNN native25 remain unfinished;
original phase order and60card-hour budget remain. The measured local-LDS
comparisons do not establish the hoped-for TPOT improvement.

Verified576 context counterexample (same new A19): Llama1B B1 normal
G/H-floor,attentionPK2 program has167MT/frame0 andactualresident1,API0.
The runtime-shape self-prefetch stagebench has PK0=137–138MT/frame0,
PK2=168MT/frame44–48. Actual queries give PK0fit4/4 andPK2reject4/4,
with0numerical launches. Section4.5 choosesPK2 fromPK0 Rnext, so that
576stage-cost item remains blocked; forcing the whole plan to512 or using
PK0 costs would violate the original rule. No single compiler cause is proved.
Locations: experiments/mx5/attention_stage_bench.cu:53 (self-next context),
DecodeAttentionV2.h:432 (common Run);raw local_lds_tail_576_context_actual_queries_v1.

Verified native solver continuation: new-source25 requests each emit an exact
independent512 top4 preview,100 candidates,0device compiles/0GPU calls. These
are predictions, not final selections or512 exceptions. Five development
requests'20 original canonical576 programs are rebuilt normally from their
actual576 P0 profiles:3 haveframe0 and17 retainframe failures. Guarded
zero-launch queries andprescribed history consumption follow;no missing retry
history or D7 evidence is filled in. The source-specific query log in the
frozen tree is added to the physical-card accounting;all prior logs andthe
60h limit are retained.

Verified first new-source exacttop4 E2E selection: Qwen0.6B B1 allfour
candidates passC1/C2512,two fresh timing processes each. Medians(ms) are
[1.976266,1.977535,1.980171,1.974518]. Anchor0 is retained by the1% rule;
pred/meas1.02291. FixedMX4 is1.983552ms(−.367%,indistinguishable). This
passes cell nonregression without claiming a measurable speedup. Current
source measurements subsequently reach5/25dense(4nonregression,1failure),0/5MoE,0/25DNN;defaults unchanged.

Verified second new-source selection: Qwen0.6B B16 allfour C-1/C2512
andtwo fresh timings each pass. Candidate medians are
[3.524260,3.523996,3.528153,3.529580]ms;anchor0 remains selected.
Pred/meas1.00856;fixedMX44.537082ms gives−22.323%,fastestbaseline4.0203ms
givesratio.876616. This is an actualsolver result,not the manual LDS P-PAIR.
Its 4selection/4correctness/3serving/8resource rows append without changing
prior bytes. Raw:local_lds_tail_qwen3_0_6b_B16_top4_execution_v1.

Verified development history closure:20canonical576 originals plus bounded
rebuilds have46actual zero-launch queries;16terminal histories support
independent512,D7 reviewed;4Llama1B outcomes retain576. Remaining20cells'
80canonical576 originals complete onCPU;all100originals'files/MIR bind
the normal same-source producer.13haveframe0;actual queries remain separate.

Verified third new-source selection: Llama8B B1 anchor0 TPOT10.585899ms,
fixedMX410.715531ms(-1.210%,distinguishable),pred/meas0.98267.
Allfour C-1/C2512 and2fresh timings each pass;strict source/raw verification
precedes append-only CSV publication. Raw:local_lds_tail_llama3_1_8b_B1_top4_execution_v1.

Verified budget-controller configuration: the60physical-card-hour cap andall
four raw job logs remain. The original58.7dispatch cutoff reserves time for
two long checks/timings. Only after allcurrent timing parents return0,and
allactive guards have timeout<=300s,does the query-only phase use59.75.
Query controllers additionally bound each whole wrapper to330s,including
preflight;cleanup20s pluspoll30s percard gives a two-card upper59.9612h.
At cutoff these two controllers keep their outer deadlines live;their explicit
budget/flag checks reject the next GPU dispatch. They are not suspended while
waiting on a guard. Other owned dispatch loops may still be suspended.
The initial300s-only reserve estimate is superseded before anyquery dispatch;
guard's own timeout starts after preflight. Original estimate/correction remain.
This changes an internal reserve,not the prompt budget ormeasurement protocol.
Original controllers/logs remain;only owned idle CPU waiters were replaced.

Verified fourth new-source selection: Llama8B B16 anchor0 TPOT13.150235ms,
fixedMX414.541197ms(−9.566%,distinguishable),/fastestbaseline.932112,
pred/meas.99287.Allfour C-1/C2512 and2fresh timings each pass;CSV/raw
independently reconciled. Fourdevelopment cells passnonregression;Llama1B
576cost item,G-B,G-PROGRAM,full25/30LLM andMoE/DNN remain open.

Verified remaining20cells'80canonical576 original queries complete,API0,
zero numerical launches. Outcomes:10use576,13qualified history,57prescribed
rebuilds. Together with the earlier20development originals,all100originals
have actual queries. The new80resource rows append with prior bytes preserved.
All57first-retry CPU builds complete normally;they overlap independent query0
work. A real reuse check preserves binary/receipt/MIR/process hashes exactly;
the later controller validates those receipts instead of recompiling them.
Neither these histories nor CPU rebuilds add E2E-passed cells. Raw:
local_lds_tail_canonical576_remaining80_v1;currentsource selection remains4/25.

Verified all100canonical histories reach prescribed terminal outcomes:84qualified
histories,16retained576 routes.207actual canonical resource-query rows now
exist inCSV;maximum3additional reductions percase,0numerical launches.
The remaining80 use80original+57first+18second+6third queries.Only thelast24
share4guard processes;each uses the unchanged frozen occupancy helper and
original per-item validator,with real raw runtime attributes andDLaddr DSO
binding. Sharedprocess/PID/guard evidence is explicit;no fresh-process claim.
A local CSV-variable shadowing error stopped export before anyappend;v4
renames that checker only,thenverifies161closedqueries/appends81newrows.
Original failed script/logs andalloldCSVbytes remain. Raw:CORRECTED_RESOURCE_CSV_EXPORT_SUCCESS.json.

Verified QwenB2 continuation:four independent512 originals haveactualresident1
andzero-frame queries,normal old-source before controls andfourREADY D7
reviews. Allfour512-step C-1/C2 checks pass;8fresh timings subsequently complete.
Actual runner source isrunner_bounded.py,withitsownhash/manifest andfull diff
fromthefrozenrunner. Only bootstrap/dispatch-budget checks andguard deadlines
change:300schild/330stotal;ISL64/OSL1024,warm1/timed2,2fresh peroriginal
candidate,allnumericalvalidators and1%selection remain. No old-runner identity
is claimed. Once the resource controller closes andonlythat live,source-bound
GPU6controller remains,dispatchcutoff59.88 reserves330+20cleanup+30poll
seconds:upper59.9856card-hours. Two-card cutoff remains59.75;cap remains60.

Verified Qwen0.6B B2 completed exacttop4,allC1/C2512 and2fresh timings each.

| Candidate | Predicted ms | Actual ms | pred/meas | Chosen |
|---|---:|---:|---:|---|
|0|2.150885|2.301333|0.9346|yes|
|1|2.151145|2.304746|0.9334|no|
|2|2.151245|2.299368|0.9356|no|
|3|2.151515|2.308777|0.9319|no|

Anchor0 TPOT2.301333ms vsfixedMX42.095492ms
(+9.823%,distinguishable regression). /FastestBF16baseline
=0.821259;TTFT11.334ms,E2E2.365598s. Performancegatefails;defaultsunchanged.
Actual performanceD7:WD0/WD1 each231MT/frame0/staticshared0,resident1;
identified steady rings have no4160/generic/calls. Prefetch andP-B issuepoints
remain source-bound;attention bmin4 givesGA12..104 andcross-head CTA0..12
(source geometry,not site latency). Matchingbenchprediction2.150885ms leaves
.150448ms unassigned. Current per-stage programDIAG andI1 timing areunmeasured;
no causal latency split orcache explanation isclaimed. Raw:D7_PERFORMANCE_FOLLOWUP_v1.json
inlocal_lds_tail_qwenB2_bounded_top4_v1. This is an unresolved regression,
not an approved default orcompleted G-LLM gate.


Verified remaining15-cell resource preparation:60 normal independent512 original
builds,60 normal old-source before controls,and60 actual zero-launch resource
queries complete. Allactual queries report223–231MT,frame0,staticshared0,
resident1. The unchanged frozen helper andoriginal per-item validator run in
8guarded processes oncard7;each DSO is bound bydladdr andsource/binary hashes.
SharedPID/guard evidence is explicit;no fresh-numerical-process claim.
Eachguard has30schild/60stotal deadline. All60 source-repair D7 reviews pass
unchanged audit_d7_v2;none claims a mathematical >227 register lowerbound.
60rows append toMX5_resources_c550.csv withallprior bytes preserved.
The15exacttop4 protocols allpass unchanged CPU capture/validation;C-1/C2 andE2E
remainunrun for these15cells. Current numerical selection remains5/25,including
QwenB2's +9.823%performance failure. Defaults remain unchanged.
Raw:local_lds_tail_remaining15_cells_cpu_v1/{CPU60_COMPLETE.json,
ACTUAL60_QUERIES_COMPLETE.json,ACTUAL60_D7_CPU_PROCESSES.json,
D7_PROTOCOLS_COMPLETE.json,ACTUAL512_RESOURCE_CSV_APPEND_1793177286362.json}.

Verified physical-card interval union:59.895/60hours after allGPUjobs close.
Allfour original job logs andclosed-interval proofs remain. The final8guards
reserve60+20cleanup+30poll seconds foronecard;cutoff59.96 givesupper59.99056h.
Only short zero-launch queries used thispost-numerical reserve. No new long
correctness/timing job is started;mandatory missing scope staysincomplete.


Stated user amendment(2026-10-28):original60total-card-hour cap iswaived;
allper-job deadlines,allowedcards,concurrency andguard rules remain.
Verified cumulative-accounting monitorv9 preserves allfour original logs and
prior cutoff evidence;no counterreset. At amendment59.895card-hours hadaccrued.
Remaining15-cell fulltop4 queue uses the original frozen runner unchanged oncard6.
B2 selected-pin DIAG preparation uses the same producer,source andeffective pin;
onlyDIAG=1 changes. No new correctness orperformance outcome isclaimed yet.


### 解除总卡时限制后的实测续跑

Verified：新增Qwen0.6B B4完整四候选C-1/C2及各两新进程计时，选中0：
TPOT2.575954ms，固定MX42.394211ms，慢7.591%，仍判G-LLM失败。
四候选TPOT为2.575954/2.578473/2.579156/2.577437ms；pred/meas0.95646。
当前源码累计6/25格，4格不退化、2格退化；默认值不变。其余队列继续，未完成格不发布测量。

Verified：B2已选计划的DIAG双胞胎仅改变DIAG=1，源码与有效坐标一致，
frame0、实际resident1、稳态MIR无4160/generic；16新进程完成两轮8mask。
attention干预差值按28层除得35.923/35.842us，bench为27.167us；不是可相加的体耗时。
12行带轮次、实现与原始引用的账本已追加，原行字节保持。当前I1仍未测量。

Verified：同源同坐标的37个长度×4个b_min×3新进程冷测量完成，所有原正确性断言保留。
原5格点复测变化0.202–0.715%；b_min4的32-token块中点加权估计为33.674us，
原5点线性插值为27.167us。Inferred：用中点估计替换诊断模型的attention项得2.333086ms，
接近实际2.301333ms；这解释了大部分预测偏差，但不是1023个长度逐一实测的精确均值。
其他b_min的中点估计(us)：1=32.040、2=33.989、8=39.179。
官方成本表、原5点插值公式、候选选择及默认值均未改。

Verified：G576失败点的插桩副本输出与原副本完整逐位一致，两次均仍只有(13,4622)失败。
实际gate累加0.249793708、inverse9.789322853，归一化gate2.445311308；
CPU归一化gate2.445315361。BF16中点2.4453125的两侧距离分别为−1.192e−6/+2.861e−6，
舍入gate分别2.4375/2.453125，最终21.75/22.125；原误差0.375>原容差0.374000013不变。
这将放大位置定位到规定的BF16/SwiGLU舍入链，不宣称两个重复就证明无竞态。
插桩只在诊断拷贝写未使用partial，未改变生产代码；原BINDING的before-source列表保留，
真实after-source闭包单列SOURCE_CLOSURE_ADDENDUM.json，仅GEMM头多诊断写出。

Verified：三个Llama1B bench参数生命周期/容量CPU诊断均仍有PK2栈帧，全部未采用。
第一个attention合并系数复用候选通过5新进程×56项及5新进程×6项×2000×104CTA litmus，
但同源37长度微基准慢4.599%；不继续该候选，不作端到端收益判断。
第二个候选把交换限制为wave内，无新增CTA屏障；隔离验证中，未进入正式计划。
Raw：local_lds_tail_remaining15_full_execution_v1；local_lds_tail_selected_qwenB2_D7_diag_v1；
local_lds_tail_qwenB2_length_geometry_d7_v1；gemm576_failed_coordinate_intermediates_v1；
attention_{merge_scales_once,wave_merge_factors}_candidate_v1。


Verified：Qwen0.6B B8完整top4完成，四候选TPOT为
2.853298/2.810956/2.859982/2.858719ms；候选1比锚点0快1.484%，按原1%规则选1。
选中TPOT2.810956ms，相对固定MX43.040548ms快7.551%；四候选C1/C2512及各两新进程均过。
原冻结执行器与独立消费者复核后才追加CSV；当前源码7/25测量、5不退化、2失败，默认不变。

Verified：wave内合并系数候选在7号卡guard被running interference拒绝；原部分日志保留，
不把已完成的四进程计为整组通过。5号卡重新完成5新进程×56数值项与5新进程完整helper litmus。
同卡5按old/new、new/old、old/new各3新进程对照，32块中点加权微基准old33.655us/new36.225us，
新候选慢7.636%；不继续此候选，未测生产TPOT、不作端到端性能结论。
Raw：attention_wave_merge_factors_candidate_v1/{checks_retry_card5_v1,B2_offgrid37_card5_pair}。

Verified：另一个仅CPU的inline operands按值传递诊断仍有PK2的44–48B栈帧，未采用。
后续候选只把既有合法geometry正数域告知编译器，Owner闭式公式、划分、布局和同步不变；
它没有固定新的运行时形状，也没有改寄存器限制。CPU编译中D64PK2仍有32–40B帧；
512数值与同卡微基准验证中，尚未采用。Raw：attention_geometry_domain_candidate_v1。

Verified（2026-10-28）：当前冻结 A19 源新增 Qwen1.7B B1/B16 正式 top-4 选择，
TPOT 为 3.270282/5.185553ms，较固定 MX-4 分别下降 4.510/17.066%。四候选各自完成
C-1/C-2 512步及双新进程计时；独立结果验证通过后追加四份 CSV。当前9/25测量，
7格不退化，Qwen0.6B B2/B4仍失败。原始基线未改，默认保持 MX-4。

Verified：geometry 正数域候选的5新进程数值/helper litmus通过，但同卡三新进程微基准
old33.715572us/new33.772014us（+0.1674%），不可分辨，未采用。packed-pair调度候选
也通过对应数值/helper测试，但微基准old33.693675us/new39.551975us（+17.3869%），未采用。
后者最初按资源日志行号误判 Norm=true 寄存器数；已按完整符号纠正，错误记录保留。
微基准加权以 ceil(L/32) 对应32块中点；旧 floor 加权计算记录被v3明确更正，原记录未删。
这些都是定位结果，没有生产 TPOT 性能结论。
Raw：{attention_geometry_domain,row_transform_packed_pair_schedule}_candidate_v1/B2_offgrid37_card5_pair/ACTUAL_D7_COMPARISON_v3.json。

Verified：新的 consumed-prefetch 候选在第一次 head 段消费后清空传入负载，并只对本 wave
真实拥有的首段块发出预取。CPU 中576/D64/Q4/PK2自预取体降为139/141MT、栈帧0，
R_cap220/248、预取公式、划分、算术、P-B与同步协议均未改变。完整336数值、空指针闲置
wave/CTA用例与5新进程helper litmus正在跑，尚未进入主源或实测成本表。
第一次包装器错误地把固定576线程的TransformKernel按512核对，数值启动前停止；
已改为按真实576/512 launch bounds核对全部六个helper内核，失败日志保留。
Raw：attention_consumed_prefetch_guard_candidate_v2；R_cap160提案未应用，当前验证无需该变更。

Verified：consumed-prefetch 修正已收齐验证并提交为4b7d83686，回归测试为d889bd292。
完整336 attention项、576/PK2空指针28项、实际helper六项以及576/PK2冷自预取12键，
每组都完成5新进程；helper每项104CTA×2000迭代、0违例，自预取每键≥2000迭代、
活跃冷工作集≥256MiB，原数值/KV/callback/LA/屏障/确定性判据未改。
实际API查询均先于对应数值启动，所选内核frame0/static0/resident1；PK2 bench为139/141MT。
已追加20组正确性与130条实际资源查询，标记真实候选源码/构建HEAD/轮次，不冒充正常新producer。
新提交的Python空指针回归入口另实际运行28项及15变换项，2测试、0skip通过；CPU无GPU发现
测试的1skip仅是环境检查，没有计入上述数值通过。R_cap220/248和预取公式保持不变。
正常新冻结producer正在全量160步构建；之后新源码800个P0/primary实际查询与合法成本冷测量。
512/576成本按独立新进程分组，原mixed CPU清单不执行；所有旧A19成本/选择仍保留旧身份。
当前主源新体没有端到端选择，9格结果仍属A19；不作TPOT改善或默认提升声明。

Verified：d889正常冻结producer230a572be9c7d98d0a085f1a91cb39e9ddf8e44e769c2ecc06a4909d7210abaa
完成160/160新构建、C++单测、源码指纹和normal选项一致性检查。A20闭包为b4555ea2e0dbde380c37c48793e2fd5ff4a9f6dc767b0ebb9606e108c4fd34b3；G14/H15闭包原样保持。
18个同源正常bench编译完成；512的400次实际P0查询API0、全部资源合法，576查询进行中。
Llama1B B1同几何诊断WD0/WD1各165MT/frame0、各14干净稳态环，WD0实际resident1；
该控制的设备检查和计时独立记录，不能替代top4终版。原MIR包装器误把host/helper函数也
计作执行kernel，CPU即停止；v2按执行kernel完整符号筛选，原失败保留，未改MIR判据。

Verified：A19又完成Qwen1.7B B2四候选C1/C2512与每候选两新进程计时，
3.610137/3.609771/3.608008/3.610523ms，按1%规则选锚点0：3.610137ms。
相对固定MX43.555443ms慢1.538%，该格G-LLM失败；对最快基线4.2658ms的比值约0.8463。
独立执行结果复核后追加原行；A19合计10/25、7不退化、3失败，A20仍无端到端选择。

Verified：新源Llama1B B1人工同几何控制两新进程TPOT2.134865/2.135300ms，
中位数2.135083ms，较固定MX42.211073ms快3.437%；TTFT分别9.045/9.112ms。
WD0/WD1均165MT/frame0/static0/actualresident1，512步C1/C2通过；计时WD0、trace0、step-events0，
ISL64/OSL1024、greedyBF16、warm1/timed2。追加3计时/1正确性/2资源行，标记manual_control；
不是终版top4、不是同卡旧/新P-PAIR，也不证明修正本身带来的TPOT幅度。
Raw：consumed_prefetch_llama1b_selected_control_v1/ACTUAL_MANUAL_CSV_APPEND_PROOF.json。

Verified：新源800 P0及800规则派生实际查询完成：512的400项合法、576的100项合法，
其余576的300项资源拒绝且未运行数值/计时。21个正常bench编译与所选MIR通过；
500合法键按线程分组进行原三新进程冷测量。25个512scope/source/dependency绑定已准备，
最终成本资格重放只在全部实际批次成功后执行，零补造、零旧A身份转移。

Verified：新A20 Llama1B B1的576请求域189=156实测成本+33实际资源拒绝，全部原成功批次
先按原aggregate数值/资源/normal门核验，再精确投影；闭包字节核验通过，原complete-outcome
判据与消费者重放均过。正常producer已生成真实exact top4，四原始576计划编译、MIR与实际
occupancy通过，0宽度缩减。正式runner执行中，尚无新源solver-selected结果。
旧的CPU包装器把measured_args=None误当已传成本，被原验证器正确报缺156；实际156行聚合
证据保留，v2传入明确的投影成本对象后才通过。未补数据或放宽判据。
第一次四query用主root wrapper，helper路径不等于可信冻结W而被协议拒绝；冻结W原wrapper
的新查询才被消费。另一次启动误传schema1，被只接受schema2的runner在GPU前拒绝；现用
既有v2 capture/validator重新生成，原四候选/ledgers/queries不变。全部失败记录保留。

Verified：A19 Qwen1.7B B4完整top4选择3.920668ms，较MX43.894393ms慢0.675%，不可分辨。
A19共11/25、8不退化、3失败；旧队列完成6/15格后，原磁盘<1GiB门停止下一格，9格未启动。
只终止已无后续工作可消费的自有CPU watcher，未信号GPU进程；新A20工作继续。
自有已完成host构建的无引用CPU .o/.a中间文件逐字节归档到/dev/shm，并在原位置保留可读
中间文件链接。原测量、日志、CU、源码、producer/solver/plan二进制和身份文件不变。
初次归档因旧producer缺DELIVERY中止；部分归档与恢复路径有独立manifest，没有删除证据。
新producer归档67,685,508B，中间文件与最终producer哈希复核；磁盘仍受持续监控。
Raw：cpu_intermediate_archive_tmpfs_v1；local_lds_tail_remaining15_full_execution_v1/ACTUAL_OLD_SOURCE_QUEUE_DISK_STOP_AND_CPU_WATCHER_HANDOFF.json。

Verified：Qwen0.6B B1新源512请求域174坐标全部有实测成本，原消费者重放通过。
已独立求解top4并按真实576 P0重新导出四个canonical原始构建：MT168，frame340/356/360/356。
实际查询及至多三次原规则缩减在卡6继续；没有直接强制512，也未转移旧A19的资源历史身份。

Verified容量更正：df显示0的是f_bavail（普通用户可用空间），并非文件系统f_bfree；
实际UID0仍有38,062,530,560B保留空闲空间，df0之后的正式批次亦成功写入。没有发生已观察
ENOSPC；未修改保留块配置。旧CPU队列使用f_bavail的1GiB守卫停止记录保留，后续工作按
实际身份和两种容量指标继续监控。仅只读比较的大edge TSV并不逐字节相同，未做原始数据去重。
Raw：storage_capacity_audit_v1；/dev/shm/tilemega_mx5_edge_dedup_audit_20261028/BYTE_COMPARE.json。

Verified：A20的500/500合法attention键三新进程冷测量全部完成；576的300拒绝仍未计时。
与逐字节一致的G14/H15组成4545实测行，经原资格工具及25个完整512请求域消费者重放全过。
25个新正常producer预览正在生成。旧早期QwenB1请求域预览具有独立身份并继续自己的真实
576历史、512原始构建、D7和端到端流程，不把其历史移给不同preview SHA。
Raw：consumed_prefetch_attention_epoch_v1/ACTUAL_COLD_THREE_FRESH_COMPLETE.json；
consumed_prefetch_attention_consumer25_v1/ACTUAL_QUALIFIED25_COMPLETE.json。

Verified：新QwenB1四条canonical576历史闭合，缩减次数1/1/0/0，最终各GEMM/head均Pw=D；
相应四个独立512原始计划各229MT/frame0/actualresident1，实际query与共享预算复核通过。
另用旧A19正常producer编译相同当前minimum物理pin作为source-before控制，真实新after是
当前canonical失败体。四份D7逐项绑定spill/wait/Pw/发出点/划分/I1静态及原I1证据，原v2审核
全过，正式runner在卡6进行四候选C1/C2与双新进程计时；当前仍未登记新源选择结果。

Verified：A20首个真正求解器选择完成并独立复核入库：Llama1B B1，576线程，原始候选0。
四候选TPOT2.135062/2.135004/2.133439/2.133940ms；最快未超过锚点0的1%阈值，选0。
对固定MX42.211073ms快3.438%，pred/meas约1.01325。四C1/C2512和每候选两新进程均过；
追加4选择/3计时/4正确性/8资源行，真实A20/source/producer/round独立标记，旧A19行前缀保留。
当前A20为1/25，不能与A19历史11格合并为12个当前格；默认仍MX4，全部门仍未关闭。
Raw：consumed_prefetch_llama1b576_full_top4_execution_v2/execution/ACTUAL_ALL_RESULT_CSV_APPEND_PROOF_v2.json。


## Verified A20 continuation and watchdog resource stop (2026-10-28)

The user waived only cumulative60GPU-hours; card/concurrency/guard/per-job limits
remain. Current A20 has two independently verified original-top4 selections:
Llama1B B1=2.135062ms(−3.438%vsMX4),Qwen0.6B B1=1.973800ms(−.492%,indistinguishable).
Each original4 passes C1/C2512 and8fresh timing processes. Fixed baselines remain
byte-identical. Commitd52857c4f is pushed; neither cell changes defaults.

The remaining80 canonical576 originals complete normal compilation. Llama1B
B2/B4/B8/B16 have independent real576 scopes,original4 normal builds and actual
resource queries; their512 origin/ranks/predictions are not transferred.
B2/B4/B8 original0 watchdog builds fail frames16/24/56B,MT168;timing originals fit.
The unchanged runner stops at mx5_resolved_runner.py:266 before numerical launch.
B16 watchdog originals fit and actual C1/C2 proceeds. These are not four accepted cells.

An isolated opaque barrier-step diagnostic gives B2frame0,B4frame20,B8frame56.
Nine further CPU builds retain all outcomes;three WD0 device images differ from
originals,so no byte-equivalence or end-to-end improvement is claimed. An isolated
lazy watchdog-record diagnostic retains16/24/56B. Both remain outside main source.
The unchanged native producer now checks the predeclared largest-P_w halving
sequence,maximum3 per diagnostic,without changing geometry,PK,threads orD.
Those are diagnostic receipts,not an approved retry history,new costs or acceptance.
Raw:watchdog_iteration_lifetime_d7_v1;watchdog_lazy_record_d7_v1.

The first generalized D7 wrapper used an assessment string outside the original
validator's exact enum and was rejected for all4 QwenB16 reviews. The actual
168MT+frame observations were already correct. Separate v2 reports use the
original supported literal;the unchanged validator still asserts168MT+positive
frame and the shared budget. Old rejected audits/logs remain. No check was
weakened. Raw:consumed_prefetch_development512_handoff_v1/ACTUAL_D7_WRAPPER_LITERAL_CORRECTION.json.

The diagnostic resource CSV appender's first file failed Python parsing before
any write (missing whitespace around numeric conditional literals). A separate
compiled v2 appends only3 actual rejected normal resources and9 CPU diagnostics,
with an old-prefix proof. No measurement row was backfilled or edited.


## Verified WD0/WD1 resource-pair continuation (2026-10-28)

The normal timing originals fit while watchdog originals B2/B4/B8 fail. The first
isolated source repairs are rejected;no kernel/header change is applied. Instead,
the continuation follows §4.5's existing largest-P_w halving branch for the pair:
same original candidate,geometry,threads576,PK,D andpositive P0-derivedR_next;
both instrumentations must meet MT≤227,frame/static0 andactual resident1.

Twelve original candidates use respective reduction counts:
B2[2,2,1,0],B4[1,0,0,1],B8[1,1,0,0]. This is9 prescribed reductions in total,
never more than2 per candidate. All24 terminal actual zero-launch queries pass
API0/resident1. Threefresh costs cover GU/Pw2 atB2/B4/B8 andheadTN32/Pw4/D2,D3 atB2.
The unchanged frozen stage qualifier and VariantCatalog check raw rounds,P0 resources,
source archives,normal commands,binary IDs andexact width keys. The catalog remains
separate from the baseline stage table. Actual all12history/cost/query replay passes.

The new CPU tool replay verifies the complete contiguous failure/reduction chain;
it cannot grant512,reset a budget,change a coordinate or invent a cost. The separate
runner uses an explicit paired protocol,retains the exact original top4/native solve,
and reuses the frozen identity,guard,C1/C2,timing and1% selection checks. It does
not disguise the new proof as a v2 protocol or weaken the original v2 runner.
Committed tools:experiments/mx5/watchdog_pairs.py andwatchdog_pair_runner.py;
23 CPU tests pass without skips. B2/B4/B8 actual C1/C2 and8timings/cell are in progress.
No endpoint performance/default acceptance is inferred from the resource repair.

Raw:watchdog_pair_policy_validation_v1/qualified_pair_histories_v4,
actual_terminal24_queries,changed_width_costs andprotocols. Earlier CPU replay
wrappers failed on a cleaned query-summary field,missing source inventory and
missing P0 binary references. Their rejected artifacts/logs remain;the original
validators were not changed. Separate v4 includes the real missing evidence.
An initial relative child-script path failed in the shell;the first queued guard
later found the created child and executed the5queries successfully. The redundant
second child refused the existing raw output with FileExists;both guards are kept.
The later ACTUAL_QUERY_ORDER_AUDIT corrects the earlier intended conditional wording.

The remaining512 dispatcher excludes Llama1B's four remaining batches because
there is already an independently validated native576 route. Only its own waiting
CPU controller was replaced before any cellwork/GPU child;no GPU process was signalled.
This avoids duplicate final cells and a blanket512 exception. Raw handoff proof is
consumed_prefetch_remaining512_handoff_v1/ACTUAL_WAITING_CPU_TRUE576_ROUTE_HANDOFF.json.


## Verified prefill-mode scope correction (2026-10-28)

MX4 scripts measure modeL1 without a prefill override;resolve_execution inherits
L1 for prefill. C1/C2 already used L1,whereas inherited E2E controllers explicitly
used L2. This was a harness deviation. Four earlier A20 selections remain append-only
evidence but do not count as final aligned L1 acceptance. Only own CPU dispatch
parents were stopped;their already running GPU children were left to complete.
All corrected controllers now explicitly pass prefill-modeL1. No kernel,geometry,
stage cost,25scope or resource history changed. Raw code/dispatch proofs are under
prefill_L1_scope_correction_v1;the old 8B1 outer return code is unknown after its
CPU parent stopped,even though actual child completion states were observed.

The earlier B2 L2 timing failed the unchanged generation equality assertion.
A separate5fresh prefill-only diagnostic (3repeats each L1/L2) observes identical
first64token KV bytes and first outputs;it launches no model decode and therefore
does not establish the cause of the L2 generation difference.

Corrected L1 B2 original4 each pass C1/C2512 and twofresh timing processes;
independent verification appends a new consumed_prefetch_L1prefill_actual_solver_median
round. Selected0 TPOT2.264236623ms versus MX4 2.272015373ms (−.342%,indistinguishable),
effective pred/meas.968690729. Across each candidate's six saved warm/timed outputs,
all1024 generated tokens match (24 saved generations total). This is the observed
scope of determinism evidence;no general race cause claim or tolerance change.
B4 and native8BB16 aligned L1 timings continue;B8 and other development cells follow.


## Verified current-body I1 diagnostic (2026-10-28)

New instruction_order_v2.cu uses Qwen0.6B B1 original0's actual body coordinates
with the frozen consumed-prefetch headers. One 512-thread kernel executes both
orders,so register allocation and placement are identical. Both modes issue each
body's incoming prefetch at entry and return an empty next;this follows the
original I1 diagnostic envelope,not the stage-cost or full-program prefetch chain.
Inputs rotate through at least256MiB W and combinedKV,with independent activation
inputs,so this is attribution evidence only. Five fresh processes all pass actual
API/resources (program234MT,frame/static0,resident1) and all ten timing rows/process.
All615 frozen source inputs still match. MIR18rings haszero banned waits/generic
loads. No synchronization correctness or E2E/default claim is inferred.

| Body | Repeat ns | Program-order ns | Difference | Wrapper code bytes | Wrapper MIR instructions |
|---|---:|---:|---:|---:|---:|
|qkv|9200|9780|+6.304%|10064|1334|
|attention|21540|22330|+3.668%|41568|5226|
|o|6540|7070|+8.104%|7824|1017|
|gate_up|12320|13210|+7.224%|15088|1797|
|down|8360|8320|-0.478%|7824|1014|

Program kernel82168bytes/10358MIR instructions. Standalone body wrapper sizes
are actual device-symbol sizes;they are not an exact decomposition of inlined
program bytes. Measured order overhead does not isolate instruction-cache causality.
Raw:current_body_I1_v1;50 process rows and6resources appended without old-row edits.
Initial compiler-path probes did not launch a compiler;the first actual compiler
invocation rejected a missing Bf16 namespace. Corrected source builds normally;
all failed attempts remain archived. No forced register cap or body change.


## Verified MX4-reference recovery correction (2026-10-29)

The resumed summary omitted earlier actual reference work. Contrary to that
summary,QwenB1 prefill had compiled successfully andthen failed the extra v4
historical-ELF equality gate;CU andthe full semantic contract matched. An existing
specification audit explicitly distinguishes this added reconstruction gate from
the prompt's record-only P-PAIR requirement. Its original FAIL stays unchanged.
A separately labelled rebuilt22a reference contract already consumed QwenB1 and
built QwenB16/Llama1B B1,with actual ABI passes. Earlier8B B1 stopped on guard75
interference;B16 had not started. These are recovered facts,not newly repeated work.

The resumed controller mistakenly attempted v4 job0 again and stopped at an
existing output directory before native compilation (root_prefill_v4.py:78).
Its old guard directory was reused:guard.jsonl was appended andpgid overwritten;
the previouspgid is not recoverable fromthat file. Binary,source,CLI receipts and
old failed-contract bytes remain intact. The duplicate's guard observations and
this data-integrity incident are separately archived under
mx4_reference_actual_completion_v1/ACTUAL_DUPLICATE_RECOVERY_AND_GUARD_PATH_AUDIT.json.
No timing/CSV row was altered. All subsequent guard outputs are required unused.

Ten fresh decode WD0/WD1 zero-launch queries now actually pass API0/resident1.
They do not qualify prefill orP-PAIR. Continuing the existing independent rebuilt22a
contract,three prior prefills are retained under their own truthful identities.
A new8B B1 attempt stopped in all52prewarm entries at the unchanged disk check:
f_bavail0 despite UID0 physicalfree about35GiB. No variant compiler launched.
A further distinct attempt uses the supported --variant-cache option in tmpfs
(about64GiB available),with new runtime/output/guard paths and unchanged source,
pins,geometry,compiler flags,resource/numerical criteria. Earlier failures stay failed.

The continuation still requires actual final-prefill ABI/occupancy and fresh
C1/C2512 for allfive reconstructed reference pairs. P-PAIR waits both those gates
and allfive current aligned development selections. Rebuilt-prefill ELF differences,
two historical host-source differences andtheir unknown timing effects remain
explicit;the fixed MX4 CSV is unchanged. These are no completed P-PAIR outcomes.
Raw:reconstructed22a_reference_completion_v2 (disk failure),
reconstructed22a_reference_completion_v3 (supported-cache correction),
aligned_development_MX4_PPAIR_v1 (unstarted CPU waiter handoffs).


## Verified cache execution and accounting correction (2026-10-29)

The tmpfs-cache attempt compiled52normal probes but could not run their host
size getters because /dev/shm is noexec. That actual PermissionError and all
compiled outputs remain in completion_v3;no resource PASS was fabricated and
no mount policy was changed. The next CPU-only step invokes the unchanged
frozen22a prewarm atmaxM64/128 and the D128/Q4 nongemm probe against the earlier
actual8B cache. All52+1source keys match andevery output reportscompiledFalse.
No compiler orGPU call occurs in these validation processes. Completion_v4 uses
that supported cache path,with new build/output/guard paths and all original
source/pin/semantic/resource/C1C2 gates. Its actual builds are still pending.
Raw:reconstructed22a_reference_cache_validation_v1/ACTUAL_8B_ALL53_SOURCE_CACHE_READY.json.

The duplicate guard-output path also exposed a bookkeeping bug:the former monitor
grouped intervals only byoutput path,so the oldcard6 andnewcard5 jobs were merged
into a spurious longcard6 interval. V10 groups by(output,physicalcard),reprocesses
the complete original logs andretains every old observation. It changes accounting
only;the cumulative history is not reset andthe user hour-cap waiver remains.
Before correction the inflated upper bound was104.16h;the same complete logs
after splitting give68.78h at that observation. Both values and the stopped own
CPU monitor's identity are retained inphysical_budget_dispatch_guard_v1/
ACTUAL_PATH_REUSE_ACCOUNTING_CORRECTION_V10.json. GPU children were not signalled.


## Verified disk-space recovery with raw bytes retained (2026-10-29)

Existing-cache reuse passed but the next unchanged native C++ resource-probe
check still rejected root f_bavail0 (8192MiB required). The reference_v4 attempt
stops there andremains failed. No disk threshold was lowered. Read-only duplicate
edge-table comparison found2.179GiB potential savings;no hardlink mutation followed.

Instead,813 completed own MIR data logs (older than1hour,nlink1,complete machine
function markers) were archived. Every file retains a persistent lossless gzip
backup on disk,a byte-identical uncompressed copy in tmpfs,and a readable symlink
at its original path. Each original/memory/decompressed-backup SHA agrees.
No binary,generated CU,source,receipt,JSON measurement row orguard log was moved.
The archive holds16.739GiB original bytes andabout1.402GiB compressed backup data.
Original stats,paths,hashes andrestore.py are retained;this is reversible.

Pilot before/after unmodified MIR parsing matches102functions/66rings/0failures.
The unchanged current fulltop4/source/native-receipt/query/timing verifier for
QwenB1 then returns0. All three fixed MX4 CSVs remain byte-identical to22a.
Actual f_bavail recovers to14.002GiB;filesystem reserve andmount policies stay
unchanged. Fresh reference_v5 starts from this observed space andsame actual
source-keyed cache. It still requires allnormal source/pin/resource/ABI/occupancy
and C1/C2512 checks beforeP-PAIR. No reference timing has been accepted yet.
Raw:completed_mir_archive_v1;reconstructed22a_reference_completion_v4/v5;
P-PAIR now waits reconstructed22a_reference_final_ready_v1's actual final proof.


## Verified aligned ledger preparation and reference progress (2026-10-29)

Both8B reconstructed22a prefills now complete normal build/static-contract/ABI.
Allfive final occupancy queries andfreshC1/C2512 remain required;onlyactual pass
files unlock internal P-PAIR. Existing fixedbaseline CSVs remain byte-identical.

The new CPU ledger appender derives model,batch,layers,card andselected original
from actual artifact bindings. It requires16distinct successful fresh DIAG children,
one physicalcard,accepted2046timed-step deltas per child,and matching-source
three-fresh stage evidence. The CPU waiter consumes actual completion receipts;
no ledger rows are claimed before execution. It retains head/token-history
confounding andnonadditive scope;no production TPOT is inferred from DIAG.
Raw:aligned_development_stage_ledger_v1/append_actual_ledger.py.


## Verified reconstructed reference ready and cellwise P-PAIR dispatch

Allfive final prefill ABI/occupancy queries andC1/C2512 now pass;an independent
CPU replay verifies every referenced hash andactual pass file. The stable readiness
receipt isreconstructed22a_reference_final_ready_v1/ACTUAL_REFERENCE5_READY.json.
It labels the reference reconstructed22a source/pins,not exact historicalELF.
No fixedbaseline file changed andno externalbaseline was rerun.

P-PAIR v7 now dispatches each fully selected development cell whenavailable,
following allfive reference checks. Its per-cell measurement body is byte-identical
tov6 except selecting the ready row. Each cell still runs old/new/old/new oncard5,
fresh processes,warm1/timed2,guard900,ISL64/OSL1024,L1prefill. Only its own idle
CPU waiter was stopped,with no GPUchildren;the handoff is recorded. QwenB1 has
started;no P-PAIR result is yet claimed. Raw:aligned_development_MX4_PPAIR_v1/
ACTUAL_READY_CELL_SCHEDULING_HANDOFF_v7.json.


## Verified first aligned internal P-PAIR result

Qwen0.6B B1 completes old/new/old/new oncard5:old1.981876/1.984387ms,
new1.976820/1.978266ms. Arm medians1.983132/1.977543ms give−.2818%;
maxrelative roundrange.1266%,threshold1%,so the result is indistinguishable.
No selection ordefault changes follow. Both arms useL1 prefill,WD0 decode;old
prefillWD1,newprefillWD0 are identified separately. The oldreference is rebuilt22a
source/pins,not the original historicalELF. Independent CPU ingestion verifies
allfour distinct guard PIDs,successful guard/outer receipts,actual accepted
warm1/timed2 requests,artifact/source/binary hashes andthe exact median/spread.
Fourfresh process rows areinMX5_ppair_c550.csv,each labelled round,executor and
implementation;the fixedMX4 CSVs areunchanged. Raw:aligned_development_MX4_PPAIR_v1/
qwen3_0_6b_B1/ACTUAL_PPAIR_COMPLETE.json andACTUAL_PPAIR_CSV_APPEND_PROOF.json.


## Verified second aligned internal P-PAIR result

Qwen0.6B B16 completes the same fourfresh alternating card5 processes. Arm
medians old4.521433ms,new3.519038ms give−22.1699%;relative roundrange.4136%,
threshold1%,so the improvement is distinguishable. Independent CPU ingestion
verifies fourfresh guard PIDs,artifact/source/binary identities,accepted requests
andthe exact median/spread;fournew labelled process rows append toMX5_ppair_c550.csv.
Raw:aligned_development_MX4_PPAIR_v1/qwen3_0_6b_B16/ACTUAL_PPAIR_COMPLETE.json.
The fixedMX4 baseline files,solver selection anddefaults remain unchanged.


## Verified third aligned internal P-PAIR and CPU ingestion corrections

Llama8B B16 completes fourfresh alternating card5 processes. Arm medians
old14.494706ms,new13.137143ms give−9.3659%;relative roundrange.0599%,threshold1%,
so the improvement is distinguishable. Independent CPU ingestion appends four
labelled process rows;MX5_ppair_c550.csv now has12fresh-process rows fromthree
completed cells. Raw:aligned_development_MX4_PPAIR_v1/llama3_1_8b_B16/.

The future DIAG resource CPU appender first fails syntax preflight (missing
whitespace afterelse);the original failed source/log remain. A versioned correction
passes syntax. Subsequent pre-execution review observes that trusted guard/query
physical_gpu is a string. The resource appender now compares exactsaved5 asstring;
the ledger output converts that actualfield tointeger. Only two idle ownCPUwaiters
were replaced before any DIAG query orCSVappend. No GPUjob,validator,resource
criterion orraw measurement changed. Raw:aligned_development_stage_ledger_v1/
ACTUAL_RESOURCE_APPEND_SYNTAX_CORRECTION_CPU.json and
ACTUAL_SAVED_GUARD_CARD_TYPE_PREEXEC_CPU_REVIEW.json.


## Verified fourth aligned internal P-PAIR result

Llama1B B1 completes fourfresh alternating card5 processes. Arm medians
old2.208144ms,new2.135033ms give−3.3110%;relative roundrange.1020%,threshold1%,
so the improvement is distinguishable. Independent CPU ingestion appends four
labelled process rows,bringing MX5_ppair_c550.csv to16rows/fourcells. The final
development8BB1 stillneeds all8formal top4timing processes andthen its ownP-PAIR.
Raw:aligned_development_MX4_PPAIR_v1/llama3_2_1b_B1/ACTUAL_PPAIR_COMPLETE.json.
No fixedbaseline,solver selection ordefault changes follow fromP-PAIR.


## Verified aligned eight cells and G-PROGRAM development gate

Llama8B B1 selects original0 at10.584414109973673ms,−1.223620% versusfixedMX4,
pred/meas.982853. Allfour original candidates passC1/C2512 andeightfresh timed
processes;the unchanged independent verifier andCSV ingestion pass. Current
aligned L1 coverage is8/25,all nonregressions;five development cells are complete.

The separate G-PROGRAM CPU replay verifies all20 actual candidate correctness
files and40 distinct timing PIDs,actual accepted warm1/timed2 atISL64/OSL1024,
L1prefill,source/artifact hashes andeach TPOT belowfixedMX4. The formal proof
passes;G-B,25cell solver,30cellLLM,forward anddefaults remain open. Stage ledger
andthe final development P-PAIR are stillpending;this is not CP-1 completion.
The new gate reader initially used the bash/P-PAIR argv offset for the direct
native wrapper;pre-execution review corrects this in a versioned CPU helper.
Only its own idle waiter was replaced;fouravailable cells/32processes precheck
then passes beforethe fullfive proof. No GPUjob,raw result orcriterion changed.
Raw:aligned_G_PROGRAM_development5_v1/ACTUAL_G_PROGRAM_DEVELOPMENT5.json and
ACTUAL_ARGV_POSITION_PREEXEC_CPU_REVIEW.json;consumed_prefetch_llama3_1_8b_B1_L1prefill_full_top4_execution_v1.


## Verified allfive aligned internal P-PAIR results and DIAG launch readiness

Llama8B B1 completes fourfresh alternating card5 processes. Arm medians
old10.715534ms,new10.589079ms give−1.180104%;relative roundrange.089222%,
threshold1%,so the improvement is distinguishable. Allfive development P-PAIR
cells now finish:QwenB1 indistinguishable,the otherfour distinguishably faster.
Independent CPU ingestion verifies andappends20fresh-process rows acrossfivecells.
Fixedbaseline CSVs,selections anddefaults remain unchanged. Raw:aligned_development_MX4_PPAIR_v1/
ACTUAL_FIVE_DEVELOPMENT_PPAIR_COMPLETE.json andACTUAL_REMAINING4_PPAIR_CSV_COMPLETE.json.

Allfive selected-artifact DIAG CPU builds pass stack/static0 and70total cleanMIR
rings;allfive independent --check-only processes passwithout GPUcalls. Resource
summary follows;these rows alone do not establish occupancy orDIAGtiming.

| Cell | Threads | MTreg | Code bytes |
|---|---:|---:|---:|
|llama3_1_8b_B1|512|229|136024|
|llama3_1_8b_B16|512|227|178272|
|llama3_2_1b_B1|576|165|121472|
|qwen3_0_6b_B1|512|228|129720|
|qwen3_0_6b_B16|512|227|130392|

The first actual DIAG query (8BB16) passesAPI0,resident1,launches0 oncard5;
only then does its fixed-mask/two-round measurement begin. CP-1 still requires
allfive actual stage ledgers. Raw:aligned_development_stage_ledger_v1/
ACTUAL_FIVE_DIAG_CPU_RESOURCES_SUMMARY.json andACTUAL_FIVE_DIAG_CHECK_ONLY_CPU_PROCESSES.json.


## Verified actual origin fit R² reporting

The eight current aligned selections retain source-bound original solver previews.
CPU replay verifies each protocol/preview SHA andextracts the actual fit values
(.9929311059–.9989661621;136/154/193 measured rows byscope). These are measured-row
fit R²,not held-out validation ornew out-of-table measurements. Current selections
use covered measured-stage rows;no least-squares fallback candidate is substituted.
MX5_llm_c550.md now reports onlyR² in the request-metrics table,as§5.7 requests;
the original fit objects andscope remain inrawcurrent_selected_fit_report_v1/
ACTUAL_SELECTED8_ORIGIN_FIT_R2_CPU.json. No GPU/compiler call orsolver change.


## Verified nine aligned cells and refreshed normal root compiler

Llama1B B16 selects original0 at3.006683667085679ms,−12.476161% versusfixedMX4,
pred/meas1.004738. Allfour original C1/C2512 andeightfresh timings passindependent
verification/CSVappend. Current aligned coverage9/25 hasno discernible regressions;
allfive Llama1B batches are complete. The native six-cell controller completes
andunlocks the already-gated remaining16-cell queue,startingQwen0.6B B2.

The primary build/tools/tilemega stillhad sourcefingerprint0bba3b96,whilethe
active frozen producer already matchedcurrent f054c811. The old root executable
is retained byte-identical (SHA8485c9f4). Normal incremental Ninja rebuilds two
CPU objects andlinks the host compiler;an actual normal Ninja confirmation,
source-fingerprint check andversion command allreturn0. New root SHA76502785,
sourcefingerprint f054c811 matchescurrent native/compiler inputs. Frozen producer
SHA230a572b remains unchanged andactive artifacts are not replaced. GPUcalls0.
The initial build wrapper captured shellrc only;the separately recorded normal
Ninja confirmation supplies its actualrc0 rather than inventing the first rc.
Raw:current_root_build_verification_v1/ACTUAL_CURRENT_ROOT_BUILD_VERIFIED.json.


## Verified current QwenB2 failure and first completed selected stage ledger

Current Qwen0.6B B2 selects original0 at2.3013748563571177ms,+9.825017%
versusfixedMX4,pred/meas.935251. Allfour originalC1/C2512 andeightfresh timings
passindependent verification/CSVappend,butperformance failsG-LLM. The oldA19
row2.301333ms is retained separately;the current measurement doesnot fix it.
Read-only source-bound replay finds231MT/frame/static0 inallfour timing artifacts,
14cleanMMA rings each,zero kernelcalls/private accesses inselected MIR. The
unchanged parser excludesCTA-barrier paths;no all-path absence claim is made.
Actualselected-pin DIAG CPU preparation begins,with unchanged producer/source,
geometry,widths,resource/MIR gates;no GPUlaunch orsolver change. Raw:current_QwenB2_D7_readonly_v1.

8BB16 completes16freshcard5 DIAGchildren andactual API0/resident1 query. The
per-cell summarizer andstrict independent CSVingestion pass;12round/classrows
append toMX5_stage_ledger_c550.csv. The CPU append waiter now consumes each
completed cell afterits successful summarizer receipt;onlythe oldidle CPU waiter
was replaced. Its actual row verifier is unchanged,with all PID/source/guard/
accepted-timestamp/bench checks retained. Raw:aligned_development_stage_ledger_v1/
ACTUAL_READY_CELL_LEDGER_APPEND_HANDOFF_v3.json.

Nonadditive two-round median interventions versusbench,us/occurrence:
QKV35.200/37.620,attention60.383/53.694,O26.931/31.620,gate/up166.632/174.180,
down82.896/90.350,head+tokenhistory655.155/656.700. The head comparison is
confounded;no additive sum orisolated-body timing isclaimed. Appendix8BB16
attention26–28us is contradicted:bench already53.694us (25.694us aboveupper),
andthe program intervention isanother6.689us higher. This places muchof the
observed gap in the prescribed self-next bench envelope;finer causal attribution
remainsopen. Otherfour development stage ledgers andCP-1 remainpending.


## Verified selected current QwenB2 D7 readiness

The actualselected DIAG CPU build nowpasses strict source/geometry/width/CU
equality andnormal build-option checks:229MT/frame0/static0,14cleanMMA rings,
no generic/private/calls inselected MIR. Independent --check-only CPU preflight
passes. This prepares diagnostics for the actual+9.825% G-LLM failure;it doesnot
change solver selection,production code orcriterion.

Its guardedcard5 query300s and16fresh per-mask900s jobs are explicitly queued
afterthe existing five development DIAG attempts complete. This preserves the
two-card limit andavoids overlapping card5 jobs. The driver uses the same
strict body/guard/PID/timestamp summarizer protocol;an independent CPU appender
waits actual D7 completion. No D7GPUquery/timing has yet run. Raw:current_QwenB2_D7_readonly_v1/
ACTUAL_D7_GPU_QUEUE_PREPARATION_CPU.json andselected_diag/DIAG_CPU_RESULT.json.


## Verified second selected development stage ledger

Qwen0.6B B1 completes16freshcard5 DIAGchildren andactual API0/resident1 query;
strict independent per-cell ingestion appends12round/classrows. Selected ledger
coverage is2/5 cells/24rows. The nonadditive two-round median interventions/bench,
us peroccurrence,are QKV6.722589/10.100,attention23.413214/22.988265,
O4.820089/8.360,gate/up9.737411/13.770,down6.371964/10.030,
head+tokenhistory190.475/195.300. Head isconfounded;no additive sum isused.
Appendix attention2.5–3.5us is contradicted:the prescribed bench itself is22.988us
andthe program intervention23.413us. Difference.425us is an observed comparison,
not isolated integration overhead;finer causal attribution remainsopen. Raw:
aligned_development_stage_ledger_v1/qwen3_0_6b_B1/ACTUAL_DIAG_SKIP_LEDGER.json and
ACTUAL_LEDGER_CSV_APPEND_PROOF.json. The finalthree development ledgers remainpending.


## Verified current QwenB4 failure and source-bound I1 extension

Qwen0.6B B4 selectsoriginal0 at2.5758779246290877ms,+7.587778% versusfixedMX4,
pred/meas.956615. Allfour originalC1/C2512 andeightfresh timings passindependent
verification/CSVappend;performance failsG-LLM. Current coverage11/25 hasnine
nonregressions andtwo failures(B2/B4),separate fromhistoricalA19 failures. The
actual selectedB4 pin/source DIAG CPU preparation starts withunchanged criteria.
Raw:current_QwenB4_D7_v1;the fullremaining queue continuesQwenB8.

Instruction-order probe nowhas compile-time B1/bmin1 orB2/bmin4 support,bound
to the actualsame QwenB1/B2 coreclass coordinates. B2 receives realA/residual row
strides,B-sized QKV andKV storage,andaccurate CSVbatch metadata. The oldB1source
is retained byte-exact undercurrent_body_I1_v1/source_before_B2_parameterization.
Both normal same-compile MIR builds pass:control234MT,B2 232MT,allframe/static0,
18cleanMMA rings each. Entire defaultB1 device.text is byte-identical to the old
measured executable;old source,binary andmeasurement rows are not relabelled.
B2 program symbol84480B;actualoccupancy/query/fivefresh timing remainpending
after actualB2 stageD7. No model body,solver,cost table ordefault changed.

An initial CPU metadata call used the decoder-only kernel_instructions helper
on I1kernels;it raises beforeproof/GPU/CSV writes. The corrected CPU tool matches
the actual I1function identities anduses the same opcode-counting expression.
It also audits old size scope:separatedevice-object program82168B differs from
oldnormal-linked symbol82944B;whole linkeddevice.text168688B. Per-wrapper linked
sizes areQKV10064,attention41872,O7888,GU15056,down7920B. The previous oldsize
values belong tothat separatedeviceobject andremain inold rows;new resourceCSV
rows explicitly record actuallinked sizes and B2CPU resources withquerypending.
The I1sizes arestandalone wrappers,not exactinlined modelbody decomposition.
Raw:current_QwenB2_D7_I1_v1/ACTUAL_NORMAL_CPU_RESOURCE_MIR_AND_CONTROL_PROOF.json,
ACTUAL_OLD_CODE_SIZE_SCOPE_AUDIT_CPU.json andACTUAL_CPU_METADATA_HELPER_CORRECTION.json.

Verified isolated last-block peeling candidate (2026-10-29): prompt §4.7 steady blocks keep same-head next loads; peeled tail retains cross-head loads and owner K/V patch. Production header unchanged. CPU sequence equivalence covers599760 consumers/34272 owners;22selected normal kernels pass resource/MIR checks. Initial112 numerical cases (addsB2/B4,unchanged oracle/tolerance/seed formula) andsix helper-litmus cases require five fresh processes each,queued oncard5 afteractualB4I1. GPU,fullmatrix andE2E remain pending;no candidate/default promotion. Raw:attention_peeled_last_block_candidate_v1.

Verified follow-up CPU preparation: all17 full-matrix artifact entries complete,
including two genuine same-candidate reuses. The original336-case plan selects
24kernels using unchanged R_cap220/248:224casesPK0 and112casesPK2. All selected
P0/unit kernels pass normal resource/MIR gates;no GPU numerical result is claimed.
The queued full336 and null-idle28 suites each require five fresh processes;
existing initial-helper evidence will be reused only for this identical candidate.
A subsequent five-fresh cold self-next test covers22keys: original12 D64Q4/576/PK2,
plus actual QwenB2/B4 five-length grids atD128Q2/512/PK0. Neither publishes costs.

Verified normal-linked CPU code comparison (same actual bench function names):
D128Q2 norm1/b_min4 and8 grows231→238MT,45368→48152B,5695→6052 MIRinstructions.
D64Q4 PK2/b_min8 grows141→150MT,30704→32656B,3866→4114 MIRinstructions.
These are standalone bench symbols from their actual linkedELFs,not separate
objects andnot an inlined model decomposition. No runtime/TPOT/cache claim follows.
Raw:attention_peeled_last_block_candidate_v1/ACTUAL_FULL_CPU_RESOURCE_MIR_GATE.json,
ACTUAL_NORMAL_LINKED_CODE_COMPARISON_CPU.json andCOMPILER_CASE_PLAN.json.

Verified candidate integration CPU experiment atactual selected QwenB2/B4 geometry:
reuseeach original generatedCU byte-exact andnormalcompile flags,changing only
theattention include dependency. Both newlinkedbinaries compile234MT/frame0/
static0,14cleanMMA rings,no deviceprivate accesses/calls;wholedevice.text128024/
128120B. These have no newnative servingidentity,query orGPU authorization;
they do not countas selected plans/E2E. Actualsource-bound producer stillrequired.
The initial metadata helper wrongly matchesboth devicekernel andhoststub witha
substring;it asserts aftersuccessful compilation andbefore result writes. The
separate v2finalizer matches the exactsame-compile resourcefunction name. No
recompile,resourcecriterion change orGPU call. Failedhelper/logsremain archived.
Raw:attention_peeled_last_block_candidate_v1/ACTUAL_SELECTED_B2_B4_INTEGRATION_CPU_COMPLETE.json
andACTUAL_CPU_INTEGRATION_METADATA_CORRECTION.json.

Verified current A20 selected QwenB2 D7:16fresh guarded processes,2rounds×8masks,
strict independent ledger append passes. Medianintervention/bench us are
QKV7.080/10.080,attention35.719/27.220,O4.857/8.500,GU9.177/13.810,
down6.614/10.290,head+tokens191.425/195.160. Interventions are nonadditive;
head+tokens isconfounded. Mask63=.333240ms retains prefetch/barriers/control/timer.
Attention ratio1.312 exceedsD7 1.2;no isolated-body/integration decomposition claimed.
Actual selected B2 I1 fivefresh order/repeat deltas:QKV5.274%,attention1.195%,
O5.178%,GU7.036%,down2.339%;same-entry diagnostic,not E2E. Theattention1.195%
delta alone doesnot establish thecause of31.2% ledger/bench gap. Actualquery
API0/frame0/resident1;232MT,programsymbol84480B,totaldevice.text171536B.
Fiftyfresh I1 measurement rows andone actualresource row append;oldprefixes unchanged.
Raw:current_QwenB2_D7_readonly_v1 andcurrent_QwenB2_D7_I1_v1.

Verified pre-GPU preparation mistake:ten candidate D128Q2 B2/B4 cold keys used
hkv4 fromgeneric fixtures;actual Qwen0.6B config has hkv8. Nonehad executed.
Oldkeys/fixture/scope preserved;ten cold keys nowhkv8,andadded initial B2/B4
numerical fixtures use hkv8. Original B1/B16 fixtures,hkv4/full336 tests,oracle,
tolerance,seedformula,compiledheaders/binaries unchanged. CPUfixture v2proof
binds these exact changes;no measurement rewritten. Raw:candidate
ACTUAL_PRE_GPU_HKV_CORRECTION_CPU.json andACTUAL_ORACLE_FIXTURE_SCOPE_CPU_PROOF_v2.json.
Current-source offgrid37 diagnostic prepared withactualhkv8/A20library,threefresh
processes queued aftercandidate self-next tokeepcard5 serial. Itsfuture results
are attribution only;officialfive-grid interpolation/cost selection stays unchanged.

Verified current Qwen1.7BB2 regression3.609036ms,+1.507% vsfixedMX4,
pred/meas.96038,original0 selected;all4 C-1/C-2 andeightfresh timingprocesses
pass independent selection/CSV replay. It is thethird currentG-LLM failure.
Same-source/CU/pin selected DIAG normalbuild passes228MT/frame0/static0/14rings,
check-only CPUpreflight passes. Dedicated I1 usesactualQKV4096×2048,GU12288×2048,
O2048×2048,down2048×6144 andselected TN/WN/D/Pw;D128Q2/HKV8/B2/b_min4/PK0.
Itsnormalcompile passes230MT/frame0/static0/18rings,programsymbol87040B.
TwoCPUresource rows append withactualGPUqueries explicitlypending. I1source is
an ignored experimentcopy;production headers andexistingI1profiles unchanged.
GPU D7 waits current-source offgrid37,thenI1 waits actualsamecellD7 completion;
card5 staysserial. Raw:current_Qwen17B2_D7_v1 andcurrent_Qwen17B2_D7_I1_v1.

Verified current Qwen0.6BB4 D7:16fresh guarded children,2rounds×8masks,
strict independent append. Medianintervention/bench us:QKV7.827/10.400,
attention42.466/36.373,O5.476/9.040,GU9.937/14.000,down7.640/11.250,
head+tokens194.065/194.370. Attentionratio1.168;G-LLM regressionstillopen.
Mask63=.332398ms iscommon work,not pure synchronization. No additivebody sum.
Actual B4 I1 fivefresh order/repeat:QKV+4.728%,attention−2.389%,O+5.103%,
GU+9.302%,down+2.041%. Same-entry diagnostic,not E2E orcache causality.
Actualquery228MT/frame0/static0/resident1;normalprogramsymbol84224B,
wholedevice.text171248B. Fiftyfresh I1rows/oneactualresource append;oldprefixes
unchanged. Raw:current_QwenB4_D7_v1 andcurrent_QwenB4_D7_I1_v1.
Candidate initial GPU numerical/helper checks nowstart withcorrected HKV8
foradded B2/B4 fixtures;no candidate acceptance/result claimed beforecompletion.

Verified tail-peel initial GPU gate:fivefresh×112 numerical/KV/callback/ticket/
repeat-bit cases pass,zero skips;added B2/B4 useactualHKV8. Fivefresh×sixlocal
LDS-helper litmus cases pass:512/576threads×0/2/8pending16Bloads perlane,
104CTAs×2000iterations each,zeroRAW/cold errors. Ten distinct childPIDs,
actualprelaunch resource/API/oneCTA query gates andfinal guard0 verified.
Complete336/null28 fivefresh matrix nowrunning;no adoption orE2E claim.
Raw:attention_peeled_last_block_candidate_v1/ACTUAL_INITIAL_GPU_CHECKS_COMPLETE.json
andINITIAL_CHECK_BULK_COMPLETE.json.

Verified candidate D128Q2/576 P0 CPU experiment,onlythread macro changed from
same candidate512 build:all8 b_min1/2/4/8×norm0/1 functions report168MT and
296–360B stack,so all8 are rejected beforequery/numericalGPUexecution. Eight
CPUrejection resource rows append;no force-register flag/limit applied. This is
an observed body-compile failure,not proof ofa plan-level512 exception. Full
plan prescribed histories andMIR allocation analysis remainrequired;no global
576 impossibility claimed. Raw:candidate/bench576_D128Q2_PK0.

Verified candidate full GPU gate:complete336 matrix andnull-idle28 fixture each
pass fivefresh guarded processes,zero skips;eachalso passes15 MX4transform
bitcases. ActualP0 queries derivePK fromoriginalR_cap220/248,then selected
kernels pass API0/frame0/static0/oneCTA beforelaunch. Ten distinctactual child
PIDs. Samecandidate initialhelper sixcases/fivefresh reused withno rerun.
Coldselfnext22keys nowrunning;normalcandidate fullhost producer staysgated on
itsactualfivefresh completion. No source integration/default orE2E acceptance.
Raw:attention_peeled_last_block_candidate_v1/ACTUAL_FULL_GPU_CHECKS_COMPLETE.json.

Verified candidatecold22 fivefresh completion andindependent append:15body/
cold validation rows,130actualselected-kernel resourcequeries;existingprefixes
unchanged. Newnormal fullsource producer9aac1b7e completes160freshbuildsteps,
C++unit/fingerprint checks. All615 nativeinputs differfromA20 onlyin tested
DecodeAttentionV2.h;G14/H15 closure hashes unchanged,newattention d1d9508b.
Producer livesinisolated worktree atbase d80486efe plus exactoneheader patch;
mainattention source stays0ccf7bc5. It isnot a committedmain sourceintegration.

Verified same-source canonical576 controls:B2 initial376Bstack,headPw2→1
terminal392B;B4 initial/terminal344B. Bothterminal planshaveallGEMM/head Pw=D.
The normalcompilerreturns1 onthese resource refusals whilecucc itselfreturns0;
initialnaive CPUcontrollerstops beforeclassifying that expectedrefusal. V2
reuses theunchangedinitialartifact,validates exactnative receipt/resourcefailure,
thencontinuesonly theprescribed headhalving;no allow_resources/flag bypass.
Samegeometry512 controls bothcompile234MT/frame0/14cleanrings;candidate WD1
checks compileB2/B4 232/236MT/frame0. Manualcomparison allocationreview records
observed compilerrequirements,not physicalregister minima orfinalsolver origins.
Actualqueries for8retained/control plans,C1/C2512 andsamecard old/new/old/new
P-PAIR arequeued afterQwen1.7BB2 I1;no control timinghasexecuted yet.
Raw:candidate selected_resource_histories andACTUAL_MANUAL_CONTROL_CPU_PREPARATION_COMPLETE.json.

Verified currentA20 B2 offgrid37 threefresh diagnostic:actualHKV8,b_min4,PK0,
source b4555ea2;block-midpoint weightedmean33.697713us vs officialfivegrid
27.220132us. Substitutingonlythisdiagnostic estimateinto oldmodelgives
2.333736ms vs actual2.301375ms. This isattribution,not a newformal table,
formula oradditive-body latencyclaim. Raw:current_QwenB2_D7_offgrid37_v1.
Candidate5fresh fivegrid same-formula diagnostic: B2 28.292600us vs A20
27.220132us (+3.940%);B4 36.459585us vs36.372639us (+.239%). Component
results donot select source/defaults;actual modelcontrast remainsrequired.

Verified actual selected Qwen1.7BB2 DIAG:16fresh children/tworounds,strict
independent ledger append. Intervention/bench us:QKV12.304/15.500,
attention36.779/27.220,O6.784/10.420,GU31.778/36.340,down16.825/20.610,
head+tokens377.020/383.540;nonadditive,head/token-history confounded.
Mask63=.417215ms retains commonwork. Attentionratio1.351 triggersD7.
Actualprofile I1 fivefresh order/repeat:QKV+3.511%,attention+1.563%,O+8.588%,
GU+4.078%,down+1.470%;diagnostic only,no cachecause ordefault claim. Fifty
I1rows/oneactualresource row and12stageledger rows append. Raw:current_Qwen17B2_D7{,_I1}_v1.

Verified candidateactualresourcequeries:eightfreshqueryPIDs,allAPI0/resident1,
zero launches. Allfour512 WD0/WD1 controlsframe0. Thefourfailed576 kernels
stillhave344/376/392B localframes andremainrejected despiteoccupancy1;no
numericalexecution ofthem. Eightactualqueryresource rows append withoutediting
priorCPUquery-pending rows. B2 WD1C1/C2512 hasstarted;no modeltiming result yet.
Raw:candidate/ACTUAL_EIGHT_CONTROL_QUERIES_COMPLETE.json.


Verified tailpeel manual end-to-end controls complete:C1/C2512 passesboth
B2/B4;A20/new/A20/new oncard5,4freshprocesses/cell,warm1/timed2,L1prefill,
WD0,noinstrumentation. B2medians2.302905/2.306382ms(+.151%);B4
2.569007/2.578243ms(+.359%);bothbelow1% threshold,indistinguishable.
Eight source-bound rows inMX5_candidate_ppair_c550.csv aremanualcontrasts,
not formal serving selections orfixedexternal baselines. No candidateadoption.
Independent guard/PID/source/identity/hash checks append2model-correctness rows.
Raw:attention_peeled_last_block_candidate_v1/ACTUAL_MANUAL_E2E_INDEPENDENT_CSV_APPEND_CPU_PROOF.json.

Verified static-tail-bound candidate(parenttailpeel) removesonlya provably
unreachable steadyloop:static capacity1088,H<=24 givesatmost1block/wave;
runtime/dynamic shapes retainthesamecode. CPU18944geometries,
19781376consumers/1221888owners preserveexactorder. Actual5fresh×112
staticshape numerical/KV/callback/ticket/repeat-bit casespasswithzeroskips;
B2/B4 actualHKV8 covered. All16queries/process API0/frame0/resident1.
Thezero-ring MIRcase isallowedonlywiththeproofthatnoloopis emitted;
no globalwait/private/calls criterionwasrelaxed. Originaloracles unchanged.
Thecompilerlog splitone longresourcefunctionname witha warning;unique
MIR-name prefix/continuation repairis preservedasaseparateparsedview,
rawlog unchanged. Fivecorrectness/80actualresource rowsappend.
Normalfresh160-step producer6b9b5d0675/source a246d0c435 isisolated,
exactheader50dc44ba patch;no hostobjectreuse orproduction/defaultchange.
Dynamic D128Q2/512P0 andD64Q4/576PK2 .text/resources matchparentbytes;
parentmeasurements arenot relabeled. Wholemodel resource/E2E controls pending.
Raw:attention_static_tail_bound_candidate_v1,frozen_catalog_producer_static_tail_bound_candidate_v1.

Verified A20 partitiondiagnostic:296keys×3fresh (B2/B4,bmin1/2/4/8,
37Lvalues,HKV8,cold>=256MiB,2000iters). Midpointmeansus B2:
32.094/33.991/33.709/39.115;B4:38.787/38.530/38.092/41.473.
Strong nonmonotoniclengthcosts explainpartoffivepointinterpolationbias;
thisisdiagnosticinference,notareplacementformula orsolver/defaultdecision.
Raw:current_A20_attention_partition_diagnostic_v1.

Verified valid-query-softmax candidate(parentstaticbound) first8dynamic
D128Q2/512P0 kernels passnormalresource/MIR gates (232–234MT,frame0,
cleansteadyloops). Itlimits max/exp worktoobserved q<Q rowswhile retaining
alllane K/Vloads,ownerstores,MMAs andsynchronization. FullmatrixCPU/GPU
validation andE2E remainpending;productionheader isunchanged.
Raw:attention_valid_query_softmax_candidate_v1.


Verified staticbound manualcontrols complete:C1/C2512 bothB2/B4 pass;
card5 A20/new/A20/new,4fresh/cell,warm1/timed2,ISL64/OSL1024,L1prefill,
WD0,no trace. B2old/new2.301054911/2.301855480ms(+.034791%);
B4 2.574111492/2.580446370ms(+.246100%). Bothindistinguishable
undermax(1%,range);candidate staysisolated/off. Independent binding/hash/
identity/PID/guard validationappends8candidate PPAIR and2C1/C2 rows;
8actualplanresource rows preservefailed576352/344Bframes (queryonly),
legal512WD0MT228/234 andWD1MT230/236,allAPI0/frame0/resident1.
No finalsolver qualification orcosttable/default promotion isclaimed.
Raw:attention_static_tail_bound_candidate_v1/ACTUAL_MANUAL_TWO_CELL_PPAIR_COMPLETE.json.

Verified validquery full336-case formula-derived CPUresource/MIR planpasses,
alongwith16staticshapekernels (B1/B2 MT226,B4/B16 MT232,frame0).
Asecond compilerstdout/stderr resource-name interleave isuniquely matched
toitsMIRname intoaseparateview;actualcompilerreturncode0,rawlog preserved.
Fivefresh full336/null28/static112 GPUchecks arequeued afterstaticbound
modelcontrols. No end-to-end result ornewsource adoption yet.

Verified segment-scale-reuse candidate(parentvalidquery) computes each
identicalper(query,wave) exp2 scalaronceinLDS for§4.7 segmentmerging,
preservingmax andwave0..7 result/lambda arithmeticorder. Itaddsone
existingpureLDS PublishWaveResults call;allCTA waves participate and
laterLA/finalpublications protectscratchreuse. Globalbarrier/LA unchanged.
Normalfirst8benchkernels andfull336-derived CPUresource/MIR planpass;
static16/P0576 CPUcheckscomplete. Actual5fresh×104CTA×2000 iteration
sparse-LDS publication/reuse litmus,fullnumerics andE2E arepending.
Header87e25d5b remainsisolated;production0ccf7bc5/defaults unchanged.
Raw:attention_segment_scale_reuse_candidate_v1;noGPU/performance claim.


Verified validquery full336 numerics nowpassall5fresh. Itsfirstnull576
attemptstopsatcheck_child.py:20 withFileNotFoundError forcurrent-source
D64Q4/576 P0benchmetadata,beforesuchcase dispatch. Thisisapreparation
omission,not anumericalpass/failure. Rawfailedguard/output andsetup-failed
CSVrow remain. ThemissingP0/PK2 benchandunit576 libraries arenow normally
compiled/audited foreachS3/S4 source;allresources/MIR pass. Null28 reruns
use5newprocesses innewfull_checks_card5_v2 paths;theactual5fullsuccesses
are retained throughhash-boundrefs,neverrelabelingthefailedattempt.
Raw:attention_valid_query_softmax_candidate_v1/ACTUAL_NULL_MISSING_INPUT_PREPARATION_CORRECTION_CPU_v2.json.


Verified validquery numerics complete:full336×5fresh,null28×5fresh,
static112×5fresh,zero skips;15correctness and210actualresource rowsappend.
Thefailednullsetup row remainsseparate. Staticappender initiallyrefused
othercompleted rounds becauseduplicatecheck wasimplementation-wide;
no writesoccurred. Scopewasfixedtoimplementation+round,thenindependent
raw/hash/PID/resource checks passedandoldprefixes remainedunchanged.
Freshnormal160-step producerbuildisrunning;no wholemodelresult yet.
Normal-linked representative D128Q2/bmin4/norm1: A20 MT231/45368B/
5695MIRinstructions;tailpeel/staticbounddynamic MT238/48152B/6052;
validquery MT234/48376B/6084;scalereuse MT234/48664B/6123.
Thesearestandalonebodyfunctionsymbols,not separatelycompiledobjects or
wholemodel inlinecounts. Eightrepresentative recordsappendinoneCPU
comparisonrow;nocachecause,TPOTgain ordefaultdecision isclaimed.
Raw:attention_candidate_normal_code_comparison_v1.


Verified validquery manual end-to-end controlcomplete:C1/C2512 bothpass,
card5 A20/new/A20/new,4fresh/cell,warm1/timed2,L1prefill,WD0,no trace.
B2old/new2.302893126/2.298254596ms(-.201422%),range.088264%;
B4old/new2.573165217/2.580484399ms(+.284443%),range.236335%.
Bothunder1% threshold,indistinguishable;candidate staysisolated/off.
Independent rawbinding/hash/identity/PID/guard checks append8candidate
PPAIR/2correctness rows. Actual8planqueries API0/resident1,zero launches;
failed576framesB2 320/320B,B4 396/356B remainrejected. Legal512WD0
MT228/234 andWD1MT226/236 frame0. Normalfreshproducerb287055b/source
e50003d8 usesexact0423a65e header;G/H closuresunchanged,attentionclosure
eeda1ece. Manualcontrast isnot newsource solver/finalqualification.
Raw:attention_valid_query_softmax_candidate_v1.

Verified MoEread-only source reviewconfirms exportedweightdtype/recipes,
rankorderedlegacyrounding,andtheexistingdense-onlyentry gaps. Additional
inferredintegration checkscoverprompthead's singlebarriercounter invocation,
conditionaltail generation,per-candidate prompt/decode recipeequality,
routerticket counts/groupedM6 producer counts,andsharedsource costidentity.
No MoEcode orGPUexecution wasstarted aheadofdense;rawmoe_native_contract_readonly_v1.


Verified segment-scale-reuse candidatevalidation complete:fivefreshprocesses
for eachof sparseLDS6cases/full336/null28/static112. SparseLDS usesall104CTAs
for2000iterations each,zeroRAW/cold errors,actualAPI0/frame0/resident1;
no inflight-overlap claim. Full/null/static originaloracles,tolerances,seeds
unchanged,zeroskips. Twentycorrectness rows (5litmus+15numeric) and210
actualbodyresource-query rowsappend independentlywitholdprefixes unchanged.
Wholemodelnormalproducer buildstartsafterthese gates;no E2Eresult yet.
Raw:attention_segment_scale_reuse_candidate_v1.

Verified additionalreadonly MoEbindings:fiveB eachhave779buffers,219stages,
121GEMMs;allboundoperand indicesvalid. Exactembeddingprefix,24×9legacy
stage sequence,head+argmax tail,allGEMMshapes,foldedexpert recipes/unit16,
f32routingweights andBF16 Q/Knorm/RoPE validate. The246external/weight
names,dtypes andrecipes areidentical acrossB. Theseare CPUfrontendfacts,
notnativecompile/correctness/routing/performance results.
Raw:moe_native_bindings_readonly_v1/ACTUAL_NATIVE_CONTRACT_VALIDATION_CPU.json.


Verified segment-scale-reuse manual end-to-end controlcomplete:C1/C2512
bothB2/B4 pass;card5 A20/new/A20/new,4fresh/cell,warm1/timed2,
ISL64/OSL1024,L1prefill,WD0,no trace. B2old/new2.301821086/
2.315481970ms(+.593482%);B4 2.571234213/2.595170951ms(+.930944%).
Bothunder1% threshold,indistinguishable,withno demonstratedbenefit;keptoff.
Eightcandidate PPAIR/twoC1C2 rows independentlyappend afterrawbinding,
identity/hash/PID/guard checks. Current-source576 B2frames332/332B,
B4 392/364B remainrejected;512WD0MT228/234,WD1MT226/236 frame0;
eightAPI0/resident1 queryrowsappend withzero failedkernel launches.
Freshnormalproducerb6d766a5/source47102229 usesexact87e25d5b header.
Allfourimplementation contrasts(tailpeel,staticbound,validquery,scale reuse)
are recordedseparatelyfromformal sourceA20 selection;noneispromoted.
Raw:attention_segment_scale_reuse_candidate_v1.

Verified remainingdense schedule:theoriginalactive Llama3BB4 childcontinues
unchanged oncard6. Onlyitsowned CPUparent dispatcher isstoppedtoprevent
futureduplicate celllaunches. Aftertheexistingchild exits,actual/proc exit
status isrecorded beforeterminatingthatstoppedCPUparent andindependently
verifying/appending B4. Nocard/GPU/measurementprocess issignalled.
Card5 startsB8onlyafterallS4 PPAIRguardsfinish;remainingfixedqueues use
cards6/5,oneguardjob/card,atmost2cards,originalsource/protocol/prefillL1/
C1/C2/top4/measurementcriteria. No finalgateis waived. Thependinghandoff
isnot claimedcomplete;rawremaining_dense_two_card_dispatch_v1.


Verified isolated MoE CPUpreparation duringremaining denseGPU validation:
AnalyzeMxMoe draftdirectly readsModelPlan andvalidates layerstages,
projectionshapes,gather/scatter/dispatch/recipe/rounding contracts;five
actualbatchplans passand50specificmutants rejectwithlayer/location.
Initialtest compilemissed DmDescriptorCodec include;then initialanalyzer
incorrectlyrequiredBF16headoutput. Bothattempts remain. Correctheadcontract
isFP32argmaxvalues/I32indices,explicitlyboundtoreducer;oraclesunchanged.
Draftsources remainunderignoredruns;no compilerentryintegration orMoEGPU
execution. Thisis CPUpreparation beforedense25finishes,notformalphaseE
acceptance. Raw:moe_analyzer_cpu_preparation_v1.

Verified isolated router/topk/dispatch helper CPUcompilation:20instances
(B1/2/4/8/16 xweightsFP32/BF16 xthreads512/576) have20–24MT,frame0,
staticSMEM0,andexactfunctionMIR generic/addrspace0/calls/private0.
Thefirstattempt usedAS1assignmentofCUDAuint2/uint4structs,whichMACArejects;
separatev2 usesexisting nativeext-vector8/16B stores,arithmeticunchanged.
RankcountusesroundedBF16logits,stablelowerindex ties;expf/sum isrankordered;
activeexperts androwsarewritteninexpert/(b,k) order. Theseareimplemented
sourcecontracts;GPUequivalence/occupancy/determinism remainuntested.
Raw:moe_router_cpu_preparation_v1;no nativeplan,measurement ordefaultclaim.


Verified ownedCPU handoffoccurredonlyaftertheexisting B4childexited0;
raw/procwaitstatus ispreserved. InitialnewCPUappendrefusedWD1 queryargv
becauseoldactualrunnerused/opt/conda/bin/python butnewappender used
/opt/conda/bin/python3. No modelmeasurement failedorCSVrowwaswrittenby
thatattempt. Exactoldinterpreter nowrerunsstrictindependentverification,
passesandappends B4;oldfailurelog/process stayunchanged,new_v2 pathsused.
No verifiercondition orartifact/data waschanged. Recoverycard6queue handles
B16/8BB4;originalnewqueue'scard5threadcontinues B8/8BB2/8BB8. Thefirst
newcard6thread failureisCPUmetadata-only,not aGPUfailure;finalcompletion
usesanexplicitv2collector instead ofclaimingthefailedparent completed.
Raw:remaining_dense_two_card_dispatch_v1.

Verified isolatedanalyzer v4adds FP32-before-BF16 RMSrounding,exactQ/Knorm
vectorlength,unitOresidualscale andF32headpartialnegativecases:fiveactual
plans and70specificmutants pass;previous5/50result andfailedattemptsretained.
No nativecompilerintegration orMoE GPUexecution yet.


Verified isolated StageProgram CPUpreparation addsanexplicitfinal-token tail
predicate forprompt,withmissingpromptpredicate rejected. Ten defaultdecode
syntheticspecs (5layercounts×dense/MoE-shapedstage lists) generatebyte-identical
text tooriginalemitter. Prompt24layer×6stagesspec keeps145stage metadata,
usespast==63 forheadtail,andwrapsnextQKVweightlayerto0. Thisis emittertext
verification,not acompiledpromptkernel,ABI,monotoniccounterorGPUlitmus
result. Rootgeneratorunchanged. Raw:moe_stageprogram_cpu_preparation_v1.
