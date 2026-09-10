# Round 5 A/B 逐项台账

基线 c1281b9，分支 tilemega。本轮未完成。每次恢复先读本表并检查实际进程。
来源完整读取：`/root/Prompt/TileMega_round5_promptA.md`、`TileMega_round5_promptB.md`。
SHA256 A=`336983f2747597a7b89efdfe563e437451f93a910543cf3ac5e8a607e4e22ed3`；
B=`50f53fe6c76d42ca01cd6f5fcb8ba1bf230bf58703e1affbb0133d1a8cf79b11`。

状态：未开始／进行中／已验证／触发停止门槛／待外部条件／经用户批准取消。
实现与验证分列；代码存在不等于完成。未验证不得打勾；历史数据不得冒充本轮运行。
commit 列记录实现/证据提交；本表自身记录提交不算功能实现。

## 执行规则与范围

- A0 最先检查；A1+A2 优先，随后 A3–A6、A9、A7、A8、A11、A12。
- B 全部登记但不提前开工，必须先通过 A6/A9；A2 投影仅实现一次供 B 复用。
- 停止门只停止相应项及依赖项，独立项继续。不得用此规则绕过 B 的入口门。
- 用户再次明确：不能因单项失败结束整份prompt；有明确修法时可在该项内尝试，
  记录失败、修复及重新验收。A2恢复为坐标修复中的局部停止，A4/A5等独立项继续。
- A10：标定残差不阻塞下游，但 A9.3 非零且符号正确的功能门必须通过。
- A9.3 用户已批准按实际κ语义修订方向判断，并要求记录分析流程：κ=0是
  aggregate特例，不是正整数粗化序列起点。保留κ0/1非零预测差，方向与精确
  runtime等待数一致，另报正整数κ粗化；不改运行时κ定义、不用系数凑方向。
  依据：gqa2 seq4/128，κ0 waits=244/4520，κ1=500/16292，后者已与归档日志一致。
- 条件7按新 prompt 关闭为“判据产物，已归因”；T2.d 数值可行域线取消。
  保留历史失败日志，不修改数值容差、不再扩展深度/宽度诊断；A0 是明确例外。
- 拒绝重启两级原子扇入、TC 永不获胜、occupancy1→2；不运行 sm_120 脚本。
- 所有对照 runner 轮内交错完整构建状态；同步证据≥50全新进程。
- 每个修改独立开关；按 CLAUDE.md 分类别 commit，单行，无身份/工具附注。

## A：实现、闸门与证据

| ID | 范围与验收（不可删减） | 实现状态 | 验证状态/剩余项 | 证据、commit |
|---|---|---|---|---|
| A0 | 原始失败 split8 同输入、同 golden 线程；CPU common-FP32 三比较及 k；≤1.003关闭，>1.2报告，其余不擅定 | 已验证 | 用户授权一次原二进制采集；hash/diff完全复现；56线程golden逐位复现；k=.9961308506568204，条件9归因关闭 | REALMODEL/condition9_noise/result.json、condition9_result.md；ab91cc8、b4a1e69 |
| A1 | wait 求交 actual producer domain；所有fixture每边参数网格 Σwait=Σfanout；前后逐边表；无512下游修正 | 已验证 | 参考8图2505/2505、生产2模型1920/1920；OFF372格不等；ON26/26 CTest；物理C同步写回保持verifier | INCIDENCE/result.md及逐边表；148ed7e、a723d9a |
| A2 | QP runtime_task_refs/runtime_wait_entries；split/ownership/attention映射；所有fixture/seq/split与runtime逐值对账；供B复用 | 当前结构已验证 | 原tile ownership矩阵7500/7500、15000计数相等；追加element ownership+双runtime tile variants矩阵7500/7500、15000计数相等。新attention chunk/placement须另过门 | 修复77c942e、1cfab978；EVENT_COST/split_order_matrix、element_variants/result.md |
| A3 | CG逐task count/read/write；从输出索引识别reduce/parallel轴；QP work；GEMM count/MainloopBytes位一致 | 当前任务族已验证 | 用户批准双域；4308组1357020项GEMM work门，648生产/48合成精确元素格；完整CG语义4/4 codegen字节不变；生产TaskCost已消费。新split attention工作另归A7 | COST_MODEL/round5_work.md、element_work.md、scalar_work/result.md |
| A4 | 单位置签名schema；GEMM/attention/RMS/RoPE/SiLU/mul/add/SwiGLU/KV/MoE/Softmax/LayerNorm/GeLU全部语义推导；缺项抛错；op-audit | 当前签名已验证 | 单表14签名schema零失败、4真实错误出口零残留；4个独立TaskBody缺失/占位明确拒绝执行定价。A6真实消费，混合签名扩展归B1.3 | 5c869fc；COST_MODEL/op_audit.txt、scalar_work/result.md |
| A5 | shared/threads/stages从TaskBody traits暴露，消除本地假设 | 当前TaskBody已验证 | traits、scalar控制流DAG与union消费已接入A6；BF16线程128；实际CUDA合约重编译通过，原10/10 SASS/资源比较保留。chunk/fusion新增资源另随对应任务验证 | TaskResources.h、ScalarDataflow.h；COST_MODEL/scalar_work、stage_price_gate |
| A6 | 统一TaskCostNs；九lane/max/尾波原序不变；QP；旧路径开关；顶层task_sum/combine/barrier/event | 硬门已验证 | 4308/4308组，显式及实际TaskStageNs入口各904680位比较相等；14标量格、13错误分支零残留。FP32全1077×2排名不降；BF16历史770/462子集排名不变（非新oracle）。四份完整DP计划与旧路径逐字节相同；默认统一路径后33/33测试通过 | 1264431e、ccdb90d8；COST_MODEL/stage_price_gate、scalar_work、unified_rank、unified_solver |
| A7.1 | attention FLOP/非TC；chunk候选/plan/runtime/iters/combine价格及复用理由 | 数值原型实施中 | 已实现四阶段QK/normalize/PV/combine原型，保留score/prob两处BF16舍入；50/50新进程数值单测通过，但生产plan/DP/投影/价格未接入。原combine占位已识别，不当成已有实现 | COST_MODEL/attention_phases，非完整模型验收 |
| A7.2 | chunk_extent shared；同步TaskSmem/static_assert/kNonGemmTaskSmem/CtasPerSm；四chunk资源与F40 | 未开始 | 未验证 | 待填 |
| A7.3 | 2模型×2seq×4chunk×50=800新进程BF16；chunk1哈希同基线；预测/实测排序；长上下文attention新旧占比 | 部分验证 | 长上下文占比诊断已完成；数值原型50进程不是800生产模型门。后者及排序仍待接入 | COST_MODEL/scalar_work/result.md、attention_phases |
| A8 | CG wait与volume定价Interface，实际使用两个tile；旧Carry开关；spread非零及per-op收益；若零解释并反事实轴 | CPU功能已验证 | 两模型真实残差边M32→128为19.2307693963ns，其余三组合0；八候选spread615.3908806976ns。跨非相邻GEMM残差使旧链不再精确，新增frontier DP；两模型×16穷举选择位相同，统一价格组合路径复核通过。不是全1077候选性能或GPU收益 | 8cb78e55；COST_MODEL/interface_work/result.md |
| A9.1 | notify/poll各拟合stages/max_worker/total结构；≥12格25轮四臂；4090运行/5090脚本；LOO残差 | 已验证 | 600/600正确性、1200四臂进程；12格LOO完成，poll误差最高95.06%。两个最长队列系数均为0，保留不凑系数。sm120只写脚本未运行 | EVENT_COST/round5_structured.md、calibration_round5_fit |
| A9.2 | coupling_metrics QP消费A2投影；L1保留；κ仅改wait；fence/fusion接口零且带缺失理由 | 具体价格已验证 | exact variant输入/缺失rates拒绝；L1独立保留。L2候选级DP转移未实现且显式拒绝，不能当作A6或符号DP通过 | b08cd50；CostModel.cpp:475 |
| A9.3 | κ0/1 event差非零且符号正确；具体数值；B后补fusion/placement两门 | A阶段已验证 | 两模型×6seq的κ0/1均非零且符号正确，κ1→2下降；36/36代入位相同，队列字段1200/1200；B两门尚未做 | EVENT_COST/calibration_round5_fit/functional_gate.json |
| A10 | 残差不作为B入口门，报告而继续 | 已验证 | 已登记执行规则；不代表A9实现 | 本表 |
| A11 | A1后SEMANTIC/P3/derive重跑；14边/44边440格逐条影响；OWNERSHIP与labeling来源只审计不乱重跑 | 已验证 | 6份derive/4份wiring/4份normalization codegen；440格仍4命名差；runtime poll与volume×count reach未受wait修正影响；补跑验证fanout/count/volume均未变 | INCIDENCE/history_audit；a723d9a |
| A12.1 | 新isl路径scoped guard，实际错误分支零残留 | 进行中 | A1 ComputeMetrics及A2无效grid实走错误分支before0/after0；工具remaining0；未覆盖全部新增错误出口，不作全量关闭 | INCIDENCE及EVENT_COST/runtime_projection；ab8ab21 |
| A12.2 | FP32-partial combine微基准实测速率替代解析extra；缺失reason | 已修复并验证 | 保留原−128ns失败；同kernel的64-launch配对graph解决分辨率，50/50新进程且每次完整1048576输出CPU位核对。四系数已发布sm89，缺目标仍明确报not_calibrated；独立旧解析开关保留。FP32预测/排名逐字节不变 | ee93cd44、77a8148e、f8088813、35a6732b；COST_MODEL/partial_combine.md |
| A12.3 | barvinok未跟踪检查；ignore或清理，保留用户内容 | 已验证 | 8个未跟踪autotools文件按精确路径移至可恢复临时目录，前后SHA256一致，未动gitlink/跟踪文件 | EVENT_COST/runtime_projection/autotools_cleanup.md |

## B：A6/A9入口已通过，后续逐项实施

| ID | 范围与验收（不可删减） | 实现状态 | 验证状态/依赖 | 证据、commit |
|---|---|---|---|---|
| B1.1 | CouplingRelation复合/I1；外部中间写回合法性；fanout重算、索引导出tile约束、max scratch+跨界tile/live regs residency、消费者task数wave四代价；事件与流量收益 | 部分实施 | 精确访问复合、fanout重算量、索引tile单生产者约束、外部唯一写回检查已实现，符号单测及5错误分支零残留；四成本及生产CG适配未完 | a496c870；FUSION/round5_access_design.md |
| B1.2 | 相邻单生产者区间DP；融合在求解内；GemmStages唯一性 | 未开始 | 待B1.1 | 待填 |
| B1.3 | L-task FusionPass重建任务及L-sched；opt独立调用verify；新拓扑A1门；混合签名组合规则进op-audit | 未开始 | 待B1.2 | 待填 |
| B1.4 | 两条真实融合BF16各50进程；稳态时间/资源/spill/事件/schedule；预测胜负相符；A9.3融合差非零正确 | 未开始 | 待B1.3 | FUSION/result.md |
| B1.5 | sm_120融合脚本，预测与实测同输出，状态轮转；只写不跑 | 未开始 | 待可执行B1路径 | FUSION/run_sm120.sh |
| B2.1 | 六例全消费者同CTA的fence_free_producers；并列same-worker边；价格只用前者 | 离线已验证 | 六例78映射通过；seq4不改善，128/256时35→541且队列22不变。fence未标定折扣仍0；未声称GPU免fence通过 | 63c366f1、9a09cf21；AFFINE_PROBE/fence_producers/result.md |
| B2.2 | 参数域worker跨度证明或L-sched强制grid≤resident；lowering检查；不能凭有限采样 | 未开始 | 写回硬前置 | 待填 |
| B2.3 | 复用A2投影覆盖30/60stage、split与attention chunk，不另写固定图 | 未开始 | 待A2 | 待填 |
| B2.4 | 纯L-sched属性/lowering，第四placement开关；原三对照；Validate/I3/E2E_SCHEDULE全字段保留 | 未开始 | 待B2.1–3 | 待填 |
| B2.5 | BF162模型×2seq×50=200进程；时间/调度字段；预测差与实测差；A9.3 placement差非零正确 | 未开始 | 待B2.4 | PLACE/result.md |
| B2.6 | sm_120 placement脚本只写不跑，状态轮转 | 未开始 | 待可执行B2路径 | PLACE/run_sm120.sh |
| B3.1 | cache实测分段曲线/CDF开关；BF16/FP32全配置ρ/top-k/逐点差；BF16变差保留报告 | 负门已触发 | BF16历史770/462子集ρ分别−.00022857/−.00003006；FP32全1077×2预测字节不变。曲线保留OFF，不默认替换CDF；不是新BF16全oracle | ebace9bf、2a0df628；PARAMETRIC/cache_curve/result.md |
| B3.2 | live-lane ≤二次闭式根，>2报错；cache/ceil/lane分段并集；符号DP包含全部决策；residency链外；有限(b)仅对照S1..16选择同 | 未开始 | 待B3.1、A6/A9 | PARAMETRIC/result.md |
| B4.1 | 承接A12.1未完成项 | 未开始 | 与A12.1同一任务，不重复计数 | 待填 |
| B4.2 | 承接A12.2未完成项 | 未开始 | 与A12.2同一任务 | 待填 |
| B4.3 | 承接A12.3未完成项 | 未开始 | 与A12.3同一任务 | 待填 |

## 容易遗漏清单（prompt A/B §1.1）

- [x] A-C1：GEMM逐stage位一致，两入口各904680位比较。
- [x] A-C2：算术签名缺项报错，无零/默认值；op-audit及新增实走错误分支。
- [ ] A-C3：chunk同步union和occupancy。
- [x] A-C4：Interface实际使用两端tile；非相邻残差因子进入精确frontier DP。
- [x] A-C5：十二格三特征结构拟合，零系数与大残差原样记录。
- [x] A-C6：历史审计限推导侧；A11证据见INCIDENCE。
- [x] A-C7：work从第一版即QP；运行时标量所有权也使用关系复合和barvinok计数。
- [ ] B-C1：fanout重算定价。
- [ ] B-C2：shared活跃期max加中间tile。
- [ ] B-C3：fence按producer而非边。
- [ ] B-C4：I3写回前解决。
- [ ] B-C5：Fusion L-task/Place L-sched分层。
- [ ] B-C6：融合拓扑重过A1恒等式。
- [ ] B-C7：两份sm120脚本仅编写。

## 交付物对账

- [ ] A-D1/B-D1：代码、独立开关矩阵、分类commit。
- [x] A-D2：REALMODEL/condition9_result.md（A0），ab91cc8、b4a1e69。
- [ ] A-D3/B-D6：EVENT_COST/result.md（A2/A9及B补齐功能门）。
- [ ] A-D4：COST_MODEL/result.md（A3–A8全部门/表/排序/占比）。
- [ ] A-D5：COST_MODEL/op_audit.txt零失败。
- [x] A-D6：EVENT_COST/run_sm120.sh（未运行）。
- [ ] A-D7/B-D8：FINDINGS本轮条目，历史负结果保留。
- [ ] A-D8：skeleton统一形式/签名层/条件4、7、9准确状态。
- [ ] A-D9/B-D10：T0_T4_STATUS更新并链接本表逐项检查。
- [ ] B-D2：FUSION/result.md实测。
- [ ] B-D3：PLACE/result.md实测。
- [ ] B-D4：AFFINE_PROBE/result.md新指标/I3。
- [ ] B-D5：PARAMETRIC/result.md实测曲线/(a)/(b)对照。
- [ ] B-D7：FUSION/PLACE两份sm120脚本。
- [ ] B-D9：skeleton Fuse四代价、Place双目标与阶段条件。

## 恢复点

A0已按单独授权完成一次原二进制采集及CPU三比较。A1恒等式参考2505/2505、
生产1920/1920；旧生产372/1920不等；A11归档完成。最新全套27/27通过，policy
通过、target-audit 5目标0失败，FP32输入2154与BF16输入1540位门通过（非A6门）。
A2原runtime split坐标错误触发局部停止后，已实现CG顺序的独立开关修复；
首条入边192/256个task漏等的反例仍保留。聚焦反例旧臂0/50、修复臂50/50，
完整修复臂150格×50进程矩阵7500/7500；完整S/P投影与全矩阵15000/15000计数相等。
A3已得到物理/名义双域QP原型：seq4的4224 vs 8192 B/iter反例已报告，
用户批准分别用于实际访存与旧collective定价，A6历史GEMM位一致门不变。
生产访问关系补全与价格消费尚未完成。split局部跨度和内层K padding已有独立检查；
全1077配置work门已两模型两dtype4308/4308组通过，不是A6价格门。A4声明/audit及A5资源traits已有分项提交与CPU验证，
不等于A6通过。A9采样、结构拟合及κ功能门完成；L2候选级DP转移仍未实现。
续跑只补齐已核验前缀之后的缺项，原中断记录未覆盖。
最新全套29/29、policy、target-audit五目标0失败；A6/A7/A8与B未完成。
B仍因A6未通过而不启动。A12.2独立局部停止：固定开销未分辨，未发布系数。
停止门槛只停止当前项及依赖项；继续独立项，有明确修法可修复后重验并记录。

### 最新检查点（上述恢复记录为历史，不覆盖失败证据）

A12.2局部停止已由配对graph修复并经50新进程独立核对，sm89实测速率发布。
A3的生产CG语义传输、split物理读集和候选级输入已接入；A6正在消费这些输入，
并非“携带即消费”。A6统一显式GEMM价格904680位门已过，实际缓存入口的完整
再核对仍在运行。非GEMM14格CPU检查（两种所有权）、13条错误出口零残留；
两次导出域拒绝原样保留：BF16 past2048超出512上界、FP32历史域只有1..8。
未放宽域，长上下文采用prompt列出的BF16512/512。
A8的非零两端shape价格和exact-frontier DP已验证；独立实验开关默认仍保留旧路径。
最新33/33 CTest，policy通过、target-audit五目标零失败。字面`ninja -C build`
因预存MLIR-OFF缓存失败；正确配置`build-portable`的policy通过，不冒称前者通过。
A2追加element ownership+运行时两variant的7500新进程矩阵仍运行中；冻结二进制
与投影工具未重编，未将未结束矩阵报满分。A7及B尚未宣告完成或解除入口门。

**入口更新**：A6完整四模型/dtype组门已结束，两入口各904680比较全过；
默认统一价格33/33 CTest、重编的BF16/FP32 TaskBody契约均通过。
A6/A9入口现已解除（1264431e、ccdb90d8）。A7与B可以启动；旧“待A6”行仅指
此前依赖，不再是当前阻塞理由。A2扩展矩阵仍独立运行。禁止把BF16历史子集的
排名失败混成A6的GEMM位门失败，或用它终止独立后续任务。
