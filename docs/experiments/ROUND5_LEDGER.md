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
| A2 | QP runtime_task_refs/runtime_wait_entries；split/ownership/attention映射；所有fixture/seq/split与runtime逐值对账；供B复用 | 触发停止门槛 | split1 3000/3000归档计数相等；split2首组100/100。补采95/150格后停止：94通过，gqa2 S512/p0/split16 的L2失配223287，原二进制未改。已证明split坐标顺序不一致，尚未修复；全矩阵未验收 | EVENT_COST/runtime_projection/result.md；f8fba8e、ab5b706、146800a、ab8ab21 |
| A3 | CG逐task count/read/write；从输出索引识别reduce/parallel轴；QP work；GEMM count/MainloopBytes位一致 | 未开始 | 未验证 | COST_MODEL/result.md待更新 |
| A4 | 单位置签名schema；GEMM/attention/RMS/RoPE/SiLU/mul/add/SwiGLU/KV/MoE/Softmax/LayerNorm/GeLU全部语义推导；缺项抛错；op-audit | 未开始 | 签名无法推导须停，不填实测常数 | COST_MODEL/op_audit.txt待生成 |
| A5 | shared/threads/stages从TaskBody traits暴露，消除本地假设 | 未开始 | 未验证 | 待填 |
| A6 | 统一TaskCostNs；九lane/max/尾波原序不变；QP；旧路径开关；顶层task_sum/combine/barrier/event | 未开始 | 每GEMM stage×1077×2×两dtype位门；非GEMM逐条新旧比值解释；BF16/FP32排名且FP32不降 | COST_MODEL/result.md待更新 |
| A7.1 | attention FLOP/非TC；chunk候选/plan/runtime/iters/combine价格及复用理由 | 未开始 | 依赖A4/A6 | 待填 |
| A7.2 | chunk_extent shared；同步TaskSmem/static_assert/kNonGemmTaskSmem/CtasPerSm；四chunk资源与F40 | 未开始 | 未验证 | 待填 |
| A7.3 | 2模型×2seq×4chunk×50=800新进程BF16；chunk1哈希同基线；预测/实测排序；长上下文attention新旧占比 | 未开始 | 未运行 | COST_MODEL/result.md待更新 |
| A8 | CG wait与volume定价Interface，实际使用两个tile；旧Carry开关；spread非零及per-op收益；若零解释并反事实轴 | 未开始 | 依赖A1，未验证 | COST_MODEL/result.md待更新 |
| A9.1 | notify/poll各拟合stages/max_worker/total结构；≥12格25轮四臂；4090运行/5090脚本；LOO残差 | 未开始 | 不复用过原点单斜率 | EVENT_COST/result.md及run_sm120.sh待更新 |
| A9.2 | coupling_metrics QP消费A2投影；L1保留；κ仅改wait；fence/fusion接口零且带缺失理由 | 未开始 | 未验证 | 待填 |
| A9.3 | κ0/1 event差非零且符号正确；具体数值；B后补fusion/placement两门 | 未开始 | A阶段必过κ门；其余待B | 待填 |
| A10 | 残差不作为B入口门，报告而继续 | 已验证 | 已登记执行规则；不代表A9实现 | 本表 |
| A11 | A1后SEMANTIC/P3/derive重跑；14边/44边440格逐条影响；OWNERSHIP与labeling来源只审计不乱重跑 | 已验证 | 6份derive/4份wiring/4份normalization codegen；440格仍4命名差；runtime poll与volume×count reach未受wait修正影响；补跑验证fanout/count/volume均未变 | INCIDENCE/history_audit；a723d9a |
| A12.1 | 新isl路径scoped guard，实际错误分支零残留 | 进行中 | A1 ComputeMetrics及A2无效grid实走错误分支before0/after0；工具remaining0；未覆盖全部新增错误出口，不作全量关闭 | INCIDENCE及EVENT_COST/runtime_projection；ab8ab21 |
| A12.2 | FP32-partial combine微基准实测速率替代解析extra；缺失reason | 未开始 | 未运行 | 待填 |
| A12.3 | barvinok未跟踪检查；ignore或清理，保留用户内容 | 已验证 | 8个未跟踪autotools文件按精确路径移至可恢复临时目录，前后SHA256一致，未动gitlink/跟踪文件 | EVENT_COST/runtime_projection/autotools_cleanup.md |

## B：入口未通过，不提前实现

| ID | 范围与验收（不可删减） | 实现状态 | 验证状态/依赖 | 证据、commit |
|---|---|---|---|---|
| B1.1 | CouplingRelation复合/I1；外部中间写回合法性；fanout重算、索引导出tile约束、max scratch+跨界tile/live regs residency、消费者task数wave四代价；事件与流量收益 | 未开始 | 待A6/A9 | FUSION/result.md |
| B1.2 | 相邻单生产者区间DP；融合在求解内；GemmStages唯一性 | 未开始 | 待B1.1 | 待填 |
| B1.3 | L-task FusionPass重建任务及L-sched；opt独立调用verify；新拓扑A1门；混合签名组合规则进op-audit | 未开始 | 待B1.2 | 待填 |
| B1.4 | 两条真实融合BF16各50进程；稳态时间/资源/spill/事件/schedule；预测胜负相符；A9.3融合差非零正确 | 未开始 | 待B1.3 | FUSION/result.md |
| B1.5 | sm_120融合脚本，预测与实测同输出，状态轮转；只写不跑 | 未开始 | 待可执行B1路径 | FUSION/run_sm120.sh |
| B2.1 | 六例全消费者同CTA的fence_free_producers；并列same-worker边；价格只用前者 | 未开始 | 待A6/A9入口 | AFFINE_PROBE/result.md |
| B2.2 | 参数域worker跨度证明或L-sched强制grid≤resident；lowering检查；不能凭有限采样 | 未开始 | 写回硬前置 | 待填 |
| B2.3 | 复用A2投影覆盖30/60stage、split与attention chunk，不另写固定图 | 未开始 | 待A2 | 待填 |
| B2.4 | 纯L-sched属性/lowering，第四placement开关；原三对照；Validate/I3/E2E_SCHEDULE全字段保留 | 未开始 | 待B2.1–3 | 待填 |
| B2.5 | BF162模型×2seq×50=200进程；时间/调度字段；预测差与实测差；A9.3 placement差非零正确 | 未开始 | 待B2.4 | PLACE/result.md |
| B2.6 | sm_120 placement脚本只写不跑，状态轮转 | 未开始 | 待可执行B2路径 | PLACE/run_sm120.sh |
| B3.1 | cache实测分段曲线/CDF开关；BF16/FP32全配置ρ/top-k/逐点差；BF16变差保留报告 | 未开始 | 待A入口，不提前开工 | PARAMETRIC/result.md |
| B3.2 | live-lane ≤二次闭式根，>2报错；cache/ceil/lane分段并集；符号DP包含全部决策；residency链外；有限(b)仅对照S1..16选择同 | 未开始 | 待B3.1、A6/A9 | PARAMETRIC/result.md |
| B4.1 | 承接A12.1未完成项 | 未开始 | 与A12.1同一任务，不重复计数 | 待填 |
| B4.2 | 承接A12.2未完成项 | 未开始 | 与A12.2同一任务 | 待填 |
| B4.3 | 承接A12.3未完成项 | 未开始 | 与A12.3同一任务 | 待填 |

## 容易遗漏清单（prompt A/B §1.1）

- [ ] A-C1：GEMM逐stage位一致，而非仅模型输入相等。
- [ ] A-C2：算术签名缺项报错，无零/默认值。
- [ ] A-C3：chunk同步union和occupancy。
- [ ] A-C4：Interface实际使用两端tile。
- [ ] A-C5：标定不是过原点四格单斜率。
- [x] A-C6：历史审计限推导侧；A11证据见INCIDENCE。
- [ ] A-C7：work从第一版即QP。
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
- [ ] A-D6：EVENT_COST/run_sm120.sh。
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
A2符号实现已提交，但原runtime split坐标错误触发停止：capture第95格失败，
没有继续GPU采集；首条入边192/256个task漏等实际所需行。修复须统一CG与runtime
坐标顺序，不可切kAll规避，随后重做逐值对账与≥50新进程正确性。详见A2报告。
A3–A9未实现，B未启动；没有后台实验进程。未完成项不得因交接被抹去。
