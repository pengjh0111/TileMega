# R13 sm_89 — Phase D 结果验收（待两轮金丝雀复测封版）

**代码与原 A/B/C/D 队列已结束；全轮性能验收尚未封版。** 终版四格 C-1/C-2 通过；验收回放发现两轮 vLLM 金丝雀超过预注册 2%，已准备同二进制、同顺序的整轮替代测量。不能把“原队列结束”写成全项 PASS。

## 基本信息

- 基线：`48c013d1863d5a5af249401be217d4aa4c15d710`。
- 外部 prompt：`/root/Prompt/TileMega_R13_prompt.md`；SHA256 `5adb32833673d7dcd18e6ec02999be9c6bd9ca3216c95f9d9f2b136f8329ace9`。
- 固定 target：`/root/r13_work/target_r12b.json`；SHA256 `c2c03ad6e9534130762cc88d423aac336077a0bd040db6d337a1e31e3c1ec8b9`。
- 设备源码冻结：`dedd849f3`；此前推送检查点 `36f17e6ee`，距基线 78 个提交。最终 HEAD/提交数待两轮替代测量接受后记录；本次只修改 CPU 验收工具与文档。
- Phase A：独立 `/root/r13_work/diag`，C++ 为基线，Python 含 TL-1。B/C/D 固定产物均登记 SHA256；未因本次工具修正重建。
- `R13_plan.md` 未找到，执行依据为保存的 R13 prompt。sm_120 路径在本机只宣称已实现、已编译。

## 逐项实现与验证

| ID | 状态 / 提交 | 关键代码与证据 |
|---|---|---|
| BL-1 | 完成；cd94bcac7 | `serving/execution.py:6`、`cli.py:438`；watchdog 默认 0，旁车和键过滤，T9 两级选择 |
| PD-1 | 完成编译/位置检查；1c99253eb、8dd3c6e12 | `ModelHarness.cuh:1348`、`ServingRuntime.cuh:381`；五架构编译、sm_89 SASS、四项 PTX CPU 回放通过；其他硬件未执行 |
| LP-1 | 完成；dd8ea6b23 | `ModelHarness.cuh:1401`、`ServingRuntime.cuh` 的 loop_modes/launch_steps；B5 4 格 C-2 与 50/50；旧 ABI 回退、实际 loop 标志已测 |
| FX-21 | 完成；aab46bb4d、a7f75b318 | `Frontend.cpp:795`、`HandoffPass.cpp:443`；48 条旧拒绝已归档，PSA 归约/四格 C-1 及 50/50 通过 |
| FX-22 | 完成；653b0e848、8954bc431 | `TraceReducer`、`page_chain.py`；实际 E2E_STAGES 与动态 LA 发布重建关键链 |
| TR-1 | 实现/采集完成，开销门未达；55d4287e2、6b482d015 | `ServingTrace.cuh:64`、`ledger.py`；T4 原始账本已收集；trace 比无插桩高 3.10%–6.63%，不能当作精确无插桩分解 |
| TR-2 | 完成；517c127b5 | `ModelHarness.cuh:2634`；非分页 grid_stride/rotate，B5 token 一致 |
| TR-3 | 完成；a95fe309e、8c76f2a03 | `ServingTrace.cuh`、`ServingPages.cuh`、T6；逐步与循环边界、lag/barrier 明确分开 |
| PG-2 | 完成；761dc161d | `SkeletonSearch.cpp:921`；分页四个 D1 shortlist 均有原种子和 seed_split1；投影单测通过 |
| SL-5 | 完成；58b80a226 | `compile.cpp:347`、`cli.py:411`、旁车及 plans.json；联合选择已实际执行 |
| TL-1 | 完成；1ff6c3535 | engine/measure；prefill 独立、step-events、decode_loop_used、旧二进制加载已测 |
| TL-2 | 完成；d207ab0a1、3db575e6b | dram_ceiling.py；5 个干净新进程，标定口径同时写 target 两处 |
| MB-1 | 完成 sm_89 测量及五架构编译 | tools/experimental/loadbench；六子项，bulk/cluster/PDL 在 sm_89 不宣称运行结论 |
| CM-1 | 报告完成，修正重排为 inferred | SkeletonFinalize.cpp、cm_report.py；T12 配对、固定/字节/几何拟合；默认代价模型未改 |
| Phase C | 完成实现、矩阵及保留决定 | 391c86843、bd8d01cd9、10ded66d3、dedd849f3；phase_c_retention.json |
| Phase D | 构建/正确性完成，性能待两轮封版 | 原 D2 12/12，D3 R13F 四格通过；queue_final_review.json |

## 一致性、访问与加载审计

verified：Carch 的 19 个编译产物全成功、FP64 为 0；sm_90/120 × trigger 0/1 的 PTX 位置检查 4/4。旧检查把 ret 后不可达冷分支算成触发后的写入；改为可达控制流检查，未修改 PTX。非分页/分页参考的已有内核源码、资源、SASS 三项均相同；资源比较按“入口内核::被调函数”区分新增循环核调用上下文。sm_89 的 auto/off PDL 原检查通过，Caps 的 kPdl=false。证据：`raw/final_review/{pdl_replay.json,Cidentity_nonpaged/,Cidentity_paged/}`；原失败保留。

LP-1 inferred 论证：步末使用与现有跨阶段相同的 leader acquire + compute/grid barrier。GEMM 可变 A 用 cp.async.cg 只经 L2；普通 C++/向量加载沿用跨阶段协议。没有新增 nc/restrict 读取。verified：五架构新 L1 loop 的 SASS NC/CONSTANT/CI 加载清单均为 0，见 loop_nc_audit.json；sm_89 新进程 50/50 支撑本机同步结论，其他硬件待执行。

FX-21 采用 c1 私有访问见证：从实例化的 partial/combine 实际访问生成 split_access_semantic，只在证明克隆中提升为 L-sem；原图与定价不变，recompute 不看到公开的新语义。证明禁止其他消费者读 partial；逻辑输出消费者等待 reducer 事件，由 LA 归约后代为发布；没有合成 GEMM 自环。verified：PSA 实际 elided 为 Llama 65（17+48）、Qwen3 113（29+84），包含 split combine；运行时表及诊断见 T2/T11，不能按 split 数推算代替日志。

## T1–T12 验收索引

全部表位于 results/；大表原件在 raw/final_review 的归档中。数值口径为 (E2E−TTFT)/1023，三轮中位数与极差；与 vLLM 比值逐轮配对。以下 D2 数字为**两轮替代测量之前的初步值**。

| 表 | 采集与结论 / 原始证据 |
|---|---|
| T1 | A1 四格三轮完成；PF-R10-noWD 四格胜出。A1 B0-pfR10 TPOT ms：2.8531/3.1795/4.3818/5.7363；TTFT ms：4.170/21.661/6.178/32.626 |
| T2 | A2+B4 执行器/分页同轮矩阵；E2E_STAGES 与被调函数 ptxas 行全部保存；PS/PSA/PR 不混为同一几何 |
| T3 | MB-1a–f 完成；5 个新进程、污染 0；标定口径 979.9776 GB/s、进程极差 0.6011 GB/s（0.0613%）；短工作集重复拷贝最大值 1102.8417 GB/s，不能称为物理 DRAM 带宽 |
| T4 | 63,424 条阶段记录，三个 past、四格、B0/PR 并列；已修复 B16 被子串误认 B1 的 CPU 汇总，31,072 条 B16 记录正确归类；trace 开销未达 2% |
| T5 | B2 放置/漂移、逐 task 等待/运行/发布、队头阻塞；原 slots/events/waits 与大表归档，HOL 重叠项不可直接相加为步时 |
| T6 | B3/TR-3 与 MB-1e：无事件、普通 launch、L1 loop、分页 loop；按步和 past 分桶，循环未成为 D1 最优 |
| T7 | 12 份分页原始 trace、LA 关键链；已按正确 B16 字节重算下界，满环且等依赖是 CTA 本地时间，不能跨 CTA 相加 |
| T8 | 五项 C 实际矩阵 54 轮；保留 C-LP2、C-PG3；C-L2b/C-WL 关闭；C-AT Qwen3 B16 固定优胜结构单列，D1 自行选 Ec/Rq |
| T9 | D1 两级全样本与 shortlist、origin、fidelity；所有搜索 shortlist 有 seed，分页还有 seed_split1；固定单候选构建不要求 seed |
| T10 | 原终版 4 格×3 轮完成；vLLM 错误的 TM-loop 后置条件已按 clean guard+原 metrics CPU 回放；2 轮金丝雀重排 |
| T11 | B/C/D 本机协议共 300 个不同 PID，300/300；D3 R13F C-1/C-2/repeat 四格均通过；对照 Llama B1 B0h/B0-D 的真实 C-1 失败保留 |
| T12 | 修复精确 batch 路径后 309,146 条 task run 价格/实测配对；分项与几何残差、候选 τ 输出；不含等待/发布，重排只作 inferred 报告 |

### 带宽口径（verified / inferred 明确分开）

verified：约 980 GB/s 的大工作集标定核在五进程间稳定，981.6 的旧读数相容，884.5 的回落本次未复现。1102.84 是 256 MiB 重复多遍的有效拷贝吞吐；受缓存/重复访问影响的可能性为 inferred，不把它用作唯一 DRAM 下界。T1–T7 保留规格要求的 884.5、981.6、原 MB 最大值三列，并明确第三列是有效吞吐参照。

单驻留 MB-1b：1/2 loader warp 最大 981.152/980.909 GB/s；C-PG1 未触发。MB-1c 连续布局收益最高 12.94%，触发 C-WL，但 E2E 同轮反而 +2.37%–6.77%，故关闭；加载指令最高改善 0.19%，C-BW 未触发。以 979.9776 作为大工作集口径，四格 DRAM 下界约 2.54/2.83/3.58/4.59 ms；距离明显大于按 884.5 的估计，不再将后者称为本机绝对上限。

### Phase C 预测与决定

| 项 | 实测 TPOT 相对控制臂 | 决定 |
|---|---|---|
| C-L2a | 放置触发条件未满足 | 未做 |
| C-L2b | −0.30% 至 +0.12%，未达到 ≥2 格 −2% | 默认关闭；50/50 |
| C-PG1 | loader 宽度条件未满足 | 未做 |
| C-PG3 | P-D64-first 四格满足 −1.5% | 保留 D=64 KiB、lookahead evict_normal、demand evict_first |
| C-LP2 | 角色循环外提四格改善，满足不慢于逐步 | 保留 page_loop_split=1；50/50；未强制最终使用循环 |
| C-AT | Qwen3 B16 N-AT-512-2 通过性能与 C-1 | 固定优胜报告；最终联合选择仍为 Ec256/Rq2 |
| C-BW | 加载指令条件未满足 | 未做 |
| C-WL | E2E +2.37%–6.77%，未达到 ≥2 格 −1% | 默认 row；50/50 |

初始预测逐项原文与止损见 predictions.json；基线 Qwen3 B1 +1.8422% 超过 1% 门槛，用户授权改为 advisory，未改记性能 PASS。MB 最大有效吞吐超出 896–985 的预测，约 980 的标定口径在区间内；其他预测须按 T1/T2/T6 的同轮控制比较，不能混合 A/B/D 编译器。

### 终版初步数据（两轮金丝雀复测前；四格均 L1 prefill）

| 格子 | 暂定默认 | TTFT ms | TPOT ms | E2E s | tok/s | TM/vLLM |
|---|---|---:|---:|---:|---:|---:|
| Llama B1 | R13F：pages/L1/逐步 | 4.165 | 2.8194 | 2.8884 | 354.52 | 1.0833 |
| Llama B16 | B0-D：非分页/L1/逐步 | 21.663 | 3.1802 | 3.2750 | 5002.82 | 1.1616 |
| Qwen3 B1 | R13F：pages/L1/逐步 | 6.161 | 4.1565 | 4.2582 | 240.48 | 1.0607 |
| Qwen3 B16 | R13F：pages/L1/逐步 | 32.658 | 5.4134 | 5.5706 | 2941.17 | 1.1069 |

TPOT p50/p90、极差、三轮样本、E2E/ΣT_floor、其他对照全部见 T10.tsv。Llama B16 的 R13F 与 B0-D 不可分辨，按原终版规则取 B0-D。分页三个胜者均为 split1：qkv/o/down TN32/TK128，gate_up/head TN128/TK64，stages2；κ1、16 KiB×5 页、D64 KiB；Llama Ec256/Rq4，Qwen Ec256/Rq2。旁车选择没有修改 cache manifest。

C-1：R13F 四格 max gap 为 0/0.25/0.125/0.125，全部通过。Llama B1 B0h 与 B0-D 近并列比例 0.982421875、max gap 10.5625，是真实失败，该格不能回退到未验证对照。D3_protocol 50/50 实际为最终非分页 Llama B16 L1 逐步；不能宣称它覆盖分页循环。LP-1/FX-21 的分页与循环覆盖来自 B/C 的独立 50 进程用例。

## Q1–Q7

- Q1 verified：去看门狗没有保证所有格子加速；Qwen B1 基线回退仍存在。prefill 几何恢复 PF-R10 对 B16 的 TTFT 有明确收益。
- Q2 verified：大工作集标定口径约 980 GB/s，旧 884.5 本次未复现；不同短工作集重复吞吐不等同于 DRAM 上限。
- Q3 verified：阶段账本与尾部已量出；inferred：Qwen attention/merge 与尾部值得下一轮处理。因 trace 3%–7% 开销，不能给出无误差的开销百分比归因。
- Q4 verified：grid_stride 没有满足放置规则，L2 瘦身也未达 2% 保留门槛；inferred：不能把全部 L2 差距归给 EFT 或两道冗余 barrier，须结合逐 task 等待/预取证据。
- Q5 verified：分页同几何与种子分开；split combine LA 实际生效；预排布/loader 微基准收益不保证 E2E，C-WL 的负结果已回退。
- Q6 verified：LP-1 正确、角色循环外提有收益，但联合选择四格都选逐步；inferred：不能以“存在 launch 空泡”推导设备循环必快。
- Q7 verified：D1 在三格选分页 L1、Llama B16 选非分页 L1，均不用循环；终版按可分辨规则回退一格。完整性能结论待两轮金丝雀封版。

## 偏离、工具修正与未达项

- 用户明确授权放宽 Phase B 基线止损，仅放行采集；原失败和 collection_policy.json 保留。
- 原 A/B 有环境 PATH、C-1 out 参数工具错误，恢复时只补缺失项；没有把错误计为正确性通过。
- 本次修正：anchor 只对显式 TM loop 检查；12 条 vLLM 历史后置条件误拒通过原 command/metrics/guard 回放恢复。真正非零退出/污染仍拒绝。
- PTX 检查改为可达控制流；ptxas 同名函数按入口调用上下文比较。六项 CPU 回归通过，原失败不覆盖。
- B16 子串误归类导致阶段字节/CM 配对错误；已精确匹配并 CPU 回放。原 Phase C 预注册输入与决定不重写，实际矩阵/正确性保持原数据。
- trace 开销 >2%，T4 的下界超出量只作定位；MB 短工作集重复速率不作为物理下界。CM 关键链残差重排是 inferred，未更新默认 target。
- 最终两格金丝雀各只允许一次整轮替代；原 round JSON 保留副本与 SHA，替代的 guard/全部臂/二进制定义/token 均通过才作废原轮。复测失败则报告未达，不无限重试性能。

## 队列与恢复

验收开始的调度器终态：198 步、188 done/10 failed、无 running/pending。failed 包含历史工具错误与真实对照 C-1 失败，不能解释成 10 项未实现代码。调度器 PID184785 存活等待队列。

queue_final_review.json：仅 D2_canary_llama_B1_r2、D2_canary_qwen3_B16_r2 两个 GPU 整轮步骤；随后 D_accept_canaries → D_final_replay → D_report_replay 三个 CPU 步骤。由原守卫持锁、空闲预检、污染 75 重排；不重建，不重跑协议或 C-1/C-2。所有输入提交后原子发布至 /root/r13_work/queue/，不另开调度器。

再次唤起先读一次 scheduler/progress.tsv，核对 D_accept_canaries/acceptance.json 与新 T10；无额外失败才记录最终 HEAD、提交数、最终四格数字与全部验收结论，补归档并推送。现在不反复轮询，只剩队列时结束会话。

## R14 方案（只写方案，不实施）

1. `ServingTrace.cuh`/ledger.py：降低每阶段写记录开销，先把插桩误差压到 2% 再做精确账本。
2. `TaskPriceParts.cpp`/StageFlowModel：以约 980 的多进程带宽和 T12 fixed/byte 残差重标定，单列 LA 内联归约定价；不把 critical-chain 修正当成已校准模型。
3. `ModelHarness.cuh` L2 等待/预取：按真实 ready/HOL 与阶段跨度定位，现有瘦身低于止损门槛，不默认推进。
4. `ServingPages.cuh`/loader：保留有同轮收益的 eviction 与角色循环；sm_120 bulk/PDL 必须在对应硬件独立测量，不能外推 sm_89。
5. `ServingEpilogue`/lm_head 几何：定位 Llama B1 对照的 C-1 失败，保留 HF teacher-forced 的原失败 token/gap；不为消融改默认数值判据。
