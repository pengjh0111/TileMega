# R14 sm_89 收尾：实现与已完成正确性验收；终版性能未验收

- Prompt：`/root/Prompt/TileMega_R14_prompt.md`；SHA256 `c7e5771383873ae450c153d873bc633cf4481d9be5cea0f65e216e5b7073330e`。
- 指定基线：`76beaea5e2d66e3311b36d020f470c4f016406d0`；开工实际 HEAD：`9aebaf6553247ec83c79bc8f101e61ad4ce564fd`，随后快进至指定基线。
- 补充测试冻结源码：`97f7a2f1d`；代码整合 HEAD：`8f4f7a6f0`。最终发布 HEAD 为本报告所在收尾提交（`git rev-parse HEAD`），交付消息给出其 hash；相对基线 522 个提交，其中 R14 侧 120 个、远端既有 402 个（含 dnn-moe）。
- 用户于 2026-10-10 要求停止未完成测试并整理推送。本轮据已有证据收口，没有重新发起 GPU 测量。
- 远端用户合并：`a3f0dc6d10c368dbbe3073c66cc750aec92a70a3`，包含 dnn-moe `0e64c56b010488a7229a75ed79814fb865aef197`；整合保留两条历史，正常推送。
- `R13_review.md` 未找到，以已保存 prompt 的审查结论为依据。

## 实现清单

位置为原 R14 实现；合并后同名入口通过 `TILEMEGA_DM_SUPPORT` 选择 serving 或 `Dm*` 兼容实现。

| ID | 状态 | 提交；关键代码位置 |
|---|---|---|
| Framework / rules | 完成 | `f6599a0c0`, `586c51b5d`；本目录 scheduler.py、gpu_guard.py、predictions.json、choose_r14.py |
| FX-23 | 完成 | `b3a6af1b3`；include/tilemega/Codegen/tasks/AttentionPageLayout.h:14、IndependentAttentionTaskBody.h、PagedAttentionTaskBody.h |
| FX-24 | 完成 | `bb42f31bd`；lib/Solver/StageFlowModel.cpp:46、test/unit/stage_flow_test.cpp |
| FX-25 | 完成 | `7c2cd2436`；python/tilemega/build/identity.py:116、tools/commands/compile.cpp、identity_join.py |
| TR-4 | 完成；补充开销验收限 Llama B1 | `377c674d2`, `5ec398310`, `686dc3afc`, `258d6e73c`；executor/ServingTrace.cuh、Backend/ServingProfiledMainloop.h、ledger_r14.py |
| AT-1 | 完成；硬件联合选择未收口 | `821c64cf6`, `9fa448f3a`；lib/Solver/SkeletonSearch.cpp、python/tilemega/serving/attention_selection.py |
| AT-2 / SK-1 | 完成；要求的三组协议各 50/50 | `c0849f8a2`, `68c592729`, `ac731bc10`, `84def10d5`；tasks/ModelHarness.cuh、executor/MonotonicLastArriver.cuh、SkeletonSearch.cpp |
| AT-3a | 完成 | `0c3d9ac45`, `68410eb88`, `dd7af352c`；Backend/ServingAttentionPVSwap.h |
| GV-1 | 完成；TN8/16 的 DN、SwiGLU 补齐并验证 | `54756a6e3`, `771aea037`, `5ae2ffc8d`, `26d7f4c14`；Backend/ServingGemv.h:11、tasks/ServingGemvTaskBody.h、ServingDeferredNorm.h |
| RW-3 | 完成；默认关闭 | `df96abb7f`；tasks/PagedGemmTaskBody.h、Backend/ServingTiledMainloop.h |
| RA-1 / EP-1 | 完成；已测候选未带来保留收益 | `5434f44bb`, `a8588fb82`, `9f0c36c2d`；PagedAttentionTaskBody.h、Backend/ServingEpilogue.h |
| SL-6 | 实现完成；完整硬件选择未验收 | `e95dbf3ee`, `96205607b`, `a28bbb629`, `06090c9ea`；SkeletonSearch.cpp、python/tilemega/cli.py、serving/integrated_selection.py:56 |
| 搜索状态恢复 | 完成 | `f5ce261bb`；SkeletonSearch.cpp:180；先验证新结构再移动当前状态，拒绝形状不破坏后续求解 |
| C-RW1 / C-EP2 / C-AT4 | 实现并检验，均不保留默认开启 | `1b00e23ef`, `bb48614c7`, `45f482860`；IndependentAttentionTaskBody.h、ServingEpilogue.h、ServingPages.cuh |
| C-AT3b / C-RW2 / C-GV2 | 未做；已收集证据未触发 | phase_c_decision.json、phase_c_retention.json |
| C-PF | 未决定；缺少完整新 D1 证据 | results/T10_final_status.json |
| D1/D2/D3 最终验收 | 未完成，依用户指示取消余项 | results/closure_acceptance.json；scheduler_remaining/state.json |

AT-2/SK-1：写 partial 后发布到单调 ticket，最后到达者 acquire 后读取有效 partial；L1 消费者通过下一阶段 barrier 排序，L2 等 reducer 的代发事件。
GEMV 选择寄存器流读、fp32 累加与 shuffle 归约，写相同的 epilogue tile；窄 DN 使用明确的 8 列平方和 ABI，SwiGLU 按窄 tile 调整交错。
SL-6：候选族覆盖、逐轮淘汰、三轮确认；past 64/575/1000 线性插值后按 64..1087 积分，1000 后保持端点。预算不足以覆盖必需维度时拒绝发布胜者。

## 证据与 T1–T12

所有下列结果为已完成原始数据的 CPU 汇总；没有用未完成运行填表。
历史原始包：raw/phase0_completed.tar.xz、phase_a_completed.tar.xz、phase_b_review.tar.xz、phase_c_completed.tar.xz、logic_completion_completed.tar.xz，均附 manifest。
本次新增：raw/supplemental_closure.tar.xz 与 supplemental_closure_manifest.json，含成功测量、guard、失败/占用重试、取消记录和产物 SHA；不提交二进制与生成 CU。

| 表 | 已交付范围与限制 |
|---|---|
| T1 | results/T1_anchor.tsv、phase_a_acceptance.json：Phase A 全部完成；见下表 |
| T2 | T2_diagnostic.json、T2_stage_ledger.tsv；新增 T2_supplemental_overhead.tsv、T2_supplemental_stage_past*.json |
| T3 | T3_phase_b_resources.tsv：100 个构建身份与资源；新增 T3_closure_controls.tsv；终版尚未选出 |
| T4 | T4_phase_b.tsv：B1–B5 同轮变化；见下表。新窄族只有正确性，未补性能矩阵 |
| T5 | T5_phase_b_tasks.tsv、T5_supplemental_tasks.tsv：抽样运行/字节/就绪数据；混合几何拟合只作诊断 |
| T6 | 部分：attention 分类数据在 T5_phase_b_tasks.tsv；完整逐 Ec/wave/层专表未完成 |
| T7 | 部分：qkv/o/down/gate_up/head 分类数据在 T5_phase_b_tasks.tsv；完整覆盖专表未完成 |
| T8 | T8_phase_c.tsv、phase_c_retention.json：三个触发候选均回退/关闭默认 |
| T9 | T9_d1_selection.tsv 保留历史 under-covered 选择；不可充当新终版，fresh D1 未完成 |
| T10 | T10_final_status.json：无完整 D2 四格终版对比，不作性能门通过声明 |
| T11 | phase_b_acceptance.json、logic_validation.json、T11_supplemental.tsv；见正确性表 |
| T12 | T12_audit.json、raw/FX24/：search-only 复现、修复与守恒检查 |

### T1：修复后基线（TPOT ms）

| 格 | 旧 R13D | R13D' | 重建漂移 | 按规则采用基线 |
|---|---:|---:|---:|---:|
| Llama B1 | 2.82274 | 2.85861 | +1.27% | N1' 2.84817 |
| Llama B16 | 3.18082 | 3.20018 | +0.61% | 3.20018 |
| Qwen3 B1 | 4.16167 | 4.23549 | +1.77% | 4.23549 |
| Qwen3 B16 | 5.41712 | 5.49964 | +1.52% | 5.49964 |

四格重建漂移均超过预测 ±0.5%；token 一致，原因尚未隔离。N1' 的可分辨性按本格极差规则判断，不能把跨会话变化归因于优化。
来源：results/phase_a_acceptance.json 与 phase_a_completed 原始包。

### T2：补充同轮 trace 开销（Llama B1，三轮）

| 臂 | TPOT 中位 ms | 相对基准 | token |
|---|---:|---:|---|
| 无 trace | 2.84684 | — | 基准 |
| stage trace | 2.87236 | +0.90% | 相同 |
| task trace | 2.88365 | +1.29% | 相同 |

本格满足 ≤2%；三臂实际内核均有 spill。不可推广为四格全通过。基准无 >2% 金丝雀标记，成功 guard 接受；这不排除未观测干扰。
早期 timer-only/store-only/full 中位开销分别 +0.13%/+1.57%/+0.53%，存在异常轮，不能唯一归因。
补充 task profile 的非零字节 GEMM 样本均观察到 first-ready；历史缺失值保留为不可用。1/8 CTA 抽样的尾部/极值是估计，负拟合截距不是可解释的固定开销。
来源：results/closure_acceptance.json、T2_supplemental_overhead.tsv、T5_supplemental_tasks.tsv；对应 raw 已打包。

### T3：冻结控制产物实际 L1 逐步内核资源

| 格 | pg | 寄存器 | 栈 B | spill 存/取 B | smem B |
|---|---|---:|---:|---:|---:|
| Llama B1 | l2 | 255 | 160 | 80/88 | 86016 |
| Llama B16 | l2 | 255 | 160 | 80/88 | 86016 |
| Qwen3 B1 | pages | 255 | 176 | 164/408 | 99328 |
| Qwen3 B16 | pages | 255 | 176 | 164/408 | 99328 |

这是 audited control，**不是新选定终版**。身份/执行 ID 见 T3_closure_controls.tsv；资源原文件在 logic_completion_completed 包。

### T4：已完成单因素矩阵

| 格 | 同轮最快的已测臂 | TPOT ms | 对同轮基线 |
|---|---|---:|---:|
| Llama B1 | baseline | 2.84820 | 0 |
| Llama B16 | AT_pv | 3.17325 | −0.98% |
| Qwen3 B1 | AT_ec128 | 4.16402 | −5.19% |
| Qwen3 B16 | AT_pv | 5.62227 | −1.48% |

来源 T4_phase_b.tsv；这些单因素结果没有经过本次完整组合终验，不直接写成最终默认。
已测 fill/GEMV、parallel argmax、noinline 未胜出；RW-pipe 无可分辨收益。Qwen3 跨阶段基准漂移未解释。

### T8：条件项

| 项 | 格 | TPOT 变化 | 正确性 | 决定 |
|---|---|---:|---|---|
| C-RW1 | Llama B16 | +2.44% | C-1/C-2；50/50 | 默认关闭 |
| C-EP2 | Qwen3 B16 | +3.80% | C-1/C-2 | 默认关闭 |
| C-AT4 | Qwen3 B1 | +4.47% | C-1/C-2；50/50 | 默认关闭 |

来源 T8_phase_c.tsv、phase_c_retention.json；resident-2 结构族仍保留在条件要求的候选集合中。

### T11 / T12：正确性与修复验收

| 检查 | 结果 | 证据 |
|---|---|---|
| FX-23 旧码反例 / 新码位置编码 | 旧码 26 项失败；初始修复 768 项零失败；最终 predicate 重验通过 | raw/FX23/、phase0_completed 包 |
| R13D' 与旧默认 | 四格 1024 token × 三轮逐位一致；四格 C-1 通过；N1'/B0h' C-1 通过 | phase_a_acceptance.json |
| Phase B | 72 步完成，100 构建身份通过；50 模型变体 C-1/C-2 通过 | phase_b_acceptance.json |
| AT-2 attention LA | Llama B1 50/50；Qwen3 B1 50/50 | phase_b_acceptance.json protocols |
| SK-1 非分页 combine LA | Llama B16 50/50 | 同上 |
| 逻辑补齐 | 20/20；10 构建、8 次 64 步 token/KV smoke 零差异；默认四路径 CU/resources/SASS 相同 | logic_validation.json、logic_completion_completed 包 |
| 新窄族长检查 | Llama/Qwen3 × TN8 row、TN16 tiled；各 1024 步 C-1 与四臂 C-2 通过，mismatch=0 | T11_supplemental.tsv |
| 多架构 | sm_80/89/90/100/120 编译通过；只执行 sm_89 | implementation/logic 原始包 |
| FX-24 | fixed/joint search-only 通过；守恒检查 6/468；保留修复前失败 | T12_audit.json、raw/FX24/ |

窄族 C-1 near-tie ratio 均为 1.0、max gap 为 0。其第四臂为非分页禁相位，不能声称覆盖分页 K-phase 协议。
FX-24 最小负残差为 −9.0949470177292824e−13 B；cohort 平均扣减因舍入超过 task 自身流量。改为逐 task 归属扣减并检查守恒，不钳零、不改代价模型默认值。
旧 joint 的完整缓存命令缺失，使用已存 config/export 重建且省略不可用 sm120 seed；不是逐字命令重放。

## 带宽、预测与 Q1–Q6

D0 补充标准标定口径：五个新进程均未标 contaminated，中位 **979.97757 GB/s**，进程极差 **0.04187%**。
target SHA256：`14b69fbb3e2111bb8d9618b12449ecf9ae03d34d588ac2f46b250f1fd0c59d79`；原 target、各进程读数/guard 在 supplemental_closure 包。
原始其他方法最大值包含缓存/协议效应，不替代标准 DRAM 上限。没有新终版 E2E，因此不计算 R14F 的四格下界距离。

| 预测 | 已测对照 / 结论 |
|---|---|
| R13D' ±0.5% 且 token 相同 | token 通过；时间漂移 +0.61%..+1.77%，区间未命中 |
| N1' ±3%、C-1 通过 | C-1 通过；旧 N1 正确性无效且不稳定，不作优化归因 |
| AT 全套 | 单因素 Ec/PV 有局部收益；全套终版未测 |
| SK-1 / GV-1 | 已测版本未达到预测收益；补齐窄族只完成正确性 |
| RW-3 / EP-1 | 未观察可保留收益 |
| RA-1 单独消除溢出 | PV 混合了实现与 spill 变化，未单独隔离；noinline 未胜出 |
| 终版对下界、TM/vLLM 门 | 未评估；四格目标不得标通过 |
| 条件项 | RW1/EP2/AT4 都未达保留门槛 |

逐项原注册值与已测样本见 results/prediction_review.json；未隔离/未完成者明确标记。

- **Q1 verified**：不受影响四格 token 一致；N1'/B0h' C-1 恢复通过；Llama B1 基线改为修正非分页。**inferred**：旧错误路径性能不能作有效正确性基线。
- **Q2 verified**：Ec128、PV 的收益局部成立，LA 测试未胜出；加载协助未触发，未实现。
- **Q3 verified**：已测 split-fill/GEMV 未胜出；完整窄族数值已通过。**stated**：补齐实现后的性能没有收集完成。
- **Q4 verified/inferred**：RW-3 未带来可分辨收益；sampled GEMM 数据触发驻留 2，实测回退；不能唯一归因 B16 超出量。
- **Q5 verified**：PV 降低部分分页 spill，Qwen3 B16 同轮改善；noinline 未胜出。**stated**：无新终版 spill 对照。
- **Q6 stated**：多 past 与扩展族已实现、CPU orchestration 通过；早期 D1 under-covered 不算终验，fresh D1 未完成，选择变化尚不能回答。

## 队列状态、偏离与整合

最终唯一补充状态：`scheduler_remaining/state.json`，**12 done / 53 cancelled / 0 pending / 0 running**。
完成：preflight、prepare、四组窄族、trace、三轮 overhead、overhead 汇总、D0。两模型 fresh D1 因 GPU 保护退出 75 多次重排，未产生完整新 plans.json。
停止记录：raw/closure_20261010/stop.json；原始失败、重试、progress 与取消标记均保留。没有把 GPU 占用误报为候选失败。
历史 scheduler 的 failed/skipped 只代表旧尝试；不得据其计数推断本次仍在运行。用户要求不再测，所有旧/补充队列保持停止。
若未来恢复，先读一次 state/progress 与 closure_acceptance.json；需显式重新定义剩余依赖，不能直接把旧 under-covered 胜者作为终版。
state/progress/stop 属运行目录，已收进补充原始包；新 clone 可用 `tar -xOf raw/supplemental_closure.tar.xz docs/experiments/SERVING_R14/scheduler_remaining/state.json` 读取冻结状态，不应据旧队列自动重启。

偏离与影响：
- 16 KiB Independent 私有双缓冲超出 sm_89 共享内存；此数值组合通过 paged transport 验证，实际 Independent 使用合法 8 KiB。
- 旧 R13 产物无新身份，明确登记旧二进制 SHA 与未知来源；新构建强制身份，trace 时间只进诊断表。
- TR-4 用 1/8 CTA 采样；补充 ≤2% 只验证 Llama B1。过去 shared-GPU 逻辑检查 timing_eligible=false，不能作性能证据。
- 早期实现缺口、拒绝形状后的状态损坏、测试 fixture/工具 PATH 问题均已修复并保留失败记录；旧 D1 不满足覆盖要求，作废其最终选择资格。
- 最终性能与 PlanFamily 留空，原因是用户取消余项；不自动启用单因素胜者，不声称四格 TM/vLLM 不退步。
- 为保留远端 dnn-moe 与已测 R14 的不同模板/运行时 ABI，用现有 `TILEMEGA_DM_SUPPORT` 选择 Dm* 兼容头，身份 schema 分流；未重写二进制身份或回填旧数据。
- 合并只进行编译与主机回归；历史 GPU 数据绑定合并前源码，不声称合并后已完成 GPU 终验。整合证据见 raw/closure_integration/。
- 两条历史通过普通 merge 保留；13 个入口按既有 DM 宏分派，R14 分支体与 `97f7a2f1d` 逐字一致，12 个 DM 头与远端逐字一致，余下 runtime 仅增加三个主机身份查询导出。
- 兼容实现暂时重复模板体，避免将两套已开发 ABI 强行混合；统一模板接口留待后续独立审查。DNN/MoE 身份 schema 保留，R14 校验器支持读取且不改写其哈希内容。
- 整合验证：14 项主机检查、45 项 Python 检查通过；sm_89 编译通过四个数值测试源、两个 DM 完整运行时入口和一个原 R14 生成控制。编译器源码印章通过；无合并后 GPU 执行/性能结论。
- 额外 MoE 40 族 CPU 搜索未在 300 s 命令窗口内完成，后续五项可选主机检查未运行；保留原日志，不列为通过，不续跑。整合验证不能替代 DNN/MoE 全量验收。

## R15 方案（仅记录，不实施）

- 先完成新 D1 必需族覆盖、同轮四格 D2、D3；修复预算/编译开销分配，保留 fail-closed，不能降低覆盖条件。
- 用 ServingTrace.cuh / ServingProfiledMainloop.h 对同几何拆分寄存器、spill 与调度漂移，补 T6/T7 专表及 Qwen3 无 spill 同轮对照。
- PagedGemmTaskBody / PageRing：多页 stage 协议与全阶段 smem 生命周期；不能从只加载微基准外推整网收益。
- ModelHarness / HandoffPass：phase 子图 handoff 与 ready-task；需要独立数值门与 50 新进程。
- StageFlowModel / codegen：共用执行描述；资源反馈与 codegen 部分求值，避免模型/实际产物偏离。
- 5090 专用 collective、prefill 另立规格；本轮仅已有多架构编译，不宣称对应硬件执行。
