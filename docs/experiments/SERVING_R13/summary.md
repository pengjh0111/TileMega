# R13 — 实现冻结与验证队列

这是执行中的记录，不是最终性能验收。Phase C、D 尚未开始，T1–T12 的实测结论尚未形成。

## 基本信息

- 开工 HEAD / origin/tilemega：`48c013d1863d5a5af249401be217d4aa4c15d710`。
- 外部 prompt：`/root/Prompt/TileMega_R13_prompt.md`。
- prompt SHA256：`5adb32833673d7dcd18e6ec02999be9c6bd9ca3216c95f9d9f2b136f8329ace9`。
- 固定 target：`/root/r13_work/target_r12b.json`，SHA256 `c2c03ad6e9534130762cc88d423aac336077a0bd040db6d337a1e31e3c1ec8b9`。
- 当前源码冻结：`cf369f5bf51268b3ad5c02b0a78dbd3450006a92`；冻结前 50 个提交见 `implementation_commits.tsv`。最终 HEAD / 最终提交数待归档。
- Phase A 独立 worktree：`/root/r13_work/diag`；C++ 使用基线，Python 包含 TL-1；主 worktree 的后续改动不用于 Phase A。
- 没有找到配套 `R13_plan.md`；本记录以已保存的 R13 prompt 为依据。

## 实现清单

“代码已实现”与“GPU 验证通过”分别列出；未经执行不宣称正确性或收益。

| ID | 当前情况 | 主要提交 | 代码位置 / 验证 |
|---|---|---|---|
| BL-1 | 代码已实现；端到端待验 | cd94bcac7 | `python/tilemega/serving/execution.py:6`、`python/tilemega/cli.py`、`configs/e2e/*_r13.json`；旁车、键过滤、缺失退回主机测试通过 |
| PD-1 | 代码已实现；架构与 PTX/SASS 待验 | 1c99253eb、8dd3c6e12 | `ModelHarness.cuh:1340`、`ServingRuntime.cuh`、`ptx_pdl_check.py`；sm_89 不作 PDL 性能或同步结论 |
| LP-1 | 代码已实现；设备正确性待验 | dd8ea6b23 | `ModelHarness.cuh:1397`、`ServingRuntime.cuh:470`、`Plan.loop_modes()`；旧 ABI 能力退回已做主机测试 |
| FX-21 | 根因 CPU verified；修复设备待验 | aab46bb4d、a7f75b318 | `lib/Frontend/Frontend.cpp` 的 `split_access_semantic`、`HandoffPass.cpp`、`--paged-la-splitk`；诊断与四个导入用例已提交 |
| FX-22 | 代码已实现；trace 数据待验 | 653b0e848、8954bc431 | `TraceReducer`、`runtime_stages.tsv`、`page_chain.py`；只读实际 `E2E_STAGES`，补动态 LA 发布者 |
| TR-1 | 代码已实现；开销检查待验 | 55d4287e2、6b482d015 | `executor/ServingTrace.cuh`、`serving/trace.py`、`ledger.py`；覆盖非分页 L1、L1 loop、分页 L1 |
| TR-2 | 代码已实现；token 待验 | 517c127b5 | `ModelHarness.cuh` 的 `E2E_PLACEMENT_ABLATION`；非分页 decode 也支持 grid_stride/rotate |
| TR-3 | 代码已实现；数据待验 | a95fe309e、8c76f2a03 | `ServingTrace.cuh`、`ServingPages.cuh`、`ledger.py`；边界、KV/token lag、最后 barrier |
| PG-2 | 代码已实现；真实 shortlist 待验 | 761dc161d | `SkeletonSearch.cpp`、`--paged-seed-from`；保留原种子与 `seed_split1`，32 KiB 投影主机用例通过 |
| SL-5 | 代码已实现；D1 待执行 | 58b80a226 | `compile.cpp` 的 candidate mode/loop、`cli.py` 两级选择；旁车与 `plans.json` 保留全部样本 |
| TL-1 | 代码已实现；测量待验 | 1ff6c3535 | `engine.py`、`measure.py`；独立 prefill、step-events、实际 loop 标志 |
| TL-2 | 代码已实现；上限待测 | d207ab0a1、3db575e6b | `dram_ceiling.py`；至少五进程，校准口径按原核实际 grid/threads，target 副本同时写两处 |
| MB-1 | 代码已实现；完整五架构与测量待验 | e0d2ca986、977812cea、f4d5c4bf6、0c91e8158 | `tools/experimental/loadbench/`；六子项；bulk/cluster/PDL 仅相应 Caps 下执行 |
| CM-1 | 部分完成；价格导出与报告工具已写，实测配对待验 | e810b9289、9486fede8、ae3910b62 | `SkeletonFinalize.cpp:45`、`cm_report.py`；按种类/几何拟合、候选重排，不修改默认模型 |

## 已执行的检查与证据

| 检查 | 结果 | 原始文件 |
|---|---|---|
| Python 主机工具套件 | verified：23 个测试通过 | `raw/host_checks/python_venv.log` |
| 止损与投影专项 | verified：5/5 | `raw/host_checks/choices_loop.log` |
| trace join / 账本专项 | verified：5/5 | `raw/host_checks/diagnostics_final.log` |
| 带宽口径专项 | verified：排除其他 grid/线程配置 | `raw/host_checks/ceiling_accounting.log` |
| FX-21 旧编译器诊断 | verified：48 对 split partial→combine 均因缺 L-sem 被拒；原来只选 17 个 LA | `raw/FX21_diagnosis/{diagnostic.log,handoff_rejects.txt,result.json}`，F-348 |
| Phase B 输入 | verified：28 个固定构建定义、68 个队列步骤；无缺失输入、重名步骤或未知依赖 | `phase_b_input_check.json`、`jobs_b.json`、`queue_b.json` |

系统 Python 缺 transformers 的首次工具检查失败也保留在 `raw/host_checks/python.log`；随后使用 Torch venv 验证。C++ 单测与 CUDA 检查以 Bpre/Barch 的终态记录为准，未把已启动的后台命令当成通过。

## 访问与一致性论证（inferred，待设备验证）

FX-21 采用 c1 的私有访问见证：从已实例化 split partial/combine 的实际访问生成 `split_access_semantic`，只在交接证明的克隆图上提升成 L-sem。原语义及定价不变；其他 recompute 证明不会看到新增的公开 combine L-sem。只允许 partial 的唯一消费者为 combine，反例额外 partial 读取者会被拒绝。逻辑输出的消费者仍等待 reducer 的事件；last-arriver 完成归约后代为发布。没有合成 GEMM 自环。实际被省阶段数与归约数须由 B0c/B4 的 `E2E_STAGES` 核对。

LP-1 使用原有 leader acquire + compute barrier 作为跨步边界，所有 CTA 在最后阶段排序后才进入下一步。GEMM cp.async 的可变 A 只经 L2；其他普通可缓存加载沿用同一次 launch 内跨阶段的 acquire/barrier 协议。该论证不是新进程检验的替代品。新循环核的 NC SASS 清单及逐条不可变数据分类待 Barch 后补齐。

CM-1 将 task 运行区间与依赖等待、事件发布分开，不把三者混为一个实测时长。阶段模型的 active-task span 与 TR-1 的 inter-barrier span 单列。候选重排目前是 critical-chain 固定项/字节残差的 inferred 修正，不是重新拟合 target 或重跑模型；缺少 task 对应关系时使用导出的价格分片平均字节并显式标注。

## T1–T12 与 Q1–Q7

| 表 | 当前情况 |
|---|---|
| T1 | A1 / prefill 选择待收集、分析 |
| T2 | A2 / B4 / 资源行待收集、分析 |
| T3 | 完整 MB-1 数据待收集；不提前选择 884.5 或 981.6 |
| T4 | B1 的阶段账本与 trace 开销待分析 |
| T5 | B2 的放置、等待、发布、队头阻塞待分析 |
| T6 | B3 与 TR-3 的步边界、每步事件待分析 |
| T7 | 页流、满环等待、LA 关键链待分析 |
| T8 | Phase C 未作决定、未执行 |
| T9 | D1 联合选择与 fidelity 未执行 |
| T10 | D2 终版四格对照未执行 |
| T11 | B5/D3 与 ≥50 新进程未验收 |
| T12 | 价格导出与配对工具就绪；数据与覆盖范围待验收 |

Q1（新基线）、Q2（可达上限）、Q3（阶段账本）、Q4（L2 开销）、Q5（分页差距）、Q6（步边界）、Q7（联合选择）均为未决；prompt 中的初始预测仅为 stated，不当成本轮实测。`predictions.json` 在对应计时之前提交。带宽结论与它对四格下界距离的影响将在 T3 完整后填写。

## 细节选择与偏离

- Phase A 的 MB-1b 原探针只固定 grid=SM 数，未用 occupancy 保证小池的单 CTA 驻留。补测 `MB-1b-resident` 在独立二进制中保留未用 smem 并检查 resident_limit=1；原数据保留，C-PG1 只用补测数据。
- Phase A 的带 cache hint 探针最初仅扫深度 8。`MB-1a-complete` 补齐规格的深度 4/8，并重新走同一个至少五进程协议；不覆写运行中的 Phase A 可执行文件。新增参数均在原规格的域内。
- 原 TL-2 汇总会混入 method 1 的不同占用配置。`Bceiling_account` 只对原始文件做 CPU 重算，保留旧汇总；完整新探针使用修正后的汇总。若修正后的进程极差触发 3% 规则，按规定补满额外五进程。
- `solver.exclude_l1_loop` 是止损结果的内部选择约束，并进入第二级测量缓存键。若任一格 B0l 比 B0-noev 慢超过 0.5%，D1 不将非分页 L1 loop 纳入默认选择；缺失证据也不启用它。显式基准工具仍可测该循环。
- 固定构建使用 `measure_stub.py`，占位 mean_ms=0 不作为性能证据；真正的性能只取守卫下的矩阵。
- 临时 binary / `.cu` / PTX / SASS 产物不提交；最终归档时登记路径与 SHA256，资源行和所引用的原始数据另行入库。

## 队列与恢复

Phase A 调度器已启动，PID 启动记录 `/root/r13_work/scheduler_launch.pid`；这里没有轮询其运行状态。Phase B 复用该单实例调度器，提交 `queue_b.json` 后原子放入 `/root/r13_work/queue/`，无需重启。所有 GPU 计时经守卫，所有构建/ctest/架构检查共用 `/root/r13_work/gpu.lock`；退出码 75 冷却后重排。Bpre 的编译、指纹与 C++ 子集测试是固定构建前置依赖。Bbaseline_gate 超出已注册止损或缺少干净轮次时会阻止 Phase B 计时，不能绕过。

再次唤起时先读一次 `scheduler/progress.tsv`，按失败步骤的摘要定位；不要重复检查在运行的步骤。Phase B 终态后运行 `analyze.py`，由 `phase_c_inputs.json` 生成、审核并提交 `phase_c_decision.json`，随后才实现/运行已触发的 Phase C 项。没有提交决定前，Phase C 不执行。Phase D 在 Phase C 保留/回退完成后安排。

目前没有形成 R14 性能方案：须以 T4/T5/T6/T7/T12 的代码位置和常数为依据。最终阶段再补齐预测对照、全部数字的原始归档、最终 HEAD，并推送 tilemega。
