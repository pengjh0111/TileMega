# R13 sm120 — 检查点并行验证（进行中）

不是完成报告；本轮在 sm89 R13 Phase D 尚未完成时启动。

## 基线与边界

- verified：同步的检查点为 `36f17e6ee`，独立工作区 `/root/tilemega-r13-sm120`、分支 `sm120-r13-validation`。
- stated：该检查点说明 Phase C 已收尾、Phase D 仍运行；不是 R13 最终通过的 HEAD。
- verified：用户粘贴的 sm120 规格已保存至 `/root/Prompt/TileMega_R13_sm120_prompt.md`。
  本地 LF 文本 SHA256：`05f3b54517180636152a0423e63d8ab110efa41410d86f957421acf7d8ec5009`。
  此值标识本机保存的文本，不冒充 sm89 记录的外部原文件 SHA。
- 用户覆盖：允许 Phase D 未完成时并行验证；保持当前 vLLM，不按原文切换版本。
- verified：当前 vLLM `0.29.0`、torch `2.13.0+cu130`、transformers `5.17.0`。
- inherited：R13 守卫、调度器、锁与统计工具来自同一检查点；仅重测本机空闲功耗并调整本机路径。
- 本轮不修改求解器/代价模型，不调优；允许的首次执行/移植修正单独提交，触及同步须各加 50 新进程。
- 不 push。原共享工作区用户改动、R12c 分支与持久化证据保持不动。

## 当前安排

E0 编译器与全部单测构建 → 全部 CPU/GPU 单测 → 硬件/能力记录与固定版本模型 → E1 全分节标定与 TL-2 → E2a MB-1b–f。
E1 的 TL-2 同时提供 MB-1a 的至少五新进程证据，避免重复同一测量。
E0 构建用三个并行编译进程；测试构建采用 keep-going，失败目标和未运行用例保留，不记为通过。
全部单测失败不隐去；E1 可在全套单测已尝试后独立采集，不将其解释为 E0 验收通过。
计时全部由 R13 scheduler 包装 gpu_guard 并持有 `/root/r13_sm120_work/gpu.lock`。
verified：6 次空闲功耗采样的中位数为 17.97 W；其余守卫阈值未变。
verified：预注册提交 `8571240f5`；本机队列准备检查 4/4 通过。
verified：调度器 PID 4919 已启动，单次启动核查确认 E0_core running、其余 13 项 pending。
CMake 已成功配置 CUDA 12.8.93、sm_120、新构建目录；完整编译与测试结果尚待收集。
启动后 verified：首次 E0_core 因稀疏检出缺 `SIMULATOR/hop_ns.tsv` 失败，13 个依赖节点 skipped。
已补齐指纹输入，并用 `_r1` 新节点原子发布恢复队列；原状态/日志完整保留，不重启调度器，不改产品代码。
`_r1` 包装脚本因局部路径变量被组件名覆盖而失败；修复此准备脚本，新增回归检查，发布 `_r2` 新节点，历史失败仍保留。
尚未发布 E3–E6；先验收本机标定与首次执行微基准，再冻结它们依赖的 target 和可用路径。
预注册数字见 `predictions_sm120.json`，必须在 E4 前提交。

## 已知风险（未改记 PASS）

- stated：上游 sm120 PDL trigger=0 的 PTX 位置检查失败，须本机生成后定位。
- stated：上游非分页源码/SASS 一致，但资源文本比较未通过。
- stated：sm89 Llama B1 B0h 的 C-1 失败，本机仍须独立核查。
- verified：上轮 tmpfs 模型与缓存已消失；固定 revision 恢复，证据写持久化目录。
- sm89 的权重 SHA 尚未随便携几何档案提供，本机先对照上轮固定 revision/SHA，跨机同权重不提前宣称 verified。
- 几何输入仅使用固定 manifest/classes，不执行或复用 sm89 的 `.so`、target。
- 旧 R12c 本地求解器/定价修正不迁入本轮。若原检查点的 fresh native calibration 有问题，保留失败并按允许范围判断，不伪造 legacy fit。

## S1–S10 与四个问题

S1–S10 尚待本机结果；没有性能或同步结论。
机制结论跨架构是否变化、bulk 分页是否胜过非分页 L1、B1 优势是否变大、PDL 对 L1/L2 回收量：均 pending，不能用预测代替实测。
sm89 最终 HEAD/表未完成时，仅报告检查点数据与限定性比较；正式结论须对齐共同代码。
R14 方案待证据，不实施。

## 恢复

进度：`/root/r13_sm120_work/scheduler/progress.tsv`、`state.json`。
单次读取状态，优先定位 failed 日志；running 的长验证不反复轮询。
调度器日志：`/root/r13_sm120_work/scheduler.log`；脚本/队列/环境见本目录。
确认原 PID 不存活且无同锁运行进程后，才可重启同一个调度器；running 节点须先核实子进程，不能直接清空状态。
