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
verified：恢复启动核查时 `_r2` 的 E0_core running（子 PID 5624）、13 个后续节点待运行；日志已推进至 73/114。准备脚本检查 5/5，通过不代表 GPU 验收。
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

## 第一次阶段验收（2026-10-05）

verified：`_r2` 14 项全部终态：4 done、4 failed、6 skipped；GPU 空闲、调度器存活。
E0_core 构建与源码指纹通过。CTest 为 84/97：12 failed/timeout、1 not run；R13 Python 为 31/31。
这不是 E0–E2a 通过：TL-2 与全部 E2a 微基准没有执行，不能报告带宽上限或首次执行路径收益。

| 项目 | 验收证据与处置 |
|---|---|
| 环境/能力 | verified：170 SM；共享内存每 SM 102400 B、每 block 默认 49152 B、opt-in 101376 B；L2 100663296 B；portable cluster 查询上限 8；CUDA 12.8.93 |
| 模型 | verified：固定 revisions 与上轮本机权重 SHA 全部一致；跨机 SHA 尚待 sm89 提供，不提前宣称一致 |
| 7 个 E2E 输入用例 | 缺 `E2E_GEN/raw/export_bridge.json`；恢复原共享目录的测试输入，记录 SHA，不改预期；待复验 |
| 2 个 interface 用例 | 缺历史 ORACLE 寄存器表导致 map::at；历史数据缺失，不造寄存器值 |
| target_audit | 仓库静态 target 的 wait_protocol 字段不满足 audit schema；不拿它覆盖本机新标定 target |
| pipeline_sigma | 包装器 180 s 超时；只将该 CPU 用例的复验上限设为 900 s，不改模型/预期 |
| norm_prologue_gemm | 旧测试接口与当前 TaskBody API 不符，未运行；不是 sm120 的共享内存失败 |
| independent_attention | 真正数值失败：D64/Q4/past63，got 0.000717163、ref −0.00453073；不修改参考或容差 |
| E1 两模型 | GPU 九分节命令均 exit 0，均有 12 类 native serving fit、19 项 paged/loader、legacy samples=0；Python 随后误拒 optional legacy fit |

移植修正 `17860b184`：target native-only 格式读写与校验兼容；缺失/nonfinite/负值数据仍拒绝。
verified：Python 合约检查通过，并在两个原实测 target 上确认缺失 native 分节为零。
C++ round-trip 回归已加入、待重建执行；没有改 GPU 同步，因此此修正不产生 50 新进程同步结论。
求解器/代价模型 diff 为空；R13 当前 in-flight 路径已经关闭重复 stage_latency 计价，不迁入 R12c 的定价修正。
原记录归档：`raw/acceptance_01/e0_e1_original_evidence.tar.xz`，SHA256 `d74dd8209e28e61992013aac60aaf2cbe4f72ff820b1079a036100094b27f692`。

追加 `_r3` 队列：重建/CPU 复验 → 两模型全九分节重测（新 cache）→ TL-2 至少五新进程 → MB-1b–f → 本机 exports → E3 固定/trace 构建。
已完成的模型下载、环境测量与非相关 GPU 单测不重复；保留旧节点，不重置状态、不重启调度器。
固定机制控制的 lookahead=0/page_loop_split=0/eviction 等保持 R13 B 的注册对照设置；R13F 使用 Phase C 保留的配置，另行构建。
E3 使用新的 target；固定构建最多三个并行，PDL off/auto 分离，prefill 固定 R13 已选 PF-R10 几何。
R13F 联合选择、E2b/E2c、E4–E6 尚未排入此队列；固定计划失败只记录该臂，后续按原规格验收首次路径。

恢复启动核查 verified：新编译器/主机单测工具重建与指纹通过；C++ target_spec 的新增 native-only round-trip 通过。
九项 CPU 复验 8/9：pipeline_sigma 295.75 s 通过，六个 E2E 输入用例通过，plan_skeleton 另缺仓库已跟踪的 COSTMODEL target。
已补齐此静态 fixture，只新增该用例的单项复验，不重复 pipeline/GPU 测量；旧失败保留。
native Python 合约与 R13 Python 31/31 也通过；准备/固定构建计划的本机检查为 9/9。

## 追加验收：r3 标定通过，微基准故障阻塞（2026-10-05）

verified：plan_skeleton 单项复验通过。合并原全套与定向复验后为 92/97；不是一次新二进制的全套 97 项通过。
剩余五项：coupling_interface_gqa2/mha4、target_audit、norm_prologue_gemm、independent_attention；原因见前表，未改参考值。
verified：两模型在新 cache 完成全部九分节 E1 标定，CLI 均 exit 0；native-only 格式移植修正通过实机回归。
verified：TL-2 五个新进程 PID 为 10682/10806/10833/10858/10886，均无污染标记。

| 本机初步测量 | 结果 | 预测核对 |
|---|---|---|
| E1 DRAM | 1691.25159609 GB/s；理论 1792.128 GB/s 的 94.371% | 在 85–97% 内 |
| 进程间极差 | 1.0337695 GB/s，0.061125% | 五进程协议通过 |
| MB-1a 最大表观带宽 | 1866.03085627 GB/s；method4/param256/grid680/threads256 | 最大值/E1=1.103343，不在 1.00–1.05 内 |
| MB-1b bulk/1 warp/只加载 | 最好 1700.82730712 GB/s，16 KiB × 5 页 | /MB-1a 最大值=0.911468，未达 0.97 |

以上是加载微基准，不是模型 TPOT。MB-1a 的表观最大值高于铭牌带宽，不能解释成实际 DRAM 字节吞吐。
verified：MB-1b 共 65 点完成；首次 bulk 路径运行成功，不等于 50 新进程同步检验通过。
verified：MB-1c 在 TN128/TK64/stages4/K2048/active50%/tiled/method5 报 `an illegal instruction was encountered`。
method5 是 evict-first 的 `cp.async.cg.shared.global.L2::cache_hint`；同点 method0–4 已输出，失败前日志保留，完整 JSON 未生成。
inferred：问题与该形状的 cache-hint 执行有关；尚未确定 PTX/驱动/设备根因，不据此修改同步或求解器。
verified：故障退出后无可见计算进程、显存仅 1 MiB，但 GPU 利用率 100%、约 105 W；观测期间持续未恢复空闲。
`dmesg` 读取无权限，不能宣称已取得 Xid 根因。现场 `nvidia-smi -q` 存于 `raw/acceptance_02/gpu_snapshot.txt`。
verified：调度器 PID 4919 存活，守卫保持原阈值；MB-1d/e/f pending，E3 exports/fixed 因 MB-1c failed 已 skipped。
阶段未通过，完整实验未完成；E2b/E2c/E4–E6 未执行，暂无端到端性能或 PDL/cluster 收益结论。
没有 GPU reset、重启、放宽守卫或改 GPU 代码；设备级恢复需用户确认，恢复后先验收设备空闲再决定最小复现/路径不可用处理。
不能简单重启调度器期待 E3 自动恢复：旧 skipped 节点必须保留，之后应发布显式记录可用路径的新依赖节点。

实测 target SHA256：`43e9e291abf6d2cdf471324e09cd063e972a4acc9e4961e8857d35cc5a8ce74c`。
loadbench 二进制 SHA256：`84bd4ecc345fb44dc10854ebd469dfb67948ce84d5094a8563c40e326ea6b68a`，未被覆盖。
本次证据归档 `raw/acceptance_02/r3_evidence.tar.xz`，SHA256 `5b97f9ee99cb918487bd33caa1665650ec8516bd9ef761f803ca89e59d09a375`。
归档包含 CPU 复验、E1 原始输出、TL-2 五进程、MB-1b 数据、MB-1c 失败与守卫/调度器快照。
求解器/代价模型仍无 diff；本轮尚未形成需 50 新进程的同步修正。所有提交仅本地保存，不 push。
