# CUDA Tile IR / TensorIR Megakernel 可行性研究报告

调研目标：判断 NVIDIA 官方 CUDA Tile IR / TensorIR 工具链能否表达"megakernel"执行模型
（persistent kernel + tile 级任务图 + kernel 内跨 tile block 事件同步），
对标 Mirage MPK 与 MLSys'26 "Event Tensor" 论文，但完全建立在 NVIDIA 官方 Tile IR 栈之上。

**证据纪律**：全文每条结论都标注 ✅ 已验证（实际编译/运行并观察到结果）/ ⚠️ 文档声明但未验证 /
❌ 推测（未经实测的合理猜想）。凡是不确定的地方，如实写"不确定/未查明根因"，不伪装成确定结论。

---

## R2-0. Round 2 对第一轮核心问题的回答

> 本节是 round 2 调研新增的内容，插在 round 1 报告的最前面，但不改动、不删除下面第 0～5
> 节的 round 1 原文（round 1 的表述保留原样，即便部分结论在 round 2 被修正或收窄，也只在
> round 2 新增章节里注明，不回头篡改 round 1 文字）。round 2 完整实验记录见
> `experiments/R2_{A,B,C,D,F,G}_*/result.md`；R2-E、R2-H 未完成，标记为 pending（见本节末尾
> 与 R2-8 节）。

round 1 报告在 E4/E5 里遗留了两个悬而未决的问题，round 2 的任务就是回答它们：

**问题 1：E4/E5 观察到的大 grid 死锁，是活锁（livelock，忙等饱和内存子系统）还是死锁
（deadlock，永久卡死）？**

**回答：都不是纯粹的一种——是局部/部分死锁，混杂着大部分 block 成功完成的假象。**
（R2-A，✅ 已验证）用 `nvidia-smi dmon` 看到挂起期间 SM 利用率 100%、时钟 2887MHz（接近满速
3090MHz）、功耗 97.71~98.33W（idle 基线 13.19W 的约 7.4 倍）——表面完全符合活锁直觉。但三次
独立 `cuda-gdb -p PID -batch -ex "info cuda threads"` 采样（间隔 >20 秒）显示：170 个 block 里
只有 57 个（约 33%）出现在线程列表里，且这 57 个 block 的 PC **逐字节完全一致、20+ 秒内一条
指令都没有前进**——这是死锁的判据（活锁哪怕循环再紧凑，20 秒内两次采样命中同一条指令地址的
概率也极低）。反汇编卡住的 PC（`spin_wait_tokenchain+24976` = 偏移 `0x6190`，一条
`LDS R4, [UR5]`）发现它**不在**自旋等待循环本体（偏移 `0x1f0~0x230`），而在自旋等待*之后*、
由 256 次静态展开构成的 consumer 归约收尾代码里，紧跟在一条 `BAR.SYNC.DEFER_BLOCKING`
硬件屏障之后。其余 113/170 个 block（约 67%，含生产者本身）**已经正常执行完毕并退出**——
它们才是 §1 里 100% SM 利用率的真实来源。`compute-sanitizer --tool synccheck/racecheck`
在插桩下 3/3 次反而不挂起（0 errors/hazards），与 E5 已经记录过的"预热能救 grid=80"是同一类
"改变时序即可掩盖问题"信号，不能反证 bug 不存在。

**这直接证伪了 round 2 的核心工作假设**——"自旋循环因为没有退避、忙等饱和内存子系统导致
活锁"——卡死的位置根本不在自旋循环里，说明 flag 的跨 block 可见性机制（token 链 +
acquire/release）在这个规模下基本是工作的，真正的死锁疑点转移到了 **`reduce` 算子在大量
并发 tile block、大量硬件屏障同时活跃场景下的 lowering**（R2-A §5，`experiments/
R2_A_livelock_vs_deadlock/result.md` 55-134 行）。

**问题 2：能否找到一个可靠的修复？**

**回答：截至本轮结束，没有找到——两个最有希望的候选修复方案都已用完整 grid-scan 统计证伪
或证实收效甚微。**

- **R2-B（test-and-test-and-set：轮询循环改用 relaxed 读 + 跳出循环后单独做一次 acquire
  重读）：❌ 已证伪。** SASS 证实改动完全达到设计目标——轮询循环体内的 `CCTL.IVALL`
  被成功移出，只在跳出循环后执行一次（`experiments/R2_B_relaxed_plus_acquire/result.md`
  §1，24-51 行）；正确性不受影响（1MB 数据 1000/1000 次、4MB 数据 200/200 次全部通过，同 §2）；
  但 grid-scan（M=50/grid）显示**挂起率没有得到任何有意义的改善**，grid=30 时甚至比 baseline
  更差（48/50=96% vs 39/50=78%），grid≥80 时两者都已接近或达到 100%：

  | 变体 | grid=30 | grid=80 | grid=120 | grid=170 |
  |---|---|---|---|---|
  | `relaxed_acquire`（R2-B 修复） | 48/50 | 46/50 | 50/50 | 50/50 |
  | `baseline_acquire_in_loop`（E4/round-1 原始版本） | 39/50 | 50/50 | 50/50 | 50/50 |

- **R2-C（自旋退避：轮询间隔插入 64/256/1024 档 LCG 非线性延迟循环，降低轮询频率）：
  ⚠️ 有一定量级上的缓解迹象，但远未达到"解决问题"的程度，且 grid=170 时完全无效。**
  见下方 R2-3 节完整数据；核心结论是 grid=30 时挂起率随退避档位增大而单调下降（backoff64
  25/30→backoff1024 16/30），但 grid=170 时全部 4 个变体（含 relaxed+backoff256 组合变体）
  均为 30/30 挂起——退避对"远超硬件驻留容量边界"的场景完全无效。

  两者失效的共同原因（R2-A §6 预告、R2-B §4 与 R2-F 发现 2 交叉印证，`experiments/
  R2_F_pipelining/result.md` 35-64 行）：R2-B/R2-C **只修改了自旋轮询循环本身的写法**，
  但 R2-F 反汇编发现，自旋轮询循环体内本身就包含一条 `BAR.SYNC.DEFER_BLOCKING`（每次轮询
  迭代都要求该 tile block 全部 256 个线程在硬件屏障上会合，而不是只有 1 个线程轮询、
  其余线程等待），R2-A 定位的死锁 PC 也紧跟在另一条 `BAR.SYNC.DEFER_BLOCKING` 之后——
  两个方案都没有触及 `BAR.SYNC.DEFER_BLOCKING` 本身，只是改变了触发它的频率/内存序，
  如果根因确实与这条硬件屏障在高并发 tile block 占用下的到达/收敛/公平性行为有关，
  这类"只改轮询写法"的方案在结构上就不可能修复它。这个"根因在 `BAR.SYNC.DEFER_BLOCKING`
  的收敛机制而非自旋轮询内存序"的判断，**目前仍是 ⚠️ 有强 SASS/统计证据支持、但未经
  cuda-gdb 在 R2-B/R2-C 变体挂起现场逐点定位 PC 予以最终坐实的假说**（R2-A/R2-B/R2-F
  均明确标注了这一遗留验证步骤，受时间预算限制均未完成）。

**问题 3：round 1 报告里"REG=228 却对应 driver 报告的 8 blocks/SM"这个数学上说不通的矛盾
是什么原因？**

**回答：是 round 1 所有 harness 的方法论错误（查询用错了 blockSize），不是 driver 或
tileiras 的 bug；但修正这个错误之后，死锁本身依然存在。**（R2-D，✅ 已验证）round 1 所有
occupancy 诊断代码（`E4/host_test_coop.cpp:15`、E5 的 harness）统一调用
`cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 1, 0)`——blockSize 传的是 1，而这个
cubin 的 `EIATTR_REQNTID` 硬编码要求 blockDim=256（`experiments/
R2_D_occupancy_audit/result.md` 11-47 行），拿 blockSize=1 去查一个真实需要 256 线程/block
的 kernel，会把资源占用低估 256 倍，算出"8 blocks/SM、max grid=1360"；用正确的
blockSize=256 重新查询，得到 numBlocksPerSm=1、max grid=170（与手算 65536/(228×256)=1.12→1
完全吻合，也与经验观测到的 hang 边界完全吻合）。用一个普通 nvcc 编译、REG=218 的对照组
kernel 做同样的 4 档 blockSize 扫描，得到完全相同的模式（同 §64-77 行）——**证明 driver 的
occupancy 算法本身没有问题，round 1 的"driver 对 tileiras cubin 算错"是误诊**。但（R2-D
§128-144 行明确标注）：用正确 blockDim=256 重新发起 `cuLaunchCooperativeKernel`，grid=170
被正确接受（因为它确实是硬件真实容量的边界）、grid=171 被正确拒绝（"too many blocks in
cooperative launch"）——**driver 的准入检查逻辑本身是对的**，只是这个"边界值" grid=170
本身依然会 hang（`rc=124`）。也就是说，blockDim mismatch 只解释了"为什么这么大的 grid
一开始没有在启动前就被拒绝"，完全没有解释"为什么跑到硬件容量边界时会卡死"——后者与
R2-A 定位的、发生在自旋循环之外的 post-spin reduce/barrier 代码里的挂起是同一个问题，
根因仍未查明。

**问题 4：E2 的"资源开销取 max 而非 sum"正面结论能否推广到更多分支？**

**回答：可以，至少推广到了 N=8（✅ 已验证），但没有测试更贴近真实 megakernel 复杂度的
loop-carried-跨分支场景（❌ 完全未测试）。** 见 R2-7 节详情。

**问题 5：tileiras 是否在 persistent loop 跨迭代之间做软件流水线/双缓冲？真实性能
（TFLOPS vs peak）如何？**

**回答：观察到真实的跨迭代批量预取现象，但是否构成经典意义的双缓冲流水线未定点验证；
真实 TFLOPS 完全无法测量（现有 kernel 都不含 MMA）。** 见 R2-6 节详情。这一节额外发现了一个
之前所有轮次都没记录过的事实：自旋轮询循环体内部本身包含 `BAR.SYNC.DEFER_BLOCKING`——这个
发现正是上面"问题 2"里解释 R2-B/R2-C 为何失效的关键证据来源。

**总体 round 2 verdict**：round 1 报告的两个阻塞项（第 3 节）在 round 2 之后**均未解除，
且第一项的严重程度被进一步坐实、复杂化**——它不再是一个笼统的"未查明根因的死锁"，而是有
具体证据链指向"局部死锁 + 位置在 post-spin reduce 收尾代码 + 疑似与 `BAR.SYNC.DEFER_BLOCKING`
硬件屏障收敛机制有关"，但两个最直接的候选修复方案（relaxed 内存序、退避）都已被证明无效
或收效甚微。**在当前 tileiras/cuda-tile 工具链版本下，"用一个 kernel 长期驻留全部 SM 做
跨 block 事件同步"这一 megakernel 核心目标场景，依然没有已知的安全实现路径**——这一
round 1 的核心悲观结论被 round 2 加固而非推翻。

**Round 2 子实验完成度一览**：

| 子实验 | 主题 | 状态 |
|---|---|---|
| R2-A | 活锁 vs 死锁判别 | ✅ 完成 |
| R2-B | test-and-test-and-set 修复验证 | ✅ 完成（结论：证伪） |
| R2-C | 自旋退避修复验证 | ✅ 完成（结论：无效/收效甚微，见 R2-3 节） |
| R2-D | REG=228 vs occupancy 矛盾溯源 | ✅ 完成 |
| R2-E | harness 卫生检查（flag 清零、预热规模扫描等） | ❌ 未开始，pending |
| R2-F | 软件流水线/双缓冲判定 | ⚠️ 部分完成（结构性问题已回答，TFLOPS/手动 unroll 对比未完成） |
| R2-G | E2 max-not-sum 规律推广到 N=3/5/8 | ⚠️ 部分完成（N 扩展已验证，loop-carried 跨分支/嵌套场景未测） |
| R2-H | 编译+查询墙钟耗时测量 | ❌ 未开始，pending（R2-G 附带观察到 tileiras 编译耗时随 N 增长明显上升，但未正式计时，见 R2-7 节） |

---

## 0. 环境

- GPU：RTX 5090，Compute Capability 12.0，170 个 SM（`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` 实测）。
- Driver 580.65.06，CUDA driver 版本 13.0；`nvcc` release 13.1 V13.1.115；另有 13.3.1 工具链于 `/data/cuda-13.3.1`。
- `cuda-tile` 仓库：`/data/cuda-tile` @ commit `8a775693b18303d6c696be6ffd06dadad1b32a8e`（2026-01-13）——
  **本次调研 E2-E7 全部手写 MLIR 实验实际使用的版本**，其预编译产物为
  `/data/cuda-tile/build/bin/{cuda-tile-opt,cuda-tile-translate,cuda-tile-tblgen}`。
- `tensor-ir` 仓库：`/data/tensor-ir` @ commit `63692d79629e6f32a1d8757695590a59e0adbafd`（2026-08-19），
  预编译产物 `/data/tensor-ir/build/bin/{tensor_ir-compiler,tensor_ir-opt}`。**通过 CMake FetchContent
  拉取了另一份更新的 cuda-tile**（commit `af2417041cc939b87ef56d92cfdcf61737c5457e`，位于
  `/data/tensor-ir/build/_deps/tensor_ir_cuda_tile-src`），两份 cuda-tile 源码 op/属性集合不完全相同
  （见 E1/E7），这是贯穿全报告的一个重要环境细节。
- `tileiras`（Tile IR 汇编器）：`/data/cuda-13.3.1/bin/tileiras` 与 `/usr/local/cuda-13.1/bin/tileiras`，
  **全程实际使用后者**。
- Profiling 工具（ncu/nsys/cuobjdump/nvdisasm/compute-sanitizer/cuda-gdb）：`/usr/local/cuda-13.1/bin/`。
- 输出目录：`/data/tensor_ir_test/megakernel_feasibility/experiments/{E0..E7}/`，每个目录含
  `.mlir` 源文件、`run.sh`（精确复现命令）、必要的 host `.cpp`、`result.md`（原始输出记录）。

---

## 1. 结论速览

| 实验 | 主题 | 结论 | 标注 |
|---|---|---|---|
| E0 | 基线工具链走通性 | verifier→translate→tileiras→实机运行全链路正常 | ✅ |
| E1 | 原语盘点（无 GPU，纯代码阅读+少量实测） | 无 device 端函数调用；tile 形状须编译期常量2的幂；无 shared memory 概念（旧版）/新版有受限 alloca；optimization_hints 仅 entry 级标量提示；原子操作支持 device-scope acquire/release；无独立 fence op，靠 ordering 属性+token；token 是纯编译期排序约束 | ✅（旧版实测）/ ⚠️（新版仅读源码未编译） |
| E2 | 异构 tile 分支 dispatch | 单 entry 内按运行时标量分支、分支内 tile 形状完全异构，全链路通过；寄存器/共享内存开销是分支间取 **max** 而非叠加求和 | ✅ |
| E3 | runtime 上界的 persistent grid-stride loop | `for` 循环下界/上界/步长全部可为运行时值，构造标准 grid-stride loop 全链路通过，100万元素 0 误差 | ✅ |
| E4 ⭐ | 跨 tile block 生产者/消费者自旋等待 | 仅用 ordering attribute（不做 token 链）会被编译器**静默消除**整个自旋循环，100% 数据竞争且无编译期警告；显式把 token 做成 loop 循环携带值后修复，1000/1000 次正确；但 grid 变大后（远低于 SM 数）**必现死锁**，`cuLaunchCooperativeKernel` 未能阻止 | ✅（均为 SASS 级 + 实机复现证据）；死锁根因 ❌ 未查明 |
| E5 | occupancy/CGA hint 是否能修复死锁 + 死锁阈值本质 | `occupancy`/`num_cta_in_cga` hint 不改变 driver 报告的并发度、不能阻止死锁；**E4 报告的"grid=80 安全"阈值被证明不是确定性边界，而是对宿主/GPU 端预热时序敏感的竞态条件**，无预热时 grid=30 即可能死锁 | ✅ |
| E6 | 最小 3 角色 megakernel（异构分派+真实跨block依赖+独立任务） | 在 E4/E5 已确认安全的小 grid（=3）范围内，单次 launch 内 3 种角色（生产者/消费者/独立任务）全部按预期工作，200/200 次正确，且有证据表明独立任务不受生产者/消费者同步阻塞 | ✅（严格限定小 grid 范围内） |
| E7 | warp-specialization 可控性 | 当前实际使用的 cuda-tile 版本完全不支持 `num_worker_warps_per_cta`（实测报错）；IR 层没有任何 warp 角色分配机制；新版 cuda-tile 里该字段的真实运行时行为**无可用工具可测试** | ✅（拒绝行为）/ ❌ 未能验证（新版实际效果） |

**总体verdict**：在表达能力（IR 语法能否描述异构任务图、跨 block 依赖、persistent loop）层面，
cuda-tile **是可行的**（E2/E3/E6 均为正面证据）。但在**正确性和可扩展性**层面存在两个都很致命的
阻塞项（详见第 3 节）：(a) 跨 block 同步存在一个容易踩、且编译器不报警的静默死代码消除陷阱（E4）；
(b) 即使用正确写法修复该陷阱，跨 block 同步在 grid 规模超过一个**不可预先可靠判定**的边界后必然死锁，
且这个边界本身是运行时竞态条件而非确定性阈值（E5）。**这意味着 Mirage MPK / Event Tensor 论文里
"用一个 kernel 吃满全部 SM 做长驻留调度"这一目标场景，在当前 cuda-tile/tileiras 工具链版本下
没有已知的安全实现路径**，即便小规模任务图（个位数 tile block）的表达和正确性已经被充分验证。

---

## 2. 逐实验详情

### E0 基线工具链走通性

最小 kernel（`load_ptr_tko weak` 读 128 个 f32 + `print`）verifier 一次通过（exit=0），
`./example` 实机运行输出 128 个步长 5.0 的浮点值，与预期模式吻合。同时确认 `nv_tensor_ir` 前端
（`add_dynamic.mlir`）与 `tensor_ir-compiler` 自动 lowering 出的 cuda_tile IR
（`matmul_lowered_cudatile.mlir`，含 `make_tensor_view`/`make_partition_view`/`load_view_tko`/`mmaf`
等标准范式）均可作为后续实验的语法参考。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E0_baseline/result.md`。

### E1 原语盘点

纯代码阅读（`grep -n "^def " Ops.td` 得到 97 个 op 定义）+ 少量 verifier 实测，核心结论：

- **无 device 端函数调用**：没有 `FuncOp`/`CallOp`，只有 `ModuleOp`/`EntryOp`/`ReturnOp`；一个
  module 可以有多个 entry（✅ 实测通过 verifier），但 entry 之间不能互相调用，只能各自独立被 host 启动。
- **tile 形状必须编译期常量、且是 2 的幂**（`TileType` 定义硬约束）；`TensorViewType`/`PartitionViewType`
  的整体形状/步幅可以是运行时值，但每次搬运的具体 tile 大小仍须编译期常量。
- **旧版（实际使用的工具链）完全没有 shared memory / alloca 概念**——`PointerType` 没有地址空间参数，
  所有可寻址内存都是 kernel 参数传入的 global pointer；新版 cuda-tile 新增了 `AllocaOp`（"private vs
  global 可见性"的逻辑属性，非独立地址空间类型），但**未在可编译工具链中验证**（⚠️）。
- `optimization_hints` 是 entry 级、按 SM 架构分组的标量字典，旧版仅 3 个字段
  （`num_cta_in_cga`、`allow_tma`、`latency`、`occupancy`，语法/取值范围经 verifier 报错实测确认）；
  **旧版没有 `num_worker_warps_per_cta`**（全仓库 grep 为空），该字段只在新版 cuda-tile 出现；两版本
  均没有找到任何区域级/warp 角色分配的 IR 机制（详见 E7）。
- 原子操作（`atomic_rmw_tko`/`atomic_cas_tko`）支持 DEVICE scope 下 RELAXED/ACQUIRE/RELEASE/ACQ_REL
  组合（✅ verifier 语法层面确认，运行时是否真正生效见 E4）。
- **没有独立的 fence/barrier op**（❌），跨线程顺序保证完全靠 `_tko` 算子自带的
  `memory_ordering_semantics` 属性 + token 数据流两者组合表达。
- `LoopOp`（无界 while）/`ForOp`（结构化 range，边界可为运行时值）/`IfOp`（不 yield 值时分支内部
  类型完全独立）三种控制流均支持 loop-carried values。
- **token 是纯编译期 SSA 排序约束、不是运行时值**；`_tko` 算子文档原文明确"不受 token 约束时可能被
  编译器重排"——这是 E4 陷阱的直接理论依据。

详见 `/data/tensor_ir_test/megakernel_feasibility/experiments/E1_primitives/notes.md`（完整版，19648 字节，逐条附 file:line 引用）。

### E2 异构 tile 分支 dispatch（⭐ 高优先级，正面证据）

单 entry 内用运行时标量 `task_type` 走 `if`，两分支分别用 `tile<128x128xf32>`+`mmaf`（矩阵乘）与
`tile<256xf32>`+`reduce`（规约求和）——完全不同形状/rank/算子。verifier/translate/tileiras
全部一次通过，两个 `task_type` 值实机运行均正确（分支A `max_abs_err=0`，分支B 相对误差 3.05e-5，
量级符合浮点求和顺序差异）。**关键定量证据**：用 `cuFuncGetAttribute` 分别测得单独编译分支A
（REG=255, SHARED=100400B）、单独分支B（REG=12, SHARED=16B）、合并后的异构 kernel（REG=255,
SHARED=98368B）——**合并后的资源开销是两分支的 MAX 而非 SUM**，未观察到"任务描述里担心的因加性资源
开销导致 occupancy 崩溃"现象（诚实标注：只测试了 2 个互斥分支，未验证 3+ 分支或更复杂嵌套下是否仍
保持这一规律，⚠️ 未验证推广)。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E2_heterogeneous_tiles/result.md`。

### E3 runtime 上界的 persistent grid-stride loop（正面证据）

`for %tile_idx in (%bx to %total_tiles, step %gx)` 构造标准 grid-stride loop，下界/上界/步长全部是
运行时 SSA 值。verifier/translate/tileiras 全部一次通过（REG:19 STACK:0 SHARED:0），实机运行
100 万元素 0 处不匹配。计时健全性检查（非严格 benchmark）显示 persistent(grid=170) 比
non-persistent(grid=1000000) 快约 4.3 倍，但 persistent(grid=10000) 又比 persistent(grid=170) 快约
8.4 倍——说明"grid=SM 数"是错误直觉，真实每 SM 可并发驻留的 block 数（E4/E5 中查得约 8 blocks/SM）
远超 1，过小的 grid 会显著欠用硬件并行度。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E3_persistent_loop/result.md`。

### E4 跨 tile block 生产者/消费者自旋等待（⭐ 最高优先级，本报告核心）

三阶段调查，结论中途两次被修正（如实保留演变过程，不事后美化）：

1. **第一版写法**（仅靠 `acquire`/`release` ordering attribute，不显式串 token）：verifier/
   translate/tileiras 全过，但 SASS 反汇编显示消费者的整个自旋循环（包括对 flag 的加载本身）**被
   编译器完全消除**——全文件唯一一次对 flag 常量 bank 偏移 `c[0x0][0x388]` 的引用只出现在生产者的
   release 处，消费者代码路径完全没有引用。1000/1000 次实机运行 **100% 数据竞争**（多数迭代读到
   `0xdeadbeef` 毒化值累加后的巨大负数，量级 `-6.00946e20`），且编译期**没有任何报错或警告**。
   换用 `relaxed` ordering 或错误的 `tl_blk` scope，结果完全一样（说明这两个属性的具体取值在当前
   后端下对这段代码的生成结果没有可观测影响——反正自旋循环本来就整个被删了）。
2. **修复**：把 token 显式做成 `loop iter_values` 的循环携带值（`loop iter_values(%tok=%init) :
   token -> token { %val,%ltok = load_ptr_tko acquire device %flag token=%tok ...; if %is_set
   { break %ltok } continue %ltok }`），构造无法被消除的跨迭代 SSA 依赖链。SASS 中出现真实的、
   会重复执行的自旋循环（`LDG`+`CCTL.IVALL`+`ISETP`+条件回跳，`@P0 BRA` 目标地址小于自身地址的
   真回边）。1000/1000 次实机运行**全部通过**（且 1000 次返回值完全一致，是强有力的"确无竞争"旁证）。
3. **新问题**：用修复后的 kernel 测试大 grid，在 120～170 之间（远低于 SM 数 170 本身，也远低于
   driver 自称的理论并发上限 1360）**必现死锁**（`timeout 30` 触发）。专为此设计的
   `cuLaunchCooperativeKernel`（本应在无法满足并发驻留时直接拒绝启动）**接受了 grid=170 的启动
   请求，但实际执行依然死锁**。每次死锁后用全新 context 跑小 grid 均能立即正常完成，排除了"GPU
   硬件本身卡死"的可能。根本机制未查明（❌ 推测：可能是 tileiras 生成的 cubin cooperative-launch
   metadata 有问题，也可能是 driver occupancy 计算对这类 cubin 失真，也可能与 role-dispatch 逻辑
   本身有关；均未坐实）。

详见 `/data/tensor_ir_test/megakernel_feasibility/experiments/E4_spin_wait/result.md`（含完整 SASS
片段与逐条排查记录，末尾附指向 E5 的更正说明）。

### E5 occupancy/CGA hint 排查 + 死锁阈值本质（本报告最重要的意外发现）

追问两个问题：`optimization_hints` 能否修复 E4 的死锁；E4 报告的"grid=80 安全"阈值是否稳定。

1. **hint 无法修复死锁**：`occupancy=1`/`num_cta_in_cga=1` 两个变体均一次通过编译；与不加 hint 的
   token 链版本相比（均为 REG=228，修正了本实验初版误将其与 E4 中*另一个*未做 token 链修复的旧
   kernel 的 REG=126 对比、错误得出"hint 导致寄存器增加"结论的错误——已在 result.md 中就地更正）
   **未观测到 hint 对代码生成的可归因影响**；driver 的 `cuOccupancyMaxActiveBlocksPerMultiprocessor`
   在两种 hint 下均报告 8 blocks/SM（与不加 hint 一致）；grid=170 仍然死锁。SASS 显示无论 hint
   取值如何，CGA（CUDA Cluster）相关指令（`SR_CgaSize`/`SR_CgaCtaId`/`CGAERRBAR`）都会被无条件生成
   （与死锁的关系未坐实，❌ 推测）。
2. **（意外发现，比原计划更重要）"grid=80 安全"不是确定性阈值，而是竞态条件**：用完全相同的
   `spin_wait_tokenchain.cubin`、完全相同的 grid=80，**仅仅因为 host harness 在启动前是否多做了
   一次 1MB 的 `cuMemsetD32` "预热"操作**，行为就从稳定成功（3/3）变为稳定挂起（2/2）；去掉预热后
   **连 grid=30 都会挂起，且结果不稳定**（3 次里 1 次成功 2 次挂起，典型的竞态特征）；用纯宿主端
   `usleep(50ms)`（不产生 GPU 端操作）替代预热**不能**复现修复效果，说明起作用的是 GPU 侧的实际
   状态变化（很可能是时钟/功耗状态被唤醒），而非单纯的墙钟延迟；用同样的预热去救更大的 grid
   （150/170）**无效**，说明预热只是把不稳定边界往上推了一截，没有从根本上消除竞态。通过交替运行
   两个 harness（成功-挂起-成功）排除了"GPU 状态随时间单调漂移"这一更简单的解释。

**结论对可行性的意义**：这比 E4 原始表述的"存在一个安全阈值，选小一点的 grid 就没事"更严重——
**没有任何已知方法能可靠地预先选出一个"安全"的 grid 大小**，哪怕远小于 SM 数的 grid（本实验里低至
30，约为 170 个 SM 的 18%）在不利时序下也可能触发死锁。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E5_coexistence/result.md`。

### E6 最小 3 角色 megakernel（正面证据，严格限定小 grid 范围）

组合 E2（异构分派）+ E4 的 token 链修复自旋等待，构造含 3 种角色的单 entry、单次 launch 任务图：
block 0（生产者，`C=A+B`，256 chunk，每 chunk 后原子自增 `progress` 计数器，完成后 release-xchg
`flag`）、block 1（消费者，token 链自旋等待 `flag` 后 `D=C*2.0`）、block 2（独立任务，**不**等待
`flag`，只做一次非循环的 `progress` 快照 + 完全独立的 `E=X*3.0`）。verifier/translate/tileiras
全部一次通过（REG:36 SHARED:1028），200/200 次实机运行 C/D/E 三个输出全部与 CPU 参考值精确匹配，
且 200/200 次 `block2_sample`（block 2 采样到的 `progress` 计数器快照值）恒为 0——证明 block 2
在生产者完成第一个 chunk 之前就已经开始执行，即三种角色确实在同一次 launch 内并发推进、独立任务
未被无关同步阻塞。

**局限性（如实标注）**：(1) `block2_sample` 恒为 0 而非分布值，只能证明"block 2 不需要等待"，不能
量化重叠程度（cuda-tile 没有设备端时钟原语，`Ops.td` 中未找到 `clock`/`globaltimer`）；(2) grid
刻意固定为 3（E4 grid=2 基础上保守 +1），**结果明确不外推到更大 grid**——按 E4/E5 的结论几乎肯定
会撞上未查明根因、且对宿主端时序敏感的死锁问题。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E6_minimal_megakernel/result.md`。

### E7 warp-specialization 可控性

用当前实际使用的工具链直接测试 `num_worker_warps_per_cta`：
```
$ /data/cuda-tile/build/bin/cuda-tile-opt entry_num_worker_warps.mlir
entry_num_worker_warps.mlir:5:123: error: custom op 'cuda_tile.entry' unknown param num_worker_warps_per_cta for sm_120
exit=1
```
`num_cta_in_cga` 被正常解析（说明是这个具体字段名不被识别，不是语法本身有问题），与 E1 grep 结论
完全吻合。进一步排查是否有办法测试新版 cuda-tile（真正定义了该字段）的实际行为：新版 cuda-tile
在 tensor-ir 的构建产物中**没有构建出独立的 `cuda-tile-opt` 可执行文件**（`tools/cuda-tile-opt/`
目录只有 CMake 中间产物，`bin/` 下只有 `cuda-tile-tblgen`）；`tensor_ir-opt`（链接新版 cuda-tile
作为库）**没有注册 `cuda_tile` 方言**（`--help` 显示 `Available Dialects: arith, builtin, func,
nv_tensor_ir`）。**结论：当前环境下没有任何可用工具能测试新版 `num_worker_warps_per_cta` 的实际
运行时效果**（❌ 未能验证，如实标注而非编造）。综合 E1 的全文 grep 结果：两版本 cuda-tile 在
IR 层均**没有任何"显式分配哪些 warp 做 MMA/哪些做 TMA/哪些做控制流"的机制**，唯一相关旋钮就是这个
仅存在于新版、取值只有 {4,8} 的 entry 级标量提示。详见
`/data/tensor_ir_test/megakernel_feasibility/experiments/E7_warp_specialization/result.md`。

---

## 3. 阻塞项（按严重程度排序）

1. **【最高严重性】跨 tile-block 同步没有可靠的、可预先判定的安全 grid 规模（E4 §10 + E5）**。
   即使用正确写法（token 链）修复了自旋等待的静默死代码消除问题，"grid 超过某个大小就死锁"这一
   现象本身被证明**不是一个由 grid 大小决定的确定性阈值，而是对宿主/GPU 端预热时序敏感的竞态
   条件**——远小于 SM 数的 grid（低至 18%）在不利时序下也可能触发。NVIDIA 官方为此设计的
   `cuLaunchCooperativeKernel` 保护机制、`optimization_hints` 里的 occupancy/CGA 相关字段，均
   实测**无法**防止或修复这一问题。这直接堵死了"用一个 kernel 吃满全部 SM 长期驻留"这一 Mirage
   MPK/Event Tensor 式 megakernel 的核心目标。✅ 已验证存在，❌ 根因未查明。

2. **【高严重性】跨 block 同步存在一个编译器不报警的静默正确性陷阱（E4 第1阶段）**。仅使用
   `memory_ordering_semantics` 属性（acquire/release）而不显式将 token 串成循环携带值，会导致
   整个自旋等待循环被编译器完全消除，产生 100% 可复现的数据竞争，且 verifier/translate/tileiras
   全程没有任何报错或警告。这是一个极易踩、后果隐蔽（编译"看起来完全正常"）的陷阱，文档
   （`Ops.td` 注释）只说"不受 token 约束时可能被重排"，未明确警告"可能被整体消除"。✅ 已验证。

3. **【中等严重性】没有 device 端函数调用能力（E1 Q1）**。没有 `FuncOp`/`CallOp`，只能靠单个大
   `entry` 内嵌套 `if`/`loop` 表达整个任务图，随着任务图节点数增加，IR 文本和维护复杂度会线性增长，
   没有代码复用/模块化机制。✅ 已验证。

4. **【中等严重性】没有 shared memory / 显式片上内存控制（旧版，E1 Q3）**，也没有 warp 角色分配
   机制（E1 Q4 + E7）。这限制了 megakernel 内部实现更精细的 warp-specialized 生产者/消费者模式
   （如 Hopper/Blackwell 上常见的 TMA+MMA+softmax 三 warp-group 流水线）——当前只能通过
   tile-block 粒度（而非 warp 粒度）做角色划分。旧版 ✅ 已验证无此能力；新版 ⚠️/❌ 文档声明但
   无法验证实际效果或工具可用性。

5. **【低严重性/待观察】没有设备端时钟/计时原语（E1 + E6 局限性）**。这不直接阻塞功能表达，但
   限制了对"重叠程度"这类问题的精确量化能力，只能靠原子计数器快照等间接代理手段。✅ 已验证不存在
  （`Ops.td` 检索无 `clock`/`globaltimer`）。

---

## 4. 意外发现

1. **E5 的竞态条件发现**：最初以为 E4 §10 报告的"grid=80 安全、120+ 死锁"是一个可复现的、由 grid
   大小唯一决定的确定性阈值，但受控消融实验（4 个 harness 变体交叉验证）证明这其实是一个对
   kernel 启动前 GPU 运行时状态（很可能是时钟/功耗唤醒状态）敏感的竞态条件——这比"有阈值"的结论
   严重得多，因为它意味着**没有任何已知方法能可靠地预先选出一个安全的 grid 规模**。这是本次调研
   中最重要、也最不符合直觉的发现。
2. **E2 的资源分配是 max 而非 sum**：异构分支合并进同一 entry 后，寄存器/共享内存开销接近其中最贵
   分支的独立开销，而非所有分支开销线性叠加——这是一个对 megakernel 可行性有利的意外正面发现，
   缓解了"任务图节点越多、occupancy 越差"的直觉担忧（但仅在 2 分支、互斥场景下验证，未验证推广到
   更多分支）。
3. **CGA（CUDA Cluster）指令无条件生成**（E5 §1.4）：无论 `num_cta_in_cga` hint 取值如何，
   tileiras 都会为涉及跨 block 同步的 kernel 生成 CGA 相关硬件指令（`SR_CgaSize`/`CGAERRBAR` 等）——
   这可能与死锁有关联，但未坐实，留作后续排查线索。
4. **persistent kernel 的 grid 大小不是越小越好**（E3）：grid=170（=SM 数）比 grid=10000 慢约
   8.4 倍，因为朴素的"grid=SM数"假设忽略了单 SM 可并发驻留多个 block（实测约 8 blocks/SM）这一
   事实，过小的 grid 反而显著欠用硬件并行度。

---

## 5. 未能验证的项

1. **新版 cuda-tile（tensor-ir 依赖版本）的 `num_worker_warps_per_cta` 字段实际运行时效果**——
   没有可用的独立编译/测试工具（E7）。
2. **新版 cuda-tile 的 `AllocaOp`（受限 shared-memory 等价物）、`AtomicRedViewTkoOp` 等新增算子的
   实际编译/运行行为**——仅读到源码定义，旧版可编译工具链中不存在这些 op（E1）。
3. **E4/E5 死锁的根本机制**——是 tileiras 生成的 cubin cooperative-launch metadata 有问题，还是
   driver 的 occupancy 计算对这类 cubin 失真，还是与 CGA/cluster 调度约束有关，均未查明，没有条件
   查看 tileiras/driver 闭源部分的实现。
4. **E2 资源分配"max 而非 sum"规律是否对 3 个以上分支、或更复杂嵌套控制流依然成立**——只测试了
   2 分支互斥场景。
5. **更大规模（几十到上百个并发角色）任务图的表达与正确性**——受限于 E4/E5 揭示的死锁问题，未尝试；
   按已有证据推断几乎肯定会触发该问题，但没有实际测试去验证"几乎肯定"这一判断本身。

---

## R2-1. Round 2 详情：R2-A 活锁 vs 死锁判别

方法：前台启动 `spin_wait_tokenchain.cubin` 的 `host_test_biggrid_tokenchain 170`（故意不加
`timeout`，等它挂起），另开终端用 `nvidia-smi dmon`、`nvidia-smi -q -d POWER,CLOCK`、三次独立
`cuda-gdb -p PID -batch -ex "info cuda threads"`（间隔约 3s/15s）、`compute-sanitizer
--tool synccheck/racecheck` 依次采集证据。

- SM 利用率 100%、时钟 2887MHz（接近满速 3090MHz）、功耗 97.71~98.33W（idle 基线 13.19W 的
  约 7.4×，但只有 TDP 575W 的约 17%）——✅ 已验证，表面符合活锁直觉，但功耗量级偏低是第一个
  疑点。
- 三次独立 cuda-gdb 采样（跨越 20+ 秒）：170 个 block 里恒定只有同一批 57 个出现在线程列表里，
  PC 逐字节完全一致（`spin_wait_tokenchain+24976` = 偏移 `0x6190`），**20+ 秒内一条指令都没有
  前进**——✅ 已验证，这是死锁的判据而非活锁。其余 113 个 block（含生产者）不在列表中，说明
  已正常执行完毕退出。
- 卡住的 PC 反汇编：`0x6190` 是一条 `LDS R4, [UR5]`，紧跟在 `BAR.SYNC.DEFER_BLOCKING` 之后，
  位于 consumer 的 256 次静态展开 reduce 收尾代码里（偏移 `0x260~0x61c0` 区间），**不在**
  自旋等待循环本体（偏移 `0x1f0~0x230`）——✅ 已验证，两处代码距离约 6000 字节。
- `compute-sanitizer --tool synccheck`（×2）/ `--tool racecheck`（×1）：grid=170 在插桩下
  3/3 次都不挂起、0 errors/hazards——✅ 已验证不挂起，❌ 不能据此断言无 race（sanitizer 插桩
  改变时序，绕开了触发条件，不代表 bug 不存在，与 E5 的"预热能救 grid=80"同一类信号）。
- **判断（不强行二选一）**：约 2/3（113/170，含生产者）成功完成——这是 100% SM 利用率的真实
  来源；约 1/3（57/170）冻结在自旋循环*之后*的 reduce 收尾代码里、真正的 PC 零漂移，是**局部
  死锁**，不是活锁式的"忙但原地打转"。这直接证伪了 round 2 的核心假设（自旋循环无退避导致
  内存子系统饱和活锁）——卡死位置根本不在自旋循环里。`kill -9` 后 GPU 立即回到 idle，
  新 context 跑小 grid 立即成功，排除硬件永久损坏。

详见 `experiments/R2_A_livelock_vs_deadlock/result.md`。

## R2-2. Round 2 详情：R2-B test-and-test-and-set 修复验证

方法：把自旋轮询循环改写为 relaxed 读（循环体内无 `CCTL.IVALL`），跳出循环后单独做一次
acquire 语义的重读（`CCTL.IVALL` 移到循环外、只执行一次）——经典 test-and-test-and-set 模式。

- SASS 证实改动完全达到设计目标：轮询循环体内不再有 `CCTL.IVALL`，只在跳出循环后的 Phase-2
  acquire 重读里出现一次；两个 cubin 的 `CCTL.IVALL` 全文件静态计数都是 225（位置不同，
  运行时执行次数从"每次轮询迭代 1 次"降到"总共 1 次"）——✅ 已验证。轮询循环体内两个变体都
  仍保留 `BAR.SYNC.DEFER_BLOCKING`（这条改动完全没有触及）。
- 正确性不受影响：1MB 数据 1000/1000 次通过，4MB 数据 200/200 次通过（原计划 500 次，因单次
  运行超过 300s 被 `timeout` 杀死且无中途输出，缩减为 200 次，如实标注）——✅ 已验证，均为
  grid=2（远低于挂起阈值）下的验证，只确认内存序改动本身不引入数据竞争。
- **核心 grid-scan（M=50/grid，6s 超时，沿用 E5 确认的"无预热 worst-case"harness）**：

  | 变体 | grid=30 | grid=80 | grid=120 | grid=170 |
  |---|---|---|---|---|
  | `relaxed_acquire`（R2-B 修复） | 48/50 | 46/50 | 50/50 | 50/50 |
  | `baseline_acquire_in_loop`（E4/round-1 原始版本） | 39/50 | 50/50 | 50/50 | 50/50 |

  **❌ 已证伪**：挂起率没有得到任何有意义的改善，grid=30 时甚至比 baseline 更差
  （96% vs 78%，⚠️ 该差异未做显著性检验），grid≥80 时两者都已接近或达到 100%。

详见 `experiments/R2_B_relaxed_plus_acquire/result.md`、`run.sh`。

## R2-3. Round 2 详情：R2-C 自旋退避修复验证

方法：轮询循环每次非 break 迭代插入一段 64/256/1024 步的编译器不可折叠 LCG
（`x' = x*1103515245 + 12345 xor i`）延迟计算，人为降低轮询频率。

**方法论陷阱（先修复才能做实验）**：第一版朴素累加式延迟循环（`addi %dacc, %bi`）被 tileiras
识别为等差数列闭式解、整体常量折叠成单条指令（如 `UIADD3 ..., 0x7e0, ...`，0x7e0=64·63/2），
产生零实际延迟，完全违背实验设计意图——只有对比不同 N 档位的 SASS 指令数是否成比例变化才能
发现这个破绽。改用 LCG 式非线性递推后，SASS 证实指令数随 N 近似线性 scale（backoff64/256/1024
分别 129/514/2050 条 UIMAD/LOP3/XOR 指令，回边地址 0xaa0/0x22b0/0x82b0 同步右移），且轮询频率
本身不受影响（`BAR.SYNC`/`CCTL.IVALL` 静态计数在三档下恒为 85/226）——✅ 已验证延迟真实存在。

**grid-scan（M=30/grid，5s 超时，矩阵较 R2-B 缩减为 grid∈{30,170} 两点，时间预算限制）**：

| 变体 | grid=30 | grid=170 |
|---|---|---|
| `backoff64` | 25/30（83%） | 30/30（100%） |
| `backoff256` | 22/30（73%） | 30/30（100%） |
| `backoff1024` | 16/30（53%） | 30/30（100%） |
| `relaxed_backoff256`（⊕ R2-B 修复） | 26/30（87%） | 30/30（100%） |

**⚠️ 部分/有限缓解**：grid=30 时挂起率随退避强度单调下降（83%→73%→53%），说明轮询频率与
挂起率确有量级相关性，但最强档位也只降到 53%，远非"解决"；grid=170（硬件驻留容量边界，
R2-D 已确认）时**全部 4 变体 100% 挂起、退避完全无效**；与 R2-B 修复组合后未见正向叠加，
反而在 grid=30 略差于单独 backoff256（87% vs 73%，⚠️ 未做显著性检验）。

详见 `experiments/R2_C_backoff/result.md`、`run.sh`。

## R2-4. Round 2 详情：R2-D REG=228 vs occupancy "8 blocks/SM" 矛盾溯源

- round 1 全部 occupancy 诊断代码（`E4/host_test_coop.cpp:15` 等）统一用
  `cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 1, 0)`——blockSize 传 1，但该 cubin 的
  `EIATTR_REQNTID` 硬编码 blockDim=256，二者不匹配——✅ 已验证（`cuobjdump -elf` +
  `occ_probe.cpp` 交叉确认）。用错误的 blockSize=1/32 查询得到 numBlocksPerSm=8（对应
  max grid=1360）；用正确的 blockSize=256 查询得到 numBlocksPerSm=1（max grid=170，与手算
  65536/(228×256)=1.12→1 吻合，也与经验挂起边界吻合）。
- 用普通 nvcc、REG=218 的对照组 kernel 做同样的 4 档 blockSize 扫描，得到完全相同的模式
  （blockSize=1/32→8, 128→2, 256→1）——✅ 已验证 **driver 算法本身没有问题**，不存在
  "driver 对 tileiras cubin 专门算错"的情况，round 1 的诊断是方法论错误（用错 blockSize），
  不是 driver 或 tileiras 的 bug。
- 因果链验证：round 1 `cuLaunchCooperativeKernel` 用同样错误的 blockDim=(1,1,1) 发起，导致
  准入检查把可驻留 block 数算成 1360，从而"放行"了 grid=170——✅ 已验证。用正确 blockDim=256
  重新验证：grid=170 被接受、grid=171/340 被正确拒绝（"too many blocks in cooperative
  launch"）——driver 准入逻辑本身是对的。**但 grid=170 本身依然 hang（rc=124）**——
  blockDim mismatch 只解释了"为什么这么大的 grid 一开始没被拒绝"，**不解释"为什么跑到硬件
  容量边界时会卡死"**，后者与 R2-A 定位的问题是同一个、根因仍未查明。

详见 `experiments/R2_D_occupancy_audit/result.md`。

## R2-5.（编号预留：R2-E harness 卫生检查，pending，见 R2-8 节）

## R2-6. Round 2 详情：R2-F 软件流水线/双缓冲判定

⚠️ 范围限制：现有全部 kernel（`spin_wait_tokenchain` 及变体）都是纯 elementwise load+reduce，
不含任何 MMA/tensor-core 矩阵乘法，因此本实验只能回答结构性问题（是否存在跨迭代加载批处理），
无法测量真实 TFLOPS-vs-peak。

- **发现 1（✅ 已验证）**：consumer 的 256 次 reduce-for 循环被完全静态展开（无 runtime 回边），
  全文件仅有的两处后向 `BRA`（自旋等待循环 `0x230→0x1f0`；producer 数据生成循环
  `0x65b0→0x6260`）之外，consumer 收尾区间（`0x260~0x61c0`，约 2400 条指令）没有任何后向跳转。
- **发现 2（✅ 已验证，本轮新发现）**：自旋等待循环体内部本身包含一条
  `BAR.SYNC.DEFER_BLOCKING`（每次轮询迭代都要求该 tile block 全部 256 线程在硬件屏障上会合，
  而不是 1 个线程轮询、其余等待）。这是解释"R2-B/R2-C 为何都未能修复挂起"的关键结构性证据：
  两个方案都只改了轮询循环的内存序/频率，都没有触及这条硬件屏障本身。R2-A 定位的死锁 PC
  也紧跟在另一条 `BAR.SYNC.DEFER_BLOCKING` 之后，两处证据指向同一模式（⚠️
  documented-but-unverified：未用 cuda-gdb 逐 warp 验证卡死线程是否正卡在某条 `BAR.SYNC`
  的到达计数上，是有强支持但未最终坐实的假说）。
- **发现 3（✅ 已验证）**：producer 的 256-chunk 写入循环以展开因子 2 编译成 128 次真实
  runtime 循环，与 consumer 的完全静态展开策略不同——同一 kernel 内 tileiras 对不同分支的
  展开决策不一致（具体依据未探究）。
- **发现 4（⚠️ documented-but-unverified）**：209/225（≈93%）条 LDG 集中在代码最前段一次性
  发出，随后才是 FADD/SHFL.BFLY/STS/LDS 交替的收尾序列——存在真实的跨迭代加载批处理/预取
  现象，但是否精确对应"预取未来 k 次迭代数据"的经典双缓冲语义未定点验证。
- ❌ 真实 TFLOPS/peak 未测量（无 MMA kernel）；❌ 手动 2 次迭代 unroll 对比未完成。

详见 `experiments/R2_F_pipelining/result.md`。

## R2-7. Round 2 详情：R2-G E2 max-not-sum 规律推广（N=2/3/5/8）

方法：N 路嵌套 if/else dispatch，工作负载从固定 4 元素调色板循环取用（`mm128` 128×128
`mmaf`、`red256`、`mm64`、`red1024`），走完整 verify→translate→tileiras→
`cuobjdump --dump-resource-usage` 链路，N=2 基线取自 round 1 E2 的同一工具口径。

| N | REG | STACK | SHARED |
|---|---|---|---|
| 2（round 1 E2） | 255 | 8984 | 99392 |
| 3 | 255 | 8984 | 99416 |
| 5 | 255 | 8984 | 99448 |
| 8 | 255 | 8984 | 99600 |

- **✅ 已验证**：REG、STACK 在 N=2/3/5/8 完全恒定（255、8984），LOCAL 全部为 0（无溢出迹象）
  ——MAX-not-SUM 规律在寄存器/栈帧维度上强力成立，从 N=2 推广到了 N=8。
- **✅ 已验证存在，⚠️ 精确归因未定点验证**：SHARED 从 99392 缓慢单调增至 99600
  （N=2→8，+208 字节，约 +0.2%），远非"分支数线性叠加各自完整 tile 缓冲区"的量级（若是真
  SUM，应是数千至上万字节增量），更接近"每个分支引入的 if/else 控制流跳转表等少量 CFG
  调度元数据"的线性小量——这是本实验相对 round 1 E2 的新发现，但没有做"只留空 if/else 骨架、
  去掉 tile 计算"的对照组来精确分离 CFG 开销与 workload 微弱累加两种解释。
- **❌ 未完成**：3 层嵌套 if-in-loop 测试；loop-carried 值跨分支是否使 MAX 退化为 SUM 测试
  ——这两个更贴近真实 megakernel 任务图复杂度的场景，round 1/2 全部产出物中都不存在对应
  kernel，本轮未新写。
- **⚠️ 附带观察（非正式计时）**：tileiras 编译耗时随 N 增长明显上升（N=5/8 均超过 2 分钟，
  需改用后台 `nohup timeout 600` 才能完成），与 R2-H 相关但未按 R2-H 的标准做严谨计时，
  只作定性记录。

详见 `experiments/R2_G_branch_generalization/result.md`、`run.sh`。

## R2-8. 未完成事项（如实标注为 pending，而非略去不提）

1. **R2-E（harness 卫生检查）：❌ 完全未开始。** 原计划内容：flag 清零/内存写入时序审计、
   host 端读回断言+复测挂起率、在挂起期间用另一进程读取 flag 内存的旁路验证、预热规模扫描
   （1KB/空 kernel/100MB 等档位）——均未进行，是本轮任务列表里唯一一个连初步探索都没有开始
   的子实验。
2. **R2-H（编译+查询墙钟耗时测量）：❌ 完全未开始。** 原计划内容：tileiras 各阶段
   （verify/translate/assemble）耗时分解、`tileiras` 各类 flag/cache 选项对耗时的影响、
   外推到真实查询规模场景下的耗时预算——均未进行。R2-G 附带观察到的"编译耗时随分支数 N
   增长明显上升"（见 R2-7 节末尾）是本轮唯一与 R2-H 相关的数据点，但只是定性记录，不满足
   R2-H 原计划的测量严谨度要求。
3. **R2-B/R2-C 挂起现场的 cuda-gdb 定点验证：⚠️ 遗留但非完全空白。** R2-A 已经在 baseline
   变体上做了完整的 cuda-gdb PC 定位（见 R2-1 节），但 R2-B/R2-C 各变体挂起时是否卡在同一条
   `BAR.SYNC.DEFER_BLOCKING` 附近的指令上，没有逐一用 cuda-gdb 重新采样确认——这是"根因在
   `BAR.SYNC.DEFER_BLOCKING` 收敛机制"这一假说目前最大的验证缺口，明确标注为可选后续加强项。
4. **R2-F 的 TFLOPS-vs-peak 与手动 unroll 对比：❌ 未完成**（见 R2-6 节，需要新写 MMA kernel，
   不在本轮时间预算内）。
5. **R2-G 的 3 层嵌套 if-in-loop、loop-carried 跨分支退化测试：❌ 未完成**（见 R2-7 节）。

上述 5 项均为诚实标注的知识空白，不影响本节已完成部分（R2-A/B/C/D/F/G 核心问题）得出的结论
的有效性，但限制了这些结论能够外推的范围——尤其是 R2-E 的完全空白意味着"挂起是否与 flag
内存本身的写入/清零时序有关"这一替代假说，round 2 全程都没有真正排除过。
