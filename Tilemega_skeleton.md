# TileMega 开发骨架

> **一句话**：在 NVIDIA CUDA Tile IR / TensorIR 栈上，把 megakernel 的任务划分、事件生成与
> 粒度选择，从人工输入提升为基于仿射依赖关系的可求解编译问题。
>
> **文档定位**：长期维护的开发指南。只写「要建什么、怎么建、注意什么、可以抄谁」。
>
> **维护约定**：完成的条目打勾并追加实测结论；推翻的假设保留原文并注明推翻原因，不删除。

---

## 目录

- [1. 设计理念](#1-设计理念)
- [2. 平台约束速查](#2-平台约束速查)
- [3. 架构](#3-架构)
- [4. 仓库结构](#4-仓库结构)
- [5. 分阶段 TODO](#5-分阶段-todo)
- [6. Codegen 规则](#6-codegen-规则)
- [7. 风险与未决问题](#7-风险与未决问题)
- [附录 A. 可借鉴实现速查](#附录-a-可借鉴实现速查)
- [附录 B. 源码位置索引](#附录-b-源码位置索引)

---

# 1. 设计理念

## 1.1 要解决的问题

Megakernel 把整个模型放进一个持久驻留的 kernel，用 kernel 内事件同步代替 kernel 边界，
从而消除 launch 开销并让跨算子的 tile 级重叠成为可能。

现有实现（Mirage MPK、MLSys'26 Event Tensor）把「任务怎么划分、事件粒度多大」当作系统的
**输入或配置**：MPK 的每个 layer API 都要求用户传 `grid_dim`，其 demo 里用的是硬编码魔数；
ETC 要求划分与事件图由用户以 Triton kernel 或 compiler builtin 的形式提供。

TileMega 要把这一层变成可求解的：**用仿射依赖关系把 task 数、事件数、同步开销、
wave quantization、数据复用表达成 `(shape, tile, 事件粒度)` 的闭式函数**，
于是划分和粒度成为可以构造求解的对象，符号形状支持是这个表示的推论。

## 1.2 三条不可动摇的原则

### 原则一：ISL 只做 task 间，不进 task 内

**ISL 用于**：access relation、dependence relation、参数化整数集、barvinok 基数计数。

**ISL 不用于**：Pluto 式仿射调度。事件驱动执行只需偏序，强加全序会损失并行度；
且目标函数不匹配（我们是 `P|prec|C_max` 调度问题，list scheduling 才是对的工具）。

task 内的 MMA 形状、TMA、warp specialization、shared memory swizzle 全部交给 CUDA Tile。
**把 ISL 塞进 task 内等于重新实现 CUDA Tile，且做得更差。**

### 原则二：不是「联合分析」，是「代价查询接口」

划分决策必须能拿到 task 内实现的真实代价（寄存器压力、shared memory 占用、是否合法）。
但因 `tileiras` 闭源，这只能是**黑盒查询**：

```
Solver 层                          Codegen/Backend 层
─────────                          ──────────────────
枚举候选粒度 g          ──查询──→   编译 (算子, g) → cubin
                       ←──返回──    是否合法 / REG / SHM / 延迟
推 Dep(g)、事件结构
算 cost(g) → 选最优 g
```

### 原则三：正确性阶梯，每一级对前一级做差分测试

```
L0    多 kernel 逐算子执行（PyTorch eager）        ← 正确性金标准
L0.5  host 端 stage 循环 + task dispatch          ← 验证编译管线，绕开 in-kernel 同步
L1    单 entry + 每 stage 后全局 barrier           ← megakernel 骨架，正确但零重叠
L2    逐边细粒度事件（ISL 推导）
L3    事件融合 + 粒度优化（Solver）
L4    符号 shape 参数化 + 运行时变体选择
```

调试 megakernel 极其困难（无设备端 printf 保证、无时钟原语、挂起后现场难保留）。
这个阶梯是唯一可控的开发方式，每一级都要保留编译开关。

## 1.3 依赖的三个 Tier（ISL 分析的分类骨架）

| Tier | 定义 | LLM 中的实例 | 处理 | 代价 |
|---|---|---|---|---|
| **0** | 纯仿射 | norm / proj / elementwise / RoPE / GEMM(含 split-K) / GQA head 映射 | ISL 精确推导 | 0 |
| **1** | 共享单射布局的间接寻址 | paged KV cache 的 block table | 布局函数在 `Dep = P⁻¹∘L⁻¹∘L∘A` 中抵消，逻辑空间精确 | 0 |
| **2** | 结构化 ragged（模式已知、extent 运行时） | split-KV chunk 数、chunked prefill 的 per-request token 数、MoE 每 expert token 数 | 索引映射保持符号闭式，extent 与 wait count 运行时填（indptr 前缀和） | 一次前缀和 |
| **3** | 真正数据相关排列 | MoE topk 路由、投机解码接受长度、动态稀疏 mask | 保守化：分组 barrier（per-sequence/per-expert）或算子级 barrier | 局部同步过度 |

**关键性质：局部非仿射不会污染整图分析。** 逐边构造事件，把任意 `Dep_e` 替换成
`Dep'_e ⊇ Dep_e` 仍然正确，保守化的代价是局部的。最坏情况（Tier 3 退化到算子级 barrier）
等价于今天所有系统的做法，永远不会更差。

---

# 2. 平台约束速查

> 本节是实现时必须随时对照的事实清单。`[R1-Ex]` / `[R2-x]` 为实测编号，`[SPEC]` 为官方规范。

## 2.1 可用能力

| 能力 | 结论 | 来源 |
|---|---|---|
| 单 entry 内异构 tile 形状分支 | 可行；**资源开销取 MAX 不是 SUM，已验证到 N=8 分支** | `[R1-E2]` `[R2-G]` |
| runtime 上界的 persistent grid-stride loop | 可行，`for %i in (%bx to %total, step %gx)` 三个边界均可为运行时 SSA 值 | `[R1-E3]` |
| 跨 tile block 生产者/消费者同步 | 语法与小规模正确性成立（grid=3，200/200） | `[R1-E4/E6]` |
| 单次 launch 内多角色并发 | 三角色（生产者/消费者/独立任务）并发，独立任务不被无关同步阻塞 | `[R1-E6]` |
| 原子操作 | `atomic_rmw_tko` / `atomic_cas_tko`，DEVICE scope × {RELAXED, ACQUIRE, RELEASE, ACQ_REL} | `[R1-E1]` |
| `load_ptr_tko` ordering | {WEAK, RELAXED, ACQUIRE} + 可选 scope | `[SPEC]` |
| 控制流 | `LoopOp`（无界 while + break/continue）/ `ForOp`（边界可运行时）/ `IfOp`，均支持 loop-carried values | `[R1-E1]` |
| 一个 module 多个 entry | verifier 通过（但 entry 之间不能互相调用） | `[R1-E1]` |
| `partition_view` 类型 | `(tensor_view, tile_size)`，`indexSpace[i] = shape[dimGroups[i]]/tileSize[i]` — **这就是 P 映射的 IR 表示** | `[SPEC §6.3.3]` |
| `tensor_view` 动态形状 | shape 与 stride 可逐维为 `?` | `[SPEC]` |
| 跨迭代批量预取 | persistent loop 中观察到，但**是否构成经典双缓冲未定点验证** | `[R2-F]` |

## 2.2 硬约束（无法绕过，影响设计）

| 约束 | 实现上的后果 |
|---|---|
| **tile 形状必须编译期静态，且为 2 的幂** | 运行时可变粒度不可能 → 只能「编译期多变体 + 运行时分支」。因资源取 MAX，此路可行 |
| **无 device 端函数调用**（tile function 规范中标注 "currently disabled"） | 整个任务图内联进单个 `entry`。用外层层循环 + 内层 stage 展开把 IR 体积压到 `O(stage 数)` |
| **无 shared memory / 片上内存显式控制** | MPK 的分页 shared memory 机制无法移植；片上缓冲完全由 tileiras 管理 |
| **无 warp 角色分配机制** | 角色分化只能到 tile-block 粒度（MPK 实际也是 block 级：`if (blockIdx.x < num_workers)`），不影响本框架 |
| **无独立 fence/barrier op** | 顺序保证只能靠 ordering 属性 + token 数据流 |
| **无设备端时钟原语** | kernel 内无法量化重叠，只能靠原子计数器快照等代理 |
| **token 是纯编译期 SSA 排序约束** | 见 §2.3 |
| **一个 tile block = 一个逻辑线程 = 一组物理线程。~~实测 256~~ → 线程数不是常量** | 由 `optimization_hints` 的 `num_worker_warps_per_cta`（**仅支持 4 或 8**）与后端版本共同决定。**自旋循环内部会生成 `BAR.SYNC.DEFER_BLOCKING`**，每次轮询要求该 block 全部线程会合。Tile IR **做不到「一个线程轮询、其余等待」**。详见下方修正 |
| entry 参数只能是 rank-0 标量 tile 或 `tile<ptr<E>>` | 所有张量以裸指针传入 |
| `LoopOp` 的 loop-carried 变量不能是 view 类型 | 影响 persistent 循环携带 partition view 的写法 |
| 循环内不能提前 return | 需先终止循环再返回 |
| Grid 每轴上限 `2^24-1` | 实践中无影响 |

### ⚠️ 修正（V0 提出，V1 实测确认）：每 block 的线程数不是常量

原文把「256」写成了实测常量。**它是后端版本 + `optimization_hints` 共同决定的**：

- `num_worker_warps_per_cta` **仅支持 4 或 8**（`AttrDefs.td:102`）
- cuda-tile 文档给 sm_120 的示例值是
  `sm_120 = {num_cta_in_cga = 16, num_worker_warps_per_cta = 4}`
  —— 4 warp × 32 = **128 线程**，与 13.3 实测的 REQNTID 完全吻合

| | tileiras 13.1 (V13.1.80) | tileiras 13.3 (V13.3.36) |
|---|---|---|
| `EIATTR_REQNTID` | **256** | **128** |
| REG | 228 | 229 |
| blocks/SM | 1 | **2** |
| cooperative launch 上限（170 SM） | **170** | **340** ✅ V1 已用 driver 实测确认 |

**这个量直接决定 occupancy 与 cooperative launch 上限，因此应作为
`BackendCostQuery` 的一个查询维度。** V1 给出了它有多敏感的量级证据：

| | V1-ctrl（含 Bug B） | V1-a（修掉 Bug B） |
|---|---|---|
| REG/thread | 229 | **24** |
| blocks/SM（同为 REQNTID=128） | 2 | **12** |
| cooperative 上限 | 340 | **2040** |

**同一台机器、同一后端，仅仅改掉一处内存序写法，驻留容量差 6 倍。**

**实现含义**：
- 任何依赖「一个 block 多少线程」的推理都必须**从 cubin 读 `EIATTR_REQNTID`**。
  §6.7 现在有了第二个理由，而且是字面意义上的必须：
  V1 期间发现 `tilemega-occupancy` 曾用 `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
  冒充 REQNTID，那是**寄存器允许的上限**（REG=24 时报 1024，而 REQNTID 是 128），
  差了 8 倍。已修正为真正解析 ELF
- §2.4 的「max grid=170」只对 13.1 成立
- grid 扫描范围要按当前后端的实际容量确定，不能沿用任何历史数字

详见 `docs/experiments/V0_toolchain_rebaseline/result.md` 与
`docs/experiments/V1_sync_corrected/SUMMARY.md`。

## 2.3 必须遵守：token 链

`[R1-E4]` 实测：仅用 `acquire`/`release` ordering、**不串 token**，则整个自旋循环
（含 flag 的 load 本身）**被编译器完全消除**。SASS 中全文件唯一一次对 flag 的引用只出现在
生产者的 release 处。1000/1000 次运行 100% 数据竞争，**编译期零报错零警告**。

正确写法见 §6.1。**这是 codegen 层必须保证的不变量，任何生成同步代码的路径都要走统一封装。**

## 2.4 ~~未解阻塞项：大 grid 挂起~~ → 测试代码 bug 导致的误判；真正的问题是静默数据损坏

> **状态更新（V1，2026-08-29）**：经对照官方内存模型规范复查，R1/R2 系列测试源码存在
> **三处 bug**，其中两处直接违反内存模型。**原结论不可信，已由 V1 重测。**
>
> - **Bug A**：生产者的 `release` 未 token-order 在数据 store 之后。
>   规范 §7.5 原文：*"Program dependencies (i.e. dependencies apparent from control flow,
>   data dependency, or address dependency) **do not** provide ordering between two memory
>   operations"*，且 *"Tokens must be used, **even where the token ordering appears
>   redundant** with program dependencies"*。§7.12.3：*"you need to token-order all memory
>   events that must stay before the release to the release itself"*。
>   **整个 E4/R2-B/R2-C 系列建立在坏掉的同步之上，1000/1000 通过是运气。**
> - **Bug B**：`acquire device` 打在 `tile<1024xf32>` 的每个元素上。规范 §7.1：
>   *"tile loads, stores, and atomic updates generate **one memory operation per element**
>   in the tile"* —— 单条指令 = 1024 个 acquire，循环 256 次 = 每 block 26 万个
>   device-scope acquire。这是 SASS 里那些 `CCTL.IVALL` 的来源。
>   **语义上完全多余**——happens-before 已由自旋循环的单次 acquire 建立（§7.12.3）。
> - **Bug C**：169 个 tile block 用 `weak` 并发写同一地址 `checksum_out[0]`。
>   §7.2：*"The compiler **may assume** that tiles accessed with `weak` are **not
>   concurrently accessed** by any other thread"*；§7.10：*"Programs with data races have
>   **undefined behaviour**"*。R2-A 定位的卡死 PC 正是该 store 的前一条指令。
> - **附带**：测试让 169 个 block 自旋在同一 flag 地址上，是最坏争用模式，
>   不代表 per-tile 事件张量的真实 megakernel。

### ✅ V1 的实测结论：挂起现象不复现，但出现了更隐蔽的问题

**1. 大 grid 挂起在 tileiras 13.3 上完全不复现。**

原封不动的 `spin_wait_tokenchain.mlir`（含全部三处 bug）：

| grid | 3 | 30 | 80 | 120 | 170 | 340 |
|---|---|---|---|---|---|---|
| 挂起 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 |

**300 次零挂起**，含 R2 报 39/50 挂起的 grid=30 与超出 R2 认知上限的 grid=340。
R1/R2「无已知安全实现路径」这一核心悲观结论，**在当前后端上不成立**。

**2. 但修好三处 bug 后，出现了静默数据损坏。**

最小复现（30 行 MLIR，一个生产者写**一个** 1024 元素 tile，token-order 后 release；
一个消费者 relaxed 自旋 + 单次 acquire，读同一 tile 求和，期望 1024.0）：

| grid | 3 | 30 | **80** | **120** | 170 | 340 |
|---|---|---|---|---|---|---|
| 校验失败 | 0/50 | 0/50 | **0/50** | **50/50** | 50/50 | 50/50 |

**阈值尖锐地落在 80 与 120 之间。** 读到的值恒为 768 或 512（而非 1024），
缺失量恒为 128 或 256 个元素 = **整数条 `STG.E` 指令的份额**（128 线程 × 1 元素）。
不是随机撕裂，是整粒度丢失。

**3. 已隔离到 Tile IR 工具链，不是硬件、不是内存模型。**

结构逐行等价的 CUDA C++ 版本（`__syncthreads()` + `__threadfence()` + `atomicExch`
作 release；消费者自旋 + `__threadfence()` 作 acquire），同一张卡、同一 driver：

| grid | 8 | 30 | 128 | 170 | 340 |
|---|---|---|---|---|---|
| 失败 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

**CUDA C++ 全过，Tile IR 在 grid≥120 全错。**

**4. per-tile 事件张量的争用模式并不能规避。**

V1-d（每个 block 等自己的 `flag[bx]`，128B padding，4 个生产者分别置位）
在 grid≥128 同样 10/10 失败。**原先寄望「真实 megakernel 争用分散、问题可能不存在」，
该寄望被证伪。**

### 已排除的解释（V1，全部实测）

| 假设 | 判定 |
|---|---|
| 自旋循环没有真的等待 | ❌ 排除。删掉生产者的 release atomic 后 kernel 按预期挂死 |
| `reduce` 算错 | ❌ 排除。host 预置数据与 flag、不经过握手时 3/3 精确正确 |
| 生产者没写完 | ❌ 排除。kernel 结束时 data 缓冲区 0/262144 个 poison 残留 |
| Bug B 是主因 | ❌ 排除。数据 load 改回 `acquire device` 仍 20/20 失败 |
| `weak` 违反 §7.2 的跨线程通信禁令 | ❌ 排除。改 `relaxed device` 后 SASS 确认变为 `STG.E.STRONG.GPU`，仍失败 |
| SASS 生成有误 | ❌ 排除。生产者 `BAR.SYNC` → `MEMBAR.ALL.GPU` → `ATOMG.E.EXCH.STRONG.GPU` 顺序正确；消费者自旋回边 → `CCTL.IVALL` → 数据 `LDG` 顺序正确 |
| 硬件或 driver | ❌ 排除。CUDA C++ 对照组全过 |

### ✅ 已确认的正面事实：跨 tile block 通信是 Tile IR 的一等能力

规范 §7.2 明确把 `device` scope 定义为「同一 GPU 内的通信」；§7.3 的 release/acquire
配对建立 happens-before；§7.11 说明整个模型是 **PTX 内存模型的严格弱化**
（*"a strict weakening of the PTX memory model"*）并可与 PTX 线程互操作。

**没有** CGA/cluster 级 scope（只有 `tile_block` / `device` / `sys` 三档），
**没有** 独立的 fence/barrier op —— 跨 block 同步的**唯一**手段就是
`device` scope 的 release/acquire + token 链。

---

## 2.4-历史 R1/R2 的原始观测（保留，不删除）

> 以下为 R1/R2 的原文记录。观测数据本身是真实的，但**产生这些数据的测试代码有三处 bug，
> 且当时用的是 tileiras 13.1**。V1 已证明挂起现象在 13.3 上不复现。

### 现象

`[R1-E4/E5]` `[R2-A/B/C/D]`

- 用正确的 token 链写法，grid 增大后大量挂起。R2 的 grid-scan（M=50）：

  | 变体 | grid=30 | grid=80 | grid=120 | grid=170 |
  |---|---|---|---|---|
  | baseline（循环内 acquire） | 39/50 挂 | 50/50 | 50/50 | 50/50 |
  | relaxed + 循环外 acquire | 48/50 挂 | 46/50 | 50/50 | 50/50 |

- 退避（LCG 延迟循环）：grid=30 随档位单调改善（backoff64 25/30 → backoff1024 16/30），
  **grid=170 时四个变体全部 30/30 挂起**

### 已排除的解释

| 假设 | 结论 |
|---|---|
| 共存性不足 | ❌ 排除。R2-D 用正确的 `blockSize=256` 查询得 `numBlocksPerSm=1`，max grid=170，与手算 `65536/(228×256)=1.12→1` 及硬件容量吻合。**grid=30 远低于容量仍然挂** |
| driver occupancy 算错 | ❌ 排除。R1 的矛盾数字源于 harness 调用 `cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, **1**, 0)` 传了 blockSize=1，而 cubin 的 `EIATTR_REQNTID` 要求 256。普通 nvcc kernel 对照组行为相同 |
| cooperative launch 准入失效 | ❌ 排除。用正确 blockDim 时 grid=170 被接受、grid=171 被拒绝，逻辑正确 |
| 活锁（忙等饱和内存系统） | ❌ 排除。cuda-gdb 三次采样（间隔 >20s）显示卡住 block 的 PC **逐字节完全一致**，20 秒内一条指令未前进 = 死锁判据 |
| 内存序问题 | ❌ 排除。relaxed 变体 SASS 确认 `CCTL.IVALL` 已移出循环，正确性 1000/1000 通过，但挂起率未改善 |
| 轮询频率 | ⚠️ 收效甚微。退避只在小 grid 有量级缓解 |

### 当前定位

- 挂起是**局部死锁**：170 个 block 中 57 个（33%）卡在**完全相同的 PC**，其余 113 个已正常退出
- 卡死 PC 在偏移 `0x6190` 的 `LDS R4, [UR5]`，**不在自旋循环本体（`0x1f0~0x230`）**，
  而在自旋之后、由 256 次静态展开构成的**消费者归约收尾代码**里，紧跟一条 `BAR.SYNC.DEFER_BLOCKING`
- **自旋轮询循环体内部本身也含 `BAR.SYNC.DEFER_BLOCKING`**（每次迭代全部 256 线程会合）
- `compute-sanitizer` 插桩下 3/3 不挂起（改变时序即可掩盖，不能反证 bug 不存在）

### 最有希望的下一步（尚未验证）

**假设 H1：挂起只发生在「自旋之后紧跟 reduce」这一特定组合，而非同步机制本身。**

依据：
- R1-E6 的消费者是逐元素（`D = C*2.0`），grid=3 时 200/200 正确
- R1-E4 / R2-B / R2-C 的消费者是 256 路展开的**归约**校验和，大量挂起
- R2-A 定位的卡死点正在归约收尾代码里，不在同步代码里

如果 H1 成立，**卡死的是测试 harness 的校验和计算，不是生产者-消费者机制**，
阻塞项的性质完全改变。验证方法见 §5.1 的 V1。

**假设 H2：根因与 `BAR.SYNC.DEFER_BLOCKING` 在高并发 tile block 占用下的收敛/公平性有关。**

R2 的两个修复方案都只改了轮询循环的写法，都没有触及这条硬件屏障，
所以在结构上就不可能修复它。目前有强 SASS/统计证据，但未经 cuda-gdb 在修复变体的
挂起现场逐点定位坐实。

### V0 重基线（2026-08-28）：换到 tileiras 13.3 之后

R1/R2 的全部证据来自 tileiras 13.1。V0 在 13.3 上做了单变量对照
（同一份 `.tilebc`，只换后端）：

| 项 | 结论 |
|---|---|
| 假设 H2 的对象（自旋循环体内的 `BAR.SYNC.DEFER_BLOCKING`） | ⚠️ **仍然存在**。屏障形式从 `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` 变为 `0x0`，但每次轮询仍要求全 block 线程会合。**H2 未被新版解决** |
| `CCTL.IVALL` 总数 | 225 → 28（−88%）。R2-B 手工优化的目标，13.3 在全局层面做掉了大半 |
| SASS 指令数 / 屏障总数 | 1664 → 1448 / 113 → 85 |
| 每 block 线程数 | 256 → 128，驻留容量翻倍（见 §2.2 的修正） |
| 挂起率本身 | ⏸ **尚未测**。GPU 统计扫描因 GPU 处于污染状态被阻塞，见下 |

**GPU 污染事故（重要操作教训）**：一次挂起中的运行被外部超时 SIGKILL 后，
`nvidia-smi` 持续报 `SM 100% / 2887MHz / 97.5W / 显存 0MiB / 无进程`——
与 R2-A 记录的挂起特征逐项一致。**杀掉宿主进程不回收 GPU 上挂死的持久
kernel**，需要 `nvidia-smi --gpu-reset`。

由此确立两条测试纪律（已写入 P0.3 并内建进 `scripts/gpu_stat_run.sh`）：
1. GPU 测试必须**独占 + 机器空闲**才能跑，否则数据不可用
2. **挂起测试的超时必须由测试脚本自己短超时管理**，绝不能让外层工具的
   超时去杀一个挂起中的 GPU 进程

### 对开发的影响

**L1（in-kernel 全局 barrier）阻塞于此。因此 L0.5（host 端 stage 循环）是必要的去风险路径**：
把 stage 边界放回 host，用多次 kernel launch 代替 in-kernel barrier，
在同一份 codegen 代码路径上验证除同步外的一切。

---

# 3. 架构

## 3.1 分层

```
┌──────────────────────────────────────────────────────────┐
│  L5  Serving Harness   continuous batching / paged KV     │
├──────────────────────────────────────────────────────────┤
│  L4  Frontend          torch.export → nv_tensor_ir        │
├──────────────────────────────────────────────────────────┤
│  L3  Analysis (ISL)    P/A 映射 → Dep → 事件张量           │
├──────────────────────────────────────────────────────────┤
│  L2  Solver            合法性剪枝 → 对齐传播 → 链上 DP      │
│                        ↕ 代价查询                          │
├──────────────────────────────────────────────────────────┤
│  L1  Codegen           TaskGraph → cuda_tile entry         │
├──────────────────────────────────────────────────────────┤
│  L0  Backend           tileiras（闭源黑盒）→ cubin          │
└──────────────────────────────────────────────────────────┘
```

## 3.2 关键设计决策

### 前端用 `torch.export`

`ExportedProgram` 提供：ATen 级 FX graph、`ShapeEnv` 中的符号维（sympy 表达式）、
FakeTensor meta。**动态维通过 `torch.export.Dim` 声明，内部已有符号推理和 guard 求解**，
这直接给了我们符号 shape 的来源，只需写 sympy → ISL parameter 的转换。

`ShapeEnv` 的 guards（`s0 % 128 == 0` 这类）可直接转成 ISL 集合约束，
相当于白送一层 shape constraint。

**LLM 特有的部分不在图里**（KV cache 管理、paged block table、continuous batching 调度），
这些属于 L5，见 §3.3。

### TaskGraph 直接扩展 `nv_tensor_ir`，不建新 dialect

复用 tensor-ir 已有的 `LayoutPropInterface`、`TensorSourceAttr`（CuTe layout + 符号维绑定）、
`ReductionSourceAttr` / `MatmulSourceAttr` 等布局传播基础设施。新增：

```mlir
// 一组并行 task 的定义（task 由多维坐标索引）
nv_tensor_ir.device_fn @gemm_tasks(%coord: index, ...) { ... }

// 事件张量：形状可含符号维
nv_tensor_ir.event_tensor @e0 : tensor<?xi32>

// 任务图：call_device 时标注 task 坐标 ↔ 事件坐标的映射
nv_tensor_ir.graph_fn @layer(...) {
    nv_tensor_ir.call_device @gemm_tasks
        in_edges  = [#nv_tensor_ir.access_map<"(m,n) -> (m)">]
        out_edges = [...]
        tier      = 0
}
```

新增属性：
- `AccessMapAttr` — 承载 `P` / `A` / `Dep`，与 ISL 互转
- `EventTensorAttr` — 形状（可符号）、wait count（可为运行时值）
- `TierAttr` — 0/1/2/3，驱动保守化策略

### 事件张量的放置：host 分配的 device buffer

**不用 `cuda_tile.global`**，理由：
- `global` 在 module load 时初始化，无法按 launch 重新初始化（除非另起清零 kernel）
- 大小依赖运行时 shape，`global` 是编译期固定的
- MPK 用的也是 device malloc 数组

**性能相关的设计规则**：
- 事件计数器**按 cache line（128B）padding**，避免不同事件的原子操作产生 false sharing
- 用**单调计数器**（借鉴 MPK）：`needed = num_triggers × iteration_num`，
  计数器从不重置，避免跨迭代 ABA，也免掉迭代边界的全局清零
- 计数器数组应尽量小以保持 L2 常驻；若事件数很大，考虑按 stage 分段复用同一块 buffer
- **待测**：`cuda_tile.global` 是否因常量 bank 寻址带来更好的 codegen。
  若测得有显著优势，可对「数量固定的少数事件」用 `global`，其余用 buffer

### 算子覆盖策略

tensor-ir 只有 63 个算子（pointwise / matmul / `reduce_ud` / reshape / concat / iota /
broadcast / slice / transpose），**无 attention、无 gather/scatter、无控制流**。

| 方案 | 适用 | 代价 |
|---|---|---|
| (a) 分解为已有算子 | attention → matmul + softmax(`reduce_ud`) + matmul | 非 paged 场景可行 |
| (b) 扩展 dialect | 加 `attention` / `gather` / `scatter` op | 需写 layout 传播规则 |
| (c) opaque task + 手写 Tile IR | 把算子当黑盒 | 逃生舱，该 task 的 Dep 只能保守化 |

**路线**：Phase 2 用 (a) 走通非 paged 最小模型；Phase 3+ 对 attention / paged attention
用 (b) 并显式提供 `P`/`A` 注解；(c) 留给 MoE 路由这类真正无法刻画的算子。

## 3.3 Serving harness（L5）

**可以直接参考 MPK 和 ETC 的实现，思想已经成熟，问题只是怎么接进我们的任务图。**

| 组件 | 参考实现 | 接入方式 |
|---|---|---|
| Paged KV cache | MPK `TaskMetadata` union 里的 `{request_id, kv_idx, merge_task_offset}` | block table 走 Tier 1（布局抵消），逻辑空间做依赖分析 |
| Continuous batching | MPK 把请求准入/完成剔除/KV 元数据更新做成 kernel 内的一个 task（`TASK_SCHD_PREPARE_BATCH`），避免每步 host-device 同步 | 作为任务图的一个特殊 stage |
| Chunked prefill | 无现成实现（MPK 用固定 prompt 离线批处理，ETC 只评估 decode） | Tier 2，per-request token 数用 indptr 参数化 |
| MoE 路由 | ETC 的 `topk` 决定事件计数器初值 + `exp_indptr` 前缀和决定 task 区间 | Tier 3 的一般化：把 MPK 硬编码的三种 metadata 特例做成统一的 indptr 机制 |
| CPU↔GPU 通信 | MPK `mpk_atoms.cuh` 的 `ld.acquire.sys.b32` / `st.release.sys.b32` + pinned ring | Tile IR 的 SYS scope 对应 |

**注意**：MPK 的这部分实现深度绑定它的 `TaskDesc`/`EventDesc` 结构和 84K 行 CUDA runtime，
不要直接移植代码，**移植的是机制设计**。

**⚠️ 不要照抄 MPK 的同步写法**：MPK 是 CUDA C++/PTX，内存模型与 Tile IR 不同 ——
Tile IR 是 PTX 模型的**严格弱化**（规范 §7.11），且**程序依赖不提供任何排序**，
必须用 token（§7.5）。`mpk_atoms.cuh` 只能作为「哪个操作该用哪个 scope/ordering」的
**语义参照**，落到 Tile IR 时必须按 §6.8–6.10 重新组织 token 链。

## 3.4 朴素 megakernel 的形态（L1）

因为没有 device 端函数调用，一切内联。控制 IR 体积的方式：

```mlir
cuda_tile.entry @megakernel(
    %weights: tile<ptr<f16>>,     // 所有层权重基址
    %kv_cache: tile<ptr<f16>>,
    %io: tile<ptr<f16>>,
    %event_buf: tile<ptr<i32>>,   // 事件张量
    %task_desc: tile<ptr<i32>>,   // 任务描述数组
    %num_layers: tile<i32>,
    %S: tile<i32>, ...            // 符号 shape 参数
) {
  %bid = get_tile_block_id
  %nblk = get_num_tile_blocks

  // 外层：层循环（运行时上界，权重指针按层偏移）
  for %layer = 0 to %num_layers {
      // 内层：stage 展开（编译期，约 17 个 stage）
      // ---- stage 0: RMSNorm1 ----
      for %t = %bid to %ntasks_s0 step %nblk {
          ... tile body（tile 形状编译期常量）...
      }
      // ---- sync ----
      // L1: 全局 barrier；L2: 细粒度事件 wait/notify
      ...
      // ---- stage 1: QKV proj ----
      ...
  }
  return
}
```

**IR 体积 = `O(stage 数 × 粒度变体数)`，不是 `O(task 数)`。**
32 层 Llama 只需展开约 17 个 stage 而非 544 个算子实例。

## 3.5 求解流程（Solver 层）

**不搜索，构造。** 因为硬件约束把每个算子的合法 tile 形状压到只有十几种。

```
层1  合法性剪枝：从硬件原生 MMA 形状扩张，撞资源墙停 → 每算子 8~20 候选
     ← 必须查询 Codegen 后端（原则二）
层2  跨算子对齐传播：wait = ⌈(m+1)Tm/Tr⌉ − ⌊mTm/Tr⌋
     Tr=Tm 时 wait=1，不整除时膨胀且变分段 → 联合空间大幅塌缩
层3  链上 DP（图是深而窄的，接近阶段链）：
     DP[i][g] = min_{g'} { DP[i−1][g'] + Cost_i(g) + Interface(g',g) }
层4  排布：分层 DAG 上的 list scheduling（关键路径优先）
```

代价函数：

```
Cost(g,κ) = max(T_compute(g), T_memory(g))   # roofline
          + T_sync(g,κ)                       # 事件数 × 原子延迟 + 自旋
          + T_quant(g)                        # wave quantization
          + T_bubble(g)                       # 流水气泡

T_quant = (⌈N(g)/M⌉·M − N(g)) / N(g) · t_task(g)     # N(g) 由 barvinok 给出
```

**必须分 regime**：

| Regime | 瓶颈 | 目标 |
|---|---|---|
| A. 低 batch decode | 权重带宽 | 最小化权重加载管道气泡（**不是 makespan**） |
| B. prefill / 大 batch | 计算 | makespan + wave quantization |
| C. 混合 batch（chunked prefill） | 异构 | 资源互补性最大化 |

**输出形态**：不是一个具体 schedule，而是**参数化的最优解区间划分**
（「`S ∈ [0,512)` 用 `g₁`；`S ∈ [512,∞)` 用 `g₂`」），区间边界来自分段拟多项式交点。
运行时 `O(1)` 查表选变体。

## 3.6 Llama decoder layer 依赖表（Phase 3 的验收基准）

符号：`H=4096, n_h=32, n_kv=8, d=128, I=14336, S=token数, L_s=序列KV长度, Tm=Tn=Tkv=128`

**ISL 自动推导的结果必须与本表逐项一致。**

| # | 边 | Dep（消费者→生产者） | 事件张量形状 | wait | fan-out | Tier |
|---|---|---|---|---|---|---|
| 1 | RMSNorm1 → Wq/Wk/Wv | `(m,n) ↦ m` | `[⌈S/Tm⌉]` | 1 | 48 | 0 |
| 2 | Wq → RoPE_q | `(m,hh) ↦ (m,hh)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 |
| 3 | RoPE_k → KVappend | `(m,hh) ↦ (m,hh)` | 同上（k 侧 8 组） | 1 | 1 | **1** |
| 4 | KVappend → Attn chunk | 仅 `j = ⌊(L_s−1)/Tkv⌋` | ragged | 1 | 运行时 | **2** |
| 5 | RoPE_q → Attn chunk | `(s,hh,j) ↦ (⌊s/Tm⌋, hh)` | ragged 域，映射仿射 | 1 | 运行时 | **2** |
| 6 | Attn chunk → Attn combine | `(s,hh) ↦ {(s,hh,j) : j<⌈L_s/Tkv⌉}` | `[B×32]` | 运行时 `⌈L_s/Tkv⌉` | 1 | **2** |
| 7 | Attn combine → Wo | `(m,n) ↦ {(s,hh) : s∈行块m, ∀hh}` | `[⌈S/Tm⌉]` | `Tm×32` | 32 | 0 |
| 8 | Wo → residual add | `(m,n) ↦ (m,n)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 |
| 9 | add → RMSNorm2 | `i ↦ {(i,n) : ∀n}` | `[⌈S/Tm⌉]` | 32 | 1 | 0 |
| 10 | RMSNorm2 → Wgate/Wup | `(m,n) ↦ m` | `[⌈S/Tm⌉]` | 1 | 224 | 0 |
| 11 | Wgate,Wup → SiLU·mul | `(m,n) ↦ {gate(m,n), up(m,n)}` | `[⌈S/Tm⌉×112]` | **2** | 1 | 0 |
| 12 | SiLU·mul → Wdown | `(m,n) ↦ {(m,nk) : ∀nk<112}` | `[⌈S/Tm⌉]` | 112 | 32 | 0 |
| 13 | Wdown → residual add2 | `(m,n) ↦ (m,n)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 |

**11/13 条边纯仿射，稠密 LLM 中 Tier 3 为零。** ragged 只在边 4–6 这一小段。

**边 7 是粒度杠杆的范例**：`wait = Tm×32 = 4096` 是一堵墙；对 Wo 做 split-K（`Kc` 段）后
`wait = Tm×32/Kc`，墙塌了。**`Kc` 只是关系里的一个参数，改粒度不用重推依赖。**

---

# 4. 仓库结构

```
tilemega/
├── CMakeLists.txt
├── TILEMEGA_SKELETON.md             ← 本文档
├── third_party/
│   ├── cuda-tile/                   submodule，跟 main
│   ├── tensor-ir/                   submodule，跟 main，会打 patch
│   └── barvinok/                    submodule（内含匹配版本的 isl）
├── patches/tensor-ir/               对 tensor-ir 的修改，按功能分文件
├── include/tilemega/
│   ├── Frontend/
│   │   ├── TorchExportImporter.h    ExportedProgram → nv_tensor_ir
│   │   └── SymbolicShapeBridge.h    sympy expr ↔ ISL param
│   ├── Analysis/
│   │   ├── ISLContext.h             isl_ctx 生命周期与错误处理
│   │   ├── CuteLayoutToISL.h        CuTe layout string ↔ isl_map
│   │   ├── AccessRelation.h         P_op / A_op 构造
│   │   ├── DependenceDerivation.h   Dep = P⁻¹ ∘ A
│   │   ├── EventSynthesis.h         Dep → 事件张量 + wait/fanout
│   │   └── Cardinality.h            barvinok 计数封装
│   ├── Partition/
│   │   ├── CandidateGenerator.h     层1 合法性剪枝
│   │   ├── AlignmentPropagation.h   层2 跨算子对齐
│   │   ├── CostModel.h              regime-aware 代价函数
│   │   ├── ChainDP.h                层3 链上 DP
│   │   └── BackendCostQuery.h       ← 原则二的接口
│   ├── Codegen/
│   │   ├── TaskGraphToCudaTile.h    主 lowering
│   │   ├── SyncLowering.h           事件 → atomic + token 链（§6 的规则）
│   │   └── PersistentLoopBuilder.h  grid-stride + stage 展开
│   └── Runtime/
│       ├── Launcher.h               grid 计算、参数打包、变体选择
│       └── EventBuffer.h            事件张量的 device 内存管理
├── lib/                             与 include 对称
├── python/tilemega/
│   ├── compile.py                   torch.compile backend 入口
│   └── serve/                       L5 serving harness
├── test/
│   ├── unit/                        ISL 转换、Dep 推导
│   ├── lit/                         MLIR lit
│   ├── correctness/                 L0/L0.5/L1/L2/L3 差分
│   └── models/                      端到端小模型
├── benchmarks/
└── docs/experiments/                验证报告归档
```

**依赖处理要点**（已实现，见 P0.1）：

- **构建模型**：TileMega 是顶层，`third_party/tensor-ir` 以 `add_subdirectory`
  嵌入（它显式支持这点：`TENSOR_IR_DOWNLOAD_LLVM` 在非顶层时默认 OFF，
  是设计好的扩展点）。tensor-ir 负责拉取并构建 LLVM 与 cuda-tile，
  因此**整个构建树里只有一份 LLVM、一份 cuda-tile**。

- ~~必须在顶层 CMake 覆盖 tensor-ir 的 FetchContent，强制统一到
  `third_party/cuda-tile`~~ —— **推翻**。覆盖是做不到的：tensor-ir 的
  FetchContent 用 `URL + URL_HASH` 强校验，指不到本地目录。而且没必要：
  让 tensor-ir 独占拉取权，天然就只有一份。`third_party/cuda-tile` 这个
  submodule **不参与构建**，它是「钉子 + 可读副本」。

- **风险 R10 靠配置期断言关闭**，不靠约定：`cmake/TileMegaVersionGuard.cmake`
  校验 submodule HEAD == tensor-ir 的 pin，且两者对 LLVM 的钉法一致。
  不符即 `FATAL_ERROR`。

- 依赖的头文件路径**不会自动传播**：tensor-ir / cuda-tile / MLIR 都用
  目录作用域的 `include_directories()`。TileMega 的 target 必须显式引入，
  见 `cmake/TileMegaDeps.cmake`。

- barvinok 是 autotools 工程，用 `scripts/build_barvinok.sh` 单独构建到
  `third_party/.install/barvinok`，**依赖解析走 pkg-config**
  （手写链接行会漏 polylib）。

- 对 tensor-ir 的修改用**独立分支 + patch 文件 + 定期 rebase upstream**，
  不要直接改 submodule 内容而不做版本管理

- `tools/tilemega-opt` 需要**自己注册 `cuda_tile` dialect**——
  `tensor_ir-opt` 虽然链接了 cuda-tile 作为库，但未注册该 dialect。
  **已实测确认**：`tensor_ir-opt` 报 ``Dialect `cuda_tile' not found``，
  `tilemega-opt` 通过。

---

# 5. 分阶段 TODO

> 状态：`[ ]` 待办 `[~]` 进行中 `[x]` 完成 `[!]` 阻塞 `[-]` 已放弃（保留并注明原因）

---

## Phase 0：基础设施

> **状态：已完成（2026-08-28）。** 本节的每一项都已落地并实测通过，
> 结论与踩到的坑一并记录在条目下方。后续开发直接在此基础上继续。

### P0.1 仓库与依赖 `[x]`

- [x] 建仓库骨架，按 §4 结构建目录 —— `/data/tilemega` 即仓库根，
      原 `megakernel_feasibility/` 已移入 `docs/experiments/R1R2/`
- [x] 加三个 submodule（cuda-tile / tensor-ir / barvinok）
- [x] 顶层 CMake 统一 cuda-tile
- [x] CI 骨架：`cmake -S . -B build -G Ninja && ninja -C build`

#### ⚠️ 重要修正：「三个依赖都跟 main」不是一个自洽的组合

原文写「submodule 跟 main」。实测后**推翻**，理由如下（保留原文并注明）：

| 仓库 | 我们钉的版本 | 说明 |
|---|---|---|
| `tensor-ir` | `63692d79`（2026-08-19） | **就是 GitHub main**，且与本地那份同 commit |
| `cuda-tile` | `af241704`（2026-07-22） | = `[Release] CUDA Tile IR 13.3.3` |
| LLVM | `57109bef` | 由上面两者共同要求 |
| `barvinok` | `dd7e6d83`（0.41.9） | 内含 isl `6c2b19a0`（0.28）、polylib `9822337a` |

关键事实：
1. `tensor-ir/cmake/TensorIRDependencyPins.cmake` 明确写着
   *"Update the CUDA Tile and LLVM revisions together. The LLVM revision
   must match the compatibility pin for the selected CUDA Tile revision."*
2. `cuda-tile` main（`be0889c`）**只比 `af241704` 多一个 commit**，
   而那个 commit 是纯粹的 `[LLVM-FIX] Breaking commit 9ebb067a` ——
   **Tile IR 本身零功能差异**，只改了它对哪个 LLVM revision 编译。
3. 因此「cuda-tile 跟 main」的唯一效果，是把 LLVM 拖到 tensor-ir
   未验证过的 `9ebb067a`，收益为零、风险为正。

**决策**：以 tensor-ir 的 pin 集为唯一真相源。`third_party/cuda-tile` 这个
submodule 不参与构建，它是「钉子 + 可读副本」；实际构建用的那份由 tensor-ir
通过 FetchContent 拉取。

#### 复查（V1，2026-08-29）：是否跟进上游 cuda-tile main？—— **不跟进**

有说法称上游 main 新增了 `ScanOp` / `AssertOp` / `GetTensorShapeOp` /
`MmaFScaledOp` / `PermuteOp`。**实际核对后不成立**：

- 上游 main 仍是 `be0889c`（2026-08-27），相对 `af241704` **只多一个
  `[LLVM-FIX]` commit**，Tile IR 零功能差异（与 P0.1 首次核对时结论一致）
- 用 `def CudaTile_*Op` 逐个对比两份 `Ops.td`：
  **main 相对 af241704 新增的 op 数为 0**
- 上述五个 op **在 `af241704` 中已经全部存在**

因此不跟进 main（跟进只会把 LLVM 拖到 tensor-ir 未验证的 `9ebb067a`）。

**`assert` op 已可用**：它就在我们钉的版本里。对调试 megakernel 可能有价值，
但**语义与设备端行为尚未验证**，列入 §7.2 的待做小实验。

#### 风险 R10（两份 cuda-tile 版本不一致）已永久关闭 `[x]`

不是靠约定，是靠 `cmake/TileMegaVersionGuard.cmake` 在**配置期**强制断言：

- 断言 1：`third_party/cuda-tile` 的 HEAD == tensor-ir 声明的 pin
- 断言 2：cuda-tile 自己钉的 LLVM == tensor-ir 声明的 LLVM

任一不符直接 `FATAL_ERROR`。实测两条断言均通过，且两者对 LLVM 的钉法
确实一致（都是 `57109bef`）——这反过来印证了版本选择正确。

> 踩坑记录：这个守卫第一版有两个 bug，都值得记住。
> (a) pin 文件把值写在变量名的**下一行**，`file(STRINGS ... REGEX)` 是
>     逐行匹配的，抽不出来 → 必须 `file(READ)` 整文件读。
> (b) **CMake 的正则不支持 `{n}` 重复语法**，`[0-9a-f]{40}` 永远匹配不上
>     → 改用 `[0-9a-f]+` 再单独校验长度。

### P0.2 工具链 `[x]`

- [x] `tools/tilemega-opt` —— 注册 `nv_tensor_ir` + `cuda_tile` 两个 dialect

  **实测确认了 §4 那条注意事项**。对同一份含 `cuda_tile.module` 的 IR：

  ```
  tensor_ir-opt : error: Dialect `cuda_tile' not found for custom op 'cuda_tile.module'
                  note: Available dialects: arith, builtin, func, nv_tensor_ir
  tilemega-opt  : 通过
                  Available Dialects: arith,builtin,cuda_tile,func,nv_tensor_ir
  ```

  实现注意：**刻意不调 `registerAllDialects()` / `registerAllPasses()`**。
  这个 LLVM revision 里没有 `MLIRRegisterAll*` 库，调它们会拖进整个上游
  MLIR 的链接。只注册我们真正处理的两个 dialect + `registerTransformsPasses()`。

- [x] `scripts/tilemega-compile` —— MLIR → cubin 一键流程

  封装 `cuda-tile-opt` → `cuda-tile-translate` → `tileiras`。
  **每次编译写出 `<out>.cubin.env.json`**，记录 tileiras 版本、三个仓库的
  commit、driver、GPU 型号、时间戳。

  > 这条不是洁癖。R1/R2 的每个实验目录各有一份 run.sh，tileiras 路径和
  > bytecode 版本互不一致，事后无法确定当时用的哪个组合；而 V0 已证明
  > **tileiras 版本会实质改变产物**。结论与工具链版本强绑定，不能追溯
  > 就等于结论作废。

- [x] `tools/tilemega-occupancy` —— 正确的 occupancy 查询（§6.7）

  内建 R1 误诊的对照输出，让「传 blockSize=1 会高估 8 倍」一眼可见。

  > **本工具第一版我自己也写错了**，值得记进来：我图省事用
  > `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` 当作「要求的 blockDim」，
  > 但那是**寄存器允许的上限**（229 regs → 65536/229≈286 → 报 256），
  > 不是 REQNTID 要求的值。两个版本恰好都报 256，差点掩盖了 V0 的核心发现。
  > **§6.7 说的「必须从 `EIATTR_REQNTID` 读」是字面意义上的必须。**

- [x] `BackendCostQuery` 最小实现 —— **推迟到 P4.1**

  理由：`tilemega-compile` 已经能返回 `{legal, REG, SHM}`，而
  `latency` 项依赖 V5(R2-H) 的编译耗时结论才能定形态（同步 or 批量 or 缓存）。
  现在定接口是猜的。

### P0.3 测试基础设施 `[x]`

- [x] **统计化执行器** `scripts/gpu_stat_run.sh`

  内建三条从 R1/R2 提炼的纪律：
  1. 默认 N=50，单次成功不算证据
  2. `timeout --signal=KILL`，挂起记为失败并保留现场（成功的输出丢弃）
  3. **GPU 独占 + 空闲检查** ← 这条是原骨架没写、但必须有的

  > **新增的注意事项**：挂起是对时序敏感的竞态（R1-E5 的核心发现：
  > 预热能「救活」grid=80）。若测试期间机器上还在跑编译或别的 GPU 负载，
  > 测出的挂起率**没有任何意义**。执行器用 `flock` 串行化，并在开跑前检查
  > GPU 利用率与 CPU 负载，不空闲就直接拒绝执行（`--allow-busy` 可强制，
  > 但报告里必须注明）。
  >
  > 这条纪律在本次 Phase 0 就立刻用上了：LLVM 编译占满 14 核期间，
  > 所有 GPU 统计实验都被推迟到编译结束。

- [x] **SASS 结构报告** `scripts/sass_report.sh`

  自动统计 `BAR.SYNC.DEFER_BLOCKING` / `CCTL.IVALL` / CGA 指令数，
  并**自动定位循环回边、报告循环体内的屏障数**——把 R1/R2 里需要人工
  反汇编去数的东西变成一条命令。

- [x] **挂起现场分析** `scripts/hang_probe.sh`

  实现 R2-A 建立的死锁/活锁判据：多次 `cuda-gdb` 采样比对 PC 集合，
  间隔 >20s 且 PC 逐字节不变 ⟹ 死锁。

- [x] **occupancy 查询封装** —— 见 P0.2 的 `tilemega-occupancy`

- [ ] **差分测试框架**（L0/L0.5/L1/L2/L3 逐元素比对）—— 推迟到 P2.1

  理由：现在没有任何两级可比。等 L0 参考实现存在时一起做才有意义。
  容差约定先记下：相对误差 3e-5 量级属正常（浮点求和顺序差异）。

### P0.4 ISL 桥接层地基 `[x]`（原属 P3.1，提前落地）

- [x] `ISLContext` —— isl_ctx 生命周期、错误策略、RAII 封装

  **接口决策：isl C API + 自建 RAII 封装，不用 `isl-noexceptions.h`。**
  理由：(1) barvinok 的基数计数接口只有 C API，混用两套所有权模型会在
  边界上出错，而 isl 的所有权错误表现为静默 use-after-free；
  (2) isl 官方 C++ 绑定的 API 覆盖面随版本变动，升级时会破损。

  错误策略选 `ISL_ON_ERROR_CONTINUE`：P4.2 的合法性剪枝本质上就是
  「构造大量可能不合法的关系再试」，失败是正常控制流，既不该 abort
  也不该往 stderr 刷警告。

- [x] 冒烟测试 `test/unit/Analysis/ISLContextSmokeTest.cpp` —— 全部通过

  已验证四件 Phase 3 会反复依赖的事：
  1. RAII 所有权正确（ctx 析构时 isl 自带的泄漏检查不 abort）
  2. 畸形输入返回空而非杀进程
  3. **`Dep = A ∘ P⁻¹` 推导跑通**，对 §3.6 的边 1 自动算出 `wait = 1` ✓
  4. **barvinok 参数化计数跑通**：`|{[i] : 0<=i<S}|` → `[S] -> { S : S > 0 }`
     ——返回的是 `S` 的符号表达式而非数值，这正是 P3.3 需要的能力

> 踩坑记录（写 CMake 时会再遇到）：
> - isl 的 `*_free` 签名是 `T *(*)(T *)` 而非 `void(*)(T *)`（返回
>   `__isl_null`），RAII 模板参数要按此声明
> - barvinok 是 autotools 工程，不产出 CMake package。**依赖解析必须走
>   pkg-config**：手写链接行第一次就漏了 polylib（报一堆 `Vector_Free`
>   未定义）。`barvinok.pc` 里的权威链接行是
>   `-lbarvinok -lpolylibgmp -lisl -lntl -lgmp`
> - `scripts/build_barvinok.sh` 一次性构建到 `third_party/.install/barvinok`，
>   不进 ninja 构建图（autotools 塞进 ExternalProject 会带来难看的顺序依赖）
> - barvinok 的 polylib submodule 用 `git://` 协议，`https://` 在本环境
>   TLS 握手失败；且它钉的 commit 不是 master tip，`--depth 1` 拉不到

### P0.5 环境基线（本次实测）

| 项 | 值 |
|---|---|
| GPU | RTX 5090，sm_120，170 SM，32GB |
| Driver | 580.65.06 |
| 编译用 CUDA | `/data/cuda-13.3.1`（CUDAToolkit 13.3.73） |
| **tileiras（默认）** | `/data/cuda-13.3.1/bin/tileiras` V13.3.36 |
| tileiras（对照） | `/usr/local/cuda-13.1/bin/tileiras` V13.1.80 |
| PyTorch | 2.12.0+cu130 |
| 主机 | 14 核，117GB RAM；`/data` 119GB 可用（**`/` 只剩 1.8G，一切构建必须落 `/data`**） |

## Phase 1：验证同步机制的真实边界（先于一切功能开发）

> **这一组实验决定 L1 能不能做、以及能做到什么规模。**
> 它们直接影响 Phase 2 之后所有设计，必须最先完成。
>
> **执行纪律（P0.3 内建，违反则数据作废）**：
> - 每格 ≥50 次，用 `scripts/gpu_stat_run.sh`，输出 CSV
> - 机器必须空闲且 GPU 独占；脚本会自己拒绝在有负载时执行
> - 单次超时由脚本管理（默认 6s），**绝不能让外层工具的超时去杀挂起进程**
>   （会把 GPU 留在污染状态，见 §2.4 的事故记录）
> - grid 扫描范围按**当前后端实测的 cooperative 上限**确定，
>   不要沿用 R2 的 170（13.3 是 340）

### V0. 工具链重基线 `[~]` ⭐ 必须最先做

**动机**：R1/R2 的全部证据建立在 tileiras 13.1 上，新基线是 13.3，
跨两个 release，而挂起的根因就在这个闭源后端里。不先做 V0，
后面 V1~V5 全部可能是在为一个已经不存在的问题设计方案。

- [x] 静态对比（同一份 `.tilebc`，只换 tileiras 版本）
      → 见 `docs/experiments/V0_toolchain_rebaseline/result.md`
  - [x] 13.3 向后兼容 13.1 的 bytecode
  - [x] **每 block 线程数 256 → 128，驻留容量 170 → 340**（推翻 §2.2 的常量表述）
  - [x] `CCTL.IVALL` 225 → 28，屏障总数 113 → 85
  - [x] **假设 H2 的对象仍在**：自旋循环体内依然有 `BAR.SYNC.DEFER_BLOCKING`
  - [x] 新工具链端到端编译 R1 时代的 MLIR 成功
- [~] **GPU 统计扫描**：13.1 侧取到 grid=30 挂起 22/50、grid=80 挂起 37/50 后
      因 GPU 被挂死 kernel 污染而中断（见 §2.4 的事故记录）。
      **13.3 侧的数据由 V1 以更严格的方式取得（V1-ctrl，300 次零挂起），
      结论已明确，本项不再单独补跑。**

**分支决策（V0 GPU 扫描出结果后立刻执行）**：

| V0 结果 | 后续 |
|---|---|
| 13.3 下挂起消失 | **风险 R1 解除**。L1 立即解锁；V1~V4 收缩为回归测试；L0.5 从必经之路降级为可选。Phase 2 直接奔 L1 |
| 挂起仍在，但边界随容量平移到 ~340 | 现象与容量强相关的假设被加强，H2 方向继续；V1~V4 按原计划做，grid 范围扩到 340 |
| 挂起仍在且边界未变（仍在 ~170 附近） | 说明与驻留容量无关，指向别的机制；V1（H1）优先级进一步提高 |

### V1. 修正三处 bug 后重测 `[x]` ⭐ 已完成

> **H1 说明**：假设 H1（挂起只发生在自旋后紧跟 reduce）提出时**尚未发现三处测试 bug**。
> V1-a 与 V1-b 的对比会同时检验 H1 和 Bug B，两者可能是同一现象的两种描述。
> 实际结果：**挂起本身在 13.3 上不复现，H1 与 Bug B 都失去了原本要解释的对象**。

矩阵与结果（每格 50 次，`scripts/gpu_stat_run.sh`）：

| 变体 | 内容 | 挂起 | 全槽位校验 |
|---|---|---|---|
| **V1-ctrl** | 原样未改的 `spin_wait_tokenchain.mlir` | **0/50 全 grid（3→340）** | 不适用（输出是竞写标量） |
| **V1-a** | 修 A+C，逐元素消费者，各 block 读不同 chunk | **0/50 全 grid** | **全 grid 通过** |
| **V1-b** | 修 A+B+C，保留 reduce，所有 block 读相同数据 | 0/50 | grid3 0/50 · 30 34/50 · 80 8/50 · 120 3/50 · 170 0/50 · 340 0/50 |
| **V1-min** | 最小复现：写 1 个 tile / 读 1 个 tile | 0/50 | grid≤80 **50/50 通过**；grid≥120 **50/50 失败** |
| **V1-c** | `num_worker_warps_per_cta` 4 vs 8 | `[ ]` 未做 | — |
| **V1-d** | 分散事件：每 block 等自己的 `flag[bx]`，4 生产者 | 0/10 | grid≥128 **10/10 失败** |

**六个必答问题的答案见 `docs/experiments/V1_sync_corrected/SUMMARY.md`。** 要点：

- [x] **挂起不复现**：V1-ctrl 在 13.3 上 300 次零挂起。R1/R2 的核心悲观结论不成立
- [x] **但出现静默数据损坏**：阈值尖锐落在 grid 80–120 之间，缺失整数条 `STG.E` 的份额
- [x] **已隔离到 Tile IR 工具链**：结构等价的 CUDA C++ 版本在所有 grid 全过
- [x] **分散事件不能规避**（V1-d）—— 对项目最不利的一条
- [x] **Bug B 的代价已量化**：修掉它让 cooperative 上限从 340 升到 2040
- [ ] `[!]` **V1-c 未做**；两个触发因素（并发读同址 / 握手数据量）未做正交分离

### V1-后续. 待办

- [ ] **V1-c**：`num_worker_warps_per_cta` = 4 与 8 两档对挂起/损坏的影响
- [ ] **正交实验**：分离「多 block 并发读同一地址」与「握手承载的数据量」两个因素
- [ ] **向 NVIDIA 报 issue**：最小复现（30 行 MLIR）+ 阈值曲线 + CUDA C++ 对照已齐备
- [ ] 静默损坏没有「现场」可采样，`hang_probe.sh` 不适用，需要新的定位方法

### V2. 消费者算子类型扫描

- [ ] 依次测：逐元素 / `reduce_ud` / `mmaf` / transpose，各 grid 各 50 次
- [ ] 目标：**画出「哪些算子可以安全地跟在自旋之后」的白名单**
- [ ] 这个白名单直接成为 Codegen 的约束条件（§6.6）

### V3. 自旋与计算解耦

若 V2 显示某些算子不安全，测试能否通过在自旋与计算之间插入「隔离」来规避：

- [ ] 自旋结束后先做一次无副作用的 dummy 操作再进入 reduce
- [ ] 把 reduce 拆成两个 stage，中间加一次同步
- [ ] 记录哪种隔离有效

### V4. 事件数量与并发规模的关系

- [ ] 固定 grid，扫事件数量（1 / 8 / 64 / 512 个独立事件）
- [ ] 目标：确认挂起与「同时活跃的同步点数量」是否相关

### V5. 补做 R2 未完成项

- [ ] **R2-E**：harness 卫生检查（flag 清零与 kernel 的 stream 同步、
      预热规模扫描 1KB/1MB/100MB、挂起时读回 flag 判断是生产者没写还是消费者没看到）
- [ ] **R2-H**：`(算子, tile)` 的编译 + 资源查询墙钟耗时
      （R2-G 附带观察到 tileiras 编译耗时随分支数明显上升，需正式计时）
      → 决定 P4.1 的剪枝规模
- [ ] **R2-F 剩余**：手动 unroll 对比 + 含 MMA 的 kernel 测 TFLOPS
      → 决定 `T_bubble` 能否建模
- [ ] **R2-G 剩余**：loop-carried 值跨分支、嵌套控制流下 max-not-sum 是否退化
      → 决定编译期变体数量上限

### V6. GPU 恢复流程（新增，运维必需）

V0 的事故暴露了一个缺口：没有从「GPU 被挂死 kernel 污染」状态恢复的标准流程。

- [ ] 确认 `nvidia-smi --gpu-reset` 在本机可用（需权限）
- [ ] 若不可用，找到替代路径（卸载/重载 nvidia 模块？重启？）
- [ ] 把恢复步骤做进 `scripts/gpu_stat_run.sh`：检测到污染特征
      （SM 100% + 无进程 + 显存 0）时给出明确指引而不是干等
- [ ] **CI 的每个 GPU 测试作业前后都要做一次污染检查**

## Phase 2：朴素端到端（正确性优先，不做任何性能优化）

> **本阶段的产出物定义**：一条能把「小 Transformer」编译成可执行 GPU 代码
> 并得到正确数值的完整管线。性能不是目标，**可对比性**才是——
> 后面每一级优化都要拿本阶段的输出做差分基准。

### P2.0 差分测试框架 `[ ]`（从 P0.3 移来，此处才有意义）

原骨架把它放在 P0.3，但那时没有任何两级可比。放在这里，与 L0 参考实现同时落地。

- [ ] `test/correctness/differential.py`：同一模型走 L0/L0.5/L1/L2/L3，逐元素比对
- [ ] 容差：**相对误差 3e-5 量级属正常**（浮点求和顺序不同）；
      比对函数要报告最大相对误差与超差元素位置，不是只给 pass/fail
- [ ] 固定随机种子；输入张量落盘复用，保证跨级用的是同一份输入
- [ ] 每一级都要有编译开关，能单独跑、能两两对比

### P2.1 L0 参考实现 `[ ]`

- [ ] PyTorch eager 逐算子执行，固定随机种子，导出参考输出
- [ ] **同时导出逐层中间张量**，不只是最终输出——
      端到端对不上时，只有最终输出无法定位是哪个算子错了

### P2.2 L0.5 host 端 stage 循环 ⭐ 去风险关键路径 `[ ]`

> 把 stage 边界放回 host，在同一份 codegen 代码路径上验证除同步外的一切。
> Phase 1（V0）结论出来后，把 host 循环换成 in-kernel 循环即升级到 L1。

**明确 L0.5 相对于「直接用 tensor-ir 编译」多了什么**（否则这一级没有意义）：
TaskGraph IR 表示、task 坐标与 partition 元数据、事件张量的分配与布局、
launcher 的参数打包与变体选择。**唯一被推迟的只有 in-kernel 同步。**

- [ ] Codegen：每 stage 一个独立 `cuda_tile.entry`，内含 grid-stride loop
- [ ] Host launcher：按 stage 顺序 launch，stage 间靠 stream 顺序保证依赖
- [ ] 事件张量照常分配并被 kernel 读写（只是 host 保证了顺序），
      **这样 L0.5 → L1 的差异被压到只有「谁来保证 stage 边界」**
- [ ] 跑通最小 Transformer block，与 L0 逐元素比对
- [ ] 跑通完整 Llama 单层
- [ ] 跑通完整小模型（TinyLlama / Llama-3.2-1B），生成正确文本

**若 Phase 1 结论是 L1 长期不可用**：L0.5 就是本项目的可交付执行后端，
Phase 3 的 ISL 事件推导仍然成立（事件在 stage 内仍有意义），
只是跨 stage 重叠拿不到。这一点要在那时明确决策，不要拖着。

### P2.3 算子覆盖 `[ ]`

按依赖顺序推进，每个算子的完成定义 = **lit 测试 + 与 L0 的数值比对**都通过：

1. [ ] pointwise（add / mul / SiLU）—— tensor-ir 已有，先验证管线
2. [ ] Linear / GEMM（`mmaf`）—— 最重要，后面一切都依赖它
3. [ ] RMSNorm（`reduce_ud`，参考 tensor-ir 文档的 Welford 例子）
4. [ ] RoPE（pointwise + 索引变换，可能需要 dialect 扩展）
5. [ ] Attention（非 paged）：matmul + softmax + matmul
      - `[?]` 待确认：`reduce_ud` 能否表达 online softmax 的 running-max 更新
      - **这是 §7.2 里应当最早做的小实验**，它决定 attention 走分解还是扩 dialect
6. [ ] Embedding lookup（gather，tensor-ir 无此算子，需扩展或 opaque task）

### P2.4 L1 单 entry megakernel `[!]`

> **阻塞于 Phase 1**：需要一个可靠的 grid 大小和安全算子白名单。

- [ ] `[!]` 全局 barrier lowering：按 §6 的规则
- [ ] stage 展开 + 层循环（§3.4 的形态）
- [ ] task_type dispatch：`if/else` 链
- [ ] 与 L0.5 逐元素比对，≥50 次统计
- [ ] 记录 grid 大小、REG、SHM、挂起率

## Phase 3：ISL 形式化（自动事件推导）

### P3.1 ISL 桥接层

- [ ] `ISLContext`：isl_ctx 生命周期、错误处理、内存管理
- [ ] **CuTe layout string → isl_map**
  - 输入形如 `"(?,?):(?,1)"` + `dynamicValueMapping`
  - 平坦 layout：`offset + Σ i_k · stride_k`，直接仿射
  - **嵌套/层次 layout 需先 flatten**
  - `[!]` 早做小实验：确认 tensor-ir 实际产生哪些 layout 形式，
        嵌套的占多少，平坦化后是否保持仿射
- [ ] 单元测试：一组 CuTe layout 的转换与语义等价性

### P3.2 访问关系构造

- [ ] `P_op`：task 坐标 → 输出元素区域
      - **可直接对应 `cuda_tile.partition_view`**
- [ ] `A_op`：task 坐标 → 输入元素区域，逐算子类别实现
      （pointwise / reduction / matmul / broadcast / concat / slice / transpose）
      - 复用 tensor-ir 的 `ReductionSourceAttr` / `MatmulSourceAttr`

### P3.3 依赖推导

- [ ] `Dep = P_prod⁻¹ ∘ A_cons`（`isl_map_apply_range` + `isl_map_reverse`）
- [ ] 事件合成：形状 = `Dep` 的 image；wait = 单事件 preimage 的势（barvinok）；
      fan-out = 单事件 image 的势
- [ ] **验收：§3.6 的 13 条边全部自动推出，wait / fan-out 逐项一致**

### P3.4 Tier 分类与保守化

- [ ] Tier 0：直接推
- [ ] Tier 1：布局抵消。识别「生产者与消费者共享同一单射布局函数」
      - `[!]` 待确认：prefix caching + CoW 下 block 共享是否破坏单射性
        （共享 block 只读 → 只产生 read-read，推测成立但未严格证明）
- [ ] Tier 2：indptr 参数化。索引映射保持符号闭式，extent 与 wait 标记为运行时值
- [ ] Tier 3：保守化。优先分组 barrier（per-sequence / per-expert），其次算子级 barrier
- [ ] 保守化正确性检查：`Dep' ⊇ Dep` ⟹ 执行仍正确（逐边可组合）

### P3.5 L2 落地

- [ ] 事件张量 → device int 数组（§3.2 的放置规则）
- [ ] `notify()` → `atomic_rmw_tko` add/release device
- [ ] `wait()` → §6 的自旋封装
- [ ] 单调计数器：`needed = num_triggers × iteration_num`
- [ ] 与 L1 逐元素比对 + ≥50 次统计

---

## Phase 4：划分与粒度求解

### P4.1 代价查询接口

- [ ] `BackendCostQuery`：`(算子, tile形状) → {legal, REG, SHM, latency}`
- [ ] 缓存层 / 批量查询（依赖 V5 的 R2-H 耗时结论）
- [ ] 兜底：解析估算 REG/SHM（从 tile 形状直接算），只对 top-k 做实际编译

### P4.2 层1 合法性剪枝

- [ ] 从硬件原生 MMA 形状出发沿各维扩张，撞资源墙停
      （shared memory / 寄存器 / TMA 128B 对齐 / **2 的幂约束**）
- [ ] 目标：每算子 8~20 候选
- [ ] 自检：候选集应覆盖 MPK demo 用的值（64/96/256）；不覆盖说明剪枝有 bug

### P4.3 层2 对齐传播

- [ ] 从输出端反向传播 tile 约束
- [ ] 计算不对齐的 wait 膨胀：`⌈(m+1)Tm/Tr⌉ − ⌊mTm/Tr⌋`
- [ ] 度量剪枝效果：联合空间塌缩比例

### P4.4 代价模型

- [ ] roofline：`max(T_compute, T_memory)`，解析
- [ ] `T_quant`：`N(g)` 用 barvinok
- [ ] `T_sync`：事件数 × 原子延迟 + 自旋开销
- [ ] `T_bubble`：依赖 V5 的软件流水结论
- [ ] **离线标定**（测硬件常数，不是搜索配置）：
      - 单 task 时长 vs tile 形状（拟合 roofline 参数）
      - atomic notify/wait 的延迟与争用曲线
      - **并发干扰系数**：GEMM task 与 memory-bound task 并发时的时长偏差
        - `[!]` 若偏差 >30%，独立时长假设失效，退化为「粗排 + 实测 top-3」
- [ ] regime 判别：从 (batch, seq_len, prefill/decode 比例) 判 A/B/C

### P4.5 层3 链上 DP

- [ ] `DP[i][g] = min_{g'} { DP[i−1][g'] + Cost_i(g) + Interface(g',g) }`
- [ ] 分叉处（gate/up 并行）：series-parallel 分解，或保守强制同 tile
- [ ] 复杂度目标 `O(层数 × |C|²)`，秒级

### P4.6 事件粒度 κ

- [ ] 实现对 Dep image 的粗投影：`E[m] → E[⌊m/κ⌋]`
- [ ] `[!]` 待验证：ISL 对含参数化整除的映射是否表达式爆炸
      （quasi-affine 支持 floor division，但配合参数化 κ 未知）。
      兜底：限制 κ 为 2 的幂，或用受限的矩形代数代替通用 ISL
- [ ] κ 消融曲线：固定其他一切只扫 κ。**曲线平坦则该维度无收益，及早放弃**
- [ ] 把 κ 纳入 DP 状态

### P4.7 排布

- [ ] 分层 DAG 上的 list scheduling（关键路径优先）
- [ ] 掩盖同步延迟：队列顺序让等待被独立 task 填充
- [ ] 时间局部性：`|A(c₁) ∩ A(c₂)|` 用 barvinok 算
- [ ] **先做 oracle 判断值不值得投入**：固定划分和事件结构，
      对比 round-robin vs 强启发式 vs oracle。差距 <2% 则跳过本节

---

## Phase 5：符号化与运行时

### P5.1 参数化解

- [ ] 代价函数以符号 shape 为参数 → DP 输出分段拟多项式
- [ ] 求分段交点 → 最优解的区间划分
- [ ] 每区间生成一个 task 实现变体，全部内联进同一 entry
      （受 V5 的 max-not-sum 推广边界约束）

### P5.2 运行时选择

- [ ] Host launcher：代入实际 shape → `O(1)` 区间查表 → 选分支
- [ ] 或 kernel 内分支：读运行时参数直接跳转
- [ ] 变体数量上限控制

### P5.3 Tier 2 运行时支持

- [ ] indptr 前缀和（kernel 内计算或 host 预计算）
- [ ] 事件张量 extent 的运行时确定
- [ ] wait count 的运行时填充

### P5.4 尾 wave

- [ ] Stream-K 式归约维切分：`Dep` 从 `m ↦ {m}` 变 `m ↦ {(m,0..K)}`，
      **同一关系换参数，不用重推依赖**
- [ ] `[-]` ~~运行时 task 拆分~~ — 已放弃，因 tile 形状必须编译期静态

---

## Phase 6：Serving 集成与评估

### P6.1 L5 Serving harness

按 §3.3 的对照表实现：

- [ ] Paged KV cache（Tier 1）
- [ ] Continuous batching（kernel 内的调度 task，参考 MPK）
- [ ] Chunked prefill（Tier 2，indptr 参数化）
- [ ] MoE 路由（Tier 3，把 MPK 的 metadata 特例一般化成统一 indptr 机制）

### P6.2 评估

- [ ] **bucketing 损失曲线**：扫 batch = 1..128，对比「逐 shape 最优」vs
      「power-of-two 向上取整复用」。不依赖 cost model 精度，纯结构性损失
- [ ] **划分 oracle**：固定排布和事件结构，穷举划分，看最优 vs 启发式魔数
      - **这是核心贡献的生死线**。可先在 MPK 上做（改一个 Python 参数即可），
        成本低一个量级
- [ ] 事件粒度 κ 消融
- [ ] 混合 batch regime vs vLLM 的 `FULL_AND_PIECEWISE` 模式
- [ ] Warmup 时间（目标：0 次 CUDA graph capture）
- [ ] 端到端 vs vLLM / SGLang / MPK

### P6.3 消融

- [ ] L1 → L2（细粒度事件的贡献）
- [ ] L2 → L3（划分优化的贡献）← **净贡献**
- [ ] 符号化的贡献（vs bucketing）

---

# 6. Codegen 规则

> 这些是从实测中提炼的硬性规则，**所有生成同步代码的路径必须走统一封装**，
> 不允许在多处手写。
>
> **本节规则的存在本身就是一个发现**：我们在 R1/R2 中同时踩了 §6.8 / §6.9 / §6.10
> 三条，全程编译期零警告，测试还长期「通过」（E4 的 1000/1000）。
> 这是「在 Tile IR 上手写 megakernel 同步逻辑不安全、需要 correct-by-construction
> 代码生成层」的实证依据。

## 6.1 自旋等待必须串 token 链

**不串 token 则整个循环被静默消除**（§2.3）。正确形式：

```mlir
// 阶段一：轮询（relaxed，不触发缓存失效）
%loop_tok = loop iter_values(%tok = %init) : token -> token {
    %val, %ltok = load_ptr_tko relaxed device %flag token=%tok : ...
    %ready = cmp uge %val, %needed
    if %ready { break %ltok }
    continue %ltok
}
// 阶段二：跳出循环后做一次 acquire，建立 happens-before
%final, %atok = load_ptr_tko acquire device %flag token=%loop_tok : ...
// 之后才能读数据
```

**注意**：R2-B 实测证明这个写法**不能**修复大 grid 挂起（那是另一个问题），
但它是正确性和性能上都更好的写法：SASS 确认 `CCTL.IVALL` 被移出循环，
正确性 1000/1000（1MB）+ 200/200（4MB）通过。**照此写就对。**

## 6.2 自旋循环内含硬件屏障，这是模型固有的

一个 tile block = 一个逻辑线程 = 一组物理线程（实测 256）。自旋循环体内部会生成
`BAR.SYNC.DEFER_BLOCKING`，每次轮询迭代要求全部线程会合。

**这意味着**：
- Tile IR **做不到**「一个线程轮询、其余线程等待」
- 轮询频率不只是内存压力问题，还是屏障会合频率问题
- **减少轮询迭代次数比减少每次迭代的内存流量更重要** → 退避有意义，但作用有限

## 6.3 退避

- [ ] 生成的自旋循环**默认带退避**（LCG 非线性延迟循环，编译器无法消除）
- 实测：grid=30 时挂起率随退避档位单调下降（backoff64 25/30 → backoff1024 16/30）
- 但 grid=170 时无效，**不要指望退避能解决可扩展性问题**
- 档位应可配置，默认值待 Phase 1 确定

## 6.4 单调事件计数器

```
needed = num_triggers × iteration_num
```

计数器从不重置，`TaskId` 编码 `iter:32 | idx:32`。避免跨迭代 ABA，
免掉迭代边界的全局清零。借鉴自 MPK。

## 6.5 事件缓冲布局

- 计数器按 128B cache line padding，避免 false sharing
- 尽量保持数组小以维持 L2 常驻；事件数大时按 stage 分段复用

## 6.6 自旋之后的算子受白名单约束

`[!]` 白名单待 Phase 1 的 V2 确定。已知风险：**归约类算子紧跟在自旋之后可能挂起**
（R2-A 定位的卡死点正在归约收尾代码里）。

Codegen 必须在生成「wait → 计算」序列时检查算子类型，
不在白名单内的要插入隔离（形式待 V3 确定）。

## 6.7 occupancy 查询

`cuOccupancyMaxActiveBlocksPerMultiprocessor` 的 `blockSize` **必须从 cubin 的
`EIATTR_REQNTID` 读取**。传 1 会把资源占用低估数百倍，得到完全错误的 grid 上限。

参考：REG=228、blockDim=256 时，`65536/(228×256) = 1.12 → 1 block/SM`。

## 6.8 release 必须 token-order 在它发布的所有写之后

规范 §7.5：程序顺序、控制流依赖、数据依赖、地址依赖**都不提供排序**，必须用 token，
且原文强调 *"even where the token ordering appears redundant with program dependencies"*。
循环内的写必须用 `iter_values` 把 token 携带出循环，再传给 release 的 `token=` 参数：

```mlir
%init = make_token : token
%chain = for %c in (%c0 to %n, step %c1) : tile<i32>
             iter_values(%tok = %init) -> (token) {
    %t = store_view_tko weak %val, %pv[%c] token=%tok : ... -> token
    continue %t : token                        // ← 必须携带出去
}
%old, %at = atomic_rmw_tko release device %flag, xchg, %one token=%chain : ...
//                                                       ^^^^^^^^^^^^^ 关键
```

违反后果：release 被重排到数据写之前，消费者读到脏数据。**编译期零警告，
且可能长期侥幸通过测试**（E4 的 1000/1000 就是这样）。

✅ V1 已验证本规则生效：修正后的 SASS 中 `MEMBAR.ALL.GPU` 正确出现在
`ATOMG.E.EXCH.STRONG.GPU` 之前；而删掉 `token=` 则不会。

## 6.9 acquire / release 只用在标量同步变量上

数据的批量 load/store **不要在数据 tile 上加 acquire**，靠 `token=` 排在那一次
acquire 之后即可。

规范 §7.1：tile 操作按元素展开成内存模型操作，一个 `tile<1024>` 的 acquire =
1024 个 acquire 操作，每个在 SASS 里对应一条 `CCTL.IVALL`。
规范 §7.12.3：acquire 之后的内存事件靠 **token 排序**放在其后，不是各自再 acquire 一遍。

✅ V1 量化了违反的代价（同一个 kernel，只改这一处）：

| | 违反（Bug B） | 遵守 |
|---|---|---|
| SASS 指令数 | 1440 | **256** |
| `CCTL.IVALL` | 29 | **1** |
| REG/thread | 229 | **24** |
| **cooperative 上限** | 340 | **2040** |

⚠️ **但数据的批量访问不能用 `weak`**：§7.2 原文 *"weak ops **cannot be used to
communicate through memory between threads**"*。跨 block 传递的数据要用
`relaxed device`（允许并发访问、但不自建 happens-before，HB 由那一次 release/acquire
提供），它同样**不**产生 `CCTL.IVALL`。

## 6.10 禁止多个 tile block 用 weak 写同一位置

并发写必须用 `atomic_*`，或按 tile block id 分槽（`out[bx]`）。

规范 §7.2：`weak` 允许编译器假设该位置无并发访问。§7.10：data race 的程序是 UB。

> 测试侧的推论：**校验必须覆盖每一个 block 的输出槽位。** R1/R2 只校验一个被竞写的
> 标量，导致算错的 block 完全不被发现。V1 的 harness 逐槽校验 + 输入预填 poison，
> 才暴露出静默数据损坏。

## 6.11 `[!]` 大 grid 下跨 block 数据可见性存在后端缺陷

⚠️ **V1 发现，尚未有 workaround。** 在 tileiras 13.3 / sm_120 上，遵守 §6.8–6.10 的
正确 release/acquire 握手，在 **grid ≥ 120** 时会静默丢失整粒度的数据
（每次 128 或 256 个元素 = 整数条 `STG.E`）。结构等价的 CUDA C++ 版本无此问题。

**Codegen 层当前无法绕过。** 在缺陷修复前，L2（细粒度事件）不可信，
L0.5（host 端 stage 边界）是唯一可信的执行路径。详见 §2.4 与
`docs/experiments/V1_sync_corrected/SUMMARY.md`。

---

# 7. 风险与未决问题

## 7.1 风险登记册

| # | 风险 | 严重度 | 状态 | 应对 |
|---|---|---|---|---|
| ~~R1~~ | ~~**大 grid 挂起未解**（§2.4）~~ | ~~高~~ | **✅ 已关闭（V1）**：三处测试 bug 已定位并证实；tileiras 13.3 上 300 次零挂起，现象不复现 | 无需应对 |
| **R1'** | **大 grid 下跨 block 数据静默损坏**（§2.4、§6.11） | **高** | **V1 已实证**：正确的 release/acquire 握手在 grid≥120 时丢失整数条 `STG.E` 份额；CUDA C++ 对照组全过，缺陷已隔离到 Tile IR 工具链；per-tile 事件模式（V1-d）不能规避 | 向 NVIDIA 报 issue（最小复现已备）；在修复前 L2 不可信，L0.5 是唯一可信路径 |
| R2 | tileiras 是否做跨迭代软件流水未定；无 workaround | 高 | 观察到批量预取，未定点验证 | V5（R2-F 剩余） |
| R3 | 划分优化的收益可能很小 | 高 | 待 P6.2 oracle | 尽早在 MPK 上做 oracle 实验 |
| R4 | 并发干扰使代价模型不可靠 | 中高 | 待 P4.4 标定 | 偏差 >30% 则退化为「粗排 + 实测 top-3」 |
| R5 | max-not-sum 在 loop-carried 跨分支下退化 | 中高 | 已验证到 N=8 分支；嵌套/携带值未测 | V5（R2-G 剩余） |
| R6 | CuTe layout ↔ ISL 转换精度损失 | 中 | 待 P3.1 | 早做小实验；必要时限制支持的 layout 形式 |
| R7 | ISL 对参数化 κ 粗投影表达式爆炸 | 中 | 待 P4.6 | 限制 κ 为 2 的幂；或用受限矩形代数 |
| R8 | 代价查询过慢限制剪枝规模 | 中 | 待 V5（R2-H） | 缓存 / 批量 / 解析估算兜底 |
| R9 | 算子覆盖不足（无 attention/gather） | 中 | 待 P2.3 | 分解 → dialect 扩展 → opaque task 三级 |
| ~~R10~~ | ~~两份 cuda-tile 版本不一致~~ | ~~低~~ | **✅ 已关闭（P0.1）** | 构建树里只有一份（tensor-ir 独占拉取）+ 配置期断言校验 pin 一致 |
| R11 | sm_120 平台特有问题 | 低 | 后期确认 | 上层逻辑与硬件无关，后期在数据中心卡交叉验证 |
| **R12** | **后端行为随 tileiras 版本实质变化** | **中高** | **V0 已实证**：同一 `.tilebc` 在 13.1/13.3 下 REQNTID 为 256/128，驻留容量 170/340，`CCTL.IVALL` 差 8 倍 | tileiras 版本作为显式参数；每个 cubin 附 `.env.json` 环境快照；任何跨版本的结论都要重测，不得沿用 |
| **R13** | **挂死的 kernel 污染 GPU，杀进程不回收** | **中** | **V0 遇到并被空闲检查挡住**（SM 100% / 无进程 / 显存 0） | `gpu_stat_run.sh` 的空闲检查 + 独占锁；短超时由测试脚本自管；V6 建立恢复流程 |
| **R14** | **`num_worker_warps_per_cta` 与后端版本会改变 tile block 线程数**，进而改变 occupancy 与 grid 上限 | **中** | 该 hint 仅支持 [4, 8]；13.1/13.3 的 REQNTID 为 256/128；V1 另测得同一后端下仅改内存序写法即让 blocks/SM 从 2 变 12 | `tilemega-occupancy` 已从 cubin 读 `EIATTR_REQNTID`（V1 期间修正）；**所有涉及 grid 规模的结论必须绑定具体工具链版本**，每次重基线复查 |
| **R15** | **手写 Tile IR 同步逻辑不安全** | **中高** | R1/R2 同时踩了 §6.8/6.9/6.10 三条，编译期零警告且测试长期「通过」 | §6 的规则必须由统一的 codegen 封装强制，不允许手写；这是 correct-by-construction 代码生成层的实证依据 |

## 7.2 需要小实验确认

- [ ] **CuTe layout → ISL 的表达能力覆盖**：tensor-ir 实际产生哪些 layout 形式？
      嵌套的占多少？平坦化后是否保持仿射？
- [ ] **ISL 参数化整除的表达式规模**：`E[⌊m/κ⌋]` 在 κ 为参数时的 `isl_map` 规模增长
- [ ] **torch.export 对目标模型的覆盖**：Llama / Qwen 能否干净导出？
      KV cache 以什么形式出现（mutation? buffer?）
- [ ] **`reduce_ud` 能否表达 online softmax**（running-max 更新）
- [ ] **paged KV 布局抵消在 prefix caching 下是否严格成立**
- [ ] **多 entry 的 module 能否共享 `cuda_tile.global`**：若能，可规避部分「一切内联」的痛苦
- [ ] **`cuda_tile.assert` 的设备端语义**：触发时的行为（终止？写错误码？）、
      是否可在 release 构建保留、对寄存器/性能的影响。
      已确认该 op 存在于我们钉的 `af241704`，但未验证行为。
      megakernel 无设备端 printf 保证，assert 若可用则是主要的调试手段
- [x] ~~**cuda-tile 与 tensor-ir 的版本兼容矩阵**~~ —— P0.1 已查清：
      tensor-ir main 钉 cuda-tile `af241704` + LLVM `57109bef`，
      cuda-tile main 仅多一个 LLVM 兼容 commit，Tile IR 零功能差异
- [ ] **`cuda_tile.global` vs device buffer 的 codegen 差异**：
      前者是否因常量 bank 寻址更快？影响 §3.2 的事件张量放置决策

## 7.3 设计决策待定

- [ ] Attention 走分解还是 dialect 扩展
- [ ] 自旋之后的安全算子白名单（Phase 1 V2 输出）
- [ ] 退避默认档位（Phase 1 输出）
- [ ] 是否需要利用新版 cuda-tile 的 `AllocaOp`（可能对 scratch buffer 有用）

---

# 附录 A. 可借鉴实现速查

## A.1 MPK（`mirage/include/mirage/persistent_kernel/`）

| 机制 | 位置 | 说明 |
|---|---|---|
| **同步原语 PTX 参照表** | `mpk_atoms.cuh`（97 行） | `atom.add.release.gpu.s32/.u64`、`atom.cas.release.gpu.b64`、`ld.acquire.gpu.u64`、`ld.acquire.sys.b32`/`st.release.sys.b32`（CPU↔GPU ring）。**哪个操作用哪个 scope/ordering 的权威参照** |
| **单调事件计数器** | `persistent_kernel.cuh` worker loop | 见 §6.4 |
| **自旋 + 退避** | 同上 | `while (actual < needed) { actual = ld_acquire_sys_u64(...); __nanosleep(10); }` |
| **共存性保证** | `__launch_bounds__(WORKER_NUM_THREADS, 1)` | 第二参数 = min-blocks-per-SM，配合 grid=SM数。**Tile IR 缺此旋钮**，只能靠资源占用自然压到 occupancy=1 |
| **角色分化** | `if (blockIdx.x < num_workers) execute_worker(); else execute_scheduler();` | block 级，Tile IR 可照搬 |
| **紧凑事件表示** | `EventDesc{event_type, num_triggers, first_task_id, last_task_id}` | fan-out 用区间编码，不存列表（需要 linearization 保证下游 task 索引连续） |
| **task descriptor** | `TaskDesc` alignas(16)，8 输入 / 3 输出 | normalization 保证 trigger/dependent event 各只有 1 个 |
| **Tier2/3 逃生舱** | `TaskMetadata` union（8 字节） | `{expert_offset}` / `{request_id, kv_idx, merge_task_offset}` / `{task_offset}`。**我们要把它一般化成 indptr 参数化** |
| **serving 集成** | `TASK_SCHD_PREPARE_BATCH` 等 | 请求准入/剔除/KV 元数据更新放进 kernel 内的 task |
| 资源预算参考 | `runtime_header.h` | 保留静态 SHM 6KB(GB/GH)/3KB；MAX 动态 SHM 207KB(B200)/163KB(A100) |

**注意**：MPK 的 task 实现是约 39.4K 行按架构手写的 kernel 库
（`tasks/{ampere,hopper,blackwell}/`），**不是 Mirage 超优化生成的**。
划分是用户参数 `grid_dim`，demo 里用硬编码魔数
（`demo/qwen3/demo.py:38-68` 的 `grid_for_rmsnorm_linear_layer`，含
`# An add-hoc workaround` 之类注释）。**这两处不要照抄，是我们要替代的部分。**

## A.2 ETC / Event Tensor

| 机制 | 说明 |
|---|---|
| **Event Tensor 的极简 lowering** | 事件张量 = 一个 int 张量；`notify()` = atomic dec；`wait()` = 自旋。**runtime 需求几乎为零**（对比 MPK 的 84K 行 CUDA） |
| **调度即 pass** | static（主机侧预计算 SM 队列）/ dynamic（GPU 中心化就绪队列）两个可替换的 pass |
| **数据相关事件更新** | expert 计数器初值由 runtime `topk` 决定 |
| **数据相关 task 触发** | `exp_indptr` 前缀和决定激活 `[indptr[i], indptr[i+1])` 区间的 task |
| **split-K 事件范例** | `task B̂_{i,j} → E[i]`（wait=4）`→ task Ĉ_i`，第二阶段只等自己那行 |

## A.3 其他

| 来源 | 借鉴点 |
|---|---|
| Welder (OSDI'23) | 从输出端反向传播 tile 约束；跨内存层级流量代价模型 → 我们换成「流量 + 同步 + 均衡」联合代价 |
| BladeDISC | Shape Constraint IR 的两类约束：dimension constraint 与 structure constraint（`d0==d1`、`d0*d1==d2`）。融合决策依赖形状关系而非具体值 |
| Roller (OSDI'22) | **构造而非搜索**：从硬件原生 rTile 出发沿维扩张至资源墙。P4.2 的直接参考 |
| Rammer (OSDI'20) | rTask/rProgram 的编译期 task→执行单元映射 |
| vLLM | `FULL_AND_PIECEWISE` CUDA graph 模式，chunked prefill 的默认参数（chunk=2048） |

---

# 附录 B. 源码位置索引

## B.1 cuda-tile

| 内容 | 位置 |
|---|---|
| 全部 op 定义 | `include/cuda_tile/Dialect/CudaTile/IR/Ops.td` |
| `TileType`（静态形状 + 2 的幂） | `Types.td` |
| `TensorViewType`（动态 shape/stride）/ `PartitionViewType` | `Types.td`；规范 §6.3.3 |
| `MemoryScope`（TL_BLK/DEVICE/SYS）/ `MemoryOrderingSemantics` | `AttrDefs.td` |
| `OptimizationHintsAttr` | `AttrDefs.td`（含未文档化的 `occupancy` 常量） |
| `EntryOp` / `LoopOp` / `AllocaOp` | `Ops.td` |
| OSS 侧 pass（只有 3 个） | `Transforms/Passes.td` |
| 官方规范 | https://docs.nvidia.com/cuda/tile-ir/latest/ |
| 关键章节 | §6.2.3 tile function disabled；§6.3.3 partition_view；§6.8.2 forward progress；§6.8.3 token ordering |

## B.2 tensor-ir

| 内容 | 位置 |
|---|---|
| 算子定义（63 个） | `include/tensor_ir/Dialect/TensorOps.td` |
| 布局属性（`TensorSourceAttr` 等） | `include/tensor_ir/Dialect/TensorAttrs.td` |
| Pass 流水线 | `lib/Compiler/CudaTile/Pipelines.cpp` |
| **tile 分析要求静态 layout** | `lib/Transform/TileAnalyzerPass.cpp:123` |
| **动态维 tile 硬编码 128** | `lib/Transform/TileAnalyzerPass.cpp:38-44` |
| **persistent grid 的 TODO** | `include/tensor_ir/Runtime/CudaTile/KernelLaunchHelpers.h:350` |
| tile 候选生成 | `lib/Analysis/TileCandidateGenerator.cpp` |
| 运行时 grid 计算 | `lib/Analysis/KernelArgLayout.cpp` |
| persistent 策略 | `lib/.../AffineMapImpl.cpp:833`，`lib/Analysis/TileAnalyzer.cpp:478` |

> 行号基于 R1 调研时的版本，跟 main 后需重新定位。

## B.3 mirage / MPK

| 内容 | 位置 |
|---|---|
| 同步原语 | `include/mirage/persistent_kernel/mpk_atoms.cuh` |
| 主 runtime（worker/scheduler 循环） | `include/mirage/persistent_kernel/persistent_kernel.cuh` |
| 数据结构 | `include/mirage/persistent_kernel/runtime_header.h` |
| 手写 task 库 | `include/mirage/persistent_kernel/tasks/{ampere,hopper,blackwell,...}` |
| 划分魔数 | `demo/qwen3/demo.py:38-68` |
| Python 前端 | `python/mirage/mpk/persistent_kernel.py` |

---

# 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-08-29 | v0.4 | **V1 重测**：定位并修正 R1/R2 测试源码的三处内存模型违反；证实 tileiras 13.3 上大 grid 挂起**不复现**（300 次零挂起），R1 关闭；但发现**新的静默数据损坏**（grid≥120，阈值尖锐），并用 CUDA C++ 对照组把缺陷**隔离到 Tile IR 工具链**；V1-d 证伪「分散事件可规避」的寄望。§2.4 整节降级重写并保留原文；§6 新增 6.8–6.11 四条规则与总纲；Phase 1 的 V1 替换为五变体矩阵；风险册 R1 关闭、新增 R1'/R14/R15；核对上游 cuda-tile main **无新增 op**，决定不跟进；新增 `docs/experiments/INDEX.md` |
| 2026-08-28 | v0.3 | **Phase 0 完成**（仓库/构建/工具链/测试基础设施/ISL 地基全部落地并实测）。**新增 V0 工具链重基线**并把结论回写进 §2.2（每 block 线程数不是常量）、§2.4（H2 在 13.3 下仍未解决）。细化 Phase 1（插入 V0/V6 与分支决策表）与 Phase 2（新增 P2.0 差分框架、明确 L0.5 的产出定义与算子推进顺序）。§4 依赖处理要点按实际落地形态重写。风险册：R10 关闭，新增 R12/R13 |
| 2026-08 | v0.2 | 重写为实现导向；整合 R2 验证结果；新增 §6 Codegen 规则、Phase 1 同步边界验证 |
| 2026-08 | v0.1 | 初版 |