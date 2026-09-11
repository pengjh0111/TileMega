# TileMega 开发骨架

> **定位**：把 megakernel 的任务划分、事件生成、粒度选择与通信归属，
> 表达为**参数化耦合图（Coupling Graph）**上的可解析优化问题。
>
> **后端**：CuTe / CUTLASS + nvcc。
>
> **配套文档**：`docs/VERIFICATION_PLAN.md`（开工前验证计划，独立维护）。
>
> **维护约定**：完成的条目打勾并追加实测结论；推翻的假设保留原文并注明推翻原因。
>
> **文档地图（v2.1）**：设计与契约 → `TileMega_skeleton.md`；实现状态 → `docs/STATUS.md`；待办 → `docs/TODO.md`；实测发现 → `docs/FINDINGS.md`；v2.0 待办原文 → `docs/archive/TODO_v2.0.md`；开工前验证计划 → `docs/VERIFICATION_PLAN.md`。每类信息只有一个权威位置，其他位置只放指针。
>
> **维护约定（v2.1 补充）**：上一条约定自 v2.1 起适用于 `docs/TODO.md`。本文件只写设计与契约：设计变更在文末变更记录登记；推翻的设计假设保留原文并注明推翻原因与证据；实现状态、打勾与新的实测数字不再写入本文件（历史段落中已有的保留原样）。

---

## 目录

- [1. 设计理念](#1-设计理念)
- [1.5 当前状态](#15-当前状态)（⚠️ v2.1：已迁至 docs/STATUS.md）
- [2. 核心抽象：Coupling Graph](#2-核心抽象coupling-graph)
- [3. 构建基础](#3-构建基础)
- [4. 架构](#4-架构)
- [5. Lowering 路径](#5-lowering-路径)
  - [5.7 执行模型与 Plan 契约](#57-执行模型与-plan-契约)
- [6. 仓库结构](#6-仓库结构)
- [7. 分阶段 TODO](#7-分阶段-todo)（⚠️ v2.1：已迁至 docs/TODO.md）
- [8. Codegen 规则](#8-codegen-规则)
- [9. 风险与未决问题](#9-风险与未决问题)
- [附录 A. 可借鉴实现速查](#附录-a-可借鉴实现速查)
- [附录 B. 源码位置索引](#附录-b-源码位置索引)

---

# 1. 设计理念

## 1.1 问题

Megakernel 把整个模型放进一个持久驻留的 kernel，用 kernel 内事件同步代替 kernel 边界，
消除 launch 开销并让跨算子的 tile 级重叠成为可能。

现有实现把「任务怎么划分、事件粒度多大」当作系统的**输入或配置**：
MPK 的每个 layer API 要求用户传 `grid_dim`（其 demo 中为硬编码常数）；
ETC 要求划分与事件图由用户以 kernel DSL 或 compiler builtin 提供。

根本原因是它们的依赖表示是**枚举出来的边集合**——任务图一旦构建就是具体的、
与形状和粒度绑定的数据结构，无法承载代价建模，也无法在粒度变化时复用。

TileMega 的出发点：**把依赖表示成闭式的参数化关系**。于是 task 数、事件数、
同步开销、wave quantization、通信量全部成为 `(符号形状, 粒度)` 的闭式函数，
划分与粒度从「靠经验设定」变成「可构造求解」。

## 1.2 四条原则

### 原则一：分析层只决定 task 的边界与关系，不进入 task 内部

分析层（CuTe 代数 + ISL/barvinok）负责访问关系、依赖关系、参数化整数集、
基数计数、代价建模。task 内部的 MMA 形状、TMA、warp specialization、
smem swizzle、软件流水全部由 CUTLASS collective 承担。

### 原则二：求解器与后端之间必须有代价查询通道

划分决策必须拿到 task 内实现的真实代价。CUTLASS 提供两级查询：

| 级别 | 方式 | 成本 |
|---|---|---|
| **编译期 traits** | `sizeof(Collective::SharedStorage)`、`Collective::is_valid()`、`size(TiledMma{})` | **零**（模板求值，不生成代码） |
| 真编译 | `nvcc --ptxas-options=-v` → `Used N registers, M bytes smem` | 一次编译 |

剪枝优先走 traits，只对候选集的 top-k 做真编译。

实测策略是**批量 constexpr traits 查询 + 幸存者并行真编译**：100 个 traits
候选耗时 17.6s 且 shared storage 误差为 0；单变体真编译中位数 4.658s，
170 候选串行约 13.2min，4 进程并行获得 3.61× 加速。缓存键必须包含源码、
目标架构、CUDA 与 CUTLASS 版本。（F-7、F-11）

### 原则三：CuTe 是表示，ISL 是求解器

| | 职责 |
|---|---|
| **CuTe layout 代数** | IR 中携带的表示；后端直接渲染；对齐静态情形可直接求解 |
| **ISL / barvinok** | 集合值关系、基数、符号形状、分段、ragged 域、代价闭式 |

cute MLIR dialect 与 CuTe C++ 模板是同一套代数，
`!cute.layout<(128,128):(1,128)>` 直接渲染为 `Layout<Shape<_128,_128>, Stride<_1,_128>>`。

### 原则四：正确性阶梯，每一级对前一级做差分测试

```
L0    多 kernel 逐算子（PyTorch eager）      金标准
L0.5  host 端 stage 循环 + task dispatch     验证编译管线，绕开 in-kernel 同步
L1    单 kernel + 每 stage 全局 barrier       megakernel 骨架，正确但零重叠
L2    逐边细粒度事件（分析层推导）
L3    事件粗化 + 粒度优化 + 通信归属（求解层）
L4    符号形状参数化 + 运行时变体选择
```

每一级保留编译开关，可随时回退比对。

## 1.5 当前状态

实现状态已迁至 [`docs/STATUS.md`](docs/STATUS.md)。v2.0 的 §1.5 状态表与 §1.5.1 残留技术债原文完整保留在该文件中，小节编号不变；代码注释中的"§1.5.1"指向该文件的同名小节。

### 1.5.1 残留技术债（本轮结束时的诚实记录）

见 [`docs/STATUS.md`](docs/STATUS.md) §1.5.1。

---

# 2. 核心抽象：Coupling Graph

> 贯穿全系统的唯一数据结构。前端构造它，分析层填充它，求解层在它上面优化，
> 代码生成层遍历它，代价模型的每一项都是它的派生量。

## 2.1 形式化

设符号形状参数向量 `θ`（来自 `torch.export` 的 `ShapeEnv`），
粒度向量 `g`（每个算子一个 tile 形状）。

**定义 1（任务空间 TaskSpace）**

算子 `op` 的任务空间是其输出张量按 `g_op` 分块后的索引空间：

```
T_op(θ, g) ≜ indexSpace( zipped_divide( OutLayout_op(θ), g_op ) )
           = { c ∈ Z^k : 0 ≤ c_i < ⌈ S_i(θ) / g_i ⌉ }
```

CuTe 的 `zipped_divide` 直接给出 `((tile 内坐标), (tile 索引))`，后者即 task 坐标。

**定义 2（写映射与读映射）**

```
W_op(θ, g) : T_op → 2^E      task 坐标 → 它写入的元素集合
R_op(θ, g) : T_op → 2^E      task 坐标 → 它读取的元素集合
```

`W` 由输出分块直接给出；`R` 按算子类别构造
（pointwise / reduction / matmul / broadcast / concat / slice / transpose）。

**定义 3（耦合 Coupling）**

生产者 `p` 与消费者 `c` 之间的耦合，是从消费者任务坐标到生产者任务坐标集合的关系：

```
C_{p→c}(θ, g) ≜ W_p⁻¹ ∘ R_c :  T_c → 2^{T_p}

C(x) = { y ∈ T_p : W_p(y) ∩ R_c(x) ≠ ∅ }
```

对齐的仿射情形，`C = composition(right_inverse(W_p), R_c)`，
**完全在 CuTe 代数内可算**。

**定义 4（派生量）**

全部具有 `ClosedForm(θ,g)` 类型，而不是代入后的标量；Phase 3 由 barvinok
对参数化多面体计数填充，Phase 1/2 使用保持同一接口的最小 AST：

```
wait(x)      = | C(x) |                    消费者 x 需要等待的生产者数
fanout(y)    = | C⁻¹(y) |                  生产者 y 解锁的消费者数
volume(y,x)  = | W_p(y) ∩ R_c(x) |         该边的通信量
count(T_op)  = | T_op(θ, g) |              task 数
tier         ∈ {0, 1, 2, 3}                可解析程度，见 §2.4
                                           （五个正交属性的摘要，见 §2.4.1）
```

字段名在 C++ 与 CG dialect 中统一为 `wait / fanout / volume / count`。
`C` 使用结构化 `AffineRelation` / `CouplingMapAttr`；字符串只用于打印诊断，
不能作为 Solver 或 Codegen 的语义输入。（F-14）

**定义 5（耦合图 CG）**

```
CG(θ, g) = ( V, E )
  V = { T_op(θ, g) }                       节点：任务空间
  E = { C_{p→c}(θ, g) }                    边：耦合
```

## 2.2 两条不变量

**I1（参数化闭合）**

> `C` 是 `(θ, g)` 的闭式表达式。改变 `g` 是**重参数化**，不是重推导；
> 改变 `θ` 只改变规模，不改变图的拓扑。

这是整个系统成立的前提。它意味着：求解器可以在 `g` 的候选集上反复求值派生量而不必重建图；
符号形状是免费的（`θ` 只是关系里的参数）；split-K 这类粒度变化表达为 `C` 的参数变化。

**I1 的可执行形式有两半，两半都已经落地**：

* `g` 无关性（L-sem）：`test/unit/semantics_test.cpp` 用两个不同的 `g` 构造同一个
  L-sem，断言 `Serialize()` 逐字节相同、且文本里不出现 `Tm`/`Tn`/`Tkv`，同时断言
  两个 L-task 实例化的 tile 确实不同（否则前一条断言是空的）。
* `θ` 参数化（派生量）：`wait`/`fanout`/`volume`/`count` 必须是 `θ` 的函数，不是
  在 `θ` 的某个取值上求得的整数。`tilemega-symbolic-probe` 把每条边推两次——一次
  让 `S`/`past`/`L_s` 保持自由、一次在具体取值上重推——用
  `QuasiPolynomial::SemanticallyEqual` 按*函数*比较（不折成标量，因为位置相关的
  度量折成标量就是错的）：✅ 21 条边 × 4 个度量 × `S ∈ {1,4,128,512,2048}` 共
  **420/420** 一致（`docs/experiments/SYMBOLIC/`）。
  这后半条此前**不成立**：前端把每个符号维度绑到区间下界再推导，63 个度量里有
  27 个是错的，`attn_chunk → attn_combine` 的 `wait = ⌈L_s/Tkv⌉` 在 4096 token 上
  低估 **32×**，而 `T_sync = |image(C_κ)| × latency` 读的正是它。
  ⚠️ 唯一仍然是整数的是等待窗口的 `div/scale/offset/count`——它们是生成表里的
  `constexpr`，符号化需要给发出的表加运行时字段，属于 P5.1 的同一件事。

**I2（保守可替换）**

> 把任意一条边的 `C_e` 替换成任何 `C'_e ⊇ C_e`，执行仍然正确。

推论：保守化的代价是**逐边局部**的，不会传播到全图。
最坏情况（整条边退化为算子级 barrier）等价于 kernel-per-operator，永远不会更差。
这是 Tier 2/3 处理策略的正确性基础。

**I3（可流式执行）**

> `resident_limit = num_sms × active_blocks_per_sm(kernel, block, smem)` 只是资源容量。
> `grid` 能否超过它取决于实现后的全局 wait-for 图：若对所有可能的常驻 CTA 子集都存在
> 有界的推进前沿，则可滚动让出 slot；只有无法证明该性质时才要求全 grid 共存。

启动序上的依赖跨度有界且小于 `resident_limit` 是一个常用的充分特例，不是完整定义。
V-A 的局部环在 2×容量仍推进，而 V-J 的反向依赖在 `resident_limit+1` 通过、
2×容量 20/20 挂起，说明合法性不能只由局部 `(θ,g)` 派生量或 `grid` 大小决定。（F-4、F-9）

## 2.3 CG 上的六种操作

系统的决策抽象是在 CG 上做以下六件事；抽象已定义不等于执行已接入：

| 操作 | 定义 | 决策变量 |
|---|---|---|
| **Reparam** 重参数化 | `g → g'`，`C` 的形式不变 | tile 形状、split-K 因子 |
| **Coarsen** 粗化 | `C_κ = ⌊·/κ⌋ ∘ C`，对 image 做粗投影 | 事件粒度 κ |
| **Label** 标注 | 每条边打 `sync_kind ∈ {global, cluster, local}` | 通信归属 |
| **Place** 放置 | `T_op` 的点 → `(worker, slot)` | 分配与顺序 |
| **Relax** 松弛 | `C → C' ⊇ C` | Tier 2/3 的保守化 |
| **Fuse** 融合 | 将链上相邻生产者/消费者合成一个 task，内部边 `C_{p→c} → ∅` | 相邻融合区间及 tile 一致性 |

**Place 的执行语义（v2.1）**：`(worker, slot)` 中的 slot 是执行语义的一部分，必须由执行器按 §5.7 消费；只给出 worker、再由 host 或 Codegen 另行决定顺序，不构成 Place 的实现。L2 执行下 Label 是 Place 的子决策：cluster 同步只在相关生产者与消费者被放进同一 cluster 时可选（§5.7.1）。

各操作的实际接入状态见 [`docs/STATUS.md`](docs/STATUS.md) §1.5.4（v2.1 迁出）。

### Fuse 的语义与代价（T0–T4，本轮实现前定义）

`Fuse(e)` 将生产者 p 与消费者 c 的匹配 task 合并，内部依赖由 task 内程序序
实现，不再生成 e 的事件、poll 或 fence；中间 tile 保留在 shared/register，
不写回 global。外部消费者若仍需要中间值，不得仅删该写回。本轮只接受链上
相邻的单生产者融合，不跨分支或合并多入边。

收益有两项：删除内部依赖的runtime事件成本（A9 event_ns，须按融合后实际
task/wait重投影，不以逻辑边数猜测）及中间global流量。L1的barrier_ns保持独立。
四项代价必须同时保留：(1) inverse fanout>1时按每producer精确计价的重算，
`recompute_ns = Σ_y (fanout(y)-1)*producer_task_ns(y)`；(2) 从读取indexing map
导出的 `g_p ≡ Π(g_c)` tile约束；(3) `max(scratch_p,scratch_c)+跨界中间tile`
的shared预算与`max(reg_p,reg_c)`，进入整个kernel的Residency并由编译资源复核；
(4) 消费者坐标决定`|T_fused|=|T_c|`，归约独占整轴带来的wave tail。
重算诊断量不重复加到已包含重复producer执行的融合wave总价上。

Residency 仍在 DP 链外固定，融合段必须满足该固定层级的寄存器/shared/线程
预算，不得让相邻状态任意选择不同 CTA/SM。T0 的实测结果决定哪个预算绑定；
题述 168 registers / 49536 B 的 256-thread 实例不能代替当前所有 BF16 变体。
若某个融合增量不跨 occupancy 台阶，其 residency 增量成本为零，但该约束仍
必须存在；不是无限 smem 免费。GEMM→add 与 GEMM→RMSNorm 的校准结果可以输，
不能为“融合成功”改变数值判据、缩减任务语义或隐藏未写回的数据。

**事件张量**由粗化后的耦合直接给出：

```
EventTensor(e) = image(C_κ)              形状
wait(e)        = |C_κ⁻¹(e)|              计数器初值
trigger(e)     = C_κ(x) 的坐标映射        谁通知谁
```

## 2.4 Tier：耦合的可解析程度

| Tier | 定义 | 实例 | `C` 的形态 | 代价 |
|---|---|---|---|---|
| **0** | 纯仿射 | norm / proj / elementwise / RoPE / GEMM(含 split-K) / GQA head 映射 | 完全闭式 | 0 |
| **1** | 共享单射布局的间接寻址 | paged KV cache 的 block table | 布局函数 `L` 在 `W⁻¹∘L⁻¹∘L∘R` 中抵消，逻辑空间闭式 | 0 |
| **2** | 结构化 ragged（模式已知，extent 运行时） | split-KV chunk 数、chunked prefill 的 per-request token 数、MoE 每 expert token 数 | 索引映射闭式，`wait` 与 image 的 extent 运行时填（indptr 前缀和） | 一次前缀和 |
| **3** | 数据相关排列 | MoE topk 路由、投机解码接受长度、动态稀疏 mask | 不可闭式，按 I2 松弛为分组 barrier（per-sequence / per-expert）或算子级 barrier | 局部同步过度 |

### 2.4.1 Tier 是五个正交属性的摘要

上表这一个数字同时回答了几个互不相干的问题，而这些答案各自独立地变化。
推导实际观察到的是五项，Tier 由它们导出（`CouplingDerivation.h`，
IR 中为 `#tilemega.coupling_attrs<...>`）：

| 属性 | 取值 | 含义 |
|---|---|---|
| `relation_kind` | `affine` / `layout_mediated` / `data_dependent` | 下标映射如何表达 |
| `extent_kind` | `static_literal` / `symbolic_static` / `runtime_dynamic` | 循环界由什么确定 |
| `exactness` | `exact` / `relaxed` | 投影是精确的还是按 I2 放宽的 |
| `runtime_requirement` | `none` / `prefix_sum` / `tensor_values` | 启动前运行时必须提供什么 |
| `countability` | `constant` / `quasipoly` / `piecewise_quasipoly` / `uncountable` | 派生量是什么形态 |

```
data_dependent 或 uncountable   → 3
runtime_dynamic 或 relaxed      → 2
layout_mediated                 → 1
其余                            → 0
```

`extent_kind` 与推导时的 `known` 绑定无关：I1 说的是 L-sem 写了什么，不是
推导时恰好绑定了哪些符号。`CouplingOp::verify` 从属性重算 Tier 并拒绝对不上
的边。§2.7 的 4 条 Tier 2 边在属性下分成三类且**全部 `exact`**——Tier 2 同时
表示"运行时 extent"（要一次前缀和）与"投影被放宽"（运行时什么都不要），
这个区别只有属性能表达。非对齐分块边（`coupling_types_test`，96 → 160）是
`affine + symbolic_static + exact + none + piecewise_quasipoly`：Tier 0，但
`wait` 以周期 3 分段，"Tier 0 ⇒ wait 是常数"这一读法因此写不出来了。
证据见 `docs/experiments/P3/attributes.md`。

## 2.5 示例：耦合的重参数化

Llama decoder layer 的 `attention combine → Wo` 边：

```
C(m,n) = { (s,hh) : s ∈ 行块 m, ∀hh }
wait   = Tm × n_h = 128 × 32 = 4096
```

`wait = 4096` 意味着任何 Wo task 都要等几乎全部 attention task，是一堵墙。

对 Wo 施加 split-K（沿 head 维分 `Kc` 段），即 `g_Wo` 增加一个 K 分片参数：

```
C(m,n,kc) = { (s,hh) : s ∈ 行块 m, hh ∈ 段 kc }
wait      = Tm × n_h / Kc
```

**`C` 的表达式形式未变，只是多了一个参数。** 墙塌了，重叠成为可能，
而依赖关系不需要重推。这是 I1 在具体场景下的体现。

## 2.6 CG 与系统各层的映射

| 层 | 在 CG 上做什么 |
|---|---|
| L4 前端 | 从计算图构造 `V` 与 `E` 的骨架（`g` 为符号，`C` 待填） |
| L3 分析 | 推导 `W` / `R` / `C`，计算派生量，标注 tier，对 Tier 2/3 施加 Relax |
| L2 求解 | 在 `(g, κ, label, placement)` 上优化，输出参数化的最优解区间划分 |
| L1 生成 | 节点 → TaskBody 模板实例；边 → 同步代码；Place → 静态调度表 |
| 代价模型 | 每一项都是派生量的函数（§4.4） |
| L5 运行时 | 代入 `θ` 实例化 Tier 2 的 extent；`O(1)` 查表选变体 |

> v2.1 注：L1 对 Place 的消费以 §5.7 为契约；当前只落地了 stage 置换与 host 端映射，见 `docs/STATUS.md` §1.5.2。

## 2.7 Llama decoder layer 的耦合表（L3 验收基准）

`H=4096, n_h=32, n_kv=8, d=128, I=14336, S=token 数, L_s=序列 KV 长度, Tm=Tn=Tkv=128`

**分析层自动推导的结果必须与本表逐项一致。**

| # | 边 | `C`（消费者→生产者） | image 形状 | wait | fanout | Tier | cluster 候选 |
|---|---|---|---|---|---|---|---|
| 1 | RMSNorm1 → Wq/Wk/Wv | `(m,n) ↦ m` | `[⌈S/Tm⌉]` | 1 | 48 | 0 | ✗ fanout 超簇容量 |
| 2 | Wq → RoPE_q | `(m,hh) ↦ (m,hh)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 | ✓ 1:1 |
| 3 | RoPE_k → KVappend | `(row,hh) ↦ (⌊row/Tm⌋,hh)` | 同上（k 侧 8 组） | 1 | **Tm = 128** | **1** | ✗ fanout 超簇容量 |
| 4 | KVappend → Attn chunk | 仅 `j = ⌊(L_s−1)/Tkv⌋` | ragged | `Tkv`（注 2） | **`G·S`**（注 4） | **2** | — |
| 5 | RoPE_q → Attn chunk | `(s,hh,j) ↦ (⌊s/Tm⌋, hh)` | ragged 域，映射仿射 | 1 | **`Tm·⌈L_s/Tkv⌉`**（注 4） | **2** | — |
| 6 | Attn chunk → Attn combine | `(s,hh) ↦ {(s,hh,j) : j<⌈L_s/Tkv⌉}` | `[B×32]` | 运行时 `⌈L_s/Tkv⌉` | 1 | **2** | ✓ 典型场景 |
| 7 | Attn combine → Wo | `(m,n) ↦ {(s,hh) : s∈行块m, ∀hh}` | `[⌈S/Tm⌉]` | `Tm×32` | 32 | 0 | split-K 后 ✓ |
| 8 | Wo → residual add | `(m,n) ↦ (m,n)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 | ✓ |
| 9 | add → RMSNorm2 | `i ↦ {(i,n) : ∀n}` | `[⌈S/Tm⌉]` | 32 | 1 | 0 | ✓ 32 ≤ 簇容量 |
| 10 | RMSNorm2 → Wgate/Wup | `(m,n) ↦ m` | `[⌈S/Tm⌉]` | 1 | 224 | 0 | ✗ fanout 超簇容量 |
| 11 | Wgate,Wup → SiLU·mul | `(m,n) ↦ {gate(m,n), up(m,n)}` | `[⌈S/Tm⌉×112]` | **2** | 1 | 0 | ✓ 理想 |
| 12 | SiLU·mul → Wdown | `(m,n) ↦ {(m,nk) : ∀nk<112}` | `[⌈S/Tm⌉]` | 112 | 32 | 0 | split-K 后 ✓ |
| 13 | Wdown → residual add2 | `(m,n) ↦ (m,n)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 | ✓ |

11/13 条边 Tier 0；稠密模型中 Tier 3 为零；ragged 仅出现在边 4–6。

**本表的三处修正**（P3 自动推导 + P3_ISL 的 isl 重推得出，见
`docs/experiments/P3/table27.md` 与 `docs/experiments/P3_ISL/result.md`；
是推导纠正表，不是把期望改成迁就实现）：

1. **边 3 的 `C` 与 fanout。** KVappend 的任务粒度是**一行一个任务**
   （tile = 1），不是 Tm 行一块，所以消费者坐标是 row，投影到生产者行块空间
   是 `⌊row/Tm⌋`——一个多对一映射。`wait` 不受影响（一行仍只需要一个生产者
   块），但 `fanout(p0) = |C⁻¹(p0)| = Tm = 128`：每个 128 行的 rope_k 块被
   128 个行任务需要。原表的 1 是按"两侧都是 Tm 块、m ↦ m 一一对应"的粗粒度
   模型写的；迁移前的实现也报 1，因为它用的是启发式（"在 C 中出现的坐标就算
   被 y 钉住，贡献因子 1"），该假设对恒等出现成立、对 floordiv 出现不成立。
   barvinok 的真实逆像计数没有这个盲区。fanout=128 超出簇容量，故 cluster
   候选一并改为 ✗。
2. **边 4 的 wait。** 表中的 1 是 decode（S=1）实例；一般（prefill）情形是
   `min(Tkv, S)`，符号下为 `Tkv`。
3. **表未列出的第 14 条边。** `add1 → add2`（第二个残差读第一个残差的输出）
   是真实耦合，已在 `table27_test` 中断言，避免它悄悄消失。
4. **边 4 与边 5 的 fanout（本轮，`docs/experiments/WIRING/result.md` (g)）。**
   表写 `S` 与 `⌈L_s/Tkv⌉`，推导给 `G·S` 与 `Tm·⌈L_s/Tkv⌉`。这不是前端与
   分析层不一致——`tilemega-derive llama` 在表自己的维度上给同样两个形状。
   ✅ 与 isl 无关地独立核对过：在 `S=7, n_h=8, n_kv=2, G=4, Tm=3, Tkv=2,
   past=5, L_s=12` 上直接按表**自己的 `C` 列**枚举 `|{c : p ∈ C(c)}|`，边 5
   内部生产者块得 18（表写 6，那是 ragged 尾块、即每消费者坐标的因子而不是
   逆像基数），边 4 得 28（表写 7）。机制与注 (f) 相同：边 5 的生产者发
   `Tm` 行块而消费者按单 token 消费，表写了 chunk 因子丢了 token 因子；
   边 4 的消费者跨 `G` 个共享同一 kv 头的 query 头，表写了 token 因子丢了
   GQA 组因子。两处都是按"两侧同粒度"的粗模型写的，与表自身的 `C` 列不符。
   按标准不迁就实现：改的是表，两个值都留在案上。

**边 2/3 成立所依赖的 tile 约束。** QKV 投影的列 tile 必须取 `d`（一个头）
而不是通用的 `Tn`。列 tile 与 `d` 无关时，RoPE 的按头读取不再 tile 对齐，
推导会（正确地）松弛——因为跨越两个头的投影 tile 确实会让一个 RoPE 任务耦合
到多个投影任务。这是求解层（L2）必须遵守的约束，不是自由选择。

---

# 3. 构建基础

> 调研基线：CUTLASS `dc45f979`（4.8.0，2026-08-25）

## 3.1 许可证边界

| 组件 | 许可证 | 可否依赖 |
|---|---|---|
| `include/cute/*`、`include/cutlass/*`（C++ 头） | **BSD-3-Clause** | ✅ |
| `cutlass_compiler/`（MLIR cute dialect） | **BSD-3-Clause** | ✅ |
| `python/CuTeDSL/` | NVIDIA EULA（专有） | ❌ 概念可参考，代码不可依赖 |
| mirage / MPK | 待确认 | ⚠️ 借鉴机制前须确认 |

## 3.2 CUTLASS C++：megakernel 骨架

| 需要的 | 现成件 |
|---|---|
| Persistent kernel + tile 调度（含 Stream-K） | `cutlass/gemm/kernel/sm90_tile_scheduler.hpp` |
| Cluster 启动 / 同步 | `cutlass/cluster_launch.hpp`、`cute/arch/cluster_sm90.hpp` |
| mbarrier 流水线 | `cutlass/pipeline/pipeline.hpp`（`PipelineTmaAsync` / `PipelineAsync`） |
| CTA 内具名 barrier | `cutlass/arch/barrier.h` 的 `NamedBarrier` |
| GEMM collective mainloop | `cutlass/gemm/collective/*` |
| Epilogue collective | `cutlass/epilogue/collective/*` |
| TMA / tcgen05 atom | `cute/atom/*`、`cute/arch/tmem_allocator_sm100.hpp`、`mma_sm100_umma.hpp` |
| layout 代数（C++） | `cute/layout.hpp`、`cute/algorithm/*` |

`CollectiveMma::operator()` 接受
`(pipeline, pipeline_state, accum, gA, gB, k_tile_iter, k_tile_count, thread_idx, smem_buf, params)`，
调用方拥有 pipeline 与 smem。这是把 collective 嵌入任务体的接口。

## 3.3 cute MLIR dialect：layout 代数一等 op

`cutlass_compiler/cute_ir/`，约 8K 行 ODS，~70 个 op。

**类型**：`IntTuple` `Coord` `Shape` `Stride` `Layout` `Tile` `ComposedLayout` `Swizzle` `ArithTuple`

| 用途 | op |
|---|---|
| **`W` 构造** | `LogicalDivide` `ZippedDivide` `TiledDivide` `FlatDivide` `TileToShape` |
| **`W⁻¹`** | `RightInverse` `LeftInverse` |
| **复合（求 `C`）** | `Composition` |
| 补集 | `Complement` |
| 坐标↔索引 | `Crd2Idx` `Idx2Crd` `LayoutEval` |
| 尺寸 | `Size` `Cosize` `TupleProduct` |
| 符号算术 | `ShapeDiv` `CeilDiv` `TupleAdd` `TupleSub` `ElemLess` `Equal` |
| 规范化（转 ISL 前置） | `Flatten` `Coalesce` `GroupModes` `RecastLayout` |
| 切片 | `Slice` `Dice` `Select` `Get` |

pass：`cute-fold-static`、`cute-expand-ops`、`cute-to-base`。
该 dialect 仅含 layout 代数，不含内存 / GPU / kernel op。

**能力边界（V-F）**：cute MLIR dialect 仅作为 Phase 3 的分析表示，不进入
codegen。动态 `Composition`、divide、flatten/coalesce 可保留为 IR，
但 `RightInverse` 的动态 shape 被 verifier 拒绝。处理顺序为：先用已选 `g`
特化 intra-tile `W`；仍动态则在 Presburger/ISL 关系上求逆；含不可消除动态
stride/swizzle 时提升 Tier。（F-12）

## 3.4 CUDA C++ 直接可用的硬件能力

- 显式 shared memory 布局控制 → 分页 smem 与 cross-task 软件流水
- `__launch_bounds__(N, min_blocks_per_sm)` → 强制 occupancy，保证 persistent 共存性
- `__cluster_dims__` / `cluster.sync()` / `cluster.map_shared_rank()` → DSMEM
- `__nanosleep()` → 自旋退避
- 内联 PTX → 任何缺口的逃生舱
- warp / lane 级控制（`__shfl_*`、`__activemask`、具名 barrier）

硬件能力必须来自 `ArchDispatch::Caps`，资源必须来自 `TargetSpec`/target JSON，
架构号不构成能力偏序：**sm_120 没有 tcgen05**，不能用 `arch >= N` 推断能力。（F-5）

`TargetSpec` 的可迁移契约还包括：SKU 标识与字段来源、每 SM 最大常驻 block、
cluster occupancy/GPC 限制、opt-in smem 状态、collective 数据类型族、
以及 CUDA/CUTLASS 版本。SM 数和 smem 上限不得作为业务代码常量。（F-13）

## 3.5 CuTe → ISL 转换规则

平坦 CuTe layout `(s₀,s₁,…):(d₀,d₁,…)` ⟷ ISL 仿射映射：

```
[i₀,i₁,…] → [offset + Σ iₖ·dₖ]    s.t.  0 ≤ iₖ < sₖ
```

层次 layout 先 `cute.flatten` → `cute.coalesce`，再读 shape/stride 构造 `isl_map`。
因为后端也是 CuTe，layout 是唯一真值，转换从真值出发。

**分工**：

| CuTe 直接提供 | 必须交给 ISL |
|---|---|
| `W`（`zipped_divide`）、`W⁻¹`（`right_inverse`）、`C`（`composition`） | 集合值关系（CuTe layout 是函数，非关系） |
| `⌈S/T⌉`（`ceil_div` / `shape_div`） | 符号形状的代数运算 |
| 坐标↔索引、补集、展平 | 基数计数（`wait` / `fanout` / `volume`） |
| Tier 0 对齐静态情形的完整求解 | 非对齐分段、ragged 域、代价闭式 |

**这张表现在是落地状态，不是计划**（P3.7 之后）。右列全部由 isl/barvinok
承担：关系代数是 `isl_map_apply_range`/`isl_map_reverse`（`CouplingRelation`），
基数计数是 `isl_pw_qpolynomial`/barvinok（`QuasiPolynomial::Card`），包含判定
是 `isl_map_is_subset`，`Coarsen` 是与 floor 映射的复合。自建的
`AffineRelation`/`ClosedForm` 关系算子已删除，不作为并行表示保留；`ClosedForm`
只剩下"符号算术构造器"这一个角色，它的输出通过 `ToIslText()` 进入 isl。

三条实测边界，写在这里是因为它们决定了转换规则的适用范围：

- **除数必须先落成字面量**。`isl_aff_div` 在 C API 层就拒绝参数化除数，所以
  tile 尺寸（`Tm`/`Tn`/`Tkv`）和任何出现在除数位置的符号（GQA 的 `G`）必须
  在建 isl 对象之前用 `known` 绑定替换掉；workload 维（`S`/`L_s`/`past`）保持
  为 isl 参数，这正是不变量 I1 的形状。`ClosedForm::ToIslText` 在除数没被替换
  时抛错，而不是生成非法文本。
- **swizzle 与 dynamic stride 必须显式抬 Tier，不能近似**。`ToIslMap` 对二者
  各抛一个 `std::domain_error`（"has no isl_map … raise the Tier instead of
  approximating it"），因为带 swizzle 的复合 layout 展平之后不是仿射 stride
  映射，`参数×坐标` 也不是 Presburger 仿射。悄悄近似会让一条 Tier 3 的边看
  起来像 Tier 0。
- **回写方向只接受具体结果**。`FromIslMap` 要求 layout 映射单值、单片、
  extent 有字面上下界；求解器给出的 layout 本来就是具体的，拒绝比猜测好。

`W⁻¹` 的三级规则（共享单射 layout 直接消去 → g-特化静态 tile 走
`cute.right_inverse` → 其余走 Presburger 关系）落在 `CuteLayoutBridge::Project`
里，Tier 由它一并给出，这是"CuTe 是表示"那一半。

---

# 4. 架构

## 4.1 分层

```
┌──────────────────────────────────────────────────────────────┐
│  L5  Serving Harness    continuous batching / paged KV / MoE  │
├──────────────────────────────────────────────────────────────┤
│  L4  Frontend           torch.export → CG 骨架                 │
├──────────────────────────────────────────────────────────────┤
│  L3  Analysis           W / R / C，派生量，tier，Relax          │
│                         CuTe 代数（表示）⇄ ISL/barvinok（求解）│
├──────────────────────────────────────────────────────────────┤
│  L2  Solver             Reparam / Coarsen / Label / Place      │
│                            ↕ 代价查询（traits + nvcc）          │
├──────────────────────────────────────────────────────────────┤
│  L1  Codegen            CG → CUDA C++（组合 CUTLASS collective）│
├──────────────────────────────────────────────────────────────┤
│  L0  Backend            nvcc → cubin / .so                    │
└──────────────────────────────────────────────────────────────┘
```

## 4.2 前端：`torch.export`

`ExportedProgram` 提供 ATen 级 FX graph、`ShapeEnv` 中的符号维（sympy）、
FakeTensor meta。动态维通过 `torch.export.Dim` 声明，内部已有符号推理与 guard 求解——
这是 `θ` 的来源。符号值表达式进入 `ClosedForm`，范围与等式/不等式/取模 guard
进入独立的 Presburger 参数域；不能因为复用了一个 `Dim` 就假设导出符号同名。
torch 私有 guard API 必须集中在一个版本锁定的 Python 适配器中。（F-15）

KV cache 管理、paged block table、continuous batching 调度不在 exported graph 中，
属于 L5（§4.6）。

## 4.3 CG dialect

自建 MLIR dialect，类型系统复用 cute dialect：

```mlir
// 任务空间
tilemega.task_space @gemm_tasks attributes {
    kind        = #tilemega.task_kind<gemm>,
    granularity = !cute.layout<"(128,128,64):(...)">,   // g_op
    write_map   = !cute.layout<...>                     // W_op
}

// 事件张量：形状可含符号维
tilemega.event_tensor @e0 : tensor<?xi32> attributes {
    extent = #tilemega.closed_form<"image_size(C_kappa)">
}

// 耦合：一条边
tilemega.coupling @c1 from @norm_tasks to @gemm_tasks attributes {
    read_map  = #tilemega.access_map<layout = !cute.layout<...>>,   // R_c
    relation  = #tilemega.coupling_map<"(m,n) -> (m)">,             // C
    wait      = #tilemega.closed_form<"1">,
    fanout    = #tilemega.closed_form<"ceil(Dq/Tn)">,
    volume    = #tilemega.closed_form<"Tm*H">,
    count     = #tilemega.closed_form<"ceil(S/Tm)*ceil(Dq/Tn)">,
    tier      = 0,
    sync_kind = #tilemega.sync<global>,                             // Label
    event     = @e0
}

// 放置
tilemega.placement @gemm_tasks map = [...] cluster = 2

// 实现契约：求解器选定的 (g, impl) 中的 impl 一侧（P4.9）
tilemega.implementation @impl_qkv_17 for @gemm_tasks attributes {
    backend    = "cutlass.sm80_cpasync.simt_f32",
    tile = array<i64: 128, 128, 16>, cluster = array<i64: 1, 1, 1>,
    stages = 3, threads = 256, smem_bytes = 49536,
    alignment = array<i64: 1, 1>, arch_required = 80,
    // regs_est 可选：只有 tier 3 的 ptxas 日志能填
    access = [#tilemega.access_map<{operand = "q_rope",
                coordinates = ["h", "q", ""], spans = array<i64: 1,128,128>}>,
              ...]
}
```

**属性**：`AccessMapAttr`、`CouplingMapAttr`、`ClosedFormAttr`（派生量的符号表达式）、
`TierAttr`、`SyncKindAttr`、`PlacementAttr`。

这些属性与 L3a C++ 类型一一对应：`ClosedFormAttr` 的 storage key 是
`analysis::ClosedForm` 本体，`CouplingMapAttr` 是包含 consumer/producers/ranges/
parameters/fiber/image 的结构化字典；Tier（可解析性）与 SyncKind（通信归属）分离。

**verifier**：事件张量形状 = `image(C_κ)`；`wait` 的闭式在 `θ` 全部代入后与
barvinok 计数一致；`tilemega.implementation` 的 threads/smem/alignment/arch
必须与后端对该 tile 的闭式相等（实现可以选形状，不能改写形状的代价），
其 `access` 必须与 task space 的 `index_map` 逐轴一致（P4.9）。

## 4.4 求解流程

### 4.4.1 L1 目标（stage 串行 + 每 stage barrier）——已实现

以下为 v2.0 的求解流程与代价模型，目标函数均为 L1 执行（stage 串行、每 stage 一次 grid barrier）下的总时长。执行感知的 L2 目标见 §4.4.2。

```
层1  合法性剪枝：从 CUTLASS TiledMma 的原生形状出发沿各维扩张，撞资源墙停
     ← 走编译期 traits，多数候选不编译
     ← 实测：一个 GEMM 224 个合法候选（× 5 个 split 因子），目标 8~20 未达到
层2  跨算子对齐传播：wait = ⌈(m+1)Tr/Tm⌉ − ⌊mTr/Tm⌋   Tm 生产者宽、Tr 消费者宽
     Tr = Tm 时 wait = 1；不整除时膨胀且变分段。Tr 从生成的 stage 表推出来
     ← 实测：层1 产出的 2 的幂轴上剪枝为 0；步长 16 的反事实轴上塌缩 143×
层3  链上 DP（图深而窄，接近阶段链）：
     DP[i][s] = min_{s'} { DP[i−1][s'] + Cost_i(s) + Interface(s',s) }
     s = (Tm,Tn,Tk; stages; split)；驻留度是整个 kernel 的属性，在链外枚举
层4  Label：带尺寸约束的图划分，最大化簇内 volume × frequency
层5  Place：分层 DAG 上的 list scheduling（关键路径优先）
```

**代价函数**。原来这里写的是 roofline 加若干附加项。标定和 2154 个实测点推翻了
它的三条，现在的形式是**资源向量 + 流水包络**——每一项仍然是 CG 派生量的函数：

```
稳态    u(o) = ⟨t_TC, t_CUDA, t_SFU, t_TMEM, t_SMEM, t_L1.5, t_L2, t_DDR, t_NET⟩
        T_steady(o) = max( u(o) )                          取 max，不是取和
        九道齐备（§2.2(a) 的顺序）。哪几道参与由 `TargetSpec::Caps` 决定，
        不参与的置零、被 max 跳过；每条零道都带一个理由
        （`LaneStatus` ∈ {kLive, kCapabilityAbsent, kNotCalibrated}），
        由 `tilemega-target-audit` 逐目标检查
波分解  CTA 按 num_sms × ctas_per_sm 分波；尾波按它自己的活跃 SM 占比
        o = active/num_sms 重新代入 u(o) 求值，而不是外加一个 T_quant 修正项
Split-K combine 用 Stream-K 归约项；BF16 FP32-partial 使用独立实测速率
任务项  TaskCostNs(CG work QP, arithmetic signature, TaskBody traits, Residency)
        GEMM 用名义 issued collective work（物理读集另外保留），保持历史位序
        SIMT 用投影到 runtime task 的物理 R/W、算术签名及控制流 DAG
        depth 是依赖 memory phase 链长，barriers 从归约/发布结构求出
总价    task_ns_sum + combine_ns + barrier_ns + event_ns
        L1 barrier_ns = stage_count × grid_barrier(grid)，历史语义保留
        L2 event_ns 消费 CG runtime refs/waits 与最长队列 QP；不是事件组×notify
接口项  可独立打开 (Σwait − |domain(C)|) × volume × dtype_bytes × memory_rate
        非相邻残差边须保留 live endpoint 的精确 frontier DP，不能冒充相邻链
```

Round5 实现/证据状态：`TILEMEGA_UNIFIED_TASK_COST` 保留独立旧路径开关；
显式 GEMM entry 已通过 904680 位比较，实际缓存入口的全量重验仍进行中。
非 GEMM 14 格 CPU 访问/价格核对、13 错误分支零残留、FP32 两模型完整
1077 配置排序不降；BF16 历史770/462可测子集排序不变，不是新全量oracle。
统一价格不等于 A7 chunk、混合 fusion 或 B3 符号 DP 已完成，详见
`docs/experiments/COST_MODEL/scalar_work/result.md` 与 Round5 台账。

算术声明层仅有 `lib/Analysis/OpArithmetic.cpp` 一张 schema 表，包含14类算术
签名，缺项/缺运行实现明确报错。`flops`/`transcendental` 使用 QP 比值表达，
attention 是 `4×total` FLOP/输出、`total/head_dim` exp/输出，当前走 SIMT CUDA
lane 而非 TC。物理读写域由访问关系计数；TaskBody traits 提供线程/shared/stages
及控制结构，不在 CostModel 中重填每类字节数。B 的混合 task 签名组合尚待实现。

被推翻的三条，以及推翻它们的证据：

- **roofline 不够。** 同一套资源常数上的纯 roofline 一级实测 MAPE 181%/185%、
  Spearman 0.444/0.430，与层2 的解析排序（0.461/0.446）无法区分；那个排序把
  实测最优排在 42/224，自己的 top-10 一个都没落进实测 top-3%。资源向量版
  ρ **0.9450 / 0.9435**。
- **`T_quant` 不能是加项。** 尾波按自己的活跃 SM 数重新求值这一改，把
  ρ 从 0.9095 抬到 0.9450、实测最优的模型排名从 33 抬到 19。加性修正
  `(⌈count/M⌉M − count)/count · t_task(g)` 做不到，因为尾波的每 CTA 时长
  本身就随活跃 SM 数变——那不是一个乘在外面的因子。
- **「各算子孤立时长可加」是假的。** 标定实测：一个 DRAM 密集邻居把 GEMM 从
  18.3 µs 拖到 27.6 µs，`interference_ratio = 1.518`，偏差 **51.8%**，越过
  本骨架自己写的 30% 门槛（P4.4）。资源道取 `max` 正是对这件事的回应。

**验收口径**（这是回归标准，不是一次性实验）。模型不按预测误差验收，按**排序**
验收；验证集是 `docs/experiments/ORACLE/` 已有的 **2154 个实测点**（两个模型各
1077 个配置），复现不需要新的 GPU 时间：

1. 模型 top-3 中至少一个落进**实测 top-3%**（rank ≤ 32 / 1077）；
2. Spearman ρ 明显高于层2 的解析排序；
3. 单配置求值 < 1 ms。

实测：✅ **3/3**（含 top-1）、0.9450/0.9435 对 0.4608/0.4462、最坏 33 µs。
绝对预测值**不**在口径里，也确实不准（gqa2 上预测 0.094 ms 对实测 0.166 ms）。
§0.3 的前提就是「落进平台期」而不是「预测得准」：top-34 个配置挤在 ±10% 带内，
25 进程中位数分不开它们。任何对代价模型的改动都要重跑这三条。

**尚未纳入**：`T_bubble`（流水气泡 + Label 的 smem 占用延长），因为 §2.2(b) 的
填充深度在 12 个标定点上不可辨识；regime 判别（下表）也还没实现。

**分 regime**：

| Regime | 瓶颈 | 目标 |
|---|---|---|
| A. 低 batch decode | 权重带宽 | 最小化权重预取管道气泡（非 makespan） |
| B. prefill / 大 batch | 计算 | makespan + wave quantization |
| C. 混合 batch（chunked prefill） | 异构 | 资源互补性最大化 |

**输出形态**：参数化的最优解区间划分（「`S ∈ [0,512)` 用 `g₁`；`S ∈ [512,∞)` 用 `g₂`」），
区间边界来自分段拟多项式的交点。运行时 `O(1)` 查表选变体。

**TileSight 适配点**（记录在此，避免下次照搬）：

- **`Topo(D)` 的枚举范围。** TileSight 的 `D` 是**一个融合 kernel** 的 tile-action
  DAG：11 个 action，`11!` 上界经合法性剪枝后落到 **132 个合法序**，穷举是可行的。
  本项目的 `D` 是**整个 megakernel 的 CG**：2 层 GQA **179 task / 222 coupling /
  30 stage**，4 层 MHA **355 task / 444 coupling / 60 stage**。同样的穷举在这里
  不是慢，是不可能。
  ✅ 已确认当前实现的范围：`Solver/ListScheduler` **不枚举**——它给出**唯一一个**
  拓扑序（层打包 + 关键路径高度优先 + 下标破平，`Schedule()` 是确定性的），
  枚举范围恒为 1。所以不存在「无界枚举需要收窄」的问题；需要收窄的是反方向：
  若将来要在序上做搜索，必须保持 `C_κ` 的偏序并通过 wait-for 环检查；跨 stage
  重排正是 task queue 打开重叠的手段，不再受 grid barrier 固定。
- **代价模型的分层消融给出的是「哪两层值钱」**（`docs/experiments/COST_MODEL/`，
  2154 个实测点）：roofline ρ 0.4435/0.4303 → `+splitk` 0.9071/0.9043 →
  `+sync` 0.9095/0.9070 → `+waves` **0.9450/0.9435**。split-K 与尾波两层拿走了
  全部增益。`+cache`（SDCM 命中率喂 DRAM 道）在两个模型上把 MAPE 与 ρ 都改动
  **恰好 0**——两个参考模型的活跃工作集都远在 L2 拐点的同一侧，命中率是常数，
  于是它只是一个乘在所有配置上的相同因子，不改排序也不改误差。保留是因为它对
  更大的模型不再是常数；但**不许把它算进本轮的收益**。
- **九道里在 sm_89 上真正起作用的只有 SMEM 道。** `full − lanes(smem only)` 与
  完整模型的差是 **0.02 个 MAPE 点、0.0006 ρ**（gqa2 26.24/0.9450 对
  26.25/0.9444，mha4 25.15/0.9435 对 25.17/0.9429）。
  ⚠️ 并且必须连带记下：**SMEM 道与 L2 道在全部已标定形状上共线**——每个形状的
  smem 流量与 L2 流量成固定比例，拟合分不开它们。把瓶颈叫作 SMEM 是一个
  微架构论证（标量 `ld.shared` 流水在这些形状上先饱和），**不是拟合结论**。
  sm_90+ 上 TMA 把两者解耦，届时必须重测，不得沿用这个命名。

**BF16 profile（本轮新增，不覆盖上述 FP32 结论）**：✅ `tc` 的标定来自真实
BF16 MMA 与 BF16 CUTLASS Stream-K，不是把 FP16 峰值改名；sm_89 实测
179.997 TFLOP/s，DRAM 981.59 GB/s，SMEM 64035.18 GB/s，L2 5111.81 GB/s，
并以 `calibration_by_dtype.bf16` 与 FP32 并存。代价查询在 BF16 时把 FMA 工作
放入 `tc` lane、按 2-byte 元素计算 traffic，并支持逐 lane 禁用。

✅ SMEM/L2 的重新检验给出负结论：当前 BF16 feature 中两者都正比于
`occupancy·2·Tk·(Tm+Tn)`，20 个标定点上的比例恒为 3.4715200776、Pearson=1，
因此**仍不可独立拟合**；这暴露的是 feature 构造限制，不是说硬件管线相同。
**重扫已完成，第一次的结果是负的，归因后修复到高于解析基线**
（1540 点，`docs/experiments/ORACLE/result.md` §6.7、`BF16/result.md`）：

| | 初测 | 修复后 | FP32 对照 |
|---|---:|---:|---:|
| gqa2 ρ | 0.5605 | **0.8984** | 0.9432 |
| mha4 ρ | 0.6239 | **0.8871** | 0.9421 |
| 未标定解析基线 tier2 | 0.8778 / 0.8738 | 同左 | 0.4608 / 0.4462 |

P4.4 的最低门槛（不劣于解析基线）**已达成**；目标（ρ≥0.94、top-1/top-3
落进实测 top-3%）**未达成**，top-1/3/10 仍为 0，阈值没有被改动。模型前十的
实测名次为 gqa2 46–232、mha4 29–145；共同误判是窄 N 与 split-K 的交互。

⚠️ **一条必须记住的自我纠正**：最初把崩溃归因于 `combine_fixed_ns` 被
`max(0, fit)` 钳成 0。那确实是缺陷、也确实修好了，但**不是**原因——修好它
ρ 只从 0.5605 走到 0.5560。真正的两条是：

1. **一个标量 `setup_ns` 给 154 个形状定价**。FP32 的标定 `a` 跨度是
   993–4501 ns，一个标量尚可；BF16 跨度 0–10112 ns，拟合报
   `setup = 1187.94 ns、rms 3247 ns`（残差是自身的 2.7 倍）。改成
   `setup = α + β·tile_m·tile_n`（同样的点、同样没有新测量）→ ρ 0.8246 / 0.8247。
2. **SMEM 道在给 BF16 不走的路径定价**。它的速率来自标量 `ld.shared` 微基准、
   工作项随 mainloop 迭代数增长，于是 split-K 除掉了一份 Tensor Core 内核
   根本不付的代价。特征是干净的单调偏差：预测/实测在 split-K 1 是 0.93、
   到 split-K 16 掉到 0.55。把该道对 BF16 标为 `kNotCalibrated` 后偏差
   **在每个 split 因子上都平到 0.46–0.48**（诊断自证），ρ 到 0.8942 / 0.8834。

标定侧还修了两处：启动基线原本是一个空 kernel（既没有同样的 smem 占用也没有
同样的 launch bounds），证据是归约在**一个**输出上的扣除后时长为 **−928 ns**；
现在基线是**同一个 kernel 什么都不做**，`a` 在零次 mainloop 迭代处直接测得而
不是外推，`ac_r2` 从 0.922 升到 0.984–0.9997（FP32 升到 0.9992–1.0），没有任何
BF16 形状再拟合出负的每 CTA setup。钳位已删除，改由 `combine_fixed_resolved`
显式说明"低于 ~100 ns 计时分辨率"，`tilemega-target-audit` 第一次运行就抓到了
四个目标文件缺该字段。

⚠️ 九道的结论修正：在修复后的模型上，**`l2` 道贡献 0.009 / 0.018、`tc` 道贡献
0.006 / 0.005**（此前对着坏模型量到的 0.001 不作数），其余六道仍然完全惰性。
"BF16 会让 `tc` 第一次成为瓶颈"这个前提在 sm_89 上**仍未被证实**——真正抬排序
的是 L2 字节道；BF16 把瓶颈推离计算道。SMEM/L2 的共线性作为拟合性质没有变，
但 BF16 下 SMEM 道已不参与，所以只剩一条字节道、不再有不可辨识的自由度
（⚠️ 这是关于"标定了什么"的决定，不是两条管线相同的证明）。

### 4.4.2 L2 目标（执行感知，v2.1，待实现）

- **目标函数**：按 §5.7 Plan 执行时的 makespan。
- **代价**：以离散事件模拟器取代 `CostModel::EventNs`（计数 × 速率）。
  - task 时长取 `TaskInstanceNs` 在实际坐标上的值。
  - 同 SM 上并发的 task 按资源向量合并各道需求后取 max。§4.4.1 的九道模型在此成为共驻干扰模型。
  - 每跳延迟与 notify/poll 开销由 trace 标定。
- **决策变量**：
  - π/σ：EFT 式 list scheduling，另以闭式模板作为候选；
  - W 与 policy；
  - sync（Label 作为 Place 的子决策）；
  - κ；
  - 与 g、split-K、residency 联合搜索：外层以 max(work 下界, 关键路径下界) 剪枝，内层用模拟器评估，最终实测 top-3。
- **现状**：ChainDP 的目标是 Σ(stage + barrier)，并在 `l2_events` 下拒绝求解。因此当前全部求解配置都是 L1 最优，其 L2 最优性未经检验（F-128）。
- **验收**：沿用 §4.4.1 的排序口径。
- **承接项**：`docs/TODO.md` EX-S1–EX-S5。

## 4.5 Label：通信归属

**问题**：把 CG 划分成大小 ≤ C（可移植 8，部分架构最多 16）的簇，
最大化簇内 `volume`，满足时间邻近约束（生产者与消费者在相近的 stage slot）。

**收益**：

- 簇内通信走 DSMEM，绕开 L2 往返
- `cluster.sync()` 是硬件原语，比全局原子 + 自旋便宜一个量级
- 簇内 CTA 由硬件保证同时驻留 → 簇内等待可证明无死锁

**代价**：把输出扣在 smem 供簇内消费者读，与「尽早释放 smem 供下一个 task 预取」冲突。
占用时长 = 生产者到消费者的时间距离 × tile 大小，计入 `T_bubble`。

**尺寸匹配**：

| 模式 | 典型规模 |
|---|---|
| split-K GEMM 的 K 分片数 | 2 – 8 |
| FlashDecoding 的 KV chunk 合并 | 4 – 16 |
| MoE 单 expert 的 tile 组 | 个位数 |

簇的粒度匹配「算子内跨 CTA 归约」，不匹配「算子间数据流」。
这一条现在有数字：把 §4.3 的 `w = Volume × Frequency` 灌进带尺寸约束的划分
（P4.7），在 stage 串行的 megakernel 唯一能实现的时间邻近半径 reach = 1 下，
簇能关住的耦合流量只有 **13.6%（2 层 GQA）/ 18.7%（4 层 MHA）**。
放宽到 reach = 4 才到 0.97~0.99，而那要求四五个算子的输出跨 stage 边界一直
待在 smem 里——正是这个 megakernel 每个 stage 复用 smem 所不做的事。
所以「算子间数据流」这条路上簇大约只值六分之一的流量。
证据：`docs/experiments/CLUSTER/result.md` §7.7。

**L2 执行下的 Label（v2.1）**：cluster 同步（DSMEM、`barrier.cluster`）要求相关生产者与消费者被放进同一个 cluster，因此在执行感知求解中 Label 是 Place 的子决策（§5.7.1）。上文 13.6% / 18.7% 的簇内流量捕获率，是在 stage 串行、reach = 1 的假设下得到的；在 Plan 驱动执行下需重新估计（EX-S2）。

## 4.6 Serving harness（L5）

| 组件 | 接入方式 |
|---|---|
| Paged KV cache | block table 走 Tier 1（布局抵消），在逻辑空间做依赖分析 |
| Continuous batching | 请求准入 / 完成剔除 / KV 元数据更新做成 kernel 内的一个 task |
| Chunked prefill | Tier 2，per-request token 数用 indptr 参数化（regime C 的载体） |
| MoE 路由 | Tier 3 的一般化：统一的 indptr 机制承载 topk 结果与 expert 计数 |

---

# 5. Lowering 路径

## 5.1 全景

```
CG（已求解：g, κ, label, placement）
        ↓  CouplingGraphToCUDA
五个生成物：
  (a) TaskBody 模板实例化           ← 节点的 g 决定模板实参
  (b) SharedStorage union           ← 各 task 类型取 max
  (c) 任务描述表 + 事件缓冲布局      ← 边的 C_κ、wait 决定
  (d) Megakernel 主体                ← Place 决定静态调度表
  (e) Host launcher                  ← 区间划分决定变体选择
        ↓
一个 .cu（#include CUTLASS 头 + TileMega runtime 头）
        ↓  nvcc
cubin / .so → torch extension
```

> v2.1 注：(d) 中的"静态调度表"以 §5.7 的 Plan 为契约；当前生成物只含 stage 置换，见 `docs/STATUS.md` §1.5.2 G1。

## 5.2 三层职责

| | 谁写 | 量级 |
|---|---|---|
| 算子内核（MMA mainloop、TMA、swizzle、软件流水） | CUTLASS | 数万行 |
| **TaskBody 模板**（把 collective 包装成可在任务循环中调用的单元） | TileMega，手写 | 每类算子 100–300 行，共约 10 个 |
| **模板参数、任务图、事件结构、放置** | 求解器生成 | 每模型 / 每形状区间不同 |

TaskBody 模板的 tile 形状、cluster 形状、pipeline 级数全部是模板参数，
由求解器填入。模板本身不含任何写死的粒度常数。

## 5.3 TaskBody 模板

```cpp
// include/tilemega/tasks/GemmTaskBody.h
template <class TileShape_MNK,     // ← Reparam
          class ClusterShape_MNK,  // ← Label
          int   Stages,            // ← 受 smem 预算约束
          class ElementA, class ElementB, class ElementAcc,
          class LayoutA, class LayoutB>
struct GemmTaskBody {
  using Collective = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape_MNK, ...>,
      TileShape_MNK, ElementA, LayoutA, ElementB, LayoutB, TiledMma, ...>;

  using SharedStorage = typename Collective::SharedStorage;
  using Pipeline      = typename Collective::MainloopPipeline;

  // 编译期 traits：求解器的零成本代价查询入口（原则二）
  static constexpr int  kSmemBytes  = sizeof(SharedStorage);
  static constexpr int  kNumThreads = size(TiledMma{});
  static constexpr bool kLegal      = Collective::is_valid();

  __device__ void operator()(TaskDesc const& t, char* smem, Params const& p) {
    auto [m, n, k_begin, k_count] = decode_coord(t);   // split-K 由 k 范围表达
    auto gA = make_tensor(p.A + t.in_ptr[0], ...);
    auto gB = make_tensor(p.B + t.in_ptr[1], ...);
    Tensor accum = partition_fragment_C(TiledMma{}, take<0,2>(TileShape_MNK{}));

    Pipeline pipeline(smem, pipeline_params);
    Collective{}.mainloop(pipeline, pipeline_state, accum,
                          gA, gB, k_tile_iter(k_begin), k_count,
                          threadIdx.x, smem, p.mainloop);

    epilogue_store(accum, p.C + t.out_ptr[0], ...);
  }
};
```

**TaskBody 清单**：`GemmTaskBody`、`GemmSplitKTaskBody`、`AttentionChunkTaskBody`、
`AttentionCombineTaskBody`、`RMSNormTaskBody`、`RoPETaskBody`、`ElementwiseTaskBody`、
`KVAppendTaskBody`、`MoERouterTaskBody`、`SchedulerTaskBody`。

TaskBody 的 `TaskDesc / context / SharedStorage / result` ABI 保持架构无关，
但允许 SM80 cp.async、SM90 TMA/GMMA、SM120 TMA/MMA 等每个 CUTLASS family
各有一个 mainloop adapter；不同 family 的 pipeline/residue/epilogue 编排并不相同。（F-6）

每个 adapter 必须显式声明逻辑操作数坐标、stride、residue 约定与 epilogue 归属。
例如 CUTLASS B 是逻辑 `(N,K)`，PyTorch 连续 `[N,K]` 权重在已验证的 SM80 adapter
中对应逻辑 stride `(K,1)`，由 `ColumnMajor` tag 表达；tag 名本身不构成布局证明。
大参数表一律以**设备端指针**传入，禁止按值复制进 kernel 参数：实测 14 个 GEMM
的参数包按值传递产生 2592B 栈帧和 5× 劣化，指针形式降到 32B。（F-17）

TaskBody 必须用 grid-stride 循环遍历自己的 task
（`for (task = blockIdx.x; task < count; task += gridDim.x)`），
不得假设 `task 数 ≤ grid`。这不是性能问题而是正确性问题：`g` 固定时该假设
恰好成立，一旦 tile 变小或 split-K 打开就不成立，多出来的 task 被静默丢弃，
表现为结果错而不是崩溃（实测 512 task / 384 grid，4090 个元素不匹配）。（F-36）

### 5.3.1 分相 ABI（目标，v2.1，未实现）

TaskBody 可选实现三段：

- `Prefetch(p, stage, task, ctx)`：只读取在 CG 中没有入边的操作数（例如 GEMM 的权重）。可在等待依赖之前或上一 task 的收尾阶段发出。
- `Wait`：由执行器负责，不属于 TaskBody。
- `Compute`：读取依赖操作数，完成计算与写回。

"没有入边的操作数"由分析层按操作数的读关系 R 与 CG 入边推出，随生成表发出，不接受手写标注。这是 CG 相对依赖标注式系统的直接收益。

sm_89 上的 Prefetch 先以 L2 预取实现：不占 shared memory，不改变 §8.6 的 union。若要把预取落到 shared memory，需要改变 §8.6 的生命周期约定（双缓冲或分页），并在采用前按 §8.6 的闭式与实测核对 occupancy。

承接项：`docs/TODO.md` EX-E4。

## 5.4 Megakernel 骨架

```cpp
// (a) 求解器决定的实例化
using T_qkv  = GemmTaskBody<Shape<_128,_128,_64>, Shape<_2,_1,_1>, 4, bf16, bf16, float, ...>;
using T_norm = RMSNormTaskBody<Shape<_128>, 4096, float>;
using T_attn = AttentionChunkTaskBody<Shape<_64,_128>, Shape<_2,_1,_1>, ...>;

// (b) smem union：取 max，不相加
union TaskSmem {
  T_qkv::SharedStorage   qkv;
  T_norm::SharedStorage  norm;
  T_attn::SharedStorage  attn;
};
static constexpr int kSmemBytes = sizeof(TaskSmem);

// (c) 事件缓冲：128B padding 防伪共享
struct alignas(128) EventCounter { unsigned long long v; char pad[128-8]; };

// (d) megakernel
__global__ __launch_bounds__(kNumThreads, 1)
__cluster_dims__(CLUSTER_X, 1, 1)
void tilemega_kernel(Params p) {
  extern __shared__ char smem[];
  int worker = blockIdx.x;
  int first = p.schedule_offsets[worker];
  int last  = p.schedule_offsets[worker + 1];

  for (int slot = first; slot < last; ++slot) {
    TaskRef const& d = p.schedule[slot];          // Place 的物化输出
    wait_task_deps(d, p.task_waits, p.events);    // 只遍历该 task 的去重入边
    switch (p.stages[d.stage].type) {
      case TASK_NORM: T_norm::RunTask(p, d.logical_task, smem); break;
      case TASK_QKV:  T_qkv ::RunTask(p, d.logical_task, smem); break;
      case TASK_ATTN: T_attn::RunTask(p, d.logical_task, smem); break;
    }
    notify_task_group(d, p.events);
  }
}
```

生成的 `RuntimeVariantDesc::schedule` 是求解器给出的紧凑 stage 顺序；host 在
`seq/past`、split-K、驻留 grid 与 Place 都绑定之后，将它展开成每个物理 CTA
一条 `TaskRef` 队列。`TaskRef` 携带 stage、逻辑 task 坐标、原始入边区间与
去重/提升后的 `TaskWait` 区间。表放在 device global memory，不占常量内存；
代码体积仍为 `O(stage 数 × 粒度变体数)`，运行时表才随 task 数增长。

事件表同样按运行时 task 数物化：`event_offsets[stage]` 是前缀和，每个 stage
只分配它的出边实际引用的聚合完成事件和/或 `ceil(task_count / κ)` 个逻辑
task 组事件。
窗口边等精细组，`kAll` 边只等聚合事件，避免把一条 CG 入边展开成
O(生产 task 数) poll；每个逻辑 task 完成同时向所属精细组和聚合组贡献
arrival。`κ=0` 明确定义为只保留聚合事件。禁止用 owner worker 代替
logical task 做精细键，也禁止延迟到 worker
在该 stage 的最后一个 task 才发布，否则 queue 形式存在但 readiness 仍是 CTA 粒度。

L0.5/L1 保留 `RunStage` 用作正确性阶梯；只有 L2 走上述队列。相邻 slot 可属于
不同 stage，CTA 不再等同 stage 的其他 CTA 全部完成。`ListScheduler::Validate`
在生成期拒绝环、非置换与反向依赖，host 在 split-K 改写后再次验证；I3 无法
证明 over-resident 安全时拒绝启动，而不是运行时等待挂死。

静态调度表在语义上只有一份，但 L0.5 的 host launch loop 与 L1 的 device loop
处于不同 CUDA 地址空间。Codegen 必须从同一个 initializer 同时生成 host
`constexpr` 表与 device `__constant__` 表，禁止手写两份；否则前者不能被 device
读取，或两级正确性阶梯可能发生调度漂移。（F-21）

**当前实现（as-built，v2.1 核实）**：

- **调度来源**：生成器只发出 stage 置换（`ScheduleStageDesc`），由 `Codegen.cpp` 的 `BuildVariantSchedule` 调用 `ListScheduler` 计算。
- **归属**：host 在绑定 θ、完成 split-K 改写与驻留 grid 之后决定归属。默认 `task_owner[stage][task] = task mod grid`；`TILEMEGA_PLACEMENT=4` 时改用 balanced 启发式。
- **队列**：按"worker → stage_order → 该 worker 在该 stage 拥有的 task"物化，因此恒为 stage-major，且默认归属与 L1 的 grid-stride 相同。
- **执行**：L2 kernel 严格按 slot 顺序执行 wait → run → notify（W = 1）。

上面的骨架代码只表达 §5.7 语义中 W = 1 的特例。它与实现的差距见 `docs/STATUS.md` §1.5.2 的 G1–G3。

## 5.5 同步的三条 lowering 路径

由边的 `sync_kind`（Label 的输出）决定：

```cpp
__device__ void wait_deps(TaskDesc const& d, EventCounter* ev, int layer) {
  switch (d.sync_kind) {

  case SYNC_GLOBAL: {                                        // 跨簇 / 大 fanout
    unsigned long long need = d.num_triggers * (layer + 1);  // 单调计数器
    if (threadIdx.x == 0) {                                  // 单线程轮询
      while (atomicAdd(&ev[d.event].v, 0ull) < need) __nanosleep(64);
    }
    __syncthreads();                                         // 集体同步在非发散点
    __threadfence();                                         // acquire
    break;
  }

  case SYNC_CLUSTER:                                         // Label 选中的边
    cooperative_groups::this_cluster().sync();               // 硬件保证共存
    break;

  case SYNC_LOCAL:                                           // 同 CTA 内
    cutlass::arch::NamedBarrier(kNumThreads, d.barrier_id).sync();
    break;
  }
}
```

### 5.5.1 当前协议（as-built，v2.1 核实）

上面的 `switch(sync_kind)` 是原始设计，不是现行实现。现行 L2 执行器的协议如下：

- **task 完成时**：
  1. 全体线程执行 `__threadfence()`，随后 `__syncthreads()`；
  2. thread0 对每个需要发布的事件行执行 `atomicAdd(arrivals)`，取返回值判断自己是否最后到达；
  3. 最后到达者执行 `__threadfence()` 与 `atomicExch(epoch)`；
  4. 再次 `__syncthreads()`。
- **双行发布**：一个 stage 同时被 kAll 型与窗口型消费者引用时，fine 行与 aggregate 行各发布一次。
- **消费者**：按 TaskWait 由多线程并行轮询 epoch，带 `__nanosleep(64)` 退避；轮询结束后执行 `__syncthreads()` 与 `__threadfence()`。默认的 `EventPoll` 是 `atomicAdd(ev, 0)`（`TILEMEGA_EVENT_LOAD_POLL=0`）。
- **屏障次数**：L2 kernel 在执行器层面每个 task 最多执行 5 次 `__syncthreads()`。
- **local**：没有对应的事件实现。κ = 1 时，同 worker 生产者以省略 poll 的方式处理，这依赖严格 FIFO（§5.7.3 L-d）。
- **cluster**：只存在于 L1 的 stage barrier 与 T1 分片 fan-in 实验（F-89）中。

### 5.5.2 目标协议 v2（待验证，v2.1）

以下步骤逐步开关、逐步验收（`docs/TODO.md` EX-E3）。在通过之前，不替换 §5.5.1 与 §8.5：

1. 单成员事件直接发布 epoch。
2. aggregate 行改为生产者无返回值的 release 归约，消费者以 acquire 加载轮询。
3. 每个 task 的 CTA 屏障不超过 2 次。
4. "CTA 屏障之后仅 thread0 执行一次 release fence"须先通过 litmus（见 §8.5 注）。
5. 发布异步化。
6. 同 CTA 依赖以 shared memory 标志实现；在具备 `caps.cluster` 的目标上评估 cluster 级同步。

## 5.6 求解结果到代码的映射

| CG 上的决策 | 落到生成代码的哪里 |
|---|---|
| Reparam：tile 形状 `g` | TaskBody 的 `TileShape_MNK` 模板实参 |
| Reparam：split-K 因子 `Kc` | `TaskDesc` 的 `k_begin` / `k_count` |
| Coarsen：事件粒度 `κ` | 事件张量 extent + `notify` / `wait` 的事件索引映射 |
| Label：簇归属 | `__cluster_dims__` + `sync_kind = SYNC_CLUSTER` + DSMEM 指针 |
| Place：task→worker | `p.schedule[worker][s]` 静态表 |
| Place：worker 内顺序 | 同表内的 task 序 |
| Stages（受 smem 预算） | TaskBody 的 `Stages` 模板实参 |
| 区间划分 | 多套实例化 + host 端 `O(1)` 查表选 kernel |

> v2.1 注：上表是目标映射；每一行的当前落地程度见 `docs/STATUS.md` §1.5.5。

## 5.7 执行模型与 Plan 契约

> 本节是求解层（L2）与生成层（L1）之间关于"如何执行"的契约（v2.1 新增）。§2.3 对 Place 的定义（`T_op` 的点 → `(worker, slot)`）不变；本节规定 slot 必须被执行器消费，并给出执行器语义、计划合法的条件以及各层的职责边界。当前实现与本节的差距见 `docs/STATUS.md` §1.5.2，承接项见 `docs/TODO.md` 的 EX 主线。

### 5.7.1 Plan

对每个运行时变体（一个 seq 区间），求解器输出：

```
Plan = ( π, σ, W, policy, sync, κ )
  π      : task → worker          worker ∈ [0, grid)，grid ≤ resident_limit（§2.2 I3 的 resident-only 形式）
  σ      : task → slot key        同一 worker 上按 σ 升序即其队列顺序
  W      : 执行窗口，W ≥ 1；W = 1 即严格 FIFO
  policy : producer stage → {aot, jit}；jit 预留给时长数据相关的 stage
  sync   : CG 边 → {local, fine(κ), aggregate, cluster}
  κ      : producer stage → 事件粒度
```

π 与 σ 有两种给出方式。两种方式都由 `lib/Solver` 中的同一实现计算，Codegen 与 host 只消费、不决定：

- **模板形式**：π、σ 是 task 坐标与 θ 的拟仿射函数（如 grid-stride、跨 stage 连续轮询、band、wavefront），模板号与参数写入 CG。
- **物化形式**：host 在绑定 θ 后，按 CG 中记录的策略与参数调用同一 Solver 例程（如 EFT list scheduling）完成物化，并按 θ 缓存。

Label 在 L2 执行下是 Place 的子决策：只有当相关生产者与消费者被放进同一 cluster 时，sync 才可以取 cluster。

### 5.7.2 执行器语义

每个 worker 持有按 σ 升序排列的队列 `q[0..n)`，并维护 `head`，即最小的未完成 slot。执行器只能选择窗口 `[head, head+W)` 中的 slot。

一个 slot 就绪，当且仅当同时满足：
1. 它的所有全局等待都已被观测到满足；
2. 它的所有本地依赖（同 worker、距离小于 W 的前驱 slot）都已完成。

多个 slot 同时就绪时，取 slot 编号最小者。执行完成后，按其出边的 sync 种类发布；`head` 前移，越过已完成的连续前缀。

推论：slot j 开始执行时，所有 i ≤ j − W 的 slot 必已完成。W = 1 时退化为严格 FIFO，即当前实现的语义。

### 5.7.3 合法性条件

- **L-a 无环**：以下三类边之并必须无环——CG 推出的 task 依赖边、同 worker 的窗口边（i → j，i ≤ j − W）、同 worker 的本地依赖边。
  - 模板形式：在生成期由 ISL 在区间上证明。
  - 物化形式：在 host 物化后检查，失败即拒绝启动。
- **L-b 驻留**：grid ≤ resident_limit。在拿到 over-resident 证明之前不放宽（§2.2 I3、§8.7）。
- **L-c 同 worker 顺序**：生产者与消费者在同一 worker 时，必须满足 σ(producer) < σ(consumer)。
- **L-d 窗口感知的等待提升与本地省略**：
  - slot j 的某个全局等待，只有在同 worker 上某个 i ≤ j − W 的 slot 已等待过同一事件时，才可以省略；
  - 同 worker 生产者只有位于 i ≤ j − W 时，才可以省略对它的全局 poll；否则必须转为本地依赖，并计入 L-a；
  - W = 1 时即现行规则（F-80、F-130）。
- **L-e 发布一致**：每条边所需的事件行必须由其全部生产者发布；不被任何消费者引用的行不发布。
- **L-f 单调 epoch**：§8.2 不变，任何协议变更都不得引入计数器重置。

### 5.7.4 层间职责

- **求解器**写入 CG：
  - 每个 task space 的 `tilemega.placement`：mode、模板参数或物化策略与参数、W、policy；
  - 每条 `tilemega.coupling` 的 `sync_kind`；
  - 每个 producer 的 κ。
  - `map=[0]` 的占位写法仅保留为 legacy 模式。
- **Codegen** 发出 `RuntimeVariantDesc` 中的 Plan 描述，作为唯一调度来源；只编码 stage 置换的 `ScheduleStageDesc` 仅作为 legacy 模式的输入保留。Codegen 不自行计算调度。
- **Host** 在绑定 θ、完成 split-K 改写与驻留 grid 之后，按 (π, σ) 生成每个 worker 的队列，按 L-d 计算 TaskWait，并按 L-a、L-b 校验。

### 5.7.5 结构性上界

设 π(stage, t) = t mod grid（与 L1 的 grid-stride 归属相同），且队列为 stage-major。此时 L1 与 L2 的每个 CTA 执行同一批 task；即使同步零成本，L2 相对 L1 的收益上界也只有 (barrier + |loop|) / L1，参考模型上为 9.1–13.6%（F-126，inferred）。

因此，L2 的性能工作必须先改变 π 与 σ；事件原语的调优只在这个上界之内起作用。

### 5.7.6 与其他章节的接口

- 分相 TaskBody 见 §5.3.1。
- 当前同步协议与目标协议见 §5.5.1 与 §5.5.2。
- 窗口规则的 codegen 约束见 §8.10，调度决策权的约束见 §8.11。

---

# 6. 仓库结构

```
tilemega/
├── CMakeLists.txt
├── TILEMEGA_SKELETON.md
├── docs/VERIFICATION_PLAN.md        开工前验证计划（独立文档）
├── third_party/
│   ├── cutlass/                     submodule，跟 main（BSD-3）
│   └── barvinok/                    submodule（内含匹配版本的 isl）
├── include/tilemega/
│   ├── Frontend/
│   │   ├── TorchExportImporter.h    ExportedProgram → CG 骨架
│   │   └── SymbolicShapeBridge.h    sympy expr ↔ ISL param（θ）
│   ├── Dialect/CouplingGraph/
│   │   ├── CGDialect.td             依赖 cute dialect 的类型
│   │   ├── CGOps.td                 task_space / coupling / event_tensor / placement
│   │   └── CGAttrs.td               AccessMap / CouplingMap / ClosedForm / Tier / SyncKind
│   ├── Analysis/
│   │   ├── ISLContext.h
│   │   ├── CuteLayoutBridge.h       !cute.layout ⇄ isl_map（双向）
│   │   ├── AccessRelation.h         W_op / R_op
│   │   ├── CouplingDerivation.h     C = W⁻¹ ∘ R
│   │   ├── DerivedMetrics.h         wait / fanout / volume / count（barvinok）
│   │   ├── TierClassifier.h         tier 判定 + Relax
│   │   └── EventSynthesis.h         C_κ → 事件张量
│   ├── Solver/
│   │   ├── CandidateGenerator.h     层1（CUTLASS traits）
│   │   ├── AlignmentPropagation.h   层2
│   │   ├── CostModel.h              regime-aware
│   │   ├── ChainDP.h                层3（Reparam + Coarsen）
│   │   ├── ClusterLabeling.h        层4（Label）
│   │   ├── ListScheduler.h          层5（Place）
│   │   └── BackendCostQuery.h       traits 优先，nvcc 兜底
│   ├── Codegen/
│   │   ├── CouplingGraphToCUDA.h    主生成器
│   │   ├── TaskBodyEmitter.h        模板实例化 + smem union
│   │   ├── SyncEmitter.h            §5.5 的三条路径
│   │   ├── ScheduleTableEmitter.h
│   │   └── HostLauncherEmitter.h
│   ├── tasks/                       手写 TaskBody 模板
│   │   ├── TaskBase.h               TaskDesc / Params / smem union 机制
│   │   ├── GemmTaskBody.h
│   │   ├── GemmSplitKTaskBody.h
│   │   ├── AttentionChunkTaskBody.h
│   │   ├── AttentionCombineTaskBody.h
│   │   ├── RMSNormTaskBody.h
│   │   ├── RoPETaskBody.h
│   │   ├── ElementwiseTaskBody.h
│   │   ├── KVAppendTaskBody.h
│   │   ├── MoERouterTaskBody.h
│   │   └── SchedulerTaskBody.h
│   └── Runtime/
│       ├── MegakernelRuntime.h      persistent 循环骨架、事件原语、DSMEM helper
│       ├── EventBuffer.h
│       └── Launcher.h
├── lib/                             与 include 对称
├── python/tilemega/
│   ├── compile.py                   torch.compile backend 入口
│   └── serve/                       L5
├── test/
│   ├── unit/                        CuTe↔ISL、耦合推导、派生量
│   ├── lit/                         MLIR lit
│   ├── correctness/                 L0/L0.5/L1/L2/L3 差分
│   └── models/                      端到端
├── benchmarks/
└── docs/design/
```

文档布局（v2.1）：`TileMega_skeleton.md`（设计与契约）、`docs/STATUS.md`（实现状态）、`docs/TODO.md`（待办）、`docs/FINDINGS.md`（发现）、`docs/archive/`（归档）、`docs/experiments/`（证据）。上方目录树是 v2.0 的规划形态，与仓库实际布局存在差异，以仓库为准。

**依赖处理**：

- CUTLASS 只用 header，不引入 `python/CuTeDSL/`
- 是否引入 `cutlass_compiler/` 在 Phase 3 决定；若不引入，
  L3 的 layout 表示用 `pycute` 或自建轻量等价物
- barvinok 优先用发行包（内含匹配 isl）

---

# 7. 分阶段 TODO

待办已迁至 [`docs/TODO.md`](docs/TODO.md)：
- §0 为常设验收条件（v2.0 前言原文 + v2.1 新增）；
- §1 为执行模型主线（EX 条目）；
- §2 为 v2.0 未关闭条目原文；
- §3 为已完成条目索引。

v2.0 的 §7 原文（含全部已完成条目的证据叙述）逐字归档于 [`docs/archive/TODO_v2.0.md`](docs/archive/TODO_v2.0.md)。P0.1–P6.3 的条目号在上述文件中保持不变，代码注释中的"§7 P0.3""P4.8"等引用据此查找。

---

# 8. Codegen 规则

> 硬性规则。所有生成同步代码的路径必须走统一封装。

## 8.1 自旋等待按独立事件分摊到 CTA，集体屏障保持非发散

```cpp
for (int i = threadIdx.x; i < task.wait_count; i += blockDim.x) {
    while (atomicAdd(&ev[task.wait_begin+i].v, 0ull) < need)
        __nanosleep(64);
}
__syncthreads();     // 集体同步放在非发散点
__threadfence();     // acquire
```

同一个 task 的 `TaskWait` 已经去重且互相独立，因此可以由线程按 `i += blockDim`
分摊；旧 stage-loop 上的受控实验为 −15.05%/−14.78%，任务队列保留该实现。
真正的正确性禁令仍是：**集体屏障不能出现在发散的自旋循环体内**。
`barrier_in_spin` 在 grid 64/128/256 共 150/150 挂起。屏障只能位于所有线程
完成各自 poll 后的非发散点。（F-2）

## 8.2 单调事件计数器

```
needed = num_triggers × iteration_num
```

计数器从不重置。避免跨迭代 ABA，免掉迭代边界的全局清零。

## 8.3 自旋必须带退避

`__nanosleep(N)`，档位可配。它是**性能规则，不是正确性规则**：单独去掉退避
0/150 失配；紧凑轮询仍可能饱和内存子系统并拖慢生产者。（F-3）

## 8.4 事件缓冲布局

计数器按 128B cache line padding，避免不同事件的原子操作伪共享。
数组尽量小以维持 L2 常驻；事件数大时按 stage 分段复用。

## 8.5 release 侧的顺序

单线程生产：

```cpp
// 数据写
__threadfence();                         // release fence
atomicExch(&ev[e].v, new_value);         // 置位
```

CTA 协作生产时，release 必须覆盖**每个 writer**：

```cpp
// 每个线程完成自己负责的数据写
__threadfence();
__syncthreads();
if (threadIdx.x == 0) atomicExch(&ev[e].v, new_value);
```

数据写与置位之间必须有 fence，不依赖程序顺序。地址复用的缺 fence 对照
150/150 读到上一轮精确值；错误率又会随 tile 增大而下降（4096→8192 出现断崖，
16384 为 0/50），因此大 GEMM tile 测不出错误不构成正确性证据，验证集必须包含
norm/RoPE 一类小 tile。（F-1、F-3、F-10）

⚠️ v2.1 待验证（不改变本规则）："CTA 屏障之后由 thread0 做一次 release fence 再发布"这一形态目前没有直接证据——F-1 的负对照是"无屏障"。在按 F-1/F-3/F-10 的要求完成 litmus 之前（地址复用、小 tile、CTA 协作写、grid 64/128/256、每格 ≥ 50 全新进程），本规则保持"每个 writer fence"。见 `docs/TODO.md` EX-E3 第 4 步。

## 8.6 smem union 取 max

各 task 类型的 `SharedStorage` 必须组成**单个显式 union**，且该 union 的生命周期
覆盖整个 dispatch，容量取 `max_i(sizeof(SharedStorage_i))`。不同 task 类型的 smem
只有在此条件下不相加；分离对象的地址逃逸会使生命周期重叠，实测退化为 36864B，
occupancy 从 6 降到 2 CTA/SM。（F-8）

union 之后的 smem 项**仍然是绑定项**，不是记账细节：在 1077 个真实候选上
smem 项单独决定 occupancy 的有 150 个、与寄存器项并列的有 322 个，只留寄存器
项会在 150 个候选上算错。闭式
`min(⌊regs_per_sm/(8·⌈regs·32/gran⌉·threads)⌋, ⌊smem_per_cta/smem⌋,
⌊threads_per_sm/threads⌋)` 在两个模型各 1077/1077 上与实测相等。（F-40，
回答了 §9.2 的对应条目）

## 8.7 共存性

资源容量公式为：

```
resident_limit = TargetSpec.num_sms
               × ActiveBlocksPerSM(kernel, block_size, dynamic_smem)
```

`resident_limit` 是上界项，不是所有图的必要 grid 上界。只有 wait-for 进度分析
无法证明流式推进（例如本轮 L1 的每-stage 全 grid barrier）时，才取
`grid = resident_limit`。可流式图允许更大 grid；cluster kernel 还须使用 cluster
occupancy 与整簇取整，而不能套普通 CTA 公式。（F-4、F-9）

`resident_limit` 是**每个 kernel** 的量，而共用一个 grid 的多个自旋 kernel
必须取其**最小值**。L2 的事件 epoch 占寄存器，同一 `g` 下比 L1 多用一档
（实测 144 vs 128 寄存器 → 1 vs 2 CTA/SM），按 L1 定的 grid 启动 L2 会让一半
CTA 不驻留，驻留的那一半永远等不到它们的 arrival。谓词
`occ(l2) < occ(l1) ∧ 最大 stage 的 task 数 > resident(l2)` 在两个模型各 1080
个候选上与实际死锁**逐个吻合**。（F-38）

## 8.8 簇内同步优先

Label 选中的边用 `cluster.sync()` 而非全局自旋：
硬件保证共存，可证明无死锁，且延迟低一个量级。

## 8.9 Reparam 不产生新模板

粒度变化（含 split-K）通过模板实参与 `TaskDesc` 字段表达，不新增 TaskBody。
split-K 用 `k_begin` / `k_count`。

## 8.10 等待提升与本地省略必须与执行窗口一致

见 §5.7.3 L-d。任何增加执行顺序自由度的变更（W > 1、jit/动态发射）都必须同时修改 host 的提升与省略规则，并提供一个负对照：沿用旧规则时必须失败。

## 8.11 调度只能由求解器决定

Codegen 与 host 只消费 Plan（§5.7.4），不得在其中新增调度决策逻辑。现有的两处属于待迁移项（`docs/TODO.md` EX-E1）：

- Codegen 内的 `BuildVariantSchedule`；
- host 端的 balanced 启发式。

---

# 9. 风险与未决问题

## 9.1 风险登记册

| # | 风险 | 严重度 | 应对 |
|---|---|---|---|
| R1 | CUTLASS collective 为独立 GEMM kernel 设计，嵌入任务循环需改造 | 中高 | 参照 MPK 的 `gemm_ws_mpk.cuh` |
| R2 | nvcc + 重模板 CUTLASS 编译慢 | 中 | traits 优先，只对 top-k 真编译；缓存 + 并行 |
| R3 | 簇内通信在实际负载下收益不明 | 中 | 若延迟收益 <2×，Label 降级为可选 |
| R4 | 并发干扰使代价模型不可靠 | 中高 | 偏差 >30% 则退化为「粗排 + 实测 top-3」 |
| R5 | ~~Reparam 的收益（划分优化）可能很小~~ **已排除** | 高 → 无 | P6.2 的 oracle 已做：6.11× / 6.75×。新的风险不是收益小，而是平台期宽（top-34 在 ±10% 内），见 R10 |
| R6 | ISL 对参数化 κ 粗投影表达式爆炸 | 中 | 限制 κ 为 2 的幂或受限矩形代数 |
| R7 | attention TaskBody 的实现工作量 | 中 | 优先复用 CUTLASS FMHA 或 FlashAttention 的 CuTe 实现 |
| R8 | `cutlass_compiler` 的现代架构路径覆盖不足 | 中 | 仅用于分析层的 layout 表示，不用于 codegen |
| R9 | mirage / MPK 许可证限制借鉴范围 | 低 | 只借鉴机制设计，不复制代码 |
| R10 | 代价模型分不出 top-10% 内部的名次（实测两次 25 进程复现选出不同冠军，冠军漂 9.04%） | 中 | 验收口径改为「落进 top 3%」而不是「命中 argmin」；别为不存在的分辨率投入 |
| R11 | 在与 L1 相同的 task 归属下，L2 结构上只能省掉 barrier（F-126） | 高 | 先改 π/σ（EX-D2、EX-E1、EX-S2）；事件原语调优只在该上界内有效 |
| R12 | 参考 fixture 处于纯延迟区，κ、窗口与放置的结论可能不外推到 real-width | 中高 | EX-V1：以 real-width 为主基准，toy 结论注明适用范围 |
| R13 | 静态计划对代价模型误差敏感，误差以 HOL 的形式放大 | 中 | EX-E2 窗口执行器；EX-S4 发射策略 |
| R14 | 执行模拟器的精度不足以对放置排序 | 中 | EX-S1 按排序验收；不足时退化为"模拟粗排 + 实测 top-k" |

## 9.2 需要小实验确认

- [ ] torch.export 对目标模型的覆盖：Llama / Qwen 能否干净导出？
      KV cache 以什么形式出现（mutation? buffer?）
- [ ] paged KV 的布局抵消在 prefix caching 下是否严格成立（Tier 1）
- [ ] CUTLASS FMHA 能否作为 attention TaskBody，还是需要自研
- [ ] `__cluster_dims__` 与 persistent grid 的交互：簇是否限制 grid 上限
- [x] smem union 对 occupancy 的实际影响：**smem 项是真正的绑定项之一**——
      1077 个真实候选里 150 个由它单独决定 occupancy、322 个与寄存器项并列，
      只留寄存器项会在 150 个上算错。闭式在 1077/1077 上与实测相等（F-40，
      §8.6）。⚠️ 只覆盖了本轮的 6 种 task 类型与 256 线程/CTA；
      「task 类型数 3 / 5 / 10」这条消融没有做。
- [ ] 跨 stage 连续轮询放置能否抬高放置吞吐上界（`docs/TODO.md` EX-D2）
- [ ] "CTA 屏障 + thread0 单次 release fence"的正确性（EX-E3 第 4 步）
- [ ] `%globaltimer` 在目标 GPU 上的分辨率（EX-D1）

## 9.3 设计决策待定

- [ ] L3 的 layout 表示：cute MLIR dialect / `pycute` / 自建
- [ ] attention TaskBody 的实现路线
- [ ] 事件缓冲是 device 全局数组还是每 worker 分段
- [ ] 退避默认档位
- [ ] 簇尺寸默认值（可移植上限 8）

---

# 附录 A. 可借鉴实现速查

## A.1 CUTLASS（BSD-3，可直接依赖）

| 需要 | 位置 |
|---|---|
| Persistent tile scheduler（含 Stream-K） | `cutlass/gemm/kernel/sm90_tile_scheduler.hpp` |
| Cluster 启动 / 同步 | `cutlass/cluster_launch.hpp`、`cute/arch/cluster_sm90.hpp` |
| mbarrier 流水线 | `cutlass/pipeline/pipeline.hpp` |
| NamedBarrier | `cutlass/arch/barrier.h` |
| GEMM / Epilogue collective | `cutlass/gemm/collective/*`、`cutlass/epilogue/collective/*` |
| layout 代数（C++） | `cute/layout.hpp`、`cute/algorithm/*` |
| layout 代数（MLIR） | `cutlass_compiler/cute_ir/` |
| Blackwell TMEM | `cute/arch/tmem_allocator_sm100.hpp`、`mma_sm100_umma.hpp` |

## A.2 MPK（机制参照，代码需确认许可证）

| 机制 | 位置 | 说明 |
|---|---|---|
| collective 嵌入任务循环 | `tasks/cute/hopper/gemm_ws_mpk.cuh` | 改造 CUTLASS GEMM 的参照 |
| 同步原语 PTX 参照 | `mpk_atoms.cuh`（97 行） | scope / ordering 的选择 |
| 单调事件计数器 | `persistent_kernel.cuh` | §8.2 |
| 自旋 + 退避 | 同上 | `__nanosleep(10)` |
| 共存性保证 | `__launch_bounds__(N, 1)` + grid=SM 数 | §8.7 |
| 角色分化 | `if (blockIdx.x < num_workers)` | block 级 |
| 紧凑事件表示 | `EventDesc{type, num_triggers, first_task_id, last_task_id}` | fanout 用区间编码 |
| 运行时元数据逃生舱 | `TaskMetadata` union（8 字节） | 一般化为 indptr |
| Serving 集成 | `TASK_SCHD_PREPARE_BATCH` | 请求准入 / 剔除放进 kernel 内 task |
| 资源预算参考 | `runtime_header.h` | 保留静态 SHM 6KB/3KB；MAX 动态 SHM 207KB(B200) |
| JIT/AOT 双队列（stated） | 论文 §5.2 | worker 优先执行已就绪的 JIT task，JIT 队列空时才检查 AOT 队首事件；仅在无就绪工作时阻塞（参照 EX-E2、EX-E5） |
| 跨 task 软件流水（stated） | 论文 §5.3 | task 拆为预取与计算两阶段，配合分页 shared memory（参照 EX-E4） |
| 轮询分发的前提（stated） | 论文 §4.1、§5.2 | 默认每个算子的 task 数与 SM 数成比例，AOT task 轮询分发；TileMega 的小 stage 不满足该前提（F-126） |
| 任务描述预取（stated） | 论文 §5.3 | 预取后续 task 的描述到 shared memory |

## A.3 ETC / Event Tensor（概念参照）

| 机制 | 说明 |
|---|---|
| 极简 lowering | 事件张量 = int 张量，`notify()` = atomic dec，`wait()` = 自旋 |
| 调度即 pass | static（主机预计算队列）/ dynamic（GPU 中心化就绪队列）可替换 |
| 数据相关事件更新 | expert 计数器初值由 runtime `topk` 决定 |
| 数据相关 task 触发 | `exp_indptr` 前缀和决定激活 `[indptr[i], indptr[i+1])` 区间 |
| split-K 事件范例 | `task B̂_{i,j} → E[i]`（wait=4）`→ task Ĉ_i` |
| 静态队列构建（stated） | 论文 §3.1：静态调度以轮询构建每个 SM 的队列 |
| 静态/动态对比（stated） | 论文 §4.5 表 3：Qwen3-32B TP=4 上，静态调度相对"算子间单事件"的 unfused megakernel 为 1.09/1.06/1.07/1.06（batch 1/16/32/128），动态为 0.83/0.82/0.85/0.89；表 2（MoE）中动态在 128–4096 token 上优于静态 |
| 权重预取 pass（stated） | 论文 §3.4：依据用户标注生成权重预取函数；TileMega 可由 CG 按操作数的读关系自动推出（§5.3.1） |
| early push（stated） | 论文附录 E：生产者派发即推送消费者，依赖由消费者侧等待保证 |

## A.4 其他

| 来源 | 借鉴点 |
|---|---|
| Welder (OSDI'23) | 从输出端反向传播 tile 约束；跨内存层级流量代价模型 |
| BladeDISC | Shape Constraint IR：dimension constraint 与 structure constraint |
| Roller (OSDI'22) | 构造而非搜索：从硬件原生 rTile 出发扩张至资源墙 |
| Rammer (OSDI'20) | rTask / rProgram 的编译期 task→执行单元映射 |
| vLLM | `FULL_AND_PIECEWISE` CUDA graph 模式；chunked prefill 默认 chunk=2048 |

---

# 附录 B. 源码位置索引

## B.1 CUTLASS（`third_party/cutlass`，4.8.0 `dc45f979`）

| 内容 | 路径 |
|---|---|
| CuTe layout 代数（C++） | `include/cute/layout.hpp`、`include/cute/algorithm/` |
| CuTe MLIR dialect ops | `cutlass_compiler/cute_ir/include/cute_ir/Dialect/Cute/IR/CuteOps.td`（6571 行） |
| CuTe MLIR dialect types | 同目录 `CuteTypes.td`（716 行） |
| cute pass | `cutlass_compiler/cute_ir/include/cute_ir/Dialect/Cute/Transforms/Passes.td` |
| base pass | `cutlass_compiler/base/include/base/Conversion/BaseToTargets/Passes.td` |
| GEMM collective | `include/cutlass/gemm/collective/` |
| tile scheduler | `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` |
| pipeline | `include/cutlass/pipeline/pipeline.hpp` |
| NamedBarrier | `include/cutlass/arch/barrier.h` |
| cluster | `include/cutlass/cluster_launch.hpp`、`include/cute/arch/cluster_sm90.hpp` |
| 端到端示例 | `examples/`（含 persistent GEMM、CLC scheduler） |

## B.2 mirage / MPK

| 内容 | 路径 |
|---|---|
| CuTe 版 warp-specialized GEMM | `include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh` |
| 同步原语 | `include/mirage/persistent_kernel/mpk_atoms.cuh` |
| 主 runtime | `include/mirage/persistent_kernel/persistent_kernel.cuh` |
| 数据结构 | `include/mirage/persistent_kernel/runtime_header.h` |
| 代码生成器 | `src/kernel/runtime.cc` |

## B.3 论文

| 论文 | 标识 | 关键内容 |
|---|---|---|
| MPK | arXiv 2512.22219 | tGraph；operator decomposition / dependency analysis / event fusion / normalization；in-kernel runtime |
| Event Tensor / ETC | arXiv 2604.13327（MLSys'26） | 三构造；static/dynamic pass；Table 3 |
| Welder | OSDI'23 | tile 传播 + 内存流量代价模型 |
| Roller | OSDI'22 | 构造法 tile 配置 |
| Rammer | OSDI'20 | rTask / rProgram |
| Mirage | OSDI'25 | multi-level superoptimizer |
| MPK（v2） | arXiv 2512.22219v2（2026-06-10）；OSDI 2026（据 mirage 仓库 README，stated） | JIT/AOT 双队列、跨 task 流水、分页 shared memory |
| ETC（补充） | arXiv 2604.13327v2 | 静态队列轮询构建；依据用户标注的权重预取 pass；early push |

---

# 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-08 | v2.0 | 引入 Coupling Graph 作为核心抽象；后端为 CuTe/CUTLASS + nvcc；验证计划拆为独立文档 |
| 2026-09 | v2.1 | 实现状态迁至 `docs/STATUS.md`，待办迁至 `docs/TODO.md`（v2.0 §7 原文归档）；新增 §5.7 执行模型与 Plan 契约；修订 §2.3、§2.6、§4.4、§4.5、§5.1、§5.3–§5.6、§6、§8、§9、附录 A/B |
