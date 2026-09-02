# TileMega 开发骨架

> **定位**：把 megakernel 的任务划分、事件生成、粒度选择与通信归属，
> 表达为**参数化耦合图（Coupling Graph）**上的可解析优化问题。
>
> **后端**：CuTe / CUTLASS + nvcc。
>
> **配套文档**：`docs/VERIFICATION_PLAN.md`（开工前验证计划，独立维护）。
>
> **维护约定**：完成的条目打勾并追加实测结论；推翻的假设保留原文并注明推翻原因。

---

## 目录

- [1. 设计理念](#1-设计理念)
- [1.5 当前状态](#15-当前状态)
- [2. 核心抽象：Coupling Graph](#2-核心抽象coupling-graph)
- [3. 构建基础](#3-构建基础)
- [4. 架构](#4-架构)
- [5. Lowering 路径](#5-lowering-路径)
- [6. 仓库结构](#6-仓库结构)
- [7. 分阶段 TODO](#7-分阶段-todo)
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

| 层 | 路径已验证 | 代码已实现 | 证据 |
|---|---|---|---|
| L5 Serving | — | ❌ | — |
| L4 Frontend | ✅ | ✅ | V-H；结构化 importer：2 层 GQA 为 30 stage，4 层 MHA 为 60 stage |
| L3a 符号类型 | ✅ | ✅ | F-14；`coupling_types_test` / `cg_attr_roundtrip` |
| L3b 耦合推导 | ✅ | ✅ | P3/P3_ISL：`W⁻¹∘R` 为 isl_map，wait/fanout 为 barvinok 计数；§2.7 全 13 行交叉验证（并纠正表中边 3 的 fanout）；Coarsen/I2/事件综合单测 |
| L2 Solver | — | ❌ | Phase 4 |
| L1 Codegen | ✅ | ✅ | E2E_GEN：表驱动 L0.5/L1/L2；2 层 GQA 50/50，4 层 MHA 通过 |
| L0 Backend | ✅ | ✅ | V-I 四架构交叉编译 |

此表是项目实现状态的唯一权威来源；每轮结束随代码和实验证据同步更新。

### 1.5.1 残留技术债（本轮结束时的诚实记录）

- **P3.1 的 CuTe ↔ `isl_map` 往返仍未实现**（求解权威迁移本身已完成）：
  `CuteLayoutBridge::Project` 仍只根据 `LayoutDescriptor` 的静态/动态/
  swizzle 标志选择 `InverseStrategy`（含 Tier 上界），`layout_bridge_test`
  覆盖五个分支；真正的 CuTe layout → `flatten`/`coalesce` → 读 shape/stride
  → `isl_map`（以及求解结果反向回写）没有实现。**注意这一条现在只剩布局桥**：
  §3.1 的求解权威迁移已经做完——`C` 是真正的 `isl_map`，`wait`/`fanout` 是
  barvinok 计数，`Contains` 是 `isl_map_is_subset`，Coarsen 是 isl 复合
  （见 §1.5 状态表与 `docs/experiments/P3_ISL/result.md`）。之所以推导层
  不需要这条布局桥就能工作，是因为 `W` 的逆按每根轴是 `⌊·/g⌋`，
  `DeriveCoupling` 直接把它写成 isl 约束，不经过 CuTe layout 对象。

- **isl 的除数必须是字面常量，这约束了参数何时可以保持符号化**：
  `isl_aff_div` 在 C API 层就拒绝参数化除数（`docs/experiments/P3_ISL/`），
  所以 tile size（`Tm`/`Tn`/`Tkv`）以及任何出现在除数位置的量（GQA 的
  `G`）必须在构造 `isl_map` 之前替换成字面量。实现上这体现为
  `DeriveCoupling(..., ParamBinding const& known, ...)`：`known` 里的符号
  被替换掉，不在 `known` 里的（`S`/`L_s`/`past` 这类工作负载维度）保留为
  真正的 isl 参数，不变量 I1 因此仍然成立。这不是缺陷，是一条必须被遵守
  的边界；`ClosedForm::ToIslText` 在除数没有被替换成常量时会显式抛错，
  而不是生成非法 isl 文本。

- **wait 与 fanout 需要不同的定界处理，这是 barvinok 的一条实测边界**：
  把生产者（range）侧的界折进 `C` 本身，会让带有真正 isl 参数、且生产者
  坐标由不等式区间导出的关系在 *wait* 方向把 barvinok 推进
  `unexpected missing (bounded) solution`（`basis_reduction_tab.c:210`）；
  完全不定界，则 *fanout* 会因为 `isl_map_card` 的分片分解保留一段"只在
  别的参数取值下可达、在当前取值下恒为 0"的尾巴而误报 `min = 0`。现在的
  做法是只把生产者盒子作用在**反向映射**上（`ProducerRangeBoxText` +
  `FanoutCard`），让两个方向各自留在自己可解的区域。这是绕过而不是修复：
  上游若换 isl/barvinok 版本，这条需要重测。

- **L2 的 `notify`/`wait` 目前只有保守的 `kAll` 松弛路径**：
  `CouplingGraphToCUDA::Lower` 为每条跨阶段耦合发出一个 `StageDependency`，
  但恒定标记为 `Map::kAll`（等一整个生产者阶段的到达计数），从不使用
  `Map::kIdentity`。原因记在代码注释里：语义上的恒等 `C`（`blockIdx.x`
  在两个独立实现的 TaskBody 里指同一个 tile）不足以证明 CTA 级别的恒等，
  除非 TaskBody ABI 显式携带 CTA→task 的归属映射，而这个 ABI 扩展还没做。
  这个选择是 I2 安全的（`C' ⊇ C`），但不是 §2.3 意义上最紧的事件；这正是
  L2 中位数比 L1 慢约 1.16×–1.36×（两个模型上的实测，见
  `docs/experiments/E2E_L2/result.md`）而不是更快的原因——事件语义是对的，
  尚未做到真正细粒度。
- **前端 `ModelPlan` 构造器泛化的范围是"decoder-layer 家族"，不是任意
  ATen 图**：`lib/Frontend/ModelPlan.cpp` 按结构匹配
  `layers.N.{input_norm,post_norm,q_proj,k_proj,v_proj,o_proj,gate_proj,
  up_proj,down_proj,inv_freq}` 参数命名和 RMSNorm→QKV→RoPE→KVAppend→
  Attention→O→残差→RMSNorm→SwiGLU→残差 的数据流形状，层数、hidden/
  intermediate 宽度、head/kv_head 比例（GQA vs. MHA）全部从权重形状和
  `inv_freq` 长度推导，不出现在生成代码里（`% 12`、`179`、`222`、
  `GeneratedLlamaRuntime` 等字面量已被移除并有 grep 回归防止复发）。
  但这条规则本身认得的仍是这一个算子家族：一个结构完全不同的模型（例如
  纯 MLP 堆叠、不同的归一化算子）需要在 `ModelPlan.cpp` 里新增一条匹配
  规则，不会自动通过。本轮验收覆盖了该家族内的两个结构差异实例（2 层
  GQA、4 层 MHA），满足任务给出的验收 B 例子，但不等于任意模型都能过。
  Analysis 层的 `MlpStack`/`GatherModel`（`lib/Analysis/ReferenceModels.cpp`）
  证明了耦合推导算法本身不依赖 Llama 结构，但它们没有经过这条 Frontend
  路径进入生成器。
- **L2 Solver、簇内/局部同步、事件粗化 κ 未实现**：按本轮任务范围显式排除
  （`g` 固定、同步全部 `global`、不做性能优化），Phase 4 待启动。
- **分析层的耦合推导尚未接入真实前端路径**（本轮新发现，且与迁移无关——
  迁移前就是如此）：`lib/Frontend/Frontend.cpp` 从 `export_bridge.json`
  构造 CG 时**并不调用** `CouplingDerivation`。它按每个 ATen `call_function`
  建 `task_space`（V-H 模型 179 个），每条张量依赖建一条 `coupling`，其
  `relation` 来自占位的 `fixedRelation()`，`wait`/`fanout`/`volume`/`count`
  一律写 `1`，`tier` 一律写 `0`。也就是说：经过验证的 `W⁻¹∘R` 推导（§2.7
  全表、Tier 分类、事件综合）目前只在 `ReferenceModels.cpp` 的算子级
  `OperatorGraph` 上运行（`tilemega-derive`、`table27_test` 等），**没有**
  参与真实模型的代码生成；生成的 `.cu` 里 L2 的 `StageDependency` 只用到
  "哪两个 stage 之间有耦合"这一结构信息，用不到 wait/fanout 的数值。
  两条路径的粒度也不同（算子级 vs. 每个 ATen 节点级），所以接上去不是改
  一行调用，而是要让前端按算子粒度建 `OperatorNode`——`ModelPlan` 已经
  结构化地识别出了这些算子，是天然的接入点。本轮任务显式把
  `Frontend.cpp` 列为"不动"，故未做；这是 L3b 距离"真正驱动生成的代码"
  之间剩下的一步。

- **Coarsen 的实测边界（P4.6 的 `[!]` 已给出结论，但覆盖有限）**：κ ∈ {1,2,4}
  下关系与拟多项式都保持**单分片**、isl 文本长度基本不随 κ 变化（保留 `S`
  为符号只多约 15 个字符），所以**没有出现表达式爆炸**，κ 不必限制为 2 的幂
  （数据见 `docs/experiments/P3_ISL/result.md`）。⚠️ 这只覆盖了一个 decoder
  层的边、κ ≤ 4、且每次只粗化一根轴；更深的嵌套没有测。

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

## 2.3 CG 上的五种操作

系统的全部决策都是在 CG 上做这五件事：

| 操作 | 定义 | 决策变量 |
|---|---|---|
| **Reparam** 重参数化 | `g → g'`，`C` 的形式不变 | tile 形状、split-K 因子 |
| **Coarsen** 粗化 | `C_κ = ⌊·/κ⌋ ∘ C`，对 image 做粗投影 | 事件粒度 κ |
| **Label** 标注 | 每条边打 `sync_kind ∈ {global, cluster, local}` | 通信归属 |
| **Place** 放置 | `T_op` 的点 → `(worker, slot)` | 分配与顺序 |
| **Relax** 松弛 | `C → C' ⊇ C` | Tier 2/3 的保守化 |

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

## 2.7 Llama decoder layer 的耦合表（L3 验收基准）

`H=4096, n_h=32, n_kv=8, d=128, I=14336, S=token 数, L_s=序列 KV 长度, Tm=Tn=Tkv=128`

**分析层自动推导的结果必须与本表逐项一致。**

| # | 边 | `C`（消费者→生产者） | image 形状 | wait | fanout | Tier | cluster 候选 |
|---|---|---|---|---|---|---|---|
| 1 | RMSNorm1 → Wq/Wk/Wv | `(m,n) ↦ m` | `[⌈S/Tm⌉]` | 1 | 48 | 0 | ✗ fanout 超簇容量 |
| 2 | Wq → RoPE_q | `(m,hh) ↦ (m,hh)` | `[⌈S/Tm⌉×32]` | 1 | 1 | 0 | ✓ 1:1 |
| 3 | RoPE_k → KVappend | `(row,hh) ↦ (⌊row/Tm⌋,hh)` | 同上（k 侧 8 组） | 1 | **Tm = 128** | **1** | ✗ fanout 超簇容量 |
| 4 | KVappend → Attn chunk | 仅 `j = ⌊(L_s−1)/Tkv⌋` | ragged | 1 | 运行时 | **2** | — |
| 5 | RoPE_q → Attn chunk | `(s,hh,j) ↦ (⌊s/Tm⌋, hh)` | ragged 域，映射仿射 | 1 | 运行时 | **2** | — |
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
```

**属性**：`AccessMapAttr`、`CouplingMapAttr`、`ClosedFormAttr`（派生量的符号表达式）、
`TierAttr`、`SyncKindAttr`、`PlacementAttr`。

这些属性与 L3a C++ 类型一一对应：`ClosedFormAttr` 的 storage key 是
`analysis::ClosedForm` 本体，`CouplingMapAttr` 是包含 consumer/producers/ranges/
parameters/fiber/image 的结构化字典；Tier（可解析性）与 SyncKind（通信归属）分离。

**verifier**：事件张量形状 = `image(C_κ)`；`wait` 的闭式在 `θ` 全部代入后与
barvinok 计数一致。

## 4.4 求解流程

```
层1  合法性剪枝：从 CUTLASS TiledMma 的原生形状出发沿各维扩张，撞资源墙停
     ← 走编译期 traits，多数候选不编译
     目标：每算子 8~20 候选
层2  跨算子对齐传播：wait = ⌈(m+1)Tm/Tr⌉ − ⌊mTm/Tr⌋
     Tr = Tm 时 wait = 1；不整除时膨胀且变分段 → 联合空间大幅塌缩
层3  链上 DP（图深而窄，接近阶段链）：
     DP[i][g] = min_{g'} { DP[i−1][g'] + Cost_i(g) + Interface(g',g) }
层4  Label：带尺寸约束的图划分，最大化簇内 volume
层5  Place：分层 DAG 上的 list scheduling（关键路径优先）
```

**代价函数**（每一项都是 CG 派生量的函数）：

```
Cost(g, κ, label) = max( T_compute(count, g), T_memory(volume) )       roofline
                  + T_sync( |image(C_κ)|, label )                      事件数 × 延迟
                  + T_quant(count)                                     wave quantization
                  + T_bubble(g, label)                                 流水气泡 + smem 占用延长

T_quant = ( ⌈count/M⌉·M − count ) / count · t_task(g)      M = SM 数 × occupancy
```

**分 regime**：

| Regime | 瓶颈 | 目标 |
|---|---|---|
| A. 低 batch decode | 权重带宽 | 最小化权重预取管道气泡（非 makespan） |
| B. prefill / 大 batch | 计算 | makespan + wave quantization |
| C. 混合 batch（chunked prefill） | 异构 | 资源互补性最大化 |

**输出形态**：参数化的最优解区间划分（「`S ∈ [0,512)` 用 `g₁`；`S ∈ [512,∞)` 用 `g₂`」），
区间边界来自分段拟多项式的交点。运行时 `O(1)` 查表选变体。

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

  for (int layer = 0; layer < p.num_layers; ++layer) {
    Params lp = p.for_layer(layer);              // 权重指针按层偏移

    #pragma unroll 1
    for (int s = 0; s < kNumStages; ++s) {       // stage 编译期展开（约 17 个）
      TaskSlot const& slot = p.schedule[worker][s];

      for (int t = slot.begin; t < slot.end; ++t) {
        TaskDesc const& d = p.tasks[t];
        wait_deps(d, p.events, layer);
        switch (d.type) {
          case TASK_NORM: T_norm{}(d, smem, lp); break;
          case TASK_QKV:  T_qkv {}(d, smem, lp); break;
          case TASK_ATTN: T_attn{}(d, smem, lp); break;
        }
        notify_deps(d, p.events, layer);
      }
    }
  }
}
```

代码体积 = `O(stage 数 × 粒度变体数)`，不随 task 数增长。

静态调度表在语义上只有一份，但 L0.5 的 host launch loop 与 L1 的 device loop
处于不同 CUDA 地址空间。Codegen 必须从同一个 initializer 同时生成 host
`constexpr` 表与 device `__constant__` 表，禁止手写两份；否则前者不能被 device
读取，或两级正确性阶梯可能发生调度漂移。（F-21）

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

**依赖处理**：

- CUTLASS 只用 header，不引入 `python/CuTeDSL/`
- 是否引入 `cutlass_compiler/` 在 Phase 3 决定；若不引入，
  L3 的 layout 表示用 `pycute` 或自建轻量等价物
- barvinok 优先用发行包（内含匹配 isl）

---

# 7. 分阶段 TODO

> `[ ]` 待办 `[~]` 进行中 `[x]` 完成 `[!]` 阻塞 `[-]` 已放弃（保留并注明原因）
>
> **前置**：`docs/VERIFICATION_PLAN.md` 的项目先完成。

---

## Phase 0：基础设施

### P0.1 仓库与依赖

- [x] 建骨架，按 §6 结构。默认 MLIR=ON 的 CMake + Ninja 构建通过。
- [x] submodule：`cutlass`（跟 main）、`barvinok`。
- [-] mirage / MPK 代码未引入；V-B 仅依据 BSD-3 CUTLASS 公共接口实现 adapter，
      因而本阶段无第三方代码许可证依赖。
- [x] CI：`ninja && ninja check-tilemega`，含 policy、unit 与 CG lit verifier 测试。

### P0.2 工具链

- [x] `tools/tilemega-opt`：注册 CG dialect；合法 round-trip 与两个负 verifier 用例通过。
- [x] `tools/tilemega-compile`：stable export JSON / CG MLIR → CG `ModuleOp` →
      `.cu`，并可一条命令调用 nvcc 生成 `.so`；E2E_GEN 同时保留可执行验收路径，
      用于逐元素与资源对照。
- [x] `BackendCostQuery`：traits 路径 + nvcc/ptxas 路径。V-D shared storage 误差 0，
      V-E 给出真编译与并行基线。

### P0.3 测试基础设施（优先级高于任何功能代码）

- [x] **差分测试框架**：L0/L0.5/L1 逐元素比对
      （容差：浮点求和顺序差异，相对误差 3e-5 量级属正常）
- [x] **挂起检测**：所有 GPU 测试带 `timeout`，挂起保留现场。
- [x] **统计化执行**：涉及同步的测试 ≥50 次全新进程并报通过率。
- [x] **交替填充校验器**：每次运行更换期望值，防止残留掩盖错误。
- [x] `ptxas -v` 解析器（REG / SHM / 溢出）。
- [x] SASS dump 脚本（回边、屏障、`__nanosleep`）。
- [x] 挂起现场分析：hang probe 多次 PC 采样区分推进与停滞；见 V-A/V-J。

---

## Phase 1：前端与 CG 骨架

### P1.1 torch.export 接入

- [x] 跑通严格模式两层 Llama decoder 的 `export`，三次图稳定（V-H）。
- [x] 提取 FX graph、`ShapeEnv` 符号维、FakeTensor meta；Python bridge 只做稳定序列化。
- [x] 30 个 target 白名单；C++ importer 对白名单外算子报告完整算子名。

### P1.2 符号形状桥（θ）

- [x] 符号维算术 → `ClosedForm`（symbol/add/mul/ceildiv/floordiv）；ISL 转换按计划留给 Phase 3。
- [x] `ShapeEnv` range/equality guards → 最小约束域并规范化符号同一性；
      torch 2.13 私有 accessor 集中在版本锁定 adapter。
- [x] 单元测试：4 guard 保留，`s61/s65 → s14`，view/transpose access map 保留。

### P1.3 CG dialect

- [x] `task_space` / `coupling` / `event_tensor` / `placement` op（§4.3）。
- [x] `AccessMapAttr` / `CouplingMapAttr` / `ClosedFormAttr` / `TierAttr` /
      `SyncKindAttr`，以及结构化 `placement` op
- [x] verifier：事件张量形状 = `image(C_κ)`；`wait` 与结构化 relation fiber
      在 θ/g 绑定后的值一致；Tier 3 + cluster 被拒绝。barvinok authority 留给 Phase 3。

### P1.4 FX graph → CG 骨架

- [x] 每个白名单 ATen `call_function` → 一个 `task_space`（固定 `g`）。
- [x] 每个张量依赖 → 一条带结构化固定规则 `C` 的 `coupling`；不冒充 Phase 3 推导。
- [x] 显式两层 Llama stage 规则：179 task / 222 coupling → 24 stage，L1 为层循环。
      **已被 P3 的 `ModelPlan` 结构化构造器取代**（见 P3.5 之后的 Phase 2/3
      generalization 记录与 §1.5.1）：层数、宽度、GQA/MHA 比例改为从
      `layers.N.*` 参数形状结构化推导，不再是写死的两层规则；这里保留
      条目是给 Phase 1 里程碑的历史记录，不代表当前生成路径。
- [x] lit + unit：合法/事件形状/Tier-sync、白名单错误、layout task 保留。

---

## Phase 2：朴素端到端（正确性优先，不做性能优化）

### P2.1 TaskBody 模板库

- [x] `TaskBase.h`：`TaskDesc` / 指针参数契约 / smem union 机制。
- [x] `ElementwiseTaskBody`、`RMSNormTaskBody`，编译期 traits 可查询。
- [x] `GemmTaskBody`：SM80 cp.async direct collective specialization；逻辑 stride 契约显式化。
- [x] `RoPETaskBody`、`KVAppendTaskBody`。
- [x] `AttentionChunkTaskBody` / `AttentionCombineTaskBody`。
      六类 body 的 union 静态断言为 `max_i(kSmemBytes_i)`；V-I 四架构仍全过。

### P2.2 L0 参考实现

- [x] PyTorch eager 逐算子，固定 seed 20260901，导出参考输出（V-H/E2E）。

### P2.3 L0.5 host 端 stage 循环

- [x] 生成版每 stage 一个独立 kernel，内含 grid-stride 任务循环。
- [x] 生成 Host launcher 按 24 个 stage 顺序 launch。
- [x] 权重无关的小配置两层 Llama（GQA/RoPE/KV/SwiGLU）端到端；
      生成 L0.5 与手写参照位哈希一致，并以 3e-5 容差匹配 PyTorch L0。

### P2.4 L1 单 kernel megakernel

- [x] 全局 barrier lowering（按 §8 规则，单线程轮询/退避/fence/CTA release）。
- [x] stage 编译期展开 + 层循环（§5.4）。
- [x] task dispatch switch + 单个显式 smem union；参数表按设备指针传入。
- [x] 生成 L1 与生成 L0.5 逐位一致，50/50 全新进程、0 timeout。
- [x] sm_89：REG 168 / SHM 49536B / occupancy 1 CTA·SM⁻¹ / grid 128；
      grid 由 `TargetSpec::Probe()` 与 occupancy API 得到。见 `docs/experiments/E2E_GEN/`。

---

## Phase 3：分析层（CG 的填充）

### P3.1 CuTe ↔ ISL 桥

- [x] isl/barvinok 依赖可行性：`docs/DEPENDENCIES.md` + `docs/experiments/P3_ISL/`。
      isl 0.28/polylib 5.22.9/barvinok 0.41.9 从匹配子模块构建（GMP 后端，
      因为 polylib 硬依赖 GMP，且已证实 MLIR 不链接 GMP、无需 imath 规避
      冲突）；`cmake/ISL.cmake` + `-DTILEMEGA_ENABLE_ISL=ON` 接入构建；
      `ctest -R isl_crosslink` 证实与 MLIR 自带 Presburger 同进程无冲突。
      同时确认 `isl_aff_div` 除数必须是字面常量——`g` 必须在构造 isl 对象
      前替换为具体值，只有 `theta` 留作 isl 参数，与 CuTe `RightInverse`
      的既有结论一致。**这一条只是依赖就绪，不是 `ISLContext` 封装本身。**
- [x] `ISLContext`：生命周期、错误处理（C API + 自建 RAII，不用
      `isl-noexceptions.h`）。`include/tilemega/Analysis/ISLContext.h` 拥有
      `isl_ctx`，`lib/Analysis/IslUtil.h` 是 isl_map/isl_set/
      isl_pw_qpolynomial/isl_val 的手写 RAII 模板（`_copy`/`_free` 对）。
- [ ] CuTe layout → `isl_map`：先 `flatten` / `coalesce`，再读 shape/stride
- [ ] `isl_map` → CuTe layout（求解结果回写）
- [ ] 单元测试：一组 layout 的往返等价性
- [x] 三级逆策略机器化：静态 `g` 走 CuTe `RightInverse`；动态 extent +
      常量 stride 走 Presburger relation；动态 stride/swizzle 明确提升 Tier。
      `layout_bridge_test` 覆盖五种分支；完整 `isl_map` 往返仍未实现。

### P3.2 访问关系构造（W / R）

- [x] `W_op`：输出 tile → 结构化 `AccessRelation`，保留 origin/runtime/layout
- [x] `R_op`：逐算子类别实现
      （pointwise / reduction / matmul / broadcast / concat / slice / transpose）
- [ ] Tier 0 对齐静态情形：走纯 CuTe 路径，验证与 ISL 路径结果一致

### P3.3 耦合推导（C）与派生量

- [x] `C = W⁻¹ ∘ R`
      （CuTe：`composition(right_inverse(W), R)`；ISL：`apply_range ∘ reverse`）
- [x] `wait` / `fanout` / `volume` / `count` 改为 barvinok 计数，类型是
      `QuasiPolynomial`（`isl_pw_qpolynomial`）：`wait = isl_map_card(C)`，
      `fanout = card(C⁻¹)`。已构造出真正需要分段拟多项式的用例（错位 tile：
      `wait(r)` 在 2 与 3 之间按周期变化，`ClosedForm` 的文法无法表达），
      见 `docs/experiments/P3_ISL/result.md` 与 `MisalignedTileModel`。
- [x] **验收：§2.7 的 13 条边全部自动推出**，并已用 isl 路径整表重推交叉
      验证：除边 3 的 fanout 外逐项一致，而边 3 是**原表算错了**（见 §2.7
      表下的修正说明），不是迁就实现改期望；见 `docs/experiments/P3/table27.md`
      与 `docs/experiments/P3_ISL/result.md`
- [x] 错位 tile 的两侧重叠条件改为精确推导（此前只能松弛）：生产者块 p 与
      读区间重叠当且仅当 `p·tile < base+span` 且 `base < p·tile+tile`，是仿射
      条件，isl 可直接承载。这类边因此从 Tier 2 松弛回到 Tier 0 精确。

### P3.4 Tier 分类与松弛

- [x] Tier 0：直接推
- [x] Tier 1：布局抵消（识别生产者与消费者共享单射布局）
      - `[!]` 待确认：prefix caching + CoW 下 block 共享是否破坏单射性
- [x] Tier 2：结构化 ragged/runtime task space 显式分类并保留 guard
- [x] Tier 3：数据依赖索引退化为算子级事件，不伪造 affine inverse
- [x] 松弛正确性检查：`Contains(C', C)` 现在就是 `isl_map_is_subset`，
      不再是手写的结构覆盖判断；未知返回“未证实”而非猜测

### P3.5 L2 落地

- [x] global 事件张量 → `(producer stage, producer CTA)` device 数组（128B padding）
- [x] global `notify` / `wait` 按 §8 release/acquire 顺序生成；cluster/local 路径待硬件验收
- [ ] 单调计数器：`needed = num_triggers × iteration_num`
- [x] 与 L1 逐位比对：50/50 全新进程，0 mismatch / 0 timeout（2 层 GQA）；
      4 层 MHA 单进程逐位一致（`docs/experiments/P3_GENERALIZATION/`）
- [x] L2 vs L1：保守 I2 松弛，中位数 2 层 GQA `1.182×`、4 层 MHA `1.355×`；
      事件表达正确但尚未优化（TaskBody ABI 尚缺 CTA→task ownership map，
      不能仅由语义 identity C 推断 block identity，见 §1.5.1）。数据见
      `docs/experiments/E2E_L2/result.md`

### P3.6 生成器一般化（去掉 Llama 结构写死）

- [x] `TaskBodyEmitter::Emit` 不再检查 `stage % 12`/六族齐全，也不再
      `#include` 手写的 `GeneratedLlamaRuntime.cuh`（该文件已删除）；
      只 `#include <tilemega/Codegen/tasks/ModelHarness.cuh>`，一个
      model-independent 运行时，模型数据全部通过生成的 `ModelSpec` 表进入。
- [x] `lib/Frontend/ModelPlan.cpp` 结构化构造 `ModelDims`/`BufferDesc`/
      `GemmDesc`/`StageDesc`/`OutputDesc`/`StageDependency` 表并作为
      `tilemega.model_plan` 模块属性挂在 CG 上；`CouplingGraphToCUDA::Lower`
      只从这个已验证属性读取，不再解析裸 JSON 或做结构假设。
- [x] `ScheduleTableEmitter::EmitStageCounts` 的 `stage % 12` 占位符改为
      `stage`（stage id 本身），任务/耦合计数不再作为编译期宏写入生成源。
- [x] CI 回归：`docs/experiments/P3_GENERALIZATION/run.sh` 对生成的 `.cu`
      grep `% 12` / `TILEMEGA_GENERATED_TASK_COUNT 179` /
      `TILEMEGA_GENERATED_COUPLING_COUNT 222` / `GeneratedLlamaRuntime`，
      全部不命中才算通过。
- [x] **验收 B**：两个结构不同的模型端到端通过，且都不是靠 `#include`
      一个手写文件满足的——2 层 GQA（179 task/222 coupling/24 stage，
      `docs/experiments/E2E_GEN/`）与 4 层 MHA（355 task/444 coupling/60
      stage/11 guard，`kv_heads == heads` 因此没有 GQA 分组，
      `docs/experiments/P3_GENERALIZATION/`）各自独立导出、独立生成、各自
      对自己的 PyTorch L0 验证，生成的 `.cu` 里没有模型结构常量（见上）。
- [!] 一般化的范围是 decoder-layer 家族（RMSNorm→QKV→RoPE→KVAppend→
      Attention→O→残差→RMSNorm→SwiGLU→残差），不是任意 ATen 图；层数/
      宽度/GQA-MHA 比例从权重形状结构化推导，但换一个不匹配这个数据流
      形状的模型（例如纯 MLP 堆叠）需要在 `ModelPlan.cpp` 里新增匹配规则。
      详见 §1.5.1。

### P3.7 求解权威迁移到 isl/barvinok（原则三的落地）

骨架**原则三**是"CuTe 是表示，ISL 是求解器"。此前关系代数、基数计数、包含
判定全部由自建的 `ClosedForm`/`AffineRelation` 承担——那是把求解权威放错了
地方，也是 Coarsen 无法实现的直接原因。

- [x] `C` 的表示换成 `isl_map`（`CouplingRelation`），`C = W⁻¹∘R` 由
      `isl_map_apply_range` / `isl_map_reverse` 给出。
- [x] `wait`/`fanout`/`volume`/`count` 换成 `isl_pw_qpolynomial`
      （`QuasiPolynomial`），由 barvinok 计数。
- [x] `Contains` 换成 `isl_map_is_subset`。
- [x] **Coarsen（`C_κ = ⌊·/κ⌋ ∘ C`）现在可以实现**——`AffineRelation` 连
      image/preimage/复合算子都没有，这是本次迁移最直接的收益。κ ∈ {1,2,4}
      验证 `wait` 精确按 κ 缩小，并断言两条代数律（κ=1 是恒等、
      `⌊⌊·/2⌋/2⌋ = ⌊·/4⌋`）。
- [x] `ClosedForm`/`AffineRelation` 作为**求解**表示已删除：`AffineRelation`
      类连同 `ProducerMap`/`AffineRange`/`StructureKey`/`SameStructure`/
      `PartitionRange` 一并移除，避免留下第二套语义权威。仍保留的
      `AffineExpr` 与 `ClosedForm` 只是符号算术的构造件（张量 extent、tile
      形状、访问基址系数），它们从不计算关系、基数或包含关系。
- [x] IR 载体改为 ISL 规范文本：`ClosedFormAttr` → `MetricAttr`
      （`isl_pw_qpolynomial` 文本），`CouplingMapAttr` 的 `DictionaryAttr`
      → `isl_map` 文本（isl 语法本身就是 schema，解析期即校验）。
- [x] CG dialect 链接 isl：`CouplingOp::verify` 现在**从关系本身推出**
      `wait` 再与存储值比较，且是按函数比较而非塌成标量比较——位置相关的
      `wait` 必须逐点相等。
- [x] isl 不再可选：`TILEMEGA_ENABLE_ISL=OFF` 直接配置失败，与既有的
      `TILEMEGA_ENABLE_MLIR` 处理一致。
- [x] 端到端不回归：2 层 GQA 与 4 层 MHA 的 L0.5/L1/L2 位哈希与迁移前完全
      相同（`5245714bc5d3ab4d` / `fd15fa2e89cdb915`），50/50 全新进程通过。
- [!] 事件张量 extent 的 verifier 交叉检查退回为"能否求值"：从 `C` 反推
      `image(C_κ)` 需要逐维回答"生产者坐标是否真的依赖这个消费者坐标"，而
      isl 唯一可用的查询（`isl_map_involves_dims`）是语法性的，会把"只是给
      域定界"的坐标也算作相关，从而高估 image。推导侧改为在构造时记录
      哪条约束引用了哪个坐标（`CouplingDetail::occurring`），verifier 拿不到
      这个上下文，故不做该项交叉推导——这是有意不上线一个不可靠的强检查。

---

## Phase 4：求解层（CG 上的优化）

### P4.1 代价查询接口

- [ ] traits 路径：`constexpr` 探针批量求值
- [ ] nvcc 路径：`--ptxas-options=-v` 解析
- [ ] 缓存与批量

### P4.2 层1 合法性剪枝

- [ ] 从 CUTLASS `TiledMma` 原生形状沿各维扩张，撞资源墙停
- [ ] 目标：每算子 8~20 候选
- [ ] 自检：候选集应覆盖 CUTLASS 官方 GEMM 配置常用的 tile

### P4.3 层2 对齐传播

- [ ] 从输出端反向传播 tile 约束
- [ ] wait 膨胀：`⌈(m+1)Tm/Tr⌉ − ⌊mTm/Tr⌋`
- [ ] 度量剪枝效果（联合空间塌缩比例）

### P4.4 代价模型

- [ ] roofline：`max(T_compute, T_memory)`
- [ ] `T_quant`：`count` 用 barvinok
- [ ] `T_sync`：`|image(C_κ)|` × 延迟；global / cluster / local 用不同常数
- [ ] `T_bubble`：软件流水气泡 + Label 带来的 smem 占用延长
- [ ] **离线标定**（测硬件常数）：单 task 时长 vs tile 形状；
      `atomicAdd` / `cluster.sync()` / `NamedBarrier` 的延迟与争用曲线；并发干扰系数
      - `[!]` 干扰偏差 >30% 则独立时长假设失效，退化为「粗排 + 实测 top-3」
- [ ] regime 判别：从 (batch, seq_len, prefill/decode 比例) 判 A/B/C

### P4.5 层3 链上 DP（Reparam + Coarsen）

- [ ] `DP[i][g] = min_{g'} { DP[i−1][g'] + Cost_i(g) + Interface(g',g) }`
- [ ] 分叉处（gate/up 并行）：series-parallel 分解或强制同 tile
- [ ] 复杂度目标 `O(层数 × |C|²)`

### P4.6 Coarsen（事件粒度 κ）

- [ ] 实现 `C_κ = ⌊·/κ⌋ ∘ C`
- [ ] `[!]` 待验证：ISL 对含参数化整除的映射是否表达式爆炸。
      兜底：限制 κ 为 2 的幂，或用受限矩形代数
- [ ] κ 消融曲线；曲线平坦则该维度无收益
- [ ] 纳入 DP 状态

### P4.7 层4 Label（簇划分）

- [ ] 通信量矩阵：逐边的 `volume`
- [ ] 带尺寸约束（≤8 或 ≤16）的图划分，最大化簇内 volume
- [ ] 时间邻近约束：生产者与消费者在相近的 stage slot
- [ ] smem 占用延长计入 `T_bubble`
- [ ] 消融：Label 开 / 关的端到端对比

### P4.8 层5 Place

- [ ] 分层 DAG 上的 list scheduling（关键路径优先）
- [ ] 掩盖同步延迟：队列顺序让等待被独立 task 填充
- [ ] 时间局部性：`|R(c₁) ∩ R(c₂)|` 用 barvinok
- [ ] oracle 判断投入价值：round-robin vs 强启发式 vs oracle，差距 <2% 则简化

---

## Phase 5：符号化与运行时

### P5.1 参数化解

- [ ] 代价函数以 `θ` 为参数 → DP 输出分段拟多项式
- [ ] 求分段交点 → 最优解的区间划分
- [ ] 每区间生成一套模板实例化（受 smem union 与编译时间约束）

### P5.2 运行时选择

- [ ] Host launcher：代入实际 `θ` → `O(1)` 区间查表 → 选 kernel
- [ ] 变体数量上限控制

### P5.3 Tier 2 运行时支持

- [ ] indptr 前缀和（kernel 内或 host 预计算）
- [ ] 事件张量 extent 与 `wait` 的运行时填充

### P5.4 尾 wave

- [ ] Stream-K 式归约维切分：`C` 从 `m ↦ {m}` 变 `m ↦ {(m,0..K)}`
- [ ] 复用 CUTLASS 的 Stream-K tile scheduler 作为参照实现

---

## Phase 6：Serving 集成与评估

### P6.1 L5 Serving harness

按 §4.6 实现：paged KV（Tier 1）、continuous batching（kernel 内调度 task）、
chunked prefill（Tier 2，regime C 的载体）、MoE 路由（Tier 3 的 indptr 一般化）。

### P6.2 评估

- [ ] **bucketing 损失曲线**：扫 batch = 1..128，「逐形状最优」vs
      「power-of-two 向上取整复用」。不依赖代价模型精度，纯结构性损失
- [ ] **划分 oracle**：固定 Place 与事件结构，穷举 `g`，看最优 vs 启发式常数
- [ ] Coarsen（κ）消融
- [ ] Label（簇）消融
- [ ] 混合 batch regime 对比
- [ ] Warmup 时间（目标：0 次 CUDA graph capture）
- [ ] 端到端对比

### P6.3 消融

- [ ] L1 → L2（细粒度事件的贡献）
- [ ] L2 → L3（Reparam + Coarsen + Label 的贡献）
- [ ] 符号化的贡献（vs bucketing）

---

# 8. Codegen 规则

> 硬性规则。所有生成同步代码的路径必须走统一封装。

## 8.1 自旋等待必须单线程轮询

```cpp
if (threadIdx.x == 0) {
    while (atomicAdd(&ev[e].v, 0ull) < need) __nanosleep(64);
}
__syncthreads();     // 集体同步放在非发散点
__threadfence();     // acquire
```

所有线程各自轮询本身是良性的，但会增加原子流量；本实验的短等待 workload
未测出稳定性能代价。真正的正确性禁令是：**集体屏障不能出现在发散的自旋循环体内**。
`barrier_in_spin` 在 grid 64/128/256 共 150/150 挂起。屏障只能位于上述轮询结束后的
非发散点。（F-2）

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

## 8.6 smem union 取 max

各 task 类型的 `SharedStorage` 必须组成**单个显式 union**，且该 union 的生命周期
覆盖整个 dispatch，容量取 `max_i(sizeof(SharedStorage_i))`。不同 task 类型的 smem
只有在此条件下不相加；分离对象的地址逃逸会使生命周期重叠，实测退化为 36864B，
occupancy 从 6 降到 2 CTA/SM。（F-8）

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

## 8.8 簇内同步优先

Label 选中的边用 `cluster.sync()` 而非全局自旋：
硬件保证共存，可证明无死锁，且延迟低一个量级。

## 8.9 Reparam 不产生新模板

粒度变化（含 split-K）通过模板实参与 `TaskDesc` 字段表达，不新增 TaskBody。
split-K 用 `k_begin` / `k_count`。

---

# 9. 风险与未决问题

## 9.1 风险登记册

| # | 风险 | 严重度 | 应对 |
|---|---|---|---|
| R1 | CUTLASS collective 为独立 GEMM kernel 设计，嵌入任务循环需改造 | 中高 | 参照 MPK 的 `gemm_ws_mpk.cuh` |
| R2 | nvcc + 重模板 CUTLASS 编译慢 | 中 | traits 优先，只对 top-k 真编译；缓存 + 并行 |
| R3 | 簇内通信在实际负载下收益不明 | 中 | 若延迟收益 <2×，Label 降级为可选 |
| R4 | 并发干扰使代价模型不可靠 | 中高 | 偏差 >30% 则退化为「粗排 + 实测 top-3」 |
| R5 | Reparam 的收益（划分优化）可能很小 | 高 | 尽早做 P6.2 的划分 oracle |
| R6 | ISL 对参数化 κ 粗投影表达式爆炸 | 中 | 限制 κ 为 2 的幂或受限矩形代数 |
| R7 | attention TaskBody 的实现工作量 | 中 | 优先复用 CUTLASS FMHA 或 FlashAttention 的 CuTe 实现 |
| R8 | `cutlass_compiler` 的现代架构路径覆盖不足 | 中 | 仅用于分析层的 layout 表示，不用于 codegen |
| R9 | mirage / MPK 许可证限制借鉴范围 | 低 | 只借鉴机制设计，不复制代码 |

## 9.2 需要小实验确认

- [ ] torch.export 对目标模型的覆盖：Llama / Qwen 能否干净导出？
      KV cache 以什么形式出现（mutation? buffer?）
- [ ] paged KV 的布局抵消在 prefix caching 下是否严格成立（Tier 1）
- [ ] CUTLASS FMHA 能否作为 attention TaskBody，还是需要自研
- [ ] `__cluster_dims__` 与 persistent grid 的交互：簇是否限制 grid 上限
- [ ] smem union 对 occupancy 的实际影响：task 类型数 3 / 5 / 10 时的 REG / SHM

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

## A.3 ETC / Event Tensor（概念参照）

| 机制 | 说明 |
|---|---|
| 极简 lowering | 事件张量 = int 张量，`notify()` = atomic dec，`wait()` = 自旋 |
| 调度即 pass | static（主机预计算队列）/ dynamic（GPU 中心化就绪队列）可替换 |
| 数据相关事件更新 | expert 计数器初值由 runtime `topk` 决定 |
| 数据相关 task 触发 | `exp_indptr` 前缀和决定激活 `[indptr[i], indptr[i+1])` 区间 |
| split-K 事件范例 | `task B̂_{i,j} → E[i]`（wait=4）`→ task Ĉ_i` |

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

---

# 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-08 | v2.0 | 引入 Coupling Graph 作为核心抽象；后端为 CuTe/CUTLASS + nvcc；验证计划拆为独立文档 |
