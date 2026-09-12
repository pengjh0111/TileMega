# TileMega 实现状态

> **文档地图（v2.1）**：设计与契约 → `TileMega_skeleton.md`；实现状态 → `docs/STATUS.md`；待办 → `docs/TODO.md`；实测发现 → `docs/FINDINGS.md`；v2.0 待办原文 → `docs/archive/TODO_v2.0.md`；开工前验证计划 → `docs/VERIFICATION_PLAN.md`。每类信息只有一个权威位置，其他位置只放指针。
>
> 本文件是实现状态的唯一权威来源。v2.1 起由 `TileMega_skeleton.md` 迁入：§1.5、§1.5.1 为 v2.0 原文（逐字保留），§1.5.4 为原 skeleton §2.3 的接入状态表原文；原文中的修正一律以"（⚠️ v2.1：……）"注记或 §1.5.2 之后的新增小节表达，不改原句。

## 1.5 当前状态

| 层 | 路径已验证 | 代码已实现 | 证据 |
|---|---|---|---|
| L5 Serving | — | ❌ | — |
| L4 Frontend | ✅ | ✅ | V-H；结构化 importer：2 层 GQA 为 30 stage，4 层 MHA 为 60 stage；FakeTensor dtype 进入 L-sem 与 `ModelSpec`，FP32/BF16 均有端到端证据（`docs/experiments/BF16/`） |
| 真实模型尺寸 | ⚠️ | ✅ | 16×2048 的 973M 参数 decoder 已生成、编译并运行，但固定 BF16 判据下 0/50；4×4096 / intermediate=14336 的真实宽度控制 50/50。失败从 8 层开始且 L0.5/L1/L2 逐位一致，是跨层数值累积而非同步错误（`REALMODEL`） |
| 多迭代 / 自回归 | ⚠️ | ⚠️ | 相同 workload 的 32 次 host launch 为两模型各 50/50；反面构建也 50/50，证明当前串行 launch harness 无法触发 ABA。增长 KV cache 的 device-side 迭代仍属于 Phase 6，不能把这项记成已验证（`AUTOREGRESSIVE`） |
| L3a 符号类型 | ✅ | ✅ | F-14；`coupling_types_test` / `cg_attr_roundtrip` |
| L3b 派生量参数化 | ✅ | ✅ | `wait`/`fanout`/`volume`/`count` 是 `S`/`past`/`L_s` 的拟多项式；符号求值与逐点重推 **420/420** 一致（`docs/experiments/SYMBOLIC/`） |
| L3b 耦合推导 | ✅ | ✅ | P3/P3_ISL：`W⁻¹∘R` 为 isl_map，wait/fanout 为 barvinok 计数；§2.7 全 13 行交叉验证（并纠正表中边 3 的 fanout）；Coarsen/I2/事件综合单测。**已驱动生产路径**：`Frontend.cpp` 按算子粒度建 `OperatorGraph` 并调 `CouplingDerivation`，`wait_map` 落到 IR、经 codegen 成为 `StageDependency`（gqa2 38 对：20 `kAll` / 3 `kIdentity` / 15 `kWindow`，`docs/experiments/WIRING/`） |
| L2 Solver | ✅ | ⚠️ | FP32：ρ = 0.9432 / 0.9421、top-3 落入实测 top-3%（`COST_MODEL`）。BF16 在修正 occupancy 特征后为 **0.8984 / 0.8871**，仍低于 FP32，top-1/3/10 仍为 0；误判集中在窄 N + split=1 与激进 split=8/16，阈值未放宽（`BF16`）。链上 DP、Label、Coarsen 见各实验；Place 的关键路径排序现已实际进入 L2 队列，不再只是求出未消费 （⚠️ v2.1：目标函数为 L1，Place 仅为 stage 置换，见 §1.5.2 G1/G8） |
| L2 执行模型 | ✅ | ✅ | 每变体携带 solver schedule；host 物化每 worker 的 `TaskRef` 队列，逐 task 等待/执行/通知，task 内 `(producer,group)` 去重、单调 epoch 提升、CTA 并行 poll。`kAll` 用聚合完成事件，窗口用逻辑 task 组，只发布被引用的行。两模型 50/50，seq×past **1500/1500**；50 进程跨 stage 提前启动平均 34.36% / 40.18%。生成期与 launch 前双重环检查，当前 resident grid 通过、不可证明的 over-resident 被拒绝（`TASKQUEUE` / `OVERLAP`） （⚠️ v2.1：正确性成立；执行能力受限——归属与 L1 相同、队列 stage-major、W=1，见 §1.5.2 G2/G3） |
| L1 Codegen | ✅ | ✅ | 单二进制最多 16 个生成器控制的粒度变体，运行时 O(1) 选取；每变体携带自己的精确依赖表与调度顺序。窗口由 isl 端点发现并做符号集合等价证明，两个参考模型 128/128 条边零 fallback 且生成结果逐字节不变；无 occupancy 损失的实测上限为 2 变体（`VARIANT` / `REALMODEL`） |
| L0 Backend | ✅ | ✅ | V-I 四架构交叉编译；FP32 SIMT 与 BF16 Tensor Core 均由 `ArchDispatch::Caps` 分发，BF16 二进制各确认 96 条 `HMMA.16816.F32.BF16` 静态指令 |

此表是项目实现状态的唯一权威来源；每轮结束随代码和实验证据同步更新。

### 1.5.1 残留技术债（本轮结束时的诚实记录）

- **CuTe 桥接的是 `LayoutDescriptor`，不是 `cute::Layout` 对象本身**
  （P3.1 的往返本身已完成）：`CuteLayoutBridge::ToIslMap` / `FromIslMap`
  实现了 §3.5 的转换规则并有往返单测，三级逆策略也有分支覆盖。剩下的边界
  是接口位置：桥的输入是本仓自己的 `LayoutDescriptor`（extent/stride 向量 +
  三个标志位），**不是** CuTe 的 `cute::Layout` C++ 对象，因此
  "先 `flatten`/`coalesce` 再读 shape/stride" 这一步目前假定调用方已经在
  CuTe 侧做完——桥不会自己去展平一个嵌套 layout。等到真的要把 CuTe 的
  layout 代数结果喂进求解器时，需要补一个 `cute::Layout → LayoutDescriptor`
  的适配层（V-F 已确认 `Flatten`/`Coalesce` 对动态维可用，所以这一步是有
  依据的，只是没写）。另外 `FromIslMap` 只接受字面 extent（见 P3.1）。
  **对 Phase 5 的阻塞影响**：区间求解本身不依赖这个适配层，现有 registry
  的候选也可生成多变体；但若 Phase 5 要把任意 CUTLASS collective 实际产生的
  嵌套 layout 纳入 `g` 候选并自动证明 access 契约，这层就是扩展候选覆盖面的
  前置。它不阻塞当前两参考模型，阻塞的是“后端 layout 自动反哺求解器”。

- **isl 的除数必须是字面常量，这约束了参数何时可以保持符号化**：
  `isl_aff_div` 在 C API 层就拒绝参数化除数（`docs/experiments/P3_ISL/`），
  所以 tile size（`Tm`/`Tn`/`Tkv`）以及任何出现在除数位置的量（GQA 的
  `G`）必须在构造 `isl_map` 之前替换成字面量。实现上这体现为
  `DeriveCoupling(..., ParamBinding const& known, ...)`：`known` 里的符号
  被替换掉，不在 `known` 里的（`S`/`L_s`/`past` 这类工作负载维度）保留为
  真正的 isl 参数，不变量 I1 因此仍然成立。这不是缺陷，是一条必须被遵守
  的边界；`ClosedForm::ToIslText` 在除数没有被替换成常量时会显式抛错，
  而不是生成非法 isl 文本。
  三层拆分之后这条边界的**作用位置**可以说得更准：它只约束 L-task 及以下。
  L-sem（`include/tilemega/Analysis/Semantics.h`）里不出现除以 `g` 的表达式
  ——indexing map 是元素级的，唯一的 floordiv 是语义本身的（GQA 的头分组
  `h/G`），除数是取值固定的 theta 符号而不是 tile size。这不再是一句设计
  意图：`test/unit/semantics_test.cpp` 用两个不同的 `g` 构造同一个 L-sem 并
  断言 `Serialize()` 逐字节相同，同时断言序列化文本里不出现 `Tm`/`Tn`/
  `Tkv`，而两个 L-task 实例化的 tile 确实不同（否则该断言是空的）。这是
  不变量 I1 的可执行形式，替代了此前的 `StructureKey()` 代理。

- **wait 与 fanout 需要不同的定界处理，这是 barvinok 的一条实测边界**：
  把生产者（range）侧的界折进 `C` 本身，会让带有真正 isl 参数、且生产者
  坐标由不等式区间导出的关系在 *wait* 方向把 barvinok 推进
  `unexpected missing (bounded) solution`（`basis_reduction_tab.c:210`）；
  完全不定界，则 *fanout* 会因为 `isl_map_card` 的分片分解保留一段"只在
  别的参数取值下可达、在当前取值下恒为 0"的尾巴而误报 `min = 0`。现在的
  做法是只把生产者盒子作用在**反向映射**上（`ProducerRangeBoxText` +
  `FanoutCard`），让两个方向各自留在自己可解的区域。这是绕过而不是修复：
  上游若换 isl/barvinok 版本，这条需要重测。✅ BF16 重标定后仍然需要：dtype
  只改变实现候选与每元素字节数，不改变这些参数化整数集合；BF16 的生产路径
  继续经过同一个双向定界实现并通过 target-audit。该结论是代码路径核对，
  不是一次新的 barvinok 数值现象。

- **~~`kElementChunk` 把精确窗口收益封顶在 5.9%~~ 已还清；所有权 Place 与
  缓存局部性 Place 必须分开讨论**：先只把 RoPE 从扁平 grid-stride chunk
  改成 `(token, head)` tile，边可窄化且 seq=128 的 L2 改善 3.23%/6.40%，
  body 本身没有显著倒退；据预先判据推广到 KVAppend、activation 与 split-K
  combiner。最终 GQA 的 38 个 stage 对从 20 `kAll` / 3 `kIdentity` /
  15 `kWindow` 变为 **10 / 7 / 21**。✅ seq=128 的精确 poll 从 482316 降到
  282636（−41.40%），exact/relaxed 从 0.999925 降到 0.520240；25 进程配对下
  L2 改善 **13.57% / 16.32%**，而 L0.5 body 代价为 1.00%/1.25%。这是
  “task body 的 work 如何归属 CTA”的 Place 契约；此前被否决的是“相邻 task
  是否发给同一 SM 以复用缓存”的 list-scheduling 目标，二者不是同一个决策。
  剩余 10 条 `kAll` 是全归约或不兼容 domain 的语义结果，不是降级回退。
  证据：`docs/experiments/OWNERSHIP/`。

- **~~`kIdentity` 的可行性判定是一个按算子名的代理~~ 已还清**：
  `tools/tilemega-identity-probe` 现在问的是生产路径本身——推导出的 isl `C`，
  以及 `LiftedOp::ownership` 与 TaskBody 自己从 `Ownership` 返回的
  `OwnershipOf`（`TaskBase.h`）逐条交叉核对。两个参考模型上
  **0 条不一致**；可容许边为 gqa2 7 / 42、mha4 15 / 86，判据从"ownership
  字符串相同"改成"两端都是 `kTilePerBlock`"，因此与旧表（1 与 4）不可比，
  旧数字作废而不是复现。
  剩下的边界是两端都是 `kElementChunk` 的边（gqa2 2 条 / mha4 4 条）：它们
  只有在"把同样多的元素按同一个 grid 线性化"时才共享 CTA→task 映射，而 CG
  的 task space 不记录这件事，所以按未判定计而不是按可容许计。

- **前端的语义恢复现在是声明式匹配，剩下的边界是"模式的覆盖面"而不是
  "名字的约定"**：`lib/Frontend/ModelPlan.cpp` 里已经没有任何 ATen target
  字面量，也没有任何参数名字面量——`layers.N.*` 正则、10 处
  `q_proj/k_proj/.../inv_freq` 参数名查表、`past_kN`/`past_vN` 命名约定、
  4 个按 target 字符串写死的查找函数、以及"head_dim = numel(inv_freq)*2"
  这条布局假设，共 21 处名字/target 相关的判断已被删除，换成
  `include/tilemega/Frontend/GraphPattern.h` 上的一条 17 槽声明式模式：
  槽之间用操作数取值（`Value`）、依赖（`Dep`）、最近同角色节点（`Near`）
  和两条轴约束互相钉住。target 字符串只在一个地方比较——`CoreRoles()`
  的角色表（13 个角色 / 30 个 target），这也是同一条模式既能匹配
  composite 导出又能匹配 `run_decompositions()` 之后的 Core ATen 图的原因
  （两个参考模型上生成的 `.cu` **逐字节相同**，见
  `docs/experiments/SEMANTIC/`）。
  剩下的边界是：仓库目前只写了一条 decoder-layer 模式，一个结构不同的
  模型家族（纯 MLP 堆叠、不同归一化）仍然需要**新增一条模式**才能被分组
  成融合的 task space。区别在于新增的是数据不是分支，且不匹配不再是错误：
  没有模式覆盖的算子降级为"一个算子一个 task space"并在
  `ImportSummary::degraded` 里报告（`frontend_degradation` 单测），
  codegen 侧则明确拒绝一个没有 model plan 的模块。
  Analysis 层的 `MlpStack`/`GatherModel`（`lib/Analysis/ReferenceModels.cpp`）
  证明了耦合推导算法本身不依赖 Llama 结构，但它们没有经过这条 Frontend
  路径进入生成器。
- **~~L2 Solver 未实现~~ 已实现；剩下的是「簇内/局部同步在这台机器上跑不了」**：
  Phase 4 把层1–层5 补齐了（标定 / 代价模型 / 对齐传播 / 链上 DP / Label /
  Place），验收口径见 §4.4。原来这条债的价签——固定 `g` 慢 **6.11×**、
  **6.75×**，排在 1077 个候选的第 951 / 960 位——**已经被还掉**：DP 现在选
  `16x64x16s2k16`，在 25 进程终选榜上排 2/15 与 1/13，逐算子 `g` 再多 ~1%。
  仍然成立的部分是：Phase 1–3 时期记录的所有 L0.5/L1/L2 **绝对**延迟数字
  都是在那个近最差四分位的 `g` 上测的，层间**相对**比较不受影响，但任何与
  外部实现的绝对对比都必须先换成 DP 的解。
  剩下的真债是同步：`sync_kind` 仍然只走 `global`，`cluster` 这条路
  **代码齐了但在 sm_89 上编不出来**（`Caps<Sm89>::kCluster` 为 false，
  `static_assert` 故意让它编译失败而不是静默退回平坦 barrier）。
  簇的端到端消融因此欠一台有簇的机器，脚本已备好
  （`docs/experiments/CLUSTER/run_on_cluster_gpu.sh`）。
- **~~分析层的耦合推导尚未接入真实前端路径~~ 已还清；剩下的是求解器消费
  参数化派生量**：`lib/Frontend/Frontend.cpp` 不再用占位的
  `fixedRelation()`。它现在把 `ModelPlan` 抬升成 L-sem，
  `Instantiate(sem, g)` 得到算子级 `OperatorGraph`（gqa2 179 task / 222
  coupling / 30 stage，mha4 355 / 444 / 60），再调
  `CouplingDerivation{}.Derive(graph, known)`，`relation` / `wait` /
  `fanout` / `volume` / `count` / `tier` 全部来自推导，`wait_map` 属性把
  §2.7 的形状带到 codegen。
  ~~新债是推导的取值点~~ **也已还清**：`known` 现在只绑 isl 真正要求是字面量
  的东西（tile size、GQA 的 `G`——`isl_aff_div` 拒绝参数化除数），
  `S`/`past`/`L_s` 保持为真正的 isl 参数，于是四个派生量是拟多项式而不是
  `S_min` 上的一个整数。✅ 符号求值与逐点重推 **420/420** 一致；旧的常量在
  63 个度量里错了 **27** 个，`attn_chunk → attn_combine` 的
  `wait = ⌈L_s/Tkv⌉` 在 4096 token 上低估 **32×**（`T_sync` 读的正是它），
  `rope_q → attn_chunk` 的 fanout 低估 4096×、count 低估 1.3e5×。
  事件张量的形状仍在最小实例化上求值——那是一次真实分配，非常量轴发
  `kDynamic`。证据：`docs/experiments/SYMBOLIC/`。
  ⚠️ **剩下的是"谁来读"**：代价模型还没有消费这些拟多项式，`T_sync` 仍是
  每 stage 一次的 grid barrier 曲线，`ModelDescription::FromGeneratedCuda`
  只从依赖表里解析生产者/消费者。接上去会改变代价模型的输出、因而必须
  重新对着 oracle 验证，属于代价模型那一摊。

- **~~等待窗口靠三个具体 S 点重拟合~~ 已还清；剩余常数是已证明的代码生成
  形式**：`FitWaitWindowSymbolic` 先用 `lexmin/lexmax` 每个 consumer 只取两个
  端点，构造 `div/scale/offset/count`，再以 isl 双向 subset 对完整参数化关系
  做等价证明。只有行主序线性化离开 Presburger 形式的单条边才退回三点拟合，
  不再让整个图回退，也不再重推 `C`。两个参考模型 42+86 条边与 16 层 350 条
  边均为 **0 fallback**，参考模型生成 `.cu` 与改造前逐字节相同。real-width
  单层从 63 s 降到 0.861 s；16 层为 362 s（余下时间在全图 coupling 派生，
  不是窗口三点枚举）。`div/scale/offset/count` 仍是每变体的整数，因为 tile
  形状已绑定；其对所有 `S/past` 的有效性来自集合证明，而不是运行时字段。
  证据：`docs/experiments/REALMODEL/`、`table27_test`、`isl_relation_test`。

- **~~GEMM 粒度是 `-D` 宏，依赖表只能 exact/degraded 二选一~~ 已还清**：
  `tilemega-compile --variants` 现在对每个 `RuntimeVariantDesc` 独立实例化 L-task、
  推导 coupling，并把 GEMM 粒度和对应的 `StageDependency` 一起写入
  `ModelSpec`；同一二进制可含 16 个 CUTLASS 模板，host 以 `seq` 做 O(1)
  区间查表。旧的粒度比较、`TILEMEGA_GENERATED_WINDOW_*` 与
  `wait_table=degraded` 已移除，每个变体都只报告 `variant_exact`。✅ 曾经
  0/50 的 `16x64x16s2k16` COARSEN 配置在两个模型上都恢复 50/50。
  新的、已量化的边界是资源：1/2 个变体保持 5 CTA/SM，4 个降到 4，8 个
  降到 2，16 个降到 1；第一个拐点是寄存器而非 smem union。Phase 5 默认
  每 binary 至多两个区间，更多区间要跨 binary 分区。证据：
  `docs/experiments/VARIANT/`。

- **~~只测 `seq=4`，抓不到 grid-stride 欠等待~~ 已还清**：两个参考模型的
  `seq∈{1,4,128,512,2048} × past∈{0,3,512}`、L0/L0.5/L1/L2 共 30 格，
  每格 50 个全新进程，✅ **1500/1500**。矩阵实际抓到并修复了 attention 与
  RMSNorm “声明 grid-stride、只执行首个 task”的问题，以及 attention score
  shared row 只按 CTA 宽度分配的问题。故意重编旧 `min(count,grid)` clamp 后，
  `seq=2048,past=0` 在 50/50 进程里失败，证明矩阵对该静默欠等待敏感。
  证据：`docs/experiments/SEQSCAN/`。

- **~~BF16 未贯通、代价模型输给解析基线~~ 已还清最低门槛；FP32 目标仍未达到**：
  dtype 来自 FakeTensor，L-sem、registry、TaskBody 与 `ModelSpec` 全程携带；
  所有 body 用 BF16 存储，GEMM/norm/softmax 用 FP32 累加。✅ 两模型各 50/50，
  SASS 各有 96 条 BF16 HMMA。FP32 标定保留，BF16 独立 profile 的 Tensor Core
  为 179.997 TFLOP/s，DRAM pin 97.37%，前后漂移 0.00073%。
  初次 1540 点重扫的 ρ 只有 0.5605 / 0.6239，且输给 tier2；但 F-70 的归因随后
  被实测推翻并更正。空 `NullKernel` 基线改成同一 kernel 的零工作量之后，负的
  per-CTA setup 消失、`ac_r2` 升到 0.984–0.9997；`combine_fixed_ns` 的 clamp
  也改成显式 `combine_fixed_resolved`。真正影响排序的是一个 `setup_ns` 标量
  覆盖 154 个形状，以及把 scalar `ld.shared` 的 SMEM lane 错用到 BF16 Tensor
  Core operand feed。修正验证 occupancy 的线程数、SMEM 预算并采用实测
  `ctas_per_sm` 后，当前 ρ = **0.8984 / 0.8871**，越过 tier2 的
  0.8778 / 0.8738；但仍低于 FP32 0.9450 / 0.9435，top-1/3/10 仍为 0，且绝对
  时间约低估 2.1×。阈值没有移动；区间解只能把它当初值。证据：
  `docs/experiments/BF16/`、`docs/experiments/ORACLE/` §6.7、F-74。

- **BF16 的逐元素判据是在单一 split 上定的，跨 split-K 轴不成立**（本轮实测）：
  `1.6e-2 + 1.6e-2·|e|` 这个界取自 SEQSCAN 矩阵，那里 split 因子是固定的。
  在 ORACLE 的 split-K 轴上，mha4 的 split_k ∈ {2,16} **每档 154/154 全判
  MISMATCH**，而 {1,4,8} 每档 154/154 全过。逐元素看是几个 BF16 ulp 刚好越界
  （split 2 时全模型只有 1 个元素超标：−0.96875 对 −0.9375，delta 0.03125 对
  容差 0.031000，超出 0.8%），而**通过的配置反而带着更大的偏差**（split 8 的
  `max_abs = 0.0625`，只因落在 `|e|` 更大的元素上）。TileMega 自己的
  L0.5/L1/L2 在这些点上逐位一致。**阈值没有被放宽**；代价是 mha4 的验证集从
  770 被截到 462，且截断方向与 `split_k` 这个决策变量相关，因此 mha4 的排序
  统计不能与 gqa2 的 770 点直接并列。诊断用 `TILEMEGA_DIFF_DUMP=n`
  逐元素打印实际值与容差。

- **~~Place 求出但没有进入执行、L2 实际按 stage 齐步走~~ 已还清**：此前
  `ModelSpec` 没有 schedule，kernel 外层是 stage，`WaitDependencies` 也在
  stage 循环外侧；所以“生成的 L0.5/L1 与手写版逐位一致”可以在两边都偏离
  §5.4 的情况下通过。现在每个运行时变体生成独立 schedule，host 物化每 CTA
  的 `TaskRef` 队列，L2 在 slot 内逐 task wait/run/notify。生成期拒绝环、缺失/
  重复 stage 与反向依赖；split-K 绑定后在 host 再验证一次。✅ 两模型 50/50、
  seq×past 1500/1500，且 50 进程直接插桩观察到平均 34.36% / 40.18% 的
  task 在更早 stage
  尚未全部完成时启动。这个缺口使旧 κ / Place / L2 数字失去被测对象，现已在
  各目录以 `raw_taskqueue` 重测，旧数字只作失败史，不作结论。

- **L2 的剩余开销仍是事件机制，而 §8.2 的单调计数器目前不可证伪**（任务队列重测，
  `docs/experiments/L2_ATTRIB/`、`docs/experiments/AUTOREGRESSIVE/`）：
  四臂消融（无事件 / +notify / +wait / L1 去掉 barrier）在 25 轮组内配对下
  重新给出完整分解。四格中 wait 为净差值的 **45–84%**，notify 为
  **137–163%**，删除 L1 barrier 与更快的 bare queue loop 抵回一部分；每轮恒等式
  `L2 = neither + notify + wait` 以 0 µs 误差闭合。旧的
  `边×owned-task×event-group` 扫描已删除：`kAll` 一条边是一个聚合事件，窗口边按
  logical task 组展开；task 内去重且把已满足 epoch 提升后，gqa2/mha4
  从 580/1284 条 raw 描述符降到 500/1076，
  剩余 poll 保持 CTA 并行。旧 F-76 的“poll 是整个 gap”是 stage-loop 的结论，
  ❌ 不再适用。
  ⚠️ 另一半是负结果：把计数器在迭代之间清零、`iteration` 恒为 0 的反面构建
  在 32 次迭代 × 50 进程下**也全过**。ABA 在当前 harness 里结构上不可达
  ——`iteration` 是**启动**参数，同流上的启动依次完成，不存在"还在收尾的
  第 i 次迭代"。所以单调计数器是 Phase 6 才开始需要的防御性代码，
  现在**没有测试覆盖**，不能因为扫描全绿就当作已验证。

- **跨会话的绝对延迟不可比，只有会话内配对可比**（本轮实测）：同一个
  字节相同的二进制（`docs/experiments/E2E_GEN/generated_e2e`）在 09-03 记录
  `l2/l1 = 1.0136×`，09-04 重测 `1.0367×`，绝对 L1 中位数在同一天的两次
  扫描间也从 0.998400 ms 走到 1.082560 ms，`nvidia-smi` 显示 SM 时钟在
  210 MHz 与 3105 MHz 之间空转。因此本仓所有跨轮次的**绝对**数字都只作为
  历史记录，任何比较都必须在同一会话内配对重测；COARSEN 的两次 κ 扫描
  中位数相差最多 1.8% 而每一条配对结论不变，是这条的第二个证据
  （`docs/experiments/COARSEN/result.md` §3.5）。

- **Coarsen 的实测边界（P4.6 的 `[!]` 已给出结论，但覆盖有限）**：κ ∈ {1,2,4}
  下关系与拟多项式都保持**单分片**、isl 文本长度基本不随 κ 变化（保留 `S`
  为符号只多约 15 个字符），所以**没有出现表达式爆炸**，κ 不必限制为 2 的幂
  （数据见 `docs/experiments/P3_ISL/result.md`）。⚠️ 这只覆盖了一个 decoder
  层的边、κ ≤ 4、且每次只粗化一根轴；更深的嵌套没有测。**硬件侧的收益现在
  也可见了**：最终队列上 κ ∈ {0,1,2,4,8,16,32} 的 argmin 在两模型上
  都是 1，相对 κ=0 快 0.285% / 0.165%。两参考模型仍共用同一选择，
  所以 κ 暂不进 DP 状态，但“结构上永远为 0”的旧论证已作废。见 P4.6 与
  `docs/experiments/COARSEN/result.md`。

### 1.5.2 v2.1 状态修订与 L2 执行模型差距

**状态修订**（追加；上表原文不改）：

| 项 | 路径已验证 | 代码已实现 | 证据 |
|---|---|---|---|
| Place（task → (worker, slot)） | ⚠️ | ⚠️ | 已执行的只有两部分：一是 stage 级置换，由 Codegen 内的 `BuildVariantSchedule` 调用 `ListScheduler` 算出；二是 host 端的 task→worker 映射，默认 `t mod grid`，`TILEMEGA_PLACEMENT=4` 时为 balanced 启发式。`BalanceTaskPlacement` 算出的 slot 无人读取；CG 的 `tilemega.placement` 仍是 `map=[0]` 占位。关键路径序对 round_robin 为 1.000000 / 0.998552（F-82）；balanced 为 1.639908 / 3.087011 / 1.396662 / 4.002253（F-118）。见 F-127、F-129 |
| L2 执行器能力 | ⚠️ | ⚠️ | 队列 stage-major，归属与 L1 的 grid-stride 相同，严格 FIFO（W=1）；L2/L1 = 1.081–1.113（`E2E_L2`）；同步零成本时的结构上界见 F-126（inferred） |
| L2 求解目标函数 | ✅（L1 目标） | ⚠️（缺 L2 目标） | `ChainDP` 的 op_cost = stage + barrier，在 `l2_events` 下拒绝求解；`CostModel::EventNs` 为计数 × 速率。见 F-128 |

**差距清单**：

| ID | 差距 | 证据（标注） | 影响 | 承接 |
|---|---|---|---|---|
| G1 | L2→L1 契约没有 Place 通道 | ✅ `Frontend.cpp` 以 `map=[0]`、`cluster=1` 生成 `tilemega.placement`，没有求解器写回；`Codegen.cpp::BuildVariantSchedule` 在生成期计算 stage 置换；`ScheduleStageDesc` 只含 `{stage, dependency_begin, dependency_count}`（F-127） | 求解器无法表达 task 级计划 | EX-E1 |
| G2 | 默认归属与 L1 相同，队列 stage-major | ✅ `harness::Create()` 的 `task_owner` 初始化与物化循环；⚠️ inferred：同步零成本时存在结构上界（F-126）；✅ 实测：`queue_lb` 占实测 `l2_ms` 的 81.2%/84.6%/83.0%/84.1%，seq=4 时 256 个 worker 只有 16 个拿到 task，最长队列 30/34/60/80（F-134） | L2 至多只能省掉 barrier；✅ 跨 stage 连续轮询（mode 5）把最长队列降到 1/18/2/47，`full` 臂配对中位比 0.6705 CI [0.6598,0.7392]（F-135） | EX-D2 已验证；EX-E1、EX-S2 承接 |
| G3 | 严格 FIFO（W=1），存在 HOL | ✅ `tilemega_l2_kernel` | 放置改变之后才显著；对代价模型误差没有容忍度 | EX-E2 |
| G4 | 每个 task 的发布协议偏重 | ✅ `NotifyTask`、`ArriveEvent`、`EventPoll`；每 task 最多 5 次 CTA 屏障（skeleton §5.5.1）；✅ 测量：notify 是 L2 超出 L1 的最大正项（`L2_ATTRIB`） | 每跳成本高于 barrier | EX-E3 |
| G5 | 等待提升与本地省略依赖 FIFO | ✅ `seen[worker]`、`per_group == 1 && owner == worker`；⚠️ inferred：一旦乱序即失效（F-130） | 阻碍窗口执行与动态执行 | EX-E2 |
| G6 | 没有跨 task 预取 | ✅ `GemmStageTaskBody::RunTask`；wait 位于整个 body 之前 | 同步延迟与加载延迟完全暴露在关键路径上 | EX-E4 |
| G7 | ~~trace 不足~~ 已补齐 | ✅ trace v2（`TILEMEGA_TRACE_V2`，默认关、关时 SASS 逐字节不变）逐 slot 记录 wait/ready/run/publish 与 `%smid`，每事件行记录发布时刻；扰动中位比 ≤ 1.0167；`%globaltimer` 实测 1024 ns、跨 SM 偏移 0 ns（F-131） | HOL、每跳延迟与关键路径已可测；仅 §3.6 节点定义漏计 publish 导致重建误差 12%–15%（F-132） | EX-D1 已验证 |
| G8 | 求解目标为 L1 | ✅ `ChainDP::Solve`（F-128） | 配置是 L1 最优，L2 最优性未经检验 | EX-S3 |
| G9 | L2 价格是计数 × 速率 | ✅ `CostModel::EventNs` | 预测不了重叠，也预测不了放置引入的串行化（F-118 的方向与预测相反） | EX-S1 |
| G10 | balanced 贪心以 affinity 优先 | ✅ 机制；~~⚠️ inferred 归因~~ ✅ 归因已实测：可回收的 HOL 阻塞占停顿时间的 18.85%/52.99%/56.80%/63.98%，EFT 重排可达实测的 0.17–0.43 倍（F-134、F-136） | 可能把整个 stage 压到少数 worker 上 | EX-S2 |
| G11 | κ 为全局编译宏，κ=0 为聚合特例 | ✅ `TILEMEGA_EVENT_KAPPA`；F-105 | κ 不能按 stage 选择 | EX-S3 |
| G12 | 参考 fixture 处于纯延迟区 | ⚠️ inferred（权重字节未核实） | 结论存在外推风险 | EX-V1 |
| G13 | SIMT task 共用 GEMM 的启动配置 | stated：128 线程、212 寄存器、2 CTA/SM（F-90、`PLACE/round5_balanced_result.md`） | 访存型 task 占用率低 | 在 EX-V1 中观测；若成立则另立条目 |
| G14 | 死字段与遗留头文件 | ✅ grep：`TaskPlacement::slot`、`kLastTaskOfStage` 只写不读；`GeneratedLlamaRuntime.cuh` 没有包含者 | 误导读者 | EX-C1 |

### 1.5.3 被执行器结构混淆的历史结论（v2.1）

以下结论作为"该执行器结构下的测量"继续有效，但不得外推为关于放置、排序、窗口或 κ 的一般结论。它们都是在 G2（归属与 L1 相同、队列 stage-major）与 G3（W = 1）下测得的（F-126）：

- F-82：关键路径序对 round_robin 为 1.000000 / 0.998552。stage 置换只在独立 stage 之间有自由度，且各 stage 的 task 仍落在相同的 worker 上。
- F-86 与 COARSEN：exact window 对 kAll 的收益随 seq 反号（seq=4 快 0.694% / 0.322%，seq=512 慢 1.171% / 1.163%）；κ=1 为 argmin（快 0.285% / 0.165%）。
- F-118：balanced 映射慢 1.639908 / 3.087011 / 1.396662 / 4.002253 倍。该测量中 slot 被丢弃（G1），贪心以 affinity 为先（G10）；它说明放置的影响量级，不说明局部性没有价值。
- OVERLAP：34.36% / 40.18% 的跨 stage 提前启动主要是非关键 worker 上的尾部重叠，不是关键路径上的重叠。

### 1.5.4 CG 操作的实际接入状态（原 skeleton §2.3，v2.0 原文）

**操作的实际接入状态**（不能把“形式化”当成“已经在决策”）：

| 操作 | 是否驱动求解器 | 为什么 |
|---|---|---|
| **Reparam** | ✅ 是 | 链上 DP 的状态就是 `g`（tile 形状、stages、split-K）。它也是唯一**不需要** `C` 的一个：代价模型读的是形状与标定表，不读耦合关系。 |
| **Coarsen** | ❌ 否 | 最终队列上重测 κ∈{0,1,2,4,8,16,32}，两模型 argmin 都为 κ=1，相对聚合-only 快 0.285%/0.165%。收益侧首次非零，但两参考模型仍共用同一最优值，本轮不进 DP；“κ=0 结构最优”的旧论证已作废。 |
| **Label** | ⚠️ 部分 | `sync_kind` 的判定与 `ClusterSync` 原语都在，簇常数也已标定，但没有一条 CG 边在 sm_89 上能取 `cluster`，所以它现在恒取 `global`；端到端消融欠一台 sm_90+ 机器。 （⚠️ v2.1：L2 执行下 Label 是 Place 的子决策，见 skeleton §5.7.1） |
| **Place** | ✅ 是（⚠️ v2.1：降级为部分——只有 stage 置换与 host 映射，slot 未被消费，见 §1.5.2） | `ListScheduler` 的关键路径优先序现在随每个 runtime variant 发进 `ModelSpec`，host 在绑定动态 task 数后物化每 worker 队列；round-robin 对照走同一二进制的 host 开关。另有 TaskBody 的 tile-aligned ownership，二者仍是正交决策。硬件价值以本轮 `PLACE` 重测为准，旧 CTA-bijection 数字全部作废。 |
| **Relax** | ✅ 是（安全方向） | `Contains(C', C)` 是 `isl_map_is_subset`，真正的 Tier 2/3 非精确边仍可取 `kAll` 超集；但“编译粒度与推导粒度不匹配”的回退已经不存在——每个运行时变体携带自己粒度下推导的精确表，不再有 `wait_table=degraded`。 |
| **Fuse** | ⚠️ 正确性已验证，定价方向门未过 | 固定域区间DP→L-task写回→exact C lowering已接；RoPE/KV完整模型400/400、GEMM两链400/400正确。add预测/实测一致，RMS预测加速却慢6.25%/35.21%，局部停止定价及依赖的联合搜索，见FUSION/runtime_result.md。 |

### 1.5.5 求解结果到代码的落地程度（v2.1）

对应 skeleton §5.6 的目标映射。

| CG 上的决策 | 目标落点（skeleton §5.6） | 当前落地 | 证据 |
|---|---|---|---|
| Reparam：tile 形状 `g` | TaskBody 的 `TileShape` 模板实参 | ✅ 每个 binary 最多 16 个模板变体 | `VARIANT`、F-66 |
| Reparam：split-K 因子 | `TaskDesc` 的 `k_begin` / `k_count` | ✅ host 改写为按 chunk 的调用，并插入 combine stage | `harness::Create()` |
| Coarsen：κ | 事件张量 extent 与索引映射 | ⚠️ 全局编译宏 `TILEMEGA_EVENT_KAPPA`；κ=0 为聚合特例 | F-105 |
| Label：簇归属 | `__cluster_dims__` + `SYNC_CLUSTER` + DSMEM | ⚠️ 仅有 L1 stage barrier 的簇形态与 T1 分片 fan-in 实验；L2 没有 cluster 边 | P3.5、F-89 |
| Place：task→worker | `p.schedule[worker][s]` | ⚠️ host 默认 `t mod grid`，或 balanced 启发式 | F-126、F-127 |
| Place：worker 内顺序 | 同一张表内的 task 序 | ❌ 恒为 stage-major，slot 未被消费 | F-127 |
| Stages | TaskBody 的 `Stages` 模板实参 | ✅ | — |
| 区间划分 | 多套实例化 + host 端 O(1) 查表 | ✅ 每个 binary 无损上限为 2 个变体 | `VARIANT`、F-66 |
