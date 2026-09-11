# TileMega v2.0 分阶段 TODO（归档）

# 7. 分阶段 TODO

> `[ ]` 待办 `[~]` 进行中 `[x]` 完成 `[!]` 阻塞 `[-]` 已放弃（保留并注明原因）
>
> **前置**：`docs/VERIFICATION_PLAN.md` 的项目先完成。
>
> **对所有"分析层"验收项的一条常设附加条件（本轮补入，代价见下）**：
> 一项分析工作只有当它的输出**进入生产路径**时才算通过——不是"算法有单测"、
> 也不是"探针能复现表格"，而是生成的 `.cu` / 求解器的决策里能指出哪一处用了
> 它，并且把它拿掉会看得见地变化。
> 这条不是事后总结的漂亮话，它有价签：P3.3 的验收（§2.7 的 13 条边全部自动
> 推出、isl 整表交叉验证）与 P3.5 的验收（global 事件张量、release/acquire
> 顺序、ABA 场景、与 L1 逐位一致 50/50）**各自都通过了**，两条合起来仍然漏掉
> 了"推导出的 `C` 必须进生成器"：`Frontend.cpp` 从不调用
> `CouplingDerivation`，IR 里的 `CouplingMapAttr` 是 `{ [0] -> [0] }` 占位，
> codegen 只能发 `kAll`，于是一条依赖退化成一次 barrier。这一个缺口同时解释
> 了三个负结果——L2 比 L1 慢、κ 的收益侧恒为零、编排层量到 −1.4%——而每一个
> 都曾被当作独立的结论记在案。验收项之间的**接缝**没有主人，就是这样漏的。
>
> 本轮又发现同一模式的第二种形态：§5.4 明写“worker schedule + 逐 task wait”，
> 实现却是“无 schedule + stage 外循环 + stage wait”。它仍通过了“生成的
> L0.5/L1 与手写版逐位一致”，因为手写版与生成版共享同一个 stage-loop 偏差。
> 因此新增常设条件：**参照实现若与被测实现共享控制流，逐位一致只能证明二者
> 一致，不能证明执行模型符合设计。** 对调度/同步的验收必须同时包含结构检查
> （生成表确实被 kernel 读取）和一个旧模型不可能产生的可观测量。本轮的可观测量
> 是跨 stage 提前启动，旧实现按构造为 0，新队列 50 个全新进程
> 的均值为 34.36% / 40.18%。
>
> 同一缺陷还出现了第三层：第一版队列虽然逐 `TaskRef` 调度，却把事件键写成
> `(producer stage, owner CTA / κ)`，并只在一个 CTA 完成该 stage 的最后一个
> task 时发布。这保证正确但没有实现逻辑 task 粒度；当 task 数大于 worker 数时，
> 消费者仍被迫等待同一 CTA 拥有的无关工作。现改为按
> 每 stage 一个聚合行加 `(stage, logical_task / κ)` 精细行的前缀表索引事件，
> 并由每个 `TaskRef` 完成时同时贡献精细与聚合 arrival。
> 第一版队列的 κ / Place / L2 数字再次全部作废并重测。由此再加一条结构验收：
> **不仅要检查 kernel 是否读取 task schedule，还要检查 readiness 的事件键与发布点
> 是否真由 logical task 定义。**

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
- [x] CuTe layout → `isl_map`：`CuteLayoutBridge::ToIslMap` 按 §3.5 的规则把
      平坦 layout `(s₀..s_k):(d₀..d_k)` 写成
      `[i₀..i_k] → [offset + Σ i_j·d_j] : 0 ≤ i_j < s_j`。extent 可以保持符号
      （只是界，`S` 之类留作 isl 参数，不变量 I1 因此保住）；stride 必须是
      字面量，否则 `参数 × 坐标` 不是 Presburger 仿射——这与 V-F 从 CuTe 一侧
      得到的边界是同一条，只是从另一边撞上。swizzle 与动态 stride 都显式抛错
      并由 `Project` 给出 Tier 后果，不做静默近似。
- [x] `isl_map` → CuTe layout（求解结果回写）：`FromIslMap` 从单值仿射映射读回
      shape/stride（extent 取自域盒，stride 取自 `isl_aff` 系数）。**要求 extent
      是字面量**——回写的是求解器"选定"的 layout，选择就是具体的；符号 extent
      属于模型而不属于求解结果，故拒绝而不猜测。
- [x] 单元测试：`layout_bridge_test` 覆盖行主序/列主序/1-D/3-D/带填充 stride
      的往返等价，以及广播模式（stride 0）由 isl 自行判出非单射、符号 extent
      的回写被拒绝、swizzle 与动态 stride 被显式拒绝、且把 stride 绑定成字面量
      后同一 layout 重新可表示。
- [x] 三级逆策略机器化：静态 `g` 走 CuTe `RightInverse`；动态 extent +
      常量 stride 走 Presburger relation；动态 stride/swizzle 明确提升 Tier。
      `layout_bridge_test` 覆盖五种分支。

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
      ⚠️ 这条验收当时通过得太轻：它只检查了推导**能**推出 13 条边，没有检查
      推出来的东西**进了**生成器。缺口与代价见 §7 开头的常设附加条件。
- [x] 推导进入生产路径：`Frontend.cpp` 按算子粒度建 `OperatorGraph` 并调用
      `CouplingDerivation`，每条 coupling 带 `wait_map`，codegen 按 stage 对
      合并成 `StageDependency`（gqa2 38 对 = 20 `kAll` / 3 `kIdentity` /
      15 `kWindow`；有一个成员松弛或不一致就整对退回 `kAll`）。
      见 `docs/experiments/WIRING/`
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
- [x] 松弛**后果**的机器验证：`TierClassifier::RelaxationCoversProducer`
      用 isl 检查"被松弛的边确实把 C 放宽到了生产者的整个任务空间"——
      即 `domain(C) × 生产者盒 ⊆ C`。注意这里是**逐消费者点**的覆盖，不是
      `range(C) == 生产者盒`：精确的恒等边同样以整个生产者任务空间为值域
      （它是到该空间的双射），只有"单个消费者点就够到全部生产者任务"才是
      松弛的特征。`containment_test` 两个方向都断言。
- [!] **Tier 本身无法由 `C` 推出，这一条按"做不到就说做不到"记录**：Tier 是
      来源属性而不是几何属性——Tier 1 取决于 layout id、Tier 2 取决于任务空间
      是否有运行时 extent、Tier 3 取决于下标是否来自张量，这三样都**不在** `C`
      里。曾经实现过一版按 `isl_map_is_single_valued` 分类的
      `TierClassifier::Classify`（一对多即判 Tier ≥ 2），实测对 §2.7 的 21 条
      派生边**误判 4 条**，其中 `attn_combine→wo`、`add1→rmsnorm2`、
      `silu→wdown` 三条是表里明确标为 Tier 0 的：它们一对多只是因为
      `wait > 1`，这对精确仿射边完全正常，与可解析性无关。该分类器已删除，
      而不是留着一个会误判的实现；Tier 仍由 `CouplingDerivation` 从访问映射
      的上下文（layout id / runtime extent / data-dependent 标志）赋值。
- [x] **Tier 正交化**：这一条的另一半——Tier 推不出来，是因为它本来就是五个
      属性的摘要。`relation_kind` / `extent_kind` / `exactness` /
      `runtime_requirement` / `countability` 现在各自由推导赋值，Tier 只由
      `DeriveTier` 从它们算出，`CouplingOp::verify` 重算并拒绝对不上的边
      （`cg_attr_test` 两个反例）。六个参考模型 221 条边重新生成后
      `tier` 列与迁移前**逐条相同**；§2.7 的 4 条 Tier 2 边分成三类且全部
      `exact`。见 `docs/experiments/P3/attributes.md`。
- [!] `docs/experiments/P3/derived-*.md` 自 `b1dbc38` 起未再生成过，`C` /
      `wait` / `fanout` 三列仍是 isl 迁移前的语法；本次一并重新生成。
      `test/Dialect/CouplingGraph/*.mlir` 三个文件引用已删除的
      `#tilemega.closed_form`，全部解析失败，且未接入 ctest（本构建无
      lit/FileCheck），**尚未修复**。

### P3.5 L2 落地

- [x] global 事件张量 → `(producer stage, producer CTA)` device 数组（128B padding）
- [x] global `notify` / `wait` 按 §8 release/acquire 顺序生成
- [~] cluster 路径：`ClusterSync<Arch>` 原语层已实现（DSMEM `map_shared_rank`
      + 单调 epoch + 两级 barrier），能力开关是 `Caps<Arch>::kCluster`，
      sm_89/sm_90/sm_120 三个目标交叉编译通过，PTX 层面确认 `mapa` /
      `barrier.cluster` 只出现在 kCluster 目标上。**生成器尚未接线**：
      `Codegen.cpp` 仍只接受 `sync_kind = "global"`，megakernel 也仍以
      `<<<grid, threads>>>` 启动。需 sm_90+ 硬件，本机未运行。
      见 `docs/experiments/CLUSTER/`（`run_on_cluster_gpu.sh` 在非 cluster 机器上
      硬失败，退出码 3，拒绝把单 CTA 回退路径当成簇结果测量）
      发现：`if constexpr (Caps<Arch>::kCluster)` **不足以**关掉 cluster 代码——
      `cooperative_groups::this_cluster` 是非依赖名，sm_89 上根本没有声明，
      被丢弃的分支照样做名字查找而编译失败；`.cluster` 作用域的 fence 同样
      要求 `.target sm_90+`。可用性只能由 `#if defined(_CG_HAS_CLUSTER_GROUP)`
      判定，策略仍由 `Caps<Arch>::kCluster` 决定，两者由 `static_assert` 绑定
      **本轮已接线**：`Codegen.cpp` 现在接受 `sync_kind = "cluster"`，并要求
      它与 placement 的簇宽**完全一致**——混用被拒绝而不是被调和（一个 grid
      只有一种 stage barrier），发出 `TILEMEGA_GENERATED_CLUSTER_DIM`；
      `ModelHarness.cuh` 的 `LaunchL1`/`LaunchL2` 相应有 `cudaLaunchKernelEx`
      变体。仍然欠一台 sm_90+ 机器来跑端到端，`run_on_cluster_gpu.sh` 的
      自检改成读 `TargetSpec::Probe().caps.cluster` 而不是比较架构号。
      ⚠️ **能力边界**：sm_120 有 Thread Block Cluster / DSMEM / TMA，但**没有**
      tcgen05，也没有 B200 的 L1.5/LRC 层——一台 5090 上跑通不等于九条资源道
      都被验证过
- [x] 单调计数器：`needed = num_triggers × iteration_num`
      （`StageArrivalTarget`）。计数器从不在迭代之间清零，`iteration` 参数
      贯穿 `GridBarrier`/`WaitTaskDependencies`/`NotifyTask` 与两个 kernel。
      ⚠️ harness 在 L2 后**不重置事件内存**并递增 iteration，正向结果正确；
      但 host launch 串行完成，清零计数器的反面构建也不会失败，所以这不是 ABA
      场景的覆盖。真正验收需要 kernel 内迭代及版本化 KV/buffer 生命周期；见
      `AUTOREGRESSIVE`，不能把绿的 host sweep 标为已验证。
- [x] 与 L1 逐位比对：50/50 全新进程，0 mismatch / 0 timeout（2 层 GQA）；
      4 层 MHA 25/25 全新进程逐位一致（`docs/experiments/E2E_L2/`）
- [x] TaskBody ABI 的 CTA→task ownership 条目（`TaskOwnership`，
      `TaskBase.h`）：每个 TaskBody 自己声明 `blockIdx.x` 归属的
      `{kind, count}`，`ActiveBlocks` 只做分发，不再重述各 TaskBody 的守卫；
      未声明的 TaskBody 触发 `static_assert` 而不是静默破坏 L2 的跳过
- [x] L2 vs L1 在 task queue 上重测：seq={4,128,512} 的六格轮内配对比值为
      **1.081–1.113**，每格 25 进程、bootstrap CI、Wilcoxon p≈1.3e−05，
      0/25 轮 L2 更快。**仍然是慢，不是快**；所有 pre-queue 数字作废。
      精确窗口与强制 `kAll` 在六格全部分离，但收益随序列反转：seq=4
      快 0.694%/0.322%，seq=512 反而慢 1.171%/1.163%。见 `E2E_L2` / `L2_ATTRIB`。
- [x] L2 还有一项此前没有计入的成本：事件 epoch 占寄存器，在某些 `g` 下比 L1
      多一整档（实测 144 vs 128 → 1 vs 2 CTA/SM），于是**整个 harness**（含
      L0.5 和 L1）的 grid 被 L2 拉到一半。这不是调度开销而是资源占用，
      `E2E_L2` 的 1.4% 差值没有覆盖它——那里 L1 与 L2 的 occupancy 恰好相同。
      按 L1 定 grid 再启动 L2 会直接死锁，修法与谓词见 §8.7（F-38）

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
      宽度/GQA-MHA 比例从权重形状结构化推导。换一个不匹配这个数据流形状
      的模型（例如纯 MLP 堆叠）需要新增一条**模式**（`GraphPattern.h` 上的
      声明式数据），而不是新增一条分支：匹配规则里已经没有参数名与 target
      字面量，不匹配的算子降级为一个算子一个 task space 而不是报错。
      详见 §1.5.1 与 `docs/experiments/SEMANTIC/result.md`。

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

- [x] traits 路径：`constexpr` 探针批量求值。300 个候选一个 translation unit，
      19.61s（65.4ms/候选）。同一族的 host 闭式（`SimtF32Traits`）2.8~3.1µs/候选，
      与 CUTLASS traits 在 300/300 上逐字段相等，二者由三条 `static_assert`
      锁死（`CutlassGemmCandidate.h`），所以剪枝走闭式而不是编译。
- [x] nvcc 路径：`--ptxas-options=-v` 解析（`ParsePtxasRegisters`）。
      2240 次真编译 32 路并行 1893.85s（0.845s/候选 wall；单编译串行实测
      21.2s，所以 32 路实际拿到 25.1 倍）。
- [x] 批量：整个候选空间「枚举 + 排序」< 1.4ms，对照真编译 13.2 CPU·小时。
- [ ] `[!]` 缓存尚未进库：目前只有实验脚本按文件缓存
      （`docs/experiments/ORACLE/run.sh`），查询接口本身每次重算。闭式 3µs
      的量级下这不是瓶颈，但 tier 3 的结果值得落盘。
      缓存键必须含源码、目标架构、CUDA 与 CUTLASS 版本（§1.2 原则二）。
- 证据：`docs/experiments/BACKEND/result.md`

### P4.2 层1 合法性剪枝

- [x] 从 CUTLASS `TiledMma` 原生形状 `(16,16,8,2)` 沿各维扩张，撞 smem 墙停。
      按非降序的规范序展开，每个形状恰好到达一次，不需要 visited 集；
      墙是单调的，所以剪的是整棵子树：`touched=279 wall_pruned=19
      rejected=36 legal=224`，对照 Cartesian 积 300。
- [!] 目标「每算子 8~20 候选」**未达到**：一个 GEMM 算子在 RTX 4090 上的
      合法候选是 **224** 个（300 → 264 形状合法 → 224 装得下 smem → 216 真
      能编译）。不改口径去凑这个数字；8~20 是层2 对齐传播（P4.3）之后才
      可能出现的量级，层1 单独做不到。
- [!] **split-K 这根轴层1 剪不掉**：它是 host 侧的粒度选择而不是 collective
      的 trait，合法性是运行时的 `split_k ≤ k_tiles`。所以 224 个形状要乘满
      5 个 split 因子 = 1120 个配置进 tier 3，而层1 只把 1500 削到 1120
      （25.3%）。P6.2 的 oracle 显示 split-K 单独值 2.0~2.1×、且出现在
      **每一个** top-10% 配置里，把它留给固定默认值就是剪错了轴。
- [!] tier 2 排序的诚实结果：实测最优 `32x32x32s3` 在解析排序里排 **42/224**，
      而排序前 8 名全部**编译失败**（`cute/int_tuple.hpp(890)`，80 例同一处）。
      collective 级 traits 合法 ≠ megakernel 可编译。缺的是访存项，
      因为 `TargetSpec::Calib` 未标定（见 P4.1 上游）。
- 证据：`docs/experiments/BACKEND/result.md`

### P4.3 层2 对齐传播

- [x] 从输出端反向传播 tile 约束：`DeriveAlignmentConstraints` 从每个 GEMM 沿
      生成的 stage 表向前走到第一个读它输出缓冲的非 GEMM stage，读出那个
      stage 的粒度（RoPE / KVAppend 取 `width`，逐元素尾取 `extent`）。
      「QKV 的列 tile 要对齐 head_dim」从来没有写进求解器，是被推出来的：
      gqa2 的三个 QKV GEMM 拿到 `Tr = 128`，因为 RoPE 在生成表里的 `width`
      就是 `d`。换模型这个数就跟着换——单测把 `head` 换成 96，可行集随之
      移动（48 和 80 变合法，112 和 128 变非法，两个排除集互不包含）。
- [x] wait 膨胀。**骨架原来写反了**：`Tm` 是生产者宽度、`Tr` 是消费者宽度，
      所以第 `m` 个消费者 task 等待的生产者 tile 数是
      `⌈(m+1)Tr/Tm⌉ − ⌊mTr/Tm⌋`，剪枝判据是它的最大值减去 `⌈Tr/Tm⌉` 为 0。
      按记录差异而不是迁就的原则，这里改的是骨架而不是实现。
- [x] `[!]` 度量剪枝效果，**实测结论是在层1 实际产出的候选轴上一个都没剪掉**：
      1077 → 1077，联合空间 10^42.451 → 10^42.451（gqa2）、10^84.902 不变
      （mha4），0 个算子无约束（即每个算子都**有**约束，只是都满足）。
      原因是算术不是缺陷：层1 只发 2 的幂（`tile_n ∈ {16,32,64,128,256}`、
      `tile_k ∈ {8,16,32}`），模型暴露的粒度也全是 2 的幂（128 / 512 / 1024），
      2 的幂不会跨越 2 的幂的读者边界，膨胀恒等于 0。
      为了让「剪不掉」和「没在算」可区分，同一套推导在步长 16 的反事实
      `tile_n` 轴上跑一遍：48 → 每算子 21..48，联合空间塌缩 **≈143×**（gqa2）
      / **≈2.0×10⁴**（mha4）。另外 1077 = 216 形状 × 5 个 split 因子，
      split 根本不是对齐轴，所以再激进的层2 也只能碰那 216。
- [x] 不移动答案的对照：`tilemega-solve` 把统一 `g` 解两遍（带掩码 / 不带），
      打印 `tier-1-only control picks 16x64x16s2k16 (same answer)`，两个模型
      都是同一答案——剪枝为空这件事被验证而不是被假定。
- 证据：`docs/experiments/SOLVER/alignment.md`；`alignment_propagation_test`

### P4.4 代价模型

P6.2 的 oracle 已给出投入判据：固定 `g` 与最优 `g` 相差 **6.11×**（2 层 GQA）
和 **6.75×**（4 层 MHA），远过 §6.4 的 10% 门槛，所以代价模型 + DP 值得做。
两条限定条件改变了要做成什么样：

- **目标是落进平台期，不是找到 argmin。** top-34 个配置挤在 ±10% 带内，
  25 进程中位数都分不开它们（两次独立复现选出不同的冠军，冠军本身漂 9.04%）。
  精确到能挑出唯一最优的模型，是在买硬件不提供的分辨率。
- **split-K 必须是模型的一个自变量**（见 P4.2）。

- [x] `[!]` **roofline 被换掉了，因为量出来不行**：同一套资源常数上的纯
      roofline 一级实测 MAPE 181%/185%、Spearman 0.444/0.430，与层2 的解析
      排序（0.461/0.446）无法区分。现在的稳态是 §2.2(a) 九条资源道
      `⟨t_TC, t_CUDA, t_SFU, t_TMEM, t_SMEM, t_L1.5, t_L2, t_DDR, t_NET⟩`
      的 `max`（§4.4）——道数是九，不是六；在 sm_89 上 tmem/l1_5/net 三道
      因 `Caps` 缺失置零、被 `max` 跳过，每条零道带一个
      `LaneStatus` 理由并由 `tilemega-target-audit` 逐目标检查。
      完整模型 ρ **0.9450 / 0.9435**。⚠️ 实测里只有 SMEM 一道决定排序
      （只留它 MAPE 差 0.02 点、ρ 差 0.0006），但 SMEM 与 L2 两道在所有已
      标定形状上共线，"是 SMEM"是微架构论证而不是拟合结论，须在 sm_90+ 重测。
- [x] `[!]` **`T_quant` 不是加项**：CTA 按 `num_sms × ctas_per_sm` 分波，
      **尾波按它自己的活跃 SM 占比 `o = active/num_sms` 重新代入资源向量
      求值**。这一改把 ρ 从 0.9095 抬到 0.9450、实测最优的模型排名从 33 抬到
      19；加性修正 `(⌈count/M⌉M − count)/count · t_task` 做不到，因为尾波
      的每 CTA 时长本身就随活跃 SM 数变。`count` 仍由 barvinok 给。
- [x] `T_sync`：grid barrier 走标定出来的 `grid_barrier_*` 曲线；split-K 的
      combine 走 Stream-K 形式 `a + b·[peers>1] + c·iters + d·(peers−1)`，
      其中 `b`/`d` 直接读标定表。cluster 常数已标定但还没有 CG 边用到它
      （P4.7 在 sm_89 上编不出簇）。
- [ ] `T_bubble`：软件流水气泡 + Label 带来的 smem 占用延长。**未纳入**。
      §2.2(b) 的填充深度 `d = stages·resident_tiles_per_SM − 1` 在 12 个标定
      点上**不可辨识**——带填充深度的包络与那条直线是同一条线的重参数化，
      恢复出来的 per-CTA setup 常数会继承占用率缩放——所以默认关闭；
      Label 的 smem 项要等簇真跑起来才有可测的东西。
- [x] **离线标定**已完成，产物是 `configs/targets/sm_89.json`：六条资源道的
      速率、L2/DRAM 延迟、smem 带宽随占用率的非平坦曲线、
      `atomicAdd`/`NamedBarrier`/grid barrier 的延迟与争用曲线、Stream-K 四
      参数、并发干扰系数。带机器空闲守卫，忙 GPU 上拒绝出证。
      - [x] `[!]` **干扰门槛被触发了**：DRAM 密集邻居把 GEMM 从 18.3 µs 拖到
        27.6 µs，`interference_ratio = 1.518`，偏差 **51.8% > 30%**。
        「各算子独立时长可加」这条假设在本目标上**是假的**，这是本条明确
        要求实测而不是假定的量，量出来越线了。回应不是退化为「粗排 + 实测
        top-3」，而是让稳态取资源道的 `max` 而不是取各算子孤立时长之和——
        并且验收改成排序验收（下条），本来就不依赖绝对预测值。
- [x] **验收口径写死为回归标准**：不按预测误差验收，按排序验收，验证集是
      `docs/experiments/ORACLE/` 已有的 **2154 个实测点**（两模型各 1077 个
      配置），不需要新的 GPU 时间。三条：模型 top-3 至少一个落进实测
      top-3%（rank ≤ 32/1077）；ρ 明显高于层2 解析排序；单配置求值 < 1 ms。
      实测 ✅ **3/3**（含 top-1）、0.9450/0.9435 vs 0.4608/0.4462、最坏 33 µs。
      绝对预测值不在口径里，也确实不准（gqa2 预测 0.094 ms 对实测 0.166 ms）。
- [ ] regime 判别：从 (batch, seq_len, prefill/decode 比例) 判 A/B/C。**未实现**。
- 证据：`docs/experiments/CALIB/result.md`；`docs/experiments/COST_MODEL/result.md`

### P4.5 层3 链上 DP（Reparam + Coarsen）

- [x] `DP[i][s] = min_{s'} { DP[i−1][s'] + Cost_i(s) + Interface(s',s) }`
      （`lib/Solver/ChainDP.cpp`，驱动 `tools/tilemega-solve.cpp`）。
      **驻留度必须留在链外**：§4.3 的 `ctas_per_sm` 是整个 kernel 的属性
      （smem 取各算子选择的并集、寄存器取最大值），不能逐层决定。求解器因此
      枚举驻留档、钉住一档、只放行够得着这一档的候选、再在所有档里取最好。
      sm_89 上这两个候选集有 3 个可行档。
- [x] 分叉处（gate/up 并行）：链的分解按 series-parallel 做，分叉两支强制同
      状态；这一条与「统一 `g`」在本模型上答案相同，被单测钉住。
- [x] 复杂度是测出来的不是断言的：`ChainDpStats` 报转移数与墙钟，
      与 `O(L·|C|²)` 一致（数字见 `SOLVER/result.md`「Complexity and solve
      time」）。通用转移循环与可分离捷径两条路径**互为对照**，目标值之差
      被打印出来。
- [x] `s` 的状态含 split 因子，`Interface` 计 split-K 的 reduction stage。
- [x] **验收 (a)**：统一 `g` 落进 oracle 实测 top 3%。DP 在两个模型上都选
      `16x64x16s2k16`、`ctas_per_sm = 2`；在 25 进程终选榜上 gqa2 排 **2/15**
      （0.183296 ms 对最好的 0.181248 ms，+1.13%）、mha4 排 **1/13**（就是它
      自己）。✅ 两个模型都通过。
- [x] **验收 (b)** 逐算子 `g` vs 统一 `g`，四臂真编译真跑、400 轮交错配对：
      gqa2 `per_op` 对 DP 自己的 `uniform` **−1.214%**
      [−1.371, −1.100]，p = 1.4e−14；mha4 **−0.769%** [−0.962, −0.645]，
      p = 1.4e−22；gqa2 对**任何人测过的最好统一配置** −2.820%。
      两条零效应对照（mha4 上 DP 的 split-only 与 per-op 是同一个计划、
      DP 的 uniform 就是 oracle 的最好统一配置）都给出中位数 0.000%、
      CI 半宽 ~0.15%，这是协议自己的地板。
      - `[!]` **对 §0.3 期望的负向结果，如实记录**：逐算子只值 ~1%，不是
        4.9×–12.1×。那个跨度是「每个算子相对自己最优的改善幅度」的范围，
        而改善幅度最大的算子同时也是主导总时间的算子，所以它已经被统一
        选择收走了；留给逐算子的是小算子的残差。不调期望去迁就，也不调
        实现去凑期望。
      - 实现代价是真的：`GemmStageTaskBody` 现在最多编 4 个 `GemmVariant`，
        共享存储是**变体的并集**——gqa2 的逐算子计划混用 `16x64x16s2` 与
        `16x32x16s2`，kernel 的 smem 停在 10496 B（大的那个），窄 tile 一个
        算子换不来占用率。这就是 §4.3 全局性的具体形态。单变体构建走
        `#if` 直连而不是变体 `switch`，所以三条单变体臂与变体特性出现之前
        逐字节相同。
      - `[!]` 协议在出数之前先被修过一次：按臂成块跑时，mha4 上两个**编译
        产物相同**的臂"显著"分开了 0.64%（p = 1.76e−4）。会话内漂移压过
        采样噪声，成块布局把漂移混叠到臂身份上。改成交错 + 配对之后才有
        意义。这条写进 F-46。
- 证据：`docs/experiments/SOLVER/result.md`；`chain_dp_test`

### P4.6 Coarsen（事件粒度 κ）

- [x] 实现 `C_κ = ⌊·/κ⌋ ∘ C`（`CouplingRelation::Coarsen`，与 floor 映射
      复合）。P3.7 的迁移把它从"做不到"变成"一行 `isl_map_apply_range`"。
- [x] `[!]` **已实测，结论是不爆炸**：κ ∈ {1,2,4}，关系与拟多项式都保持
      单片，文本长度近乎不变（100 → 96 字符），带符号参数 `S` 只多约 15 个
      字符。**兜底不需要：κ 不必限制为 2 的幂**，受限矩形代数也不需要。
      数据见 `docs/experiments/P3_ISL/`（`coarsen_probe.cpp`）。
      两条代数律作为回归断言：κ=1 是恒等、`⌊⌊·/2⌋/2⌋ = ⌊·/4⌋`——后者是
      发现"新鲜坐标名与已粗化的 range 名撞车、isl 读成 `q1 = floord(q1,2)`
      从而静默塌成单点"这个 bug 的那条断言。
- [x] κ 消融曲线已在**最终任务队列实现上全部重测**：
      κ∈{0,1,2,4,8,16,32}，两模型各 25 轮交错、每臂每轮一个全新进程，
      350/350 PASS。κ=1 是两者的实测 argmin，相对 κ=0 快 **0.285%**
      [0.031,0.481]（gqa2）与 **0.165%** [0.000,0.401]（mha4）。旧
      stage-loop 及两个中间队列实现的数字均不得复用。
- [x] 收益侧第一次真实存在：κ=1 下强制 `kAll` 对 exact window 的六格
      配对 CI 全部分离，但方向随序列长度反转。seq=4 时 exact 快
      0.694%/0.322%，seq=128 时 all 快 0.182%/0.326%，seq=512 时 all 快
      1.171%/1.163%。它证明窗口被执行消费，也证明更细并不必然更快。
- [x] 纳入 DP 状态——当前结论仍是**不纳入**，但依据已重写：两个模型的新
      queue-era argmin 都是固定 κ=1，一个未观察到随模型变化的选择暂不增加 DP
      维度。保留所有 κ 编译开关，以便图结构或负载改变时重新检验。
- 证据：`docs/experiments/COARSEN/result.md`；`docs/experiments/P3_ISL/`

### P4.7 层4 Label（簇划分）

- [x] 通信量矩阵：权重是 §4.3 的 `w(A,B) = Volume × Frequency`，两个因子都从
      `CouplingDerivation` 的派生量里取（`volume` = `|W_p(y) ∩ R_c(x)|` 每对
      生产/消费的元素数，`count` = 消费者 task space 的基数），不是估的。
      gqa2 36 节点 44 边、mha4 64 节点 72 边，0 条推不出来。
- [x] 带尺寸约束的图划分（`lib/Solver/ClusterLabeling.cpp`）：重边聚合，
      三条约束都是硬约束——尺寸 `Caps<Arch>::kMaxClusterSize`、时间邻近、
      smem 预算。最大化簇内 volume 是 NP-hard（含最大权 k 路匹配），所以这
      是启发式并**被标成启发式**：`internal_weight / total_weight` 把它与
      够不着的最优之间的差距显示出来，而不是假定为零。越界的边抛异常，
      不是丢掉它再拿一个偷偷变小的图去报捕获率。
- [x] 时间邻近约束按**整组**的 stage 跨度检查（`first`/`last`），不是按提出
      合并的那一对，否则约束会被链式合并绕过。
- [x] smem 占用计入约束（预算 = `max_smem_per_sm × max_cluster_size`）。
      计入 `T_bubble` 的那一半仍未做，见 P4.4。
- [ ] `[!]` 消融：Label 开 / 关的端到端对比 **未执行**——手上是 sm_89，
      `Caps<Sm89>::kCluster` 为 false，簇 kernel 在这台机器上编不出来
      （这是特意的：`static_assert` 让簇形状的 kernel 无法静默退回平坦
      barrier 再继续报时间）。已经建好并交叉验证过的部分：生成器发
      `TILEMEGA_GENERATED_CLUSTER_DIM`，`GridBarrier` 在 dim>1 时走
      `ClusterSync::StageBarrier`，启动走 `cudaLaunchKernelEx`，grid 被裁成
      簇的整数倍。sm_90 与 sm_120 交叉编译：dim 1 有 0 条 `UCGABAR`、
      dim 2 与 dim 8 各 8 条（PTX `barrier.cluster` 0/4/4）；默认构建的 SASS
      与加簇之前逐字节相同。`docs/experiments/CLUSTER/run_on_cluster_gpu.sh` 把
      megakernel 臂也备好了，缺的只是一台有簇的机器。
- [x] `[!]` 解析上的**先行结论，方向是负的**：在本 ABI 下时间邻近只能取
      reach = 1（stage 串行的 megakernel 每个 stage 复用 smem），此时簇能
      关住的流量只有 **13.6%（gqa2）/ 18.7%（mha4）**。把 reach 放到 4 才
      到 0.97~0.99，而那要求四五个算子的输出跨 stage 边界一直待在 smem 里，
      正是这个 megakernel 不做的事。这正面支持 §4.5 自己的论断
      （簇的粒度匹配「算子内跨 CTA 归约」，不匹配「算子间数据流」）——
      现在它是一个数而不是一句断言。
- 证据：`docs/experiments/CLUSTER/result.md` §7.5 / §7.7；`cluster_labeling_test`

### P4.8 层5 Place

Round 5 更新：新增 L-sched `mapping_mode="balanced", resident_only=true`，
以共享 runtime 投影的 task DAG 最大化本地入边，受原最大队列长度上限约束。
映射4已被 host/queue 消费；35/35 CPU 测试及 BF16 400/400 新进程正确性通过。
但四格 L2 配对比值 1.640/3.087/1.397/4.002，显著变慢，虽 wait 数下降且与
CG 200/200 对账相等。原 `ListScheduler` 单独不足以优化 locality，新版仅按
task 数均衡同样不足以优化执行时间；不能宣布 Place 最优求解完成。
默认映射不变，负结果见 `docs/experiments/PLACE/round5_balanced_result.md`。
这是当前 Place 性能门的局部失败，不阻塞独立 Fusion/(a) 实现；Phase 5 的
完整参数化求解条件不因映射接入而关闭。

**当前状态（任务队列实现）**：

- [x] `ListScheduler` 的关键路径优先序按 runtime variant 写入
      `ScheduleStageDesc`；host 绑定 `seq/past`、split-K 和 resident grid 后，
      展开成每个物理 worker 的有序 `TaskRef` 列表。L2 kernel 实际读取这张表，
      不再以 `stage=0..N` 作为外层循环。
- [x] 生成期 `Validate` 拒绝有环、非置换和反向边；host 对 split-K 展开后的
      最终顺序再验证。5-node 小图穷举 120 个排列、3 个可行拓扑序，关键路径
      序达到 oracle 最小依赖跨度 2。
- [x] 硬件重测使用同一二进制的 `TILEMEGA_SCHEDULE_POLICY=critical_path` /
      `round_robin` 开关，各 25 轮配对、50/50 正确。轮内比值为
      **1.000000 / 0.998552**；gqa2 CI 跨 1，mha4 bootstrap CI 轻微偏向
      round-robin 但 Wilcoxon p=0.063，两者都远低于 2% 判据；Place 已成为真实
      执行变量，但关键路径序未带来可声明的性能收益。下方旧实验全部作废。

#### 已作废的 pre-queue Place 实验（只保留失败史）

- [x] 时间局部性：`|R(c₁) ∩ R(c₂)|` 用 barvinok（`place_probe.cpp`，
      `isl_map_card` 精确计数，不抽样）。两模型 126 个操作数、0 个数不出来：
      gqa2 26 退化 / 20 有结构，mha4 44 / 36。有结构的只有一种事实的三种写法
      ——GEMM 的 A 操作数沿 `n` 全共享（`n=524288`、宽的那条 `1835008`）、
      沿 `m` 零共享；注意力的 K/V 沿 chunk 轴全共享、跨 head 零共享。
- [x] oracle 判断投入价值。**先做这一条，结论与预期相反，两条都记下来**：
      60 轮交错配对、5 个臂、两模型 600 个全新进程，全部 60/60 PASS
      （置换会同时改 task 下标、`active` 与事件组，任何一个不同步就是竞态，
      所以每个臂都必须每次都对）。噪声底由 `ident2`——与 `ident` 逐字节相同
      的第二个二进制——现场测出：`l2_ms` 上 0.148% / 0.144%。
      - **§4.3 那个目标函数值 <2%，按骨架自己的判据应当简化**：`scatter`
        （`31·b mod g`，把下标邻接彻底打散）在 L1 上只有 **+1.204% / −0.751%**
        ——一个模型略慢、另一个略快。若下标邻接真承载复用，它应该是最贵的臂。
        所以基于时间局部性的 list scheduling 不值得建。
      - **但 `pair`——唯一被该目标函数推荐的置换——是最差的臂，L1 上
        +17.017% / +18.172%（p = 1.7e−11）**，而且原因可从映射本身推出：
        `grid=256, ctas_per_sm=2, num_sms=128`，pair 把逻辑下标 `L` 放在
        SM `⌊L/2⌋` 上，于是有 `A` 个活跃 task 的 stage 只用 `⌈A/2⌉` 个 SM，
        恒等映射用 `min(A,128)` 个——**凡是填不满 grid 的 stage 都只剩半台机器**。
        这不是启发式的副作用，这就是启发式本身：让同 SM 的 CTA 拿连续下标，
        等价于把低位下标挤到少数 SM 上。§4.3 的目标函数看不见占用率。
      - **`reverse` 在 L1 上快 7.053% / 6.939%（p = 1.7e−11），机制未查明**。
        已排除：不是局部性（reverse 逐位保持邻接，而毁掉邻接的 scatter 只值
        ~1%）；不是占用率（`g−1−b` 下活跃集落在硬件 CTA `{g−A…g−1}`，每 SM
        的 CTA 数与恒等映射同一个多重集，只换了驻留槽）；不是 barrier 结构
        （`GridBarrier` 没有主 CTA，最后到的那个负责 notify）。L1 上的效应比
        L05 大 5~6 倍，指向反复驻留而不是单次启动的调度。这条**不上车**：
        7% 是真的、可复现的、两模型一致的，但机制不明的置换换个 fixture 或
        换个架构就可能反号。记为已测出的余量，不记为已解决。
- [ ] 分层 DAG 上的 list scheduling（关键路径优先）——**未做，且测量说不要在
      这个目标函数上做**。留空而不是打勾：真正该建的是什么还没确定。
- [ ] 掩盖同步延迟：队列顺序让等待被独立 task 填充——未做。
- [x] **TaskBody 所有权 Place（与上面的时间局部性目标正交）**：RoPE 的
      小实验满足“边窄化、poll 显著转移、body 代价可接受”三项判据后，推广到
      KVAppend / activation / split-K combiner。seq=128 时 exact poll −41.40%，
      L2 会话内配对改善 13.57% / 16.32%，body-only 代价 1.00% / 1.25%。
      这改变的是 CTA 对逻辑 task 的所有权映射，未引入或声称 cache-local list
      scheduling；详见 `docs/experiments/OWNERSHIP/`。
- [!] 解析半边与硬件半边测的不是同一个尺寸：`place_probe` 绑 `S = 512`，
      E2E fixture 是 `seq = 4`。后者在 `tile_m = 16` 下**只有一个 M 分片**，
      于是 `w(c₁,c₂)` 是常数、目标函数在整个置换群上恒等——五个臂的目标值
      完全相同，而硬件上它们相差 24 个百分点（L1 上 pair +17.0 到 reverse −7.1）。这条先于测量写下：预测对了
      "目标函数是平的"，错在由此推出"置换不值钱"。目标函数是平的只说明它是
      瞎的。两个陈述都不外推到对方的尺寸上。
- 证据：`docs/experiments/PLACE/result.md`、`docs/experiments/OWNERSHIP/result.md`；
  `raw/affinity.txt`、`raw/place_summary.txt`

### P4.9 实现契约与一致性校验

- [x] `tilemega.implementation @impl for @task`：backend / tile / cluster /
      stages / threads / smem / alignment / arch_required + 逐操作数的
      access（坐标 + span）。`regs_est` 是**可选**属性而不是 0——CUTLASS 的
      `constexpr` traits 不给寄存器数，只有 tier 3 的 ptxas 日志给。
- [x] `(g, impl)` 是一个决策：`SelectImplementation` 选中的 tile 就是给
      task space 的 access map 绑定 `g` 符号的那个 tile，契约里的 span 由它
      导出而不是另写一份。
- [x] verifier 两半：`VerifyTraits` 从后端闭式重算 threads/smem/alignment/arch，
      实现可以选形状但不能改写这个形状的代价；`VerifyAccessAgainst` 把声明的
      访问模式与 task space 的 `index_map` 逐轴比对，能抓 **tile 形状不匹配**
      与 **操作数坐标转置**（F-17 的形状：CUTLASS 的 B 操作数逻辑上是
      `(N,K)`，弄反一次产生 6143 个不匹配元素）。转置这条 negative 特意跑在
      `Tm = d = 128` 的粒度上，交换的两轴等宽，因此 span 比较**不可能**是
      拒绝它的原因。
- [!] IR 侧的访问校验只在 task space 带 `index_map` 时生效，而前端 importer
      目前不发这个属性（它直接从 FX 节点建 task space，没有 `OperatorNode`）。
      没有 `index_map` 时访问模式是不可证伪的——正是 F-17 被发现时的状态。
      不用一个静默通过的检查掩盖它。
- 证据：`docs/experiments/CONTRACT/result.md`；`impl_contract_test`

---

## Phase 5：符号化与运行时

**进入条件（Phase 5 不得绕过）**：

1. ✅ 生成器可在一个 `ModelSpec` 中携带同一算子的多粒度实例，并为每个
   `seq` 区间携带独立、精确的依赖表；host 已是 O(1) 区间查表。
2. ✅ smem/register/occupancy 曲线已量化：1/2/4/8/16 变体对应
   5/5/4/2/1 CTA/SM，因此第一版区间划分以 **2 个变体/binary** 为无损上限；
   不能只根据 C++ `sizeof(union)` 判断，4 变体的首个拐点来自寄存器。
3. ⚠️ **部分满足。** BF16 Tensor Core 已贯通并有独立 sm_89 标定（✅ 两模型
   50/50，SASS 各 96 条 BF16 HMMA）。修正 128-thread occupancy 与 per-SM smem
   预算后 ρ = **0.8984 / 0.8871**，仍低于 FP32 0.9432 / 0.9421，且
   top-1/3/10 全为 0，**区间边界只能当作初值而不是定论**。模型同时高估
   窄-N split=1 与激进 split=8/16；不是可用全局 scale 修掉的误差。
   **c8be09e 后状态修正**：上述 BF16 排名数字来自旧 partial 路径的有偏 PASS
   子集（mha4 的 308 个 RUNFAIL 实为数值判据失败），仅作历史记录，不能用作
   本轮达标判据；修复 partial 后的完整稳态排序尚未重测。
4. ✅ **Round5 A9.3 事件价格功能门通过（不是 A6/P5.1 全部完成）。**
   两模型十二格 κ0/1 预测差非零且方向符合实际等待数；1200/1200 队列字段对账，
   36/36 价格代入位一致。sm_89 使用十二格稳态结构标定，最长队列系数为零，
   poll 的 LOO 最大相对误差仍约95%；不将残差包装成高精度。
   `EVENT_COST/round5_structured.md` 记录完整证据。A6统一任务价格已通过：
   两个GEMM入口各904680位比较相等、FP32排名不降、四份DP计划不变；
   B入口已开放。L2候选级DP未完成；真实task符号价格5120 GEMM/768 scalar点
   已对照，完整CG-interface DP因KVAppend→attention的二次重复计数乘一次
   cache曲线产生seq³，触发B3.2停止门，(b)未退役。这不是以采样替代符号求解
   的理由。证据：`PARAMETRIC/task_prices/result.md`。以下为旧输入阶段的历史状态：
   `wait`/`fanout`/`volume`/`count` 是 `S`/`past`/`L_s` 的
   拟多项式而不是 `S_min` 上的整数（420/420 逐点复核）。这是 P5.1「代价函数
   以 θ 为参数 → DP 输出分段拟多项式」的输入前提，此前不成立。
   `div/scale/offset/count` 的三点拟合也已换成 isl 端点发现 + 全参数域集合等价
   证明，参考模型与 16 层均 0 fallback。⚠️ 本轮新增直接 CG 输入与 θ 绑定，
   输入侧 2154/2154 FP32 double 位模式相等；四个指标尚未完整进入价格计算，
   分段 DP 未完成，不能据此勾选 P5.1（`PARAMETRIC/result.md`）。
   c8be09e 后 BF16 输入 gate 为 1540/1540（含全部 308 个旧数值失败配置），
   FP32 2154/2154 保留。新功能审计在 seq=4 的首边发现 wait_sum=512、
   fanout_sum=16：前者含 nominal tile 的 producer 范围，后者限 actual domain。
   这两个 QP 不能直接当作同一物理图定价；event_ns 尚未接入，T1.5 未通过。
   详见 `EVENT_COST/result.md`。方案 (a) 是终态目标，旧有限枚举 (b) 不作完成标记。
5. ⚠️ L2 已改为 worker task queue 并在 50 进程观察到平均 34.36%/40.18%
   跨 stage 提前启动；
   κ=1 的六格 L2/L1 为 **1.081–1.113**，仍未胜过 L1。新的四臂归因见
   `docs/experiments/L2_ATTRIB/`，旧 stage-loop 分解已作废。
6. ✅ `seq×past` 矩阵 1500/1500，且直接删除 `TaskWait` 的新负测试 0/50；
   修改已退役 stage wait 的旧 clamp 仍为 50/50，两者共同证明守门覆盖活跃路径。
7. ✅ **Round5 定性关闭：判据产物，已归因。** 依据同输入的深度扫描、
   common-FP32 噪声地板及 golden 线程敏感性，不再作为本轮数值修复前置；
   未修改逐元素容差、未把历史失败改写为 PASS。历史记录：
   4×4096 / intermediate=14336 为 50/50；完整
   16×2048 的 973M 参数 decoder 虽然可生成、编译和运行，但固定 BF16 判据为
   0/50。✅ c8be09e 后固定输入/权重前缀深度扫描为 6 深度×50=300 进程：
   2/4/6 层各 50/50，8/12/16 层各 0/50；max_abs 随深度为
   .015625/.03125/.03125/.046875/.0625/.078125，未见定性跳变。
   16 层相对共同 FP32 golden 的 L2 误差，TileMega/PyTorch BF16 比为
   .9913284393，支持累积/噪声地板归因，但没有采用新的 k 判据。
   CPU golden 线程数 8→56 单独复现 mismatch 162→198，TileMega 哈希不变；
   这不是运行时精度提升。`REALMODEL/depth_result.md` 保留全部证据与局限。
   不能仅由层间 bit-identical 排除所有共同实现误差。论文端到端主张不能把这项写成完整
   1B 已验证。
8. ⚠️ 32 次 host 迭代的正反构建都 50/50，说明当前 harness 结构上触发不了
   ABA；增长 KV cache 与 kernel 内迭代必须在 Phase 6 的 serving loop 一起验证。
9. ✅ **Round5 A0 定性关闭：判据产物，与条件7同因。** 原始失败的
   4×4096 split8 相对共同 FP32 的 k_L2=.9961308506568204，落在用户指定的
   [0.989,1.003] 归因区间；56线程CPU BF16 golden 逐位复现。
   原二进制经用户授权一次采集，hash/diff 复现原始失败；不修改容差。
   `REALMODEL/condition9_result.md` 留存方法、三比较和边界。T2.d 数值可行域线取消。
   以下旧结论保留为历史，不再代表本轮待办：历史代价模型在 4×4096 上排名最高的 split-K 16/8 配置都违反未放宽的
   BF16 判据；最佳 split=1 候选则 50/50，并在 25 轮配对中使 L1 快
   28.05%。因此 P5 的 DP 在比较代价之前必须有数值可行性约束；不能让
   不可验收的 split-K 方案成为分段最优。本轮 FP32 partials 修复在两参考模型
   seq=128/past=3、split=1/2/4/8/16 各 50/50（500/500）；该修复消除了这组
   配置的失败，未重跑的 4×4096 最优配置不能直接宣布通过。流量模型同步增加
   partial 字节数；973M 条件 7 与数值可行域的全域证明仍未解决。
   c8be09e 已提供 sm_120 同组 500/500 的跨架构原始证据。✅ 本轮原始
   4×4096 最优 split8 在 FP32 partials 下仍失败：1 个超差元素，
   max_abs=.046875，0/1 后触发停止，split16 未运行；不是 50 进程验收。
   **条件 9 仍未关闭**，数值可行域设计与原始日志见
   `REALMODEL/condition9_result.md`，未禁用大 split 或放宽容差。
   FP32 dtype 的 partial 开关 GPU 回归另为 400/400；combine 新速率未实测。

⚠️ 这些是 Phase 5 前置的**实测状态**，不是全绿的启动条件。尚未完成的硬项是：
代价模型消费拟多项式、BF16 达到 FP32 排名/命中目标、真实规模 per-operator DP
验证，以及能触发 ABA 的增长 KV-cache 迭代。下面的参数化 DP、交点求解或 serving
本身也尚未实现。

### P5.1 参数化解

- [ ] 代价函数以 `θ` 为参数 → DP 输出分段拟多项式
- [ ] 求分段交点 → 最优解的区间划分
- [ ] 每区间生成一套模板实例化（受 smem union 与编译时间约束）
- [ ] 区间数上限：P6.2 的平台期（top-34 在 ±10% 内）说明相邻区间的最优解
      多半互相可用，所以「求精确交点」应当让位于「合并差异 <10% 的相邻
      区间」——这直接决定要编多少个变体，而变体数是 P5.2 的硬约束

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
- [x] **划分 oracle**：固定 Place 与事件结构，穷举 `g`（tile 形状 × stages ×
      split 因子），两个模型各 1077 个可运行且逐位正确的配置。固定值
      `128x128x16s3k1` 排 951/1077 与 960/1077，最优比它快 **6.11× / 6.75×**；
      差距全部落在 GEMM stage（占比 93% → 67%，两模型的 14 个 / 28 个 GEMM 无一劣化）；
      分布是「少数配置显著好」——5% 内只有 10 个，中位数是最优的 2×，
      最差 1489× / 1572×。总耗时 4846.56s。
      结论与两条限定见 P4.4；数据见 `docs/experiments/ORACLE/result.md`。
- [x] Coarsen（κ）消融：最终 logical-task 事件队列上完成 7 臂、350/350；
      κ=1 为两参考模型的实测 argmin，详见 P4.6 与 `COARSEN`
- [ ] Label（簇）消融
- [ ] 混合 batch regime 对比
- [ ] Warmup 时间（目标：0 次 CUDA graph capture）
- [ ] 端到端对比

### P6.3 消融

- [x] L1 → L2（细粒度事件的贡献）：六格各 25 轮配对，L2/L1 =
      1.081–1.113；四臂分解逐轮精确闭合，详见 `E2E_L2` 与 `L2_ATTRIB`
- [ ] L2 → L3（Reparam + Coarsen + Label 的贡献）
- [ ] 符号化的贡献（vs bucketing）
