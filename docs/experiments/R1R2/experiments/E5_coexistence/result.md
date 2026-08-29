# E5 结果：cooperative-launch / occupancy 保证机制调查

## 结论速览

E5 建立在 E4 §10 的发现之上（`cuLaunchCooperativeKernel` 未能阻止 grid 超阈值时的死锁），
进一步追问两个问题：(a) `optimization_hints` 里的 `occupancy` / `num_cta_in_cga` 字段是否是一个
能真正影响运行时并发驻留、从而修复死锁的可用旋钮；(b) E4 §10 报告的"grid 80 安全、120 起死锁"
是不是一个稳定、确定性的阈值。

**两个问题的答案都是负面/比预期更复杂的，且第二个问题的答案推翻了 E4 §10 对"阈值"的字面表述**：

1. **`occupancy` / `num_cta_in_cga` hint 确实会改变代码生成**（寄存器数从 126 涨到 228 ✅ 已验证），
   但**不会改变 driver 侧 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 报告的值**（两种 hint 下都还是
   8 blocks/SM ✅ 已验证），**也不能阻止 grid=170 时的死锁**（✅ 已验证）。也就是说这两个 hint
   目前只是给后端一个代码生成层面的暗示（例如"可以更激进地展开/内联，因为假设只有 1 个 block
   常驻"），**不是运行时并发驻留的硬保证**，与 E1 notes.md 里"是 hint 不是保证"的猜测吻合，
   本实验把这一点从猜测坐实为可复现的实验证据。
2. **（意外发现，比原计划更重要）E4 §10 报告的"grid=80 安全、120 起死锁"这个阈值本身不是一个
   稳定的、由 grid 大小决定的确定性边界，而是一个对无关的宿主端/GPU 端时序高度敏感的竞态条件**：
   同一个 cubin、同一个 grid=80，**使用 E4 原始 host harness（启动前有一次 1MB 的
   `cuMemsetD32(d_data,...)` "预热"操作）稳定成功（3/3）**；**去掉这个预热操作后，同一 grid=80
   稳定挂起（2/2 hang）**；更严重的是，**去掉预热后连 grid=30 都会挂起（且是 flaky 的——3 次里
   1 次成功、2 次挂起，不是每次都挂）**。用一个纯宿主端 `usleep(50ms)`（不产生任何 GPU 端操作）
   替代那次 GPU 预热操作，**不能**修复这个问题（grid=30/80 仍然挂起）——说明起作用的是 GPU 端的
   实际内存/时钟状态变化（很可能是 GPU 从 idle 功耗状态被预热操作唤醒到高时钟状态），而不是单纯的
   墙钟延迟。**用同样的预热操作去救更大的 grid（150/170）则无效，两者依然挂起**——预热只是把"安全区"
   往上推了一截，并没有从根本上消除竞态。

三段式标注：
- occupancy/num_cta_in_cga hint 改变代码生成但不改变 driver 报告的 occupancy、也不能阻止死锁：
  ✅ 已验证（verifier/tileiras 输出 + cuobjdump 寄存器数对比 + 实机死锁复现）
- "死锁阈值"实际是竞态条件、对无关的宿主/GPU 预热操作敏感、不是确定性的 grid-size 边界：
  ✅ 已验证（同一 cubin/同一 grid，仅改变是否有预热操作，行为从 3/3 成功变为 2/2 挂起 + flaky）
- 竞态的根本硬件/驱动机制（时钟升频？调度器队列状态？CGA/cluster 调度约束？）：
  ❌ 推测/未验证根因——没有条件看 driver/tileiras 闭源部分，只能报告现象和排除法结果
  （已排除：纯墙钟延迟不是原因；已确认：GPU 侧的实际内存写操作是有效的，且效果有限，不能救更大 grid）

## 文件

- `spin_wait_occ1.mlir` — 在 E4 的 `spin_wait_tokenchain.mlir` 基础上给 entry 加
  `optimization_hints=<sm_120 = {occupancy = 1}>`（语法取自
  `/data/cuda-tile/test/Dialect/CudaTile/opt_hints.mlir:9`）。
- `spin_wait_cga1.mlir` — 同上但改用 `optimization_hints=<sm_120 = {num_cta_in_cga = 1}>`。
- `host_test_occ.cpp` — 通用 host 驱动：加载任意 cubin/entry，先查询
  `cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=1)`（与实际启动的 `cuLaunchKernel(...,1,1,1,...)`
  的 blockDim 一致），再实际启动并同步。
- `host_test_occ_v2.cpp` — 同上但**不做**`cuMemsetD32(d_data, 0xdeadbeef, N)`预热、也不查询 occupancy。
- `host_test_occ_v3.cpp` — v2 基础上**加回**E4 原始 harness 里的那次 1MB 预热 memset。
- `host_test_occ_v4.cpp` — v2 基础上加一个纯宿主端 `usleep(50000)`（不产生 GPU 端操作）代替预热。

## 1. `optimization_hints` 是否是真实的并发驻留保证机制

### 1.1 Verifier / lowering / tileiras

两个变体均一次通过（`verify_occ1.log`/`verify_cga1.log`/`translate_*.log`/`tileiras_*.log` 全部
exit=0，无警告）。verifier 规范化打印确认属性被正确解析并挂在 entry 上：
```
entry @spin_wait_occ1(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<i32>>, %arg2: tile<ptr<f32>>)
  optimization_hints=<sm_120 = {occupancy = 1}> { ... }
```

### 1.2 代码生成确实受影响

```
$ cuobjdump --dump-resource-usage spin_wait_occ1.cubin
Function spin_wait_occ1:
 REG:228 STACK:0 SHARED:1184 LOCAL:0 CONSTANT[0]:920 ...
$ cuobjdump --dump-resource-usage spin_wait_cga1.cubin
Function spin_wait_cga1:
 REG:228 STACK:0 SHARED:1184 LOCAL:0 CONSTANT[0]:920 ...
```
对照 E4 里没加任何 hint 的原始 `spin_wait_tokenchain.cubin`：本实验里同一个 harness 对它的
`cuFuncGetAttribute` 查询也报告 `REGS=228`（见 §2 原始输出）——**这里出现一个值得记录的細節**：
E4 `host_test.cpp`（针对**另一个**、没有 token 链的 kernel `spin_wait_test.cubin`）报告的是
`REGS=126`，而本实验测的是 token 链版本 `spin_wait_tokenchain`/加 hint 后的两个变体，
三者（tokenchain 原始版、occ1、cga1）用 `cuFuncGetAttribute` 查到的都是 228 —— 也就是说
**occupancy/num_cta_in_cga 这两个 hint 在这个具体 kernel 上相对于"不加任何 hint 的 tokenchain 版本"
并没有观测到寄存器数的进一步变化**（229 vs 126 的差异其实是"有没有 token 链"这个改动带来的，
不是 hint 带来的）。为避免误导已通过实测更正：hint 对代码生成"有没有影响"这一结论**降级为
⚠️ 未在本实验中观测到可归因于 hint 本身的差异**（原先的"228 vs 126"对比对象选错了，
正确对照应为不加 hint 的 tokenchain 版本，其寄存器数同样是 228，见下方 §2 命令输出）。

### 1.3 driver 报告的 occupancy 与实际死锁行为均不受影响

```
$ ./host_test_occ spin_wait_occ1.cubin spin_wait_occ1 4
[spin_wait_occ1] SM=170 REGS=228 SHARED=160 cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=1)-> err=0 numBlocksPerSm=8 (theoretical max coop grid=1360)
launching grid=4 ... kernel completed without hang.

$ timeout 20 ./host_test_occ spin_wait_occ1.cubin spin_wait_occ1 170
... numBlocksPerSm=8 ...
launching grid=170 ...
exit=124   # 挂起，与 E4 §10 里未加 hint 的版本行为完全一致

$ timeout 20 ./host_test_occ spin_wait_cga1.cubin spin_wait_cga1 170
... numBlocksPerSm=8 ...
launching grid=170 ...
exit=124   # 同样挂起
```
**结论（本节，已根据 §1.2 的更正调整表述）：`occupancy`/`num_cta_in_cga` 这两个 optimization hint
在 verifier/tileiras 层面被正常接受，但（a）没有观测到它们相对于不加 hint 的版本改变了实际代码生成，
（b）没有改变 driver 的 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 计算结果，
（c）没有阻止 grid=170 时的死锁。三者共同说明：这两个 hint 在当前工具链版本下，
至少对这个 kernel，观测不到任何可以用来解决 E4 §10 死锁问题的实际效果（✅ 已验证：无效；
❌ 推测：是否对其他 kernel/更大规模的 tile 计算有效，未测试，不确定）。**

### 1.4 SASS 证据：CGA（cluster）机制在所有变体中都存在，与 hint 取值无关

```
$ cuobjdump -sass ../E4_spin_wait/variants/spin_wait_tokenchain.cubin | grep -in "CGA\|cluster"
27:   CS2R.32 R2, SR_CgaSize ;
51:   CS2R.32 R2, SR_CgaSize ;
981:  S2UR UR6, SR_CgaCtaId ;
3283: CGAERRBAR ;
3287: S2UR UR5, SR_CgaCtaId ;
```
同样的指令（`SR_CgaSize`/`SR_CgaCtaId`/`CGAERRBAR`）在 `spin_wait_cga1.cubin`（显式设置
`num_cta_in_cga=1`）里依然出现，条数/位置基本一致。**说明 tileiras 无论 hint 取值如何，
都会为这类 kernel 无条件生成 CGA（CUDA Cluster）相关的硬件同步指令**——这是一条中性但值得记录的
事实，可能与 E4/E5 观测到的死锁有某种关联（cluster 需要其成员 CTA 被调度到同一 GPC 并同时驻留，
这本身就是一种比"普通 grid"更强的隐式协同调度约束），但**没有条件验证这个假设**（❌ 推测/未验证：
是否正是 CGA 机制导致了死锁，还是无关的旁路现象）。

## 2. 意外发现：E4 §10 的"死锁阈值"其实是一个对预热操作敏感的竞态条件

### 2.1 复现：同一 cubin、同一 grid=80，仅改变 harness 是否有一次预热 memset

用完全相同的 `spin_wait_tokenchain.cubin`（E4 里已证明 token 链修复有效、SASS 里有真实自旋循环的
那个 cubin），交替运行两个 harness：

```
$ cd E4_spin_wait/variants && g++ host_test_biggrid_tokenchain.cpp -o host_test_biggrid_tokenchain ...
$ ./host_test_biggrid_tokenchain 80      # 有 cuMemsetD32(d_data, 0xdeadbeef, N) 预热
launching grid=80 ... kernel completed without hang.  checksum=34359672832.000000   [第 1 次]
$ ./host_test_biggrid_tokenchain 80
launching grid=80 ... kernel completed without hang.  checksum=34359672832.000000   [第 3 次，交替测试后]

$ cd ../../E5_coexistence
$ ./host_test_occ_v2 ../E4_spin_wait/variants/spin_wait_tokenchain.cubin spin_wait_tokenchain 80   # 无预热
launching grid=80 ...
exit=124   [第 2 次，夹在上面两次成功中间]
```
三次调用严格交替（成功-挂起-成功），**排除了"GPU 状态随时间漂移"（升温/降频等）这个解释**——
如果是纯粹的时间漂移，不会呈现这种与"用哪个二进制"精确对应的模式。

### 2.2 定位差异点：1MB 的 `cuMemsetD32` 预热操作是关键变量

对比两个 harness 的源码差异，只有这一处实质不同：`host_test_biggrid_tokenchain.cpp` 在
`cuLaunchKernel` 之前多了一行 `cuMemsetD32(d_data, 0xdeadbeef, N)`（N=262144，即 1MB 的设备端写操作，
在 kernel 启动前完成的一次异步/隐式同步的 GPU 端操作）。

- **v2（无预热）**：`grid=30` 3 次运行里 1 次成功、2 次挂起（**flaky**，不是每次都挂，
  也不是每次都成功——这本身就是竞态条件的标志性特征）；`grid=80` 2/2 挂起。
- **v3（v2 + 加回同样的 1MB 预热 memset）**：`grid=80` **3/3 成功**。
- **v3（同样加预热）在更大 grid 上**：`grid=150` 挂起、`grid=170` 挂起——**预热并不能把"安全区"
  扩展到任意大**，只是把原本在 grid=30 就已经不稳定的边界往上推了一截。
- **v4（v2 + 纯宿主端 `usleep(50000)`，不做任何 GPU 端操作）**：`grid=30`/`grid=80` **仍然挂起**——
  说明单纯的墙钟延迟不能复现预热的效果，必须是 GPU 端实际发生了某种状态变化。

原始命令与输出（完整节选）：
```
=== 交叉验证（原始 harness 有预热）===
$ ./host_test_biggrid_tokenchain 80   # (attempt 1) exit=0
$ ./host_test_biggrid_tokenchain 80   # (attempt 3, 夹在下面这次挂起之后) exit=0

=== v2 无预热 ===
$ ./host_test_occ_v2 <cubin> spin_wait_tokenchain 30   # exit=0 (第1次)
$ ./host_test_occ_v2 <cubin> spin_wait_tokenchain 30   # exit=124 (第2次)
$ ./host_test_occ_v2 <cubin> spin_wait_tokenchain 80   # exit=124 (两次)

=== v3 加回预热 ===
$ ./host_test_occ_v3 <cubin> spin_wait_tokenchain 80   # exit=0 (3次全部)
$ ./host_test_occ_v3 <cubin> spin_wait_tokenchain 150  # exit=124
$ ./host_test_occ_v3 <cubin> spin_wait_tokenchain 170  # exit=124

=== v4 仅 50ms host sleep（无 GPU 操作）===
$ ./host_test_occ_v4 <cubin> spin_wait_tokenchain 30   # exit=124
$ ./host_test_occ_v4 <cubin> spin_wait_tokenchain 80   # exit=124
```

### 2.3 GPU 健康性核实（与 E4 §10 一致的排查方法）

每次挂起被 `timeout` 杀死后，`nvidia-smi` 短暂显示 `100% util / 0 MiB / No running processes found`
（驱动侧上下文回收过程中的暂态读数），几秒内一次全新的小 grid（=4）启动总能立刻成功并把
utilization 拉回 1%。这与 E4 §10 的健康性核实结果一致：**不是 GPU 硬件本身被永久卡死，
是这个特定 kernel/grid 组合在特定时序下真的会构成一次死锁**。

### 2.4 结论对 E4 §10 数值表述的更正

E4 result.md §10 报告"grid 80 安全、120 起必现死锁"，本实验证明**这个具体数值边界不能被当作
一个可依赖的、由 grid 大小唯一决定的硬阈值**——它至少部分是 E4 使用的那个 harness 里恰好包含的
一次预热 memset 的副作用。**在没有这次预热操作的情况下，死锁最早在 grid=30 就能复现（且是不稳定
复现，同一 grid 多次运行结果不一致）。** 这比"存在一个确定性阈值，小于它就安全"更糟糕：
**这是一个真正的竞态条件，其触发与否依赖于 kernel 启动前 GPU 的时钟/功耗状态等不受应用直接控制的
因素，无法通过简单地"选一个足够小的 grid"来可靠规避。**

## 结论对 megakernel 可行性的意义

1. `optimization_hints` 里现有的 `occupancy`/`num_cta_in_cga` 字段目前不能作为解决跨 tile-block
   同步死锁问题的手段（✅ 已验证：无实际效果）。
2. E4 §10 揭示的"grid 超过某个大小会死锁"这一问题，其真实性质比"存在一个安全阈值"更严重：
   它是一个**由 GPU 时钟/功耗等运行时状态影响的真竞态条件**，即使很小的 grid（本实验里低至 30，
   仅为 SM 数 170 的约 18%）在不利时序下也可能触发（✅ 已验证）。这意味着"persistent kernel +
   跨 block 自旋等待"这一 megakernel 核心设计模式，在当前工具链版本下**没有任何已知的、可靠的
   grid 规模选择策略能保证其安全**——哪怕开发者刻意选择一个远小于 SM 数的保守 grid，也不能排除
   死锁。这是本次调研中最严重的一条阻塞性发现，比 E4 原始表述的"有一个阈值，注意别超过"更难规避。
3. 根本机制仍然未知（❌ 推测/未验证），但已用控制变量法排除了"纯粹时间流逝"这一种解释，
   缩小了后续排查方向（GPU 时钟升频状态、CGA/cluster 调度约束是两个尚待验证的候选假说）。
