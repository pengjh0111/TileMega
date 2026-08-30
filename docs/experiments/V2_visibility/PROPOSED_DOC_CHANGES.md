# 建议的骨架文档改动（未直接修改）

> 按任务书 §3.1：**没有编辑 `Tilemega_skeleton.md`、`README.md`，也没有改 `INDEX.md`
> 里已有的结论，没有改 R2-A / R2-D 的任何文件。** 以下全部是建议，由人复核后落笔。
>
> 每条包含：位置 / 现文 / 建议改为 / 依据 / 置信度。
> 行号基于本轮阅读时的 `Tilemega_skeleton.md`，落笔前请以关键字定位为准。

---

## C1. §6 需要新增一条规则：数据 load/store 不得用 `weak`

**位置**：`Tilemega_skeleton.md` §6「Codegen 规则」——建议新增 **§6.8**。

> 说明：任务书里写的是「§6.9」。§6.9 确实一度存在——它是 V1 直接写进骨架的，
> 已被 `5597008 "Revert skeleton edits from V1"` 整体回退（该 commit 的说明里
> 明确列出了 6.9 / 6.11 / R1' / R14）。**当前骨架 §6 只到 6.7**（6.1 token 链 / 6.2 循环内屏障 /
> 6.3 退避 / 6.4 事件计数器 / 6.5 事件缓冲布局 / 6.6 白名单 / 6.7 occupancy 查询），
> **并不存在关于 bulk data load/store 的规则条目**，`weak` 一词在骨架全文中一次都没出现。
> 所以这不是「改一句」，而是「补一条」。

**现文**：无（§6 缺少这条规则）。`weak` 的用法出现在这些**示例**里，它们是实际的传染源：

- `third_party/cuda-tile/README.md:241` — `%data, %token = load_ptr_tko weak %data_ptr_tensor : ...`
- `docs/experiments/R1R2/experiments/E3_persistent_loop/result.md:29` — `store_view_tko weak %1, %pview[%loopIdx] : ...`
- `docs/experiments/R1R2/experiments/E0_baseline/result.md:12,28`

**建议改为**（新增 §6.8）：

```markdown
## 6.8 数据 load/store 的 ordering

**跨 tile-block 通信的数据 load/store 一律用 `relaxed device`，禁止 `weak`。**

规范 §7.2：「Weak operations cannot be used to communicate through memory between
threads... The compiler may assume that tiles accessed with `weak` are not
concurrently accessed by any other thread.」

`weak` 只允许用于**确定不跨线程共享**的私有缓冲。事件标志本来就该用
`relaxed`（轮询）/ `acquire`（跳出后）/ `release`（生产者），见 §6.1；
本条约束的是**数据体**，此前没有明文规定，多处示例里写的是 `weak`。

SASS 层面的可见差别：`weak` → `STG.E` / `LDG.E`；
`relaxed device` → `STG.E.STRONG.GPU` / `LDG.E.STRONG.GPU`。
```

**依据**：**Tile IR 规范 §7.2**（以及 §7.10「数据竞争是 UB」）。
⚠️ **这一条是规范符合性要求，不是实验结论。** 实测上它对当前这个缺陷**没有影响**：
V2-d′ 把 `weak` 改成 `relaxed device` 后失败率纹丝不动（50/50 → 50/50，见
[V2_d_fixed/result.md](V2_d_fixed/result.md)）。两件事必须分开陈述，
不要让读者以为改 `weak` 能解决问题。

**置信度：高。** 规范条文明确，不依赖任何实验。唯一的不确定是条目编号（6.8 还是别的）。

---

## C2. §2.2 修正表：REQNTID 256→128 的归因错了

**位置**：`Tilemega_skeleton.md` §2.2 的「⚠️ 修正（V0，2026-08-28）」表（约 137–152 行），
以及 **`scripts/tilemega-compile` 的头注释第 14–18 行和 usage 里的 Note（第 48–51 行）**，
以及 §7.1 风险表的 **R12** 行。

**现文**：

| | tileiras 13.1 (V13.1.80) | tileiras 13.3 (V13.3.36) |
|---|---|---|
| `EIATTR_REQNTID` | **256** | **128** |

> 「同一份 `.tilebc`，只换 tileiras 版本，硬件驻留容量翻倍。」

`tilemega-compile` 头注释：
> 「tileiras ... BEHAVIOUR DIFFERS MATERIALLY BETWEEN VERSIONS (measured in V0:
> 13.1 and 13.3 emit REQNTID 256 and 128 respectively...)」

R12 行：
> 「V0 已实证：同一 `.tilebc` 在 13.1/13.3 下 REQNTID 为 256/128」

**建议改为**：把归因从 **tileiras 版本** 改成 **cuda-tile 前端版本**。

> `EIATTR_REQNTID` 的 256 → 128 来自 **cuda-tile 前端**（`8a775693` → `af241704`），
> **不是** tileiras 13.1 → 13.3。V2 用同一份 `v1_min.mlir` 在 tileiras **13.1** 下
> （bytecode 也降到 13.1）编译，得到的 REQNTID 同样是 **128**。
>
> V0 的对照实验里两个变量同时变了（前端 commit 与 tileiras 版本），把差别记到了
> tileiras 头上。tileiras 版本仍然应当作为显式参数记录（R12 的**应对措施**依然正确），
> 但**这条特定证据不成立**。

**依据**：[SASS_AUDIT.md](SASS_AUDIT.md) Q9。同一 `.mlir`，13.1 + bc13.1 vs 13.3 + bc13.3，
两边 REQNTID 均为 `0x80`(=128)，总指令数均 120，REG 均 15，barrier 形态均
`BAR.SYNC.DEFER_BLOCKING 0x0`，`STG`/`LDG` 展开与位置逐条一致。

**置信度：高**（对「13.1 下也是 128」这个观测）。
⚠️ **中**（对「所以是前端带来的」这个归因）——我没有回退到 cuda-tile `8a775693`
重新编译验证，那需要重建前端。严格说我证明的是「不是 tileiras 13.1→13.3」，
「是前端」是排除法得出的推论。**建议落笔时按这个强弱分别措辞。**

---

## C3. §2.4「已排除的解释」表：cooperative launch 那一行要加限定

**位置**：`Tilemega_skeleton.md` §2.4「已排除的解释」表的第 3 行（约 184 行）。

**现文**：
> 「cooperative launch 准入失效 | ❌ 排除。用正确 blockDim 时 grid=170 被接受、
> grid=171 被拒绝，逻辑正确」

**建议改为**：
> 「cooperative launch 准入失效 | ❌ 排除。R2-D 实测 grid=170 接受、171 拒绝。
> **⚠️ 该数字只对 R2-D 那个 cubin（REQNTID=256、REG=228）成立，不可外推。**
> driver 只接受 blockDim ∈ {1, REQNTID}，其余返回 `invalid argument`；
> 准入计算用的始终是 **REQNTID**，即使以 blockDim=1 启动。R2-D 传的 blockDim=256
> 恰好等于该 kernel 的 REQNTID 才没被拒。V0 rebaseline 后 REQNTID=128，
> 同样的 blockDim=256 会被**直接拒绝**，正确上限是 **2040**（= occ(128) × 170 SM）。」

**依据**：[SUMMARY.md](SUMMARY.md) §5，探针 `blockdim_probe.cpp` / `occ_probe.cpp` / `bd2.cpp`。
实测一个 REQNTID=128 的 kernel 以 blockDim=1 启动，`cuLaunchCooperativeKernel`
最大接受 grid = 2040。

**置信度：高**（driver 行为与 2040 都是直接实测）。

**注**：§2.2 修正块里已有「§2.4 的『max grid=170』只对 13.1 成立」和「不能沿用 170」
两句，方向是对的；这条只是把**为什么**补上，并把它加到 §2.4 表格本身
（读者常常只看那张表）。**R2-D 的文件按要求未改动。**

---

## C4. §2.4「当前定位」：256 次静态展开的描述不成立

**位置**：`Tilemega_skeleton.md` §2.4「当前定位」第 2 条（约 196 行）。

**现文**：
> 「卡死 PC 在偏移 `0x6190` 的 `LDS R4, [UR5]`，**不在自旋循环本体（`0x1f0~0x230`）**，
> 而在自旋之后、由 **256 次静态展开**构成的**消费者归约收尾代码**里，紧跟一条
> `BAR.SYNC.DEFER_BLOCKING`」

**建议改为**：把「由 256 次静态展开构成」改成「**部分展开的归约循环之后的收尾代码**」。

> 卡死 PC `0x6190` 在自旋之后的**归约收尾**代码里，紧跟一条 `BAR.SYNC.DEFER_BLOCKING`。
> （原文说归约循环被「256 次完全静态展开」，实测不成立：同一个 cubin 里存在回边
> `0x30d0 -> 0x1e30`，跨 4768 字节、体内含 8 个 `BAR.SYNC`，是**部分展开**。
> 当时的回边检测器有 bug。`0x30d0 < 0x6190 < 0x6260`，卡点确实在该循环之后。）

**依据**：[SASS_AUDIT.md](SASS_AUDIT.md) Q10，对象是 R2-A 复用的同一个
`R1R2/experiments/E4_spin_wait/variants/spin_wait_tokenchain.cubin`。

**置信度：高**（回边是 `cuobjdump -sass` 直接可读的）。

⚠️ **R2-A 的核心结论没有被推翻**（卡在归约收尾、不在自旋循环），
被推翻的只是对代码形状的描述。**按要求未改 R2-A 的 result.md 与 INDEX.md。**

---

## C5. §6.6 白名单：可以从「待定」升级为「已确定」

**位置**：`Tilemega_skeleton.md` §6.6（约 1106–1112 行），
以及 §7.3「设计决策待定」里的「自旋之后的安全算子白名单（Phase 1 V2 输出）」。

**现文**：
> 「`[!]` 白名单待 Phase 1 的 V2 确定。已知风险：**归约类算子紧跟在自旋之后可能挂起**
> （R2-A 定位的卡死点正在归约收尾代码里）。」

**建议改为**：

```markdown
## 6.6 自旋之后的算子受白名单约束

`[V2 已确定]` **判据是「归约是否需要共享内存跨 warp 级」，不是算子类别。**

- **安全**（实测 0 失败）：elementwise、整块 load/store、单元素 load、
  以及**能在单 warp 内用 `SHFL` 完成的归约**（V2 实测 `tile<32xf32>` 的 reduce
  30/30 通过，SASS 里没有 `STS`/`LDS`）
- **不安全**（实测 100% 失败）：任何宽到需要 `STS` / `BAR.SYNC` / `LDS`
  把 warp partial 拼起来的 `reduce`（V2 实测 `tile<Nxf32>`，N ≥ 64 全部 30/30 失败）

原因不是「跨 block 数据可见性」，而是自旋循环的 CTA 集体 `BAR.SYNC 0x0` 与归约级的
`BAR.SYNC 0x0` 是同一个硬件 barrier，落后 warp 的到达会错误地满足归约 barrier。
详见 `docs/experiments/V2_visibility/SUMMARY.md`。

**这不是一个可以靠布局规避的问题**：V2-e 证明事件完全分布 + 数据完全独占的
真实任务图链条在 **grid=8** 就失败。在 tileiras 修复之前，
「spin-wait 之后跟跨 warp 归约」这个组合在 Tile IR 上不可用。
```

**依据**：[V2_f/result.md](V2_f/result.md) 尺寸扫描 + [V2_e/result.md](V2_e/result.md)
2×2 析因 + [SASS_AUDIT.md](SASS_AUDIT.md) Q11。

**置信度：中高。**
- 「N=32 通过 / N≥64 失败」「无 reduce 就通过」——✅ 高，多个独立变体一致。
- 「判据是共享内存跨 warp 级」——**中高**：迄今为止**所有** V1/V2 变体的失败集合
  恰好等于「归约里有共享内存跨 warp 级」的集合，但有一个未解释的例外
  `v2_f_const`（有自旋、有完整跨 warp 归约、**不读内存** → 30/30 通过）。
  我的解释（没有 load 延迟 → warp 间拉不开时间差）**未独立坐实**。
  建议白名单按现有判据落笔，但把 `v2_f_const` 作为已知未解释点记在旁边。

---

## C6. §7.1 风险表：R1 的措辞与 R6.6 的关系

**位置**：`Tilemega_skeleton.md` §7.1 风险表 **R1** 行（约 1128 行）。

**现文**：
> 「**大 grid 挂起未解**（§2.4）| 高 | 两个候选修复已证伪；定位到 post-spin reduce +
> `BAR.SYNC.DEFER_BLOCKING`。**V0：13.3 下屏障仍在循环内，H2 未被解决；挂起率本身待测**」

**建议改为**：

> 「**spin-wait + 跨 warp 归约的 barrier 配对缺陷**（§2.4）| 高 |
> **V2 已定性为 tileiras 缺陷**：自旋 `loop` 的回边被 lower 成逐线程谓词化，
> 而循环体内的 CTA 集体 `BAR.SYNC 0x0` 与归约级用的是同一个 barrier。
> R1/R2 的**挂起**与 V1/V2 的**静默数据错误**是同一个缺陷的两种表现，
> 取决于 REQNTID 决定的 barrier 形态（256 → 显式计数 `BAR.SYNC 0x1, 0x80` → 死锁；
> 128 → 隐式全 CTA `BAR.SYNC 0x0` → 静默错误）。
> **V0 的 rebaseline 没有修复任何东西，只是把死锁降级成了静默错误——这更危险。**
> 已上报草稿：`docs/experiments/V2_visibility/NVIDIA_ISSUE_DRAFT.md`。
> 规避手段见 §6.6。」

**依据**：[SASS_AUDIT.md](SASS_AUDIT.md) Q10 附带发现 + Q11，
[SUMMARY.md](SUMMARY.md) §2 的交替填充实验。

**置信度：中。**
- 「V1/V2 的静默错误是 barrier 配对错误」——✅ **高**（交替填充实验直接抓到
  「返回值 2048 > 正确答案 1024」，只能来自上一次 launch 遗留在共享内存槽里的 partial）。
- 「R1/R2 的 hang 与它是**同一个**缺陷」——⚠️ **中**：证据链很强
  （同一对 `STS`/`BAR.SYNC`/`LDS` 指令、barrier 形态随 REQNTID 变化可解释两种表现），
  但**我没有做交叉版本对照实验**（例如在 REQNTID=256 的构建上复现静默错误，
  或在 REQNTID=128 的构建上复现死锁）。这一条落笔时应保留「推断」标记。

---

## C7. §2.4 的 grid 阈值表：不要外推为「阈值」

**位置**：`Tilemega_skeleton.md` §2.4「现象」的 grid-scan 表（约 168–176 行），
以及 §5 计划里「grid ∈ {3, 30, 80, 120, 170, 260, 340}，各跑 50 次」的扫描设计。

**现文**：给出 grid=30/80/120/170 的挂起率，读起来像一条随 grid 单调上升的曲线。

**建议补一句**（不改原数据）：

> ⚠️ **grid 不是驱动变量。** V2-g 把 grid **固定为 340**、只改参与握手的消费者数 K，
> 失败率 K=2 → 0/50、K=8 → 2/50、K=32 → 22/50、K=128 → 50/50、K=339 → 50/50。
> 真正的驱动变量是**同时在自旋并归约的 block 数**。
> 后续扫描应把 grid 与并发参与者数**分开**作为两个自变量。

**依据**：[V2_g/result.md](V2_g/result.md)。

**置信度：高**（grid 恒定 340，唯一变量是 K，单调 0→100%）。

**注**：这条**不涉及**改动 R1/R2 已有的挂起率数字，只是提醒不要把它读成阈值曲线。
V1 SUMMARY 里「尖锐阈值落在 grid 80 与 120 之间」的说法同样不成立
（本轮用全新进程重扫得到 80→19/50、120→3/50、170→36/50、340→50/50，**非单调**），
但按要求 **V1 的文件未改动**，只在 [SUMMARY.md](SUMMARY.md) §4 对账。

---

## C8. 测试方法学：进程内 rep 聚合会掩盖错误

**位置**：`Tilemega_skeleton.md` §5 的实验计划（凡是写「各跑 N 次」的地方）。

**现文**：只写次数，未规定「同进程重复」还是「进程per次」。

**建议补充一条方法学要求**：

```markdown
**统计方法学**：正确性实验必须 **每次一个全新进程**，或至少**逐 rep 单独判定并报告**，
不得把同进程内多次 launch 的结果聚合。

原因（V2 实测）：本缺陷读到的是**共享内存里上一次 launch 遗留的 warp partial**。
若每次 launch 的正确答案相同，则从第 2 次起「陈旧值恰好等于正确值」，
错误被完美掩盖 —— 表现为「只有 rep 0 会错」。把 host 填充值改成逐 rep 交替
（1.0 / 3.0）后，**每一个 rep 都错**，且出现 `expect=1024 → 实测 2048` 这种
**比正确答案更大**的值，直接暴露了掩盖机制。

复现：`docs/experiments/V2_visibility/mechanism_altfill.sh`
```

**依据**：[SUMMARY.md](SUMMARY.md) §2，脚本 [mechanism_altfill.sh](mechanism_altfill.sh)。

**置信度：高**（配对实验，唯一变量是填充值是否逐 rep 变化，三次重跑完全一致）。

---

## 汇总

| # | 位置 | 性质 | 置信度 |
|---|---|---|---|
| C1 | 骨架 §6 新增 6.8（`weak` → `relaxed device`） | 规范符合性，**非本缺陷成因** | 高 |
| C2 | 骨架 §2.2 修正表 + `tilemega-compile` 头注释 + R12 | 归因更正 | 高（观测）/ 中（归因） |
| C3 | 骨架 §2.4 已排除表第 3 行 | 加限定，防止外推 170 | 高 |
| C4 | 骨架 §2.4 当前定位第 2 条 | 描述更正（核心结论不变） | 高 |
| C5 | 骨架 §6.6 + §7.3 | 白名单从「待定」升为「已确定」 | 中高 |
| C6 | 骨架 §7.1 R1 行 | 定性 + 合并 R1/R2 与 V1/V2 | 高（V1/V2 机理）/ 中（同一缺陷） |
| C7 | 骨架 §2.4 现象表 + §5 扫描设计 | 补一句，防止读成阈值 | 高 |
| C8 | 骨架 §5 实验计划 | 新增统计方法学要求 | 高 |
