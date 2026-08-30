# 任务 A：V1_min cubin 的逐指令反汇编审计

**目的**：验证 §0.4(2) 提出的假设——"128 个元素 = 恰好一条 32-bit `STG.E`/`LDG.E` 指令的份额，
说明有一两条访存指令落到了内存屏障的错误一侧"。

**结论先行**：❌ **该假设被直接观测证伪。** 8 条 store 全部在屏障之前，8 条 load 全部在
`CCTL.IVALL` 之后，限定符完全一致，没有任何一条落在错误一侧。缺陷不在 producer/consumer 的
数据通路上。审计过程中发现的真正结构性疑点是 **自旋循环体内的 `BAR.SYNC` 与归约阶段的
`BAR.SYNC` 是同一个 barrier 0**，见 §Q11（审计计划外的发现，但它解释了全部现象）。

## 环境与可复现命令

```
GPU        : NVIDIA RTX 5090 (sm_120a), 170 SM, driver 580.65.06
cuda-tile  : af241704
tileiras   : 13.3（对照组 13.1）
```

```bash
cd /data/tilemega/docs/experiments/V2_visibility/sass_audit
# 注意：tilemega-compile 的输出走 -o，不是位置参数（写成位置参数会报
# "error: no such file: <out>.cubin"）。
# 13.3 构建
/data/tilemega/scripts/tilemega-compile -o v1_min_133.cubin v1_min.mlir
cuobjdump -sass v1_min_133.cubin > v1_min_133.sass

# 13.1 对照（注意：必须同时降 bytecode 版本，否则 tileiras 13.1 会报
# 具有误导性的 "invalid GPU architecture: 120"）
/data/tilemega/scripts/tilemega-compile \
  --tileiras /usr/local/cuda-13.1/bin/tileiras --bc-version 13.1 \
  -o v1_min_131_bc13.1.cubin v1_min.mlir
cuobjdump -sass v1_min_131_bc13.1.cubin > v1_min_131_bc13.1.sass

bash /data/tilemega/scripts/sass_report.sh v1_min_133.cubin
```

精确版本：tileiras 13.3 = `release 13.3, V13.3.36`（`/data/cuda-13.3.1/bin/tileiras`）；
13.1 = `/usr/local/cuda-13.1/bin/tileiras`。每个 cubin 旁的 `.env.json` 是权威记录。

---

## Producer 侧

### Q1 — `tile<1024xf32>` 的 `store_view_tko relaxed device` 展开成几条 `STG.E*`？

✅ **已验证：恰好 8 条 `STG.E.STRONG.GPU`**，全部是 32-bit 标量 store。

1024 元素 ÷ 128 线程（REQNTID）= 8 元素/线程，与 8 条指令一一对应。步长 `0x200` = 512 字节
= 128 个 float，即每条指令跨越一整个 warp-块的宽度（线程 t 的第 i 个元素位于 `i*128 + t`）。

### Q2 — 每条 store 的地址/限定符/相对 `MEMBAR.ALL.GPU` 的位置；有没有落在它*之后*的？

✅ **已验证：8 条全部在 `MEMBAR.ALL.GPU` 之前。没有任何一条落在之后。**

```
/*0120*/ STG.E.STRONG.GPU desc[UR6][R2.64],       R7 ;   <- chunk base + 0x000
/*0130*/ STG.E.STRONG.GPU desc[UR6][R2.64+0x200], R7 ;
/*0140*/ STG.E.STRONG.GPU desc[UR6][R2.64+0x400], R7 ;
/*0150*/ STG.E.STRONG.GPU desc[UR6][R2.64+0x600], R7 ;
/*0160*/ STG.E.STRONG.GPU desc[UR6][R2.64+0x800], R7 ;
/*0170*/ STG.E.STRONG.GPU desc[UR6][R2.64+0xa00], R7 ;
/*0180*/ STG.E.STRONG.GPU desc[UR6][R2.64+0xc00], R7 ;
/*0190*/ STG.E.STRONG.GPU desc[UR6][R2.64+0xe00], R7 ;   <- 最后一条 store
/*01a0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;                   <- CTA 内汇合
/*01b0*/ @P0 BRA 0x260 ;                                 <- P0 = (tid != 0)，非 0 号线程跳走
/*01e0*/ MEMBAR.ALL.GPU ;                                <- 设备级 release 屏障
/*01f0*/ ERRBAR;
/*0200*/ CGAERRBAR ;
/*0210*/ ATOMG.E.EXCH.STRONG.GPU PT, R2, desc[UR6][R2.64], R5 ;   <- 置 flag
```

顺序完全正确：**8×store → BAR.SYNC（让全 CTA 的 store 都已发出）→ MEMBAR（设备可见）→
ATOMG 置 flag**。这正是 release 语义要求的形状。

### Q3 — 限定符是否一致？

✅ **已验证：8 条全部是 `.E.STRONG.GPU`，完全一致。** 无一条退化为弱序 `STG.E`。
`relaxed device` → `.STRONG.GPU` 的映射符合预期。

（对照：consumer 末尾写 `out[bx]` 的那条是 `weak`，SASS 里就是裸 `STG.E`（`/*0690*/`），
说明工具链确实按 ordering 区分生成，`.STRONG.GPU` 不是无差别加上去的。）

### Q4 — `BAR.SYNC` 在哪里？它之前的 store 是否覆盖了全部 1024 个元素？

✅ **已验证：`BAR.SYNC.DEFER_BLOCKING 0x0` 在 `/*01a0*/`，8 条 store 全在它之前，
覆盖 8×128 = 1024 个元素，无遗漏。**

---

## Consumer 侧

### Q5 — 展开成几条 `LDG.E*`？

✅ **已验证：恰好 8 条 `LDG.E.STRONG.GPU`**，与 producer 对称（步长 `0x200`）。

### Q6 — 每条相对 `CCTL.IVALL` 的位置；有没有落在它*之前*的？

✅ **已验证：8 条数据 load 全部在 `CCTL.IVALL` 之后。没有任何一条落在之前。**

```
/*02d0*/ LDG.E.STRONG.GPU R3, desc[UR6][R4.64] ;   <- 剥离出来的首次 poll
/*02f0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0310*/ @!P0 BRA 0x360 ;                          <- 已看到 flag 的线程跳出循环
/*0320*/ LDG.E.STRONG.GPU R3, desc[UR6][R4.64] ;   <- 自旋循环体：poll
/*0330*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;             <- ！循环体内的 barrier
/*0350*/ @P0 BRA 0x320 ;                           <- 谓词化回边（48B）
/*0370*/ CCTL.IVALL ;                              <- acquire：作废 L1
/*0380*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0390*/ LDG.E.STRONG.GPU R3,  desc[UR6][R6.64] ;        <- 8 条数据 load 从这里开始
/*03a0*/ LDG.E.STRONG.GPU R4,  desc[UR6][R6.64+0x200] ;
/*03b0*/ LDG.E.STRONG.GPU R8,  desc[UR6][R6.64+0x400] ;
/*03c0*/ LDG.E.STRONG.GPU R5,  desc[UR6][R6.64+0x600] ;
/*03d0*/ LDG.E.STRONG.GPU R9,  desc[UR6][R6.64+0x800] ;
/*03e0*/ LDG.E.STRONG.GPU R10, desc[UR6][R6.64+0xa00] ;
/*03f0*/ LDG.E.STRONG.GPU R12, desc[UR6][R6.64+0xc00] ;
/*0400*/ LDG.E.STRONG.GPU R11, desc[UR6][R6.64+0xe00] ;
```

### Q7 — acquire load 是否被 CSE 掉了？

✅ **已验证：是，而且这是正确行为。** MLIR 里循环退出后有一条独立的
`load_ptr_tko acquire device`，SASS 里找不到对应的第 9 条 flag load——它被 CSE 进了循环里那条
`relaxed` load（`/*0320*/`），只把 acquire 的**屏障部分**留下，就是 `/*0370*/ CCTL.IVALL`。

这正是 test-and-test-and-set 想要的结果：值来自 relaxed load，acquire 只贡献屏障。
**不是缺陷。**

### Q8 — `reduce` 的展开形状（用修好的 `sass_report.sh`）

✅ **已验证。** `reduce : tile<1024xf32> -> tile<f32>` 在 128 线程（4 warp）下展开成
**两级归约**：

```
第 1 级 warp 内（SHFL 蝶形，5 步覆盖 32 lane）：
  /*04e0*/ SHFL.BFLY PT, R4, R3, 0x10, 0x1f ;
  /*0510*/ SHFL.BFLY PT, R5, R4, 0x8,  0x1f ;
  /*0540*/ SHFL.BFLY PT, R6, R5, 0x4,  0x1f ;
  /*0570*/ SHFL.BFLY PT, R7, R6, 0x2,  0x1f ;
  /*0590*/ SHFL.BFLY PT, R8, R7, 0x1,  0x1f ;

第 2 级 warp 间（共享内存，4 个 warp partial）：
  /*05b0*/ @!P0 STS [R3], R8 ;      <- 每个 warp 的 lane0 写自己的 partial
  /*05c0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;   <- 分隔 "写 partial" 与 "读 partial"
  /*05f0*/ @!P1 LDS R5, [R4] ;      <- 前 4 个 lane 读回 4 个 partial
  /*0610*/ SHFL.BFLY PT, R6, R5, 0x2, 0x1f ;
  /*0630*/ SHFL.BFLY PT, R7, R6, 0x1, 0x1f ;
  /*0650*/ @!P0 STS [R4], R9 ;      <- 写回最终标量
  /*0660*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
  /*0680*/ LDS R7, [UR4] ;          <- 全体广播读
  /*0690*/ @!P0 STG.E desc[UR6][R2.64], R7 ;   <- weak store 到 out[bx]
```

`sass_report.sh` 汇总：`total 120 / BAR.SYNC 9 / CCTL.IVALL 1 / REG 15 / REQNTID 0x80 (=128)`。

**这一级 `STS → BAR.SYNC → LDS` 就是后来 V2 定位到的出错点。**

---

## Q9 — 换回 tileiras 13.1，13.3 改了什么？

✅ **已验证：在本审计关心的所有方面，13.1 与 13.3 生成的代码结构相同。**

| 指标 | 13.1 (bc 13.1) | 13.3 |
|---|---|---|
| 总指令数 | 120 | 120 |
| 数据 `STG.E.STRONG.GPU` | 8（全在屏障前） | 8（全在屏障前） |
| 数据 `LDG.E.STRONG.GPU` | 8（全在 `CCTL.IVALL` 后） | 8（全在 `CCTL.IVALL` 后） |
| `CCTL.IVALL` | 1 | 1 |
| REG/thread | 15 | 15 |
| REQNTID | 0x80 (=128) | 0x80 (=128) |
| barrier 形态 | `BAR.SYNC.DEFER_BLOCKING 0x0` | `BAR.SYNC.DEFER_BLOCKING 0x0` |
| `BAR.SYNC` 条数 | 8 | 9 |
| 自旋回边 | `0x0100 -> 0x00d0`，48B，体内含 1 个 `BAR.SYNC` | `0x0350 -> 0x0320`，48B，体内含 1 个 `BAR.SYNC` |

⚠️ **重要推论**：`REQNTID = 128` 在 tileiras **13.1** 下同样成立。所以 V0 记录的
"`EIATTR_REQNTID` 从 256 变成 128" **不是 tileiras 13.1→13.3 带来的**，而是 cuda-tile 前端
（`8a775693` → `af241704`）带来的。这一点与 V0 的归因不同，建议在骨架里更正
（见 `PROPOSED_DOC_CHANGES.md`）。

---

## Q10 — 复核 R2-A 的"tileiras 把 256 次迭代的收尾 reduce 循环完全展开"

> 用户要求：结果写进本报告，**不要改 R2-A 的 result.md 或 INDEX.md**。此处仅报告。

❌ **该论断不成立（当时的回边检测器是坏的，修好后结论反转）。**

对 R2-A 复用的同一个 cubin
（`R1R2/experiments/E4_spin_wait/variants/spin_wait_tokenchain.cubin`）用修好的检测器：

```
total instructions           1656
BAR.SYNC.DEFER_BLOCKING      112
CCTL.IVALL                   225
REG/thread                   228
REQNTID (blockDim)           0x100 (=256)
-- loop back edges --
  0x0230 -> 0x01f0  size 64B     BAR.SYNC in loop body = 1     <- 自旋循环
  0x30d0 -> 0x1e30  size 4768B   BAR.SYNC in loop body = 8     <- ★ 归约循环，仍然是循环！
  0x65b0 -> 0x6260  size 848B    BAR.SYNC in loop body = 0
```

`0x30d0 -> 0x1e30` 这条回边横跨 4768 字节、体内含 8 个 `BAR.SYNC`，**证明归约循环并没有被
完全展开，只是被部分展开（unroll factor 使每趟带 8 个 barrier）**。R2-A 看到的"地址等间隔
递增的大量 `BAR.SYNC`"是部分展开后的循环体，不是 256 次的直线展开链。

**R2-A 中不受影响、依然成立的部分**：卡住的 PC `0x6190` 确实落在 `0x30d0` 归约循环*之后*的
直线收尾代码里（`0x30d0 < 0x6190 < 0x6260`），也确实是紧跟 `BAR.SYNC` 的一条 `LDS`。
所以 "卡的地方不在自旋循环里，而在归约收尾" 这个**核心结论仍然正确**，被推翻的只是
"完全展开" 这个对代码形状的描述。

### Q10 附带发现：R1/R2 的 hang 与 V1/V2 的 corruption 是同一个缺陷

⚠️ **推断（证据链见下，但未做交叉版本的对照实验，故不标 ✅）**

R2-A 的 result.md §5 把死锁疑点指向 "`reduce` 的 lowering……可能是某个 barrier 的到达计数与
实际线程数不匹配"，并明确标注为 ❌ 推测。V2 的实验把这条推测坐实了，而且解释了 hang 为什么会
变成 corruption。关键是 **barrier 形态随 REQNTID 改变**：

| 构建 | REQNTID | barrier 形态 | 到达计数不匹配时的后果 |
|---|---|---|---|
| R1/R2 (`8a775693`) | 256 | `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` — 具名 barrier 1，**显式计数 128** | 计数永远凑不齐 → **死锁** |
| V0/V1/V2 (`af241704`) | 128 | `BAR.SYNC.DEFER_BLOCKING 0x0` — barrier 0，隐式全 CTA | 不死锁，但**配对错乱 → 静默读到陈旧共享内存** |

R2-A 卡住的 PC 是 `LDS`，紧跟 `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` 和 `@!P2 STS`；
V2 定位到的出错点是 `@!P0 STS → BAR.SYNC 0x0 → @!P1 LDS`。**是同一对指令。**
两轮看到的是同一个 barrier 配对错误在两种 barrier 形态下的两种表现。

这也解释了 V0 rebaseline 为什么"修好了"hang 却引入了 corruption：它没有修复任何东西，
只是把显式计数 barrier 换成了隐式全 CTA barrier，把**死锁降级成了静默错误**。

---

## Q11 — 审计计划外的发现：自旋循环体内的 barrier 与归约的 barrier 是同一个

✅ **已验证（直接从 SASS 观测）。** 这是本次审计真正有价值的产出。

Tile IR 的 `loop` 被 lower 成一个 **CTA 集体（collective）** 构造——循环体里被塞进了一条
`BAR.SYNC.DEFER_BLOCKING 0x0`（`/*0330*/`），意思是"每轮 poll 全 CTA 汇合一次"。
但它的**回边是逐线程谓词化的**：

```
/*0320*/ LDG.E.STRONG.GPU R3, desc[UR6][R4.64] ;   每个线程独立 load 同一个 flag 地址
/*0330*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0350*/ @P0 BRA 0x320 ;                           <- 谓词化：P0 因线程而异
```

128 个线程各自 load 同一个地址，但**在时间上可以在不同轮次观察到 flag 的翻转**，
所以 P0 不是 CTA 一致的。于是 CTA 会分裂：一部分 warp 还在 `/*0330*/` 的 barrier 上，
另一部分已经走到 `/*0380*/`、`/*05c0*/`、`/*0660*/` 的 barrier 上。

**而这些全都是 barrier 0。** 硬件的 `BAR.SYNC 0x0` 按到达的 warp 计数汇合，
它无法区分"这个 warp 到达的是自旋循环里的那个 barrier"还是"归约里的那个 barrier"。
于是：

> `/*05c0*/` 那条本应分隔 "所有 warp 已写完自己的 partial" 与 "warp 0 读回 partial" 的
> barrier，可以被**还在自旋的落后 warp 在 `/*0330*/` 的到达**所满足。
> warp 0 于是在别的 warp 尚未 `STS` 之前就 `LDS` 了，读到共享内存里的旧内容。

这是一条**结构性**的错误，与 grid 大小、数据是否跨 block 共享都无关——只要
(a) 自旋循环的退出条件来自逐线程的内存 load，且 (b) 后续归约需要跨 warp 走共享内存，
它就成立。V2 的全部实验结果都与这个模型一致，详见 `SUMMARY.md`。

---

## 任务 A 小结：对 §0.4(2) 的裁决

| §0.4(2) 的推理 | 裁决 |
|---|---|
| "poison=−1、期望 1024 ⇒ sum = 2N − 1024" | ✅ 算术正确 |
| "768/512 ⇒ 缺 128/256 个元素" | ✅ 算术正确 |
| "128 元素 = 一条 `STG.E` 的份额" | ✅ 数值巧合属实（1024/8 = 128） |
| **"所以有一两条访存指令落在了屏障的错误一侧"** | ❌ **证伪**。8 条 store 全在屏障前，8 条 load 全在 `CCTL.IVALL` 后 |

而且这个数值巧合在 V2 的尺寸扫描里就崩了：把 tile 宽度换成 N=64 时，丢失量是 **32**，
既不是 128 也不是任何一条 store 指令的份额——它始终是 **正确答案的一半**，
也就是 4 个 warp partial 里丢了 2 个。真正的量子是 **warp partial**，不是 `STG.E`。
