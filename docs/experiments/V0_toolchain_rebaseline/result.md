# V0：工具链重基线（tileiras 13.1 vs 13.3）

**日期**：2026-08-28
**状态**：静态对比 ✅ 已完成；GPU 统计扫描 ⏸ 阻塞（见末尾）

## 动机

R1/R2 的全部结论建立在 **cuda-tile `8a775693`（2026-01）+ tileiras 13.1 (V13.1.80)** 上。
新基线是 **cuda-tile `af241704`（2026-07，CUDA Tile IR 13.3.3）+ tileiras 13.3 (V13.3.36)**——
跨越 7 个月、两个 release。而 §2.4 那个阻塞 L1 的挂起，根因就在**闭源的 tileiras** 里。

所以在按 R1/R2 的结论安排全部后续工作之前，必须先问：这些结论现在还成立吗？

## 方法

单变量对照：取 R2-B 产出的**同一份 `.tilebc`**（由旧版 cuda-tile 生成），
分别用两个版本的 tileiras 汇编，比较产物。这样唯一的变量就是闭源后端。

```bash
SRC=../R1R2/experiments/R2_B_relaxed_plus_acquire/spin_wait_relaxed_acquire.tilebc
/usr/local/cuda-13.1/bin/tileiras --gpu-name sm_120 $SRC -o sw_ti131.cubin
/data/cuda-13.3.1/bin/tileiras   --gpu-name sm_120 $SRC -o sw_ti133.cubin
```

## 结果 1：13.3 能吃下旧 bytecode ✅

两个版本都无错接受 13.1 时代的 `.tilebc`。向后兼容成立。

## 结果 2：后端 codegen 有实质变化 ✅

| 指标 | tileiras 13.1 | tileiras 13.3 | 变化 |
|---|---|---|---|
| cubin 大小 | 52304 B | 44776 B | −14% |
| SASS 指令数 | 1664 | 1448 | −13% |
| `BAR.SYNC.DEFER_BLOCKING` | 113 | 85 | −25% |
| **`CCTL.IVALL`** | **225** | **28** | **−88%** |
| CGA 相关指令 | 3 | 2 | −1 |
| REG/thread | 228 | 229 | +1 |
| static SHM | 1184 B | 1052 B | −11% |

`CCTL.IVALL` 降低近一个数量级值得注意：R2-B 当年手工改写轮询循环，
目标就是把 `CCTL.IVALL` 移出循环；13.3 似乎在全局层面大幅减少了它。

## 结果 3 ⭐：每 tile block 的线程数从 256 变成 128

这是本次最重要的发现。

| | tileiras 13.1 | tileiras 13.3 |
|---|---|---|
| `EIATTR_REQNTID` | `0x100` = **256** | `0x80` = **128** |
| REG | 228 | 229 |
| driver 报 blocks/SM（按各自 REQNTID 查询） | **1** | **2** |
| cooperative launch 上限 | **170** | **340** |

手算吻合：`65536/(228×256)=1.12→1` 对 `65536/(229×128)=2.23→2`。

**这直接推翻了骨架 §2.2 的一条硬约束表述**：
「一个 tile block = 一个逻辑线程 = 一组物理线程（实测 256）」——
**256 不是常量，是后端版本相关的选择。**

而它正是 §2.4 挂起分析的核心量：R2-D 用它算出「max grid=170，与经验挂起边界吻合」。
现在同一份 IR 在 13.3 下的硬件驻留容量翻倍了。

## 结果 4：假设 H2 未被新版解决 ⚠️

`BAR.SYNC.DEFER_BLOCKING` **仍在自旋轮询循环体内部**。

tileiras 13.1（回边 `0x220 → 0x1f0`，48B 循环体）：
```
/*01f0*/  BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 ;
/*0200*/  LDG.E.STRONG.GPU R4, desc[UR8][R2.64] ;
/*0210*/  ISETP.NE.AND P0, PT, R4, 0x1, PT ;
/*0220*/  @P0 BRA 0x1f0 ;
```

tileiras 13.3（回边 `0xf40 → 0xf10`，同样 48B）：
```
/*0f10*/  LDG.E.STRONG.GPU R0, desc[UR8][R2.64] ;
/*0f20*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0f30*/  ISETP.NE.U32.AND P0, PT, R0, 0x1, PT ;
/*0f40*/  @P0 BRA 0xf10 ;
```

屏障的形式变了（`0x1, 0x80` → `0x0`，指定屏障+计数 → 默认屏障），
但**每次轮询迭代仍要求整个 tile block 的全部线程会合**。
骨架 §6.2 的结论不变：Tile IR 做不到「一个线程轮询、其余等待」。

## 结果 5：新工具链端到端可用 ✅

`scripts/tilemega-compile` 用 **cuda-tile `af241704` → bytecode 13.3 → tileiras 13.3**
成功编译 R1 时代的 `E3_persistent_loop/persistent_loop.mlir`：

```
[1/3] 验证 IR
[2/3] MLIR -> tilebc (bytecode 13.3)
[3/3] tilebc -> cubin (sm_120)
      Function persistent_loop:
       REG:19 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:912
```

## ⏸ 阻塞：GPU 统计扫描未完成

计划：`(tileiras 13.1, 13.3) × (grid 30, 80, 120, 170, 340) × 50 次`，
用 `scripts/gpu_stat_run.sh` 采集挂起率。**尚未执行。**

原因是一个本身值得记录的事故：

第一次前台尝试时，一个 grid 较大的运行进入挂起，宿主进程随后被外部超时
SIGKILL。**宿主进程被杀掉并不回收 GPU 上挂死的持久 kernel**——
此后 `nvidia-smi` 持续报告：

```
utilization.gpu=100%   clocks.sm=2887MHz   power.draw≈97.5W   memory.used=0MiB
（且 --query-compute-apps 列表为空，无任何进程）
```

这组数字与 R2-A 记录的挂起特征**逐项一致**（R2-A：SM 100%、2887MHz、
97.71~98.33W，idle 基线 13.19W）。GPU 仍能接受新工作（grid=3 通过），
但部分 SM 被占住。

`gpu_stat_run.sh` 的空闲检查因此拒绝执行全部 10 组——**这是它按设计工作，
挡住了一批会被污染的数据**。

### 恢复所需

```bash
sudo nvidia-smi --gpu-reset -i 0     # 需要权限；本次被拦截
```

恢复后直接 `./run_v0.sh` 即可，脚本与两个 cubin 都已就绪。

### 由此得到的两条测试纪律（已写入骨架 P0.3）

1. **GPU 测试必须独占且在空闲机器上跑。** 挂起是时序敏感的竞态
   （R1-E5：预热能「救活」grid=80），有负载时的数据不可用。
2. **挂起测试的超时必须由测试脚本自己管，且要短。** 让外层工具的超时
   去杀一个挂起中的 GPU 进程，会把 GPU 留在污染状态，后续所有实验作废。

## 对骨架的影响

| 影响项 | 结论 |
|---|---|
| §2.2「实测 256」 | **需修正为版本相关**；13.3 是 128 |
| §2.4 max grid=170 | 仅对 13.1 成立；13.3 推算为 340 |
| §6.2 自旋循环内含屏障 | **不变，13.3 仍然如此** |
| 假设 H2 | **未被新版解决**，仍是最可能的根因方向 |
| 假设 H1 | 不受影响，V1 仍需做 |
| 风险 R1（大 grid 挂起） | 严重度维持；但容量边界翻倍，V1~V4 的 grid 扫描范围要相应扩到 340 |
