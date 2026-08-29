# R2-B 结果：relaxed 轮询 + acquire re-read（test-and-test-and-set）能否消除挂起

## 结论速览

**❌ 不能——已用完整 grid-scan 挂起率统计证伪。** SASS 层面确认了预期的改动确实生效
（轮询循环内的 `CCTL.IVALL` 被成功移除，只在跳出循环后的单次 acquire 重读里保留），
正确性也完全没有受影响（1MB 数据 1000/1000 次通过，4MB 数据 200/200 次通过），
**但挂起率没有得到任何有意义的改善**——在全部 4 个测试的 grid 规模（30/80/120/170）上，
relaxed+acquire 变体的挂起率与 round-1/E4 的 acquire-in-loop 基线挂起率同一量级，
在 grid=30 时甚至更高（96% vs 78%）。这直接印证了 R2-A §6 基于 SASS 证据做出的预测：
挂起的真正机制不在轮询循环的内存序上，而在 R2-F 发现的、轮询循环和 consumer 归约路径里
都存在的块内硬件屏障 `BAR.SYNC.DEFER_BLOCKING`——R2-B 的改动完全没有触及这条指令。

## 文件

- `spin_wait_relaxed_acquire.mlir` / `.cubin` / `.sass` — 核心变体（1MB 数据，对应 E4 baseline 的 `data` 大小 262144 float）。
- `spin_wait_relaxed_acquire_4mb.mlir` / `.cubin` — 同一变体的 4MB 数据版本（1048576 float），用于正确性压力测试。
- `host_test_correctness.cpp` / `host_test_correctness_4mb.cpp` — 小 grid（=2，不会挂起）、多次迭代（每次全新 memset+launch+sync+回读）正确性验证。
- `host_test_single.cpp` — 单次启动、无预热 memset 的 harness（对应 round-1 E5 §2.2 确认过的"更容易复现挂起"的 worst-case 配置），配合 `grid_scan.sh` 做统计。
- `grid_scan.sh` — 对 `relaxed_acquire` 和 `baseline_acquire_in_loop`（即 E4 的 `spin_wait_tokenchain.cubin`）两个变体，在 grid∈{30,80,120,170} 各跑 50 次、6 秒超时，统计挂起次数。
- `correctness_1mb_1000.log`、`correctness_4mb_200.log` — 正确性测试原始输出。
- `/tmp/r2b_gridscan.log` — grid-scan 完整逐次记录（本机 `/tmp`，未拷贝进实验目录，命令输出已摘录于下方）。

## 1. SASS 证据：轮询循环内的 CCTL.IVALL 确实被移除

```
$ grep -n -B2 -A6 "BAR.SYNC.DEFER_BLOCKING" spin_wait_relaxed_acquire.sass | head -20
...
/*01f0*/  BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 ;
/*0200*/  LDG.E.STRONG.GPU R4, desc[UR8][R2.64] ;
/*0210*/  ISETP.NE.AND P0, PT, R4, 0x1, PT ;
/*0220*/  @P0 BRA 0x1f0 ;                        <- 轮询循环回边，循环体内没有 CCTL.IVALL
...
/*0240*/  BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 ;     <- Phase 2：跳出循环后的单次 acquire 重读
/*0250*/  NOP ;
/*0260*/  CCTL.IVALL ;                            <- CCTL.IVALL 出现在这里，且只出现一次（非循环体内）
```

对照 baseline（`../E4_spin_wait/variants/spin_wait_tokenchain.cubin`）的轮询循环体
（R2-F result.md 发现 2 已记录）：循环体本身就包含 `CCTL.IVALL`，每次轮询迭代都执行一次。

**结论（✅ 已验证）：relaxed+acquire 的改动在代码生成层面完全达到了设计目标**——轮询循环体内
不再有 `CCTL.IVALL`（不再每次迭代都做缓存失效），`CCTL.IVALL` 被移到了循环外、只在跳出循环后
执行一次的 Phase-2 acquire 重读里。全文件 `CCTL.IVALL` 静态出现次数在两个 cubin 里都是 225
（`grep -c` 结果），这是符合预期的：baseline 是"循环体内 1 处静态 CCTL.IVALL"，relaxed_acquire
是"循环体外 1 处静态 CCTL.IVALL"——静态计数不变，但**运行时执行次数被从"每次轮询迭代 1 次"
降到了"整个自旋等待过程总共 1 次"**，这正是 test-and-test-and-set 模式的设计意图。

**但两个变体的轮询循环体内都保留了 `BAR.SYNC.DEFER_BLOCKING`**（R2-F result.md 发现 2 已经
记录并验证过这一点，此处不重复反汇编），即每次轮询迭代仍然要求该 tile block 的全部 256 个线程
在硬件屏障上会合。R2-B 的改动完全没有触及这条指令。

## 2. 正确性：完全不受影响

```
$ ./host_test_correctness spin_wait_relaxed_acquire.cubin spin_wait_relaxed_acquire 1000
[spin_wait_relaxed_acquire] over 1000 iterations: PASS=1000 FAIL=0 observed_range=[34359672832.000000,34359672832.000000] ref=34359607296.000000

$ ./host_test_correctness_4mb spin_wait_relaxed_acquire_4mb.cubin spin_wait_relaxed_acquire_4mb 200
[spin_wait_relaxed_acquire_4mb] over 200 iterations: PASS=200 FAIL=0 observed_range=[549755748352.000000,549755748352.000000] ref=549755289600.000000
```

（诚实标注：4MB 测试原计划 500 次迭代，第一次尝试因单次运行耗时超过 300 秒被 `timeout` 杀死、
未产生任何输出——该 harness 只在全部迭代结束后打印一行汇总，中途被杀无法获得部分结果；受时间
预算限制，改为 200 次迭代重新运行，200/200 全部通过。① 两次测试都使用 grid=2（远低于挂起阈值，
不会遇到 R2-A/R2-F 发现的挂起问题），只验证 relaxed+acquire 内存序改动本身不引入数据竞争/
错误结果；② relerr 判定容差沿用 round-1/E4 建立的 1e-5 相对误差阈值，浮点求和顺序导致的舍入
误差量级正常。）

**结论（✅ 已验证）：relaxed 轮询 + acquire 重读的内存序改动，在不挂起的 grid 规模下，
1MB 数据 1000/1000 次、4MB 数据 200/200 次全部产生正确结果——没有观察到因为改用 relaxed
读取而引入的数据竞争或过早读取问题。**

## 3. 核心结果：grid-scan 挂起率统计（M=50/grid，两个变体 × 4 个 grid）

沿用 round-1 E5 §2.2 已确认的"worst case"配置：`host_test_single.cpp` 不做任何预热 memset
（即去掉了 round-1 E4 原始 harness里那次意外掩盖问题的 1MB `cuMemsetD32` 预热操作），
每次运行都是全新进程、全新 CUDA context，6 秒超时判定为挂起。

| 变体 | grid=30 | grid=80 | grid=120 | grid=170 |
|---|---|---|---|---|
| `relaxed_acquire`（R2-B 修复） | 挂起 48/50 | 挂起 46/50 | 挂起 50/50 | 挂起 50/50 |
| `baseline_acquire_in_loop`（E4/round-1 原始版本） | 挂起 39/50 | 挂起 50/50 | 挂起 50/50 | 挂起 50/50 |

原始命令与逐次输出摘录（完整记录见 `/tmp/r2b_gridscan.log`，由 `grid_scan.sh` 生成）：
```
$ ./grid_scan.sh /tmp/r2b_gridscan.log 6 50
...
SUMMARY relaxed_acquire grid=30: 挂起 48/50 次
SUMMARY relaxed_acquire grid=80: 挂起 46/50 次
SUMMARY relaxed_acquire grid=120: 挂起 50/50 次
SUMMARY relaxed_acquire grid=170: 挂起 50/50 次
SUMMARY baseline_acquire_in_loop grid=30: 挂起 39/50 次
SUMMARY baseline_acquire_in_loop grid=80: 挂起 50/50 次
SUMMARY baseline_acquire_in_loop grid=120: 挂起 50/50 次
SUMMARY baseline_acquire_in_loop grid=170: 挂起 50/50 次
ALL DONE
```

**结论（✅ 已验证，M=50 per grid）：relaxed+acquire 修复没有降低挂起率，在 grid=30 这一个点上
甚至比 baseline 更差（96% vs 78%）**，在 grid≥80 时两者都已经接近或达到 100% 挂起、看不出
统计上有意义的差异。grid=30 处 96% vs 78% 这个差异本身有多大意义未定点验证（⚠️
documented-but-unverified：M=50 在接近饱和的挂起率下置信区间较宽，48/50 vs 39/50 差异是否
反映真实效应还是运行间噪声，本实验没有做统计显著性检验，只如实报告两个观测比例）。
**无论如何，两个变体在全部 4 个 grid 点上都远未达到"挂起率显著降低甚至消失"这个 R2-B
原计划要验证的目标**——这一目标已被证伪。

## 4. 为什么修复失败：与 R2-A / R2-F 的交叉印证

R2-A（`../R2_A_livelock_vs_deadlock/result.md` §3/§5）用 cuda-gdb 定位到的挂起 PC 在
consumer 归约收尾的 `LDS` 指令，紧跟在 `BAR.SYNC.DEFER_BLOCKING` 之后，**不在**自旋轮询
循环本体（0x1f0-0x230）——这从一开始就已经暗示问题不在轮询的内存序上。R2-A §6 据此
明确预测"R2-B/C（只改轮询循环）可能无法修复这个挂起"。

R2-F（`../R2_F_pipelining/result.md` 发现 2）进一步发现，自旋轮询循环体本身也包含一条
`BAR.SYNC.DEFER_BLOCKING`（每次轮询迭代都要求该 tile block 全部 256 线程会合），并指出
R2-B 的 relaxed+acquire 改动只去掉了轮询循环里的 `CCTL.IVALL`，**没有**去掉这条硬件屏障。

本实验的 grid-scan 数据是这两个预测的**实证结果**：relaxed+acquire 在 SASS 层面完全达到了
设计目标（§1）、正确性完全没问题（§2），却**没有**降低挂起率（§3）——三者放在一起，
强有力地支持"挂起根因不是缓存一致性/内存可见性问题，而是与 `BAR.SYNC.DEFER_BLOCKING`
在高并发 CTA 占用下的行为有关"这一假说（⚠️ 仍未做最终定点验证：没有用 cuda-gdb 在
relaxed_acquire 变体挂起时抓取冻结线程的 PC，逐点确认冻结位置是否与 R2-A 在 baseline 上
定位到的 PC 属于同一类指令；这是本实验遗留的一个可以低成本补做但受时间预算限制未做的验证步骤）。

## 结论对 megakernel 可行性的意义

1. **❌ R2-B 提出的"标准 test-and-test-and-set 模式能解决挂起"这一最有希望的候选修复方案，
   已被证伪。** 这是本轮调研目前为止最重要的负面结果：round 1 遗留的核心阻塞问题（跨 tile
   block 自旋等待在较大 grid 规模下挂起）**在切换到更"教科书"的内存序模式后依然存在**，
   说明问题比"round 1 用错了内存序"更深——很可能是 tileiras 为"tile block"这一抽象生成的
   块内线程收敛机制（256 线程 SPMD + 硬件屏障）本身的问题，这是 megakernel 设计模式（持久
   kernel + 跨块自旋同步）在当前工具链版本下的一个更根本性的障碍。
2. 唯一还没有测试过的候选方向是 R2-C 的退避（backoff）——**降低轮询频率本身不能去掉每次
   轮询都执行的 BAR.SYNC，但会减少 BAR.SYNC 的执行次数（总轮询次数变少）**，如果根因确实与
   BAR.SYNC 在高频/高并发下的某种竞争或公平性问题有关，退避可能有部分缓解效果（也可能完全
   无效，因为哪怕次数减少，仍然存在同一失败模式）——这是 R2-C 需要实际检验、而不能从 R2-B
   的失败直接推断的问题。

## 状态：R2-B 完成 ✅（核心问题已有明确、可复现的负面结论；唯一遗留项是"用 cuda-gdb 精确定位
relaxed_acquire 挂起 PC 并与 baseline 挂起 PC 比对"这一步定点验证，标记为可选的后续加强项）
