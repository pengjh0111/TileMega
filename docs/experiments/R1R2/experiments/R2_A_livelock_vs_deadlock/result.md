# R2-A：活锁 vs 死锁的一击判别

## 方法

沿用 E4 的 `spin_wait_tokenchain.cubin` / `host_test_biggrid_tokenchain`（源码见
`../E4_spin_wait/variants/host_test_biggrid_tokenchain.cpp`），前台启动 `grid=170`
（无 `timeout`，故意不 kill，等它挂起），在另一终端用 `sudo sysctl -w kernel.yama.ptrace_scope=0`
放开 ptrace 限制后依次采集：
1. `nvidia-smi dmon -s u -c 15`（SM 利用率）
2. `nvidia-smi -q -d POWER,CLOCK`（功耗/时钟）
3. 三次独立 `cuda-gdb -p <pid> -batch -ex "info cuda threads"`，间隔约 3s / 15s，比较 block 集合与 PC 是否漂移
4. `cuda-gdb` 反汇编卡住的 PC 附近指令，与 `spin_wait_tokenchain.sass` 里已知的自旋循环地址比对
5. `compute-sanitizer --tool synccheck` / `--tool racecheck`，各跑一次（timeout 90/120s）

## 1. SM 利用率 / 功耗 / 时钟（挂起期间）

```
$ nvidia-smi dmon -s u -c 15
# gpu     sm    mem    enc    dec    jpg    ofa
    0    100      0      0      0      0      0   (×15，全部 100%)

$ nvidia-smi -q -d POWER,CLOCK
    Instantaneous Power Draw : 97.71~98.33 W   (idle 基线 13.19W，TDP 575W，约17%)
    SM Clock                 : 2887 MHz         (Max 3090 MHz，接近满速)
```
✅ 已验证：GPU 在挂起期间**不是空闲**——SM 利用率读数 100%、时钟被拉到接近满速、功耗明显高于
idle（约 7.4×），表面上完全符合"活锁"的直觉判据。但功耗只有 TDP 的 17%，远低于"全部 170 个
block × 256 线程持续打 LDG+CCTL.IVALL"这种内存子系统饱和场景下预期的功耗量级——这是第一个和
假设不完全吻合的细节，留到下面 §3 解释。

## 2. cuda-gdb：block 集合与 PC 三次独立采样，间隔 >20 秒，完全不变

三次独立的 `cuda-gdb -p PID -batch` 调用（每次都是全新 attach/detach，无状态复用），
覆盖约 20+ 秒的时间窗口：

| 采样 | 时间点 | 驻留 block 数 | PC |
|---|---|---|---|
| #1 | T+0s | 57 个（block 14,15,16,...,167，不连续） | `spin_wait_tokenchain+24976` (0x6190) |
| #2 | T+3s | 同一组 57 个 block，逐行完全一致 | 同一 PC 0x6190 |
| #3 | T+15s（累计 T+18s+） | 同一组 57 个 block，逐行完全一致 | 同一 PC 0x6190 |

三次采样的 `info cuda threads` 输出逐字节相同（block 列表、每行的 thread range、PC 全部一致）。
**这是死锁的判据，不是活锁的判据**：如果是自旋循环在自由运行（哪怕只是几条指令的紧凑循环），
GPZ 时钟下每纳秒都在推进 PC，20 秒后两次独立采样命中同一条指令地址的概率极低；实际观测到的是
**同一批线程在 20+ 秒内一条指令都没有前进**。

同时，170 个 block 里只有 57 个（约 33%）出现在 `info cuda threads` 列表里，其余 113 个
（约 67%，包括 block 0 即生产者，以及 block 1~13）**完全不在列表中**——在这种 grid 规模下
driver 自己报告的 occupancy（8 blocks/SM，理论 grid 上限 1360，见 E5）远超 170，所有 170 个
block 应该都能同时驻留；缺席只能解释为**这些 block 已经执行完毕并退出**。

✅ 已验证：170 个 block 里约 2/3 成功完成（含生产者本身），约 1/3 永久卡死在同一条指令，
而不是"全部 170 个 block 都在无休止自旋"。

## 3. 卡住的 PC 不在自旋循环里，而在自旋之后的 reduce 收尾代码里

反汇编卡住地址（`spin_wait_tokenchain+24976` = 偏移 `0x6190`）：

```
/*6180*/  FADD R5, R12, R15 ;
/*6190*/  LDS R4, [UR5] ;              <- 卡住的 PC
/*61a0*/  FADD R5, R5, R4 ;
/*61b0*/  STG.E desc[UR8][R2.64], R5 ; <- 写 checksum_out（第3个 kernel 参数，c[0x0][0x390]）
/*61c0*/  BRA 0x65c0 ;
```
往前追溯几条指令能看到 `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` 和一条条件 `@!P2 STS [...]`——这是
consumer 分支里 `reduce %chunk_data ... -> tile<f32>` 的 warp 间归约收尾（`SHFL.BFLY` warp shuffle
+ 部分线程 `STS` 写共享内存 + 另一部分线程 `LDS` 读回 + `BAR.SYNC.DEFER_BLOCKING` 做块内同步），
**不是** round 1 SASS 里记录的自旋等待循环本体：

```
# 对照：round 1 记录的自旋循环（偏移 0x1f0~0x230，函数最前部）
/*0200*/  LDG.E.STRONG.GPU R4, desc[UR8][R2.64] ;
/*0210*/  CCTL.IVALL ;
/*0220*/  ISETP.NE.AND P0, PT, R4, 0x1, PT ;
/*0230*/  @P0 BRA 0x1f0 ;
```
卡住的偏移 `0x6190` 远在函数尾部（`cuobjdump -sass` 全量反汇编共 3321 行，`0x6190` 出现在文件
接近末尾处），而自旋循环体在偏移 `0x1f0~0x230`，两者相差约 6000 字节的代码距离。全量 SASS 里
`BAR.SYNC.DEFER_BLOCKING` 出现次数极多、地址等间隔递增（`0x1f0, 0x260, 0x370, 0x480, ...`），
与 `%chunk2` 循环上界固定为编译期常量 256 吻合——**tileiras 把这个 256 次迭代的收尾 reduce for
循环完全展开（unroll）成了直线代码**，每次展开都带一次真实的硬件 `BAR.SYNC`；卡住的位置正是
这条展开链的最后几条指令附近。

**这条证据直接推翻了"卡在自旋循环本身"的直觉假设**：57 个卡死的 block 已经越过了自旋等待（已经
看到了 flag），是在自旋等待*之后*的、由 256 次展开的块内归约（每次都要素 `BAR.SYNC`）构成的收尾
代码里卡住的。是否所有 57 个 block 都卡在 reduce 循环的同一条 SASS 指令上未逐个验证（只对 block
14 做了完整反汇编定位，但三次快照里所有可见 block 行的 PC 列都是同一个 0x6190，间接说明其余
56 个 block 大概率也卡在同一处）。

## 4. compute-sanitizer：在 sanitizer 插桩下，同样的 grid=170 反而 3/3 没有挂起

```
$ timeout 90  compute-sanitizer --tool synccheck  ./host_test_biggrid_tokenchain 170
launching grid=170 ... kernel completed without hang.  ========= ERROR SUMMARY: 0 errors

$ timeout 120 compute-sanitizer --tool racecheck   ./host_test_biggrid_tokenchain 170
launching grid=170 ... kernel completed without hang.  ========= RACECHECK SUMMARY: 0 hazards

$ timeout 90  compute-sanitizer --tool synccheck  ./host_test_biggrid_tokenchain 170   (第二次)
launching grid=170 ... kernel completed without hang.  ========= ERROR SUMMARY: 0 errors
```
三次独立运行（synccheck ×2、racecheck ×1），**在 sanitizer 插桩下 grid=170 一次都没有挂起**，
且两个工具都报告 0 个错误/hazard。这与 E5 的"预热能救 grid=80"现象是同一类信号——sanitizer
插桩显著改变了指令调度节奏/时序，恰好绕开了触发条件，而不是说明 bug 不存在（sanitizer 没有
观测到失败状态，自然也无法在失败状态里检测到 hazard）。✅ 已验证：sanitizer 插桩下 3/3 不挂起；
❌ 不能据此断言没有 race/sync 违规——只能说这次侥幸没跑进坏状态。

## 5. 判断：这是什么？

**证据本身存在张力，不强行调和成一个干净结论**：

- 支持"活锁"的证据：SM 利用率 100%、时钟被拉到接近满速、功耗高于 idle 约 7×（§1）。
- 支持"（部分）死锁"的证据：同一组 57/170 block 在跨越 20+ 秒的三次独立采样中，PC **完全没有
  移动**（§2）；卡住的位置是一次性收尾代码里的 `BAR.SYNC` 附近，不是循环体（§3）。

**给出的判断（不强行二选一，如实报告为混合现象）**：

1. 170 个 block 里约 2/3（113 个，含生产者）**成功完成**，这部分工作是 §1 里 100% SM 利用率 /
   高时钟 / 高功耗的来源——它们在挂起期间持续做了实际工作（写数据、计算 checksum、自旋+跳出），
   不是"全体卡在自旋"。
2. 剩下约 1/3（57 个观测到的样本）**冻结在自旋循环*之后*的一段代码里，且是真正的冻结（PC
   20+ 秒不变），不是活锁式的"忙但原地打转"**。这部分是**局部死锁**，判定依据是 cuda-gdb 反复
   采样 PC 零漂移这个直接证据，比 SM 利用率这种粗粒度间接指标更可信（nvidia-smi 的
   "utilization.gpu" 只表示采样窗口内"至少有一个 kernel 在跑"，不代表所有 SM 都在有效推进指令，
   与冻结的 warp 仍然占着 SM 资源但不前进完全兼容）。
3. **对第一轮假设的直接证伪**：round 2 的假设是"自旋循环本身因为没有退避而饱和内存子系统，
   导致活锁"。本实验的直接反例是：卡住的位置根本不在自旋循环里（§3），而是在自旋循环*之后*，
   由 256 次展开的块内归约构成的收尾代码、且卡在一条紧跟 `BAR.SYNC.DEFER_BLOCKING` 的
   `LDS` 指令上。这意味着 flag 的可见性机制（token 链 + acquire/release）在这个更大规模的测试里
   **看起来是工作的**——至少 113/170 个 block（含生产者）证明了这一点——真正的死锁疑点转移到了
   **`reduce` 算子在大量并发 block 同时执行、且大量硬件 barrier 同时活跃的场景下的 lowering**
   上（❌ 推测：可能是某个 barrier 的到达计数与实际线程数不匹配、也可能是大规模并发 CTA 下
   `BAR.SYNC.DEFER_BLOCKING` 这种"延迟阻塞"语义在 GPU 调度器层面有公平性/资源竞争问题，均未坐实）。

## 6. 对后续 R2-B/C 的直接影响（预告，如实标注为推测）

如果 §3/§5 的分析正确，那么 R2-B（relaxed 自旋 + acquire）和 R2-C（自旋退避）**这两个只修改
自旋循环本身写法的方案，理论上不应该能修复这次观测到的死锁**——因为死锁根本不在自旋循环里。
这是一个可以被 R2-B/C 的 grid 扫描统计直接检验的预测；如果 R2-B/C 依然在同样的 grid 规模挂起，
就印证了"根因在 reduce 收尾代码而非自旋本身"这个判断；如果 R2-B/C 意外地解决了问题，则说明
自旋循环的写法（哪怕只是编译器围绕它生成的指令调度/寄存器分配副作用）间接影响了下游 reduce
代码的行为，这将是一个值得进一步深挖的意外结果。❌ 推测，留给 R2-B/C 检验。

## 7. GPU 健康性核实

`kill -9` 杀掉挂起进程后，`nvidia-smi` 立即回到 idle（0%/13W），随后一次全新的 `grid=4` 调用
立即成功完成（`checksum=34359672832.000000`），与 E4/E5 的健康性核实结果一致：这是这个具体
kernel/grid 组合的死锁，不是 GPU 硬件层面的永久损坏。

## 原始命令记录

- 挂起进程：`./host_test_biggrid_tokenchain 170`（后台启动，PID 记录于 `/tmp/r2a_hang.pid`，
  未加 timeout，故意不 kill 直到采集完成）
- `sudo sysctl -w kernel.yama.ptrace_scope=0`（默认值为 1，禁止非父进程 ptrace 附加；本环境有
  passwordless sudo，放开后 `cuda-gdb -p PID` 才能工作）
- 完整 dmon / POWER,CLOCK / 三次 `info cuda threads` / 反汇编 / compute-sanitizer 原始输出见本
  result.md 正文引用（未截断的完整日志因体积原因未单独存档，均为本次会话内的实时命令输出）。
