# R2-C 结果：自旋退避（backoff）能否消除挂起

## 结论速览

**⚠️ 部分/有限缓解，远未达到"解决问题"的程度；grid=170（硬件驻留容量边界）时完全无效。**
grid=30 时挂起率随退避强度增大呈单调下降（backoff64 25/30=83% → backoff256 22/30=73% →
backoff1024 16/30=53%），说明降低轮询频率确实有一定的量级缓解效果；但即便是最强的 1024 档，
挂起率仍高达 53%，远不是"消除"。更重要的是，grid=170（即 R2-D 确认的硬件真实驻留容量边界）
时，**全部 4 个变体（含与 R2-B relaxed+acquire 组合的变体）无一例外 30/30 挂起、100%**，
退避在这个规模上没有观察到任何缓解迹象。与 R2-B 的 relaxed+acquire 修复组合后
（`relaxed_backoff256`）在 grid=30 反而略差于单独的 backoff256（26/30=87% vs 22/30=73%），
说明两种修复手段叠加并不产生正向增益，甚至可能有轻微的负面交互（⚠️ M=30 时的噪声也可能是
部分原因，未做显著性检验）。

## 一个关键方法论陷阱（先于结果报告，因为它一度让整个实验的第一版设计完全失效）

**朴素的累加式"delay loop"会被编译器识别为等差数列闭式解，整体常量折叠掉，产生零延迟。**
第一版设计用 `%dacc2 = addi %dacc, %bi`（对循环变量 0..N-1 逐次累加）实现延迟循环，
本意是让每次轮询之间插入 N 步计算延迟。但 tileiras 把这个循环识别成了等差数列求和
（sum = N(N-1)/2 的闭式解），编译期直接算出结果，生成类似
`UIADD3 UR4, UPT, UPT, UR4, 0x7e0, URZ`（0x7e0 = 2016 = 64×63/2，即 N=64 时的闭式常量）
这样的单条指令——循环整体消失，产生的实际运行时延迟是**零**，完全违背实验设计意图，
且这个错误在不对比不同 N 档位的 SASS 之前很难发现（表面上看 verifier/translate/tileiras/
运行全部正常，只有把 N=64 和 N=1024 两个变体的 SASS 摆在一起比较，才会发现二者指令数/
跳转地址完全相同——这才是真正的破绽）。

**修复**：把递推关系换成不可被识别为闭式的非线性 LCG（linear congruential generator）式
链条：`%dmul = muli %dacc, 1103515245`（LCG 乘数）→ `%dmulc = addi %dmul, 12345`（LCG 加数）
→ `%dacc2 = xori %dmulc, %bi`（引入循环变量依赖，防止被识别为纯粹的线性递推）。修复后用
SASS 重新验证，确认延迟循环不再被折叠（见下方"SASS 证据"一节）。

**这个陷阱本身是本实验除主结论外最值得记录的产出**：它说明在这条工具链上写"故意拖慢"的
微基准代码时，不能想当然地认为一个语法上是循环的结构就一定会在运行时循环执行——闭源
编译器后端的常量折叠/强度削减能力足以让某些循环整体消失，必须用 SASS 反汇编逐个变体
交叉验证指令数量是否真的随参数（这里是 N）成比例变化，才能确认延迟是真实的。

## SASS 证据：修复后的延迟循环确认未被折叠、且不改变轮询频率

```
$ grep -n "41c64e6d\|0x3039" backoff64.sass | head -5
95:  /*02c0*/  UMOV UR4, 0x41c64e6d ;                    # 0x41c64e6d = 1103515245（LCG 乘数）
101: /*02f0*/  UIMAD UR5, UR5, UR4, 0x3039 ;              # 0x3039 = 12345（LCG 加数）
103: /*0300*/  ULOP3.LUT UR5, UR5, 0x3, URZ, 0x3c, !UPT ; # 0x3=循环变量，逐条不同（xor）
...
```

**指令数随 N 线性scale**（`grep -cE "UIMAD|ULOP3|LOP3|XOR"`）：

| 变体 | N | SASS 总行数 | UIMAD/LOP3/XOR 指令数 | 外层轮询循环回边地址 |
|---|---|---|---|---|
| backoff64 | 64 | 3561 | 129 | `0xaa0 @P0 BRA 0x270` |
| backoff256 | 256 | 4329 | 514 | `0x22b0 @P0 BRA 0x270` |
| backoff1024 | 1024 | 7401 | 2050 | `0x82b0 @P0 BRA 0x270` |

指令数（129→514→2050）与回边目标地址（0xaa0→0x22b0→0x82b0）都随 N 近似线性增长
（≈2 条指令 ×(N-1) 步的 LCG 链），**证实延迟循环这次是真实存在、未被折叠的**。

**轮询频率（每轮外层 while 迭代只做一次 flag 读取）不受延迟档位影响**——`BAR.SYNC`
与 `CCTL.IVALL` 的静态出现次数在三个 N 档位下完全一致（均为 85 和 226），说明退避链只是
在同一个外层循环体内、每次轮询之间插入了更长的纯计算延迟，没有改变"每次外层迭代恰好读一次
flag、做一次 CCTL.IVALL/BAR.SYNC"这个结构：

```
$ for f in backoff64 backoff256 backoff1024; do grep -c "BAR.SYNC" $f.sass; grep -c "CCTL.IVALL" $f.sass; done
85 226   (backoff64)
85 226   (backoff256)
85 226   (backoff1024)
```

REG=228（`cuobjdump --dump-resource-usage`，四个变体一致，与 baseline/R2-B 同一量级，
受控对比）。

## 正确性

未单独重跑正确性测试——本实验四个变体的延迟逻辑只影响轮询间隔，不改变 flag 的
读取/校验/reduce 逻辑本身（延迟循环的累加结果通过 `itof`→`mulf` by 0.0 无副作用地折入
checksum，与 R2-B 使用的同一 DCE-avoidance 技巧），且 grid-scan 里 4 个变体在 grid=30/170
凡是"OK"退出的运行，`host_test_single.cpp` 内部已经做了参考值比对（沿用 R2-B 同一 harness，
`../R2_B_relaxed_plus_acquire/host_test_single.cpp` 逻辑不变），没有观察到任何 "ERR" 状态
（grid_scan.sh 会把非 HANG 非 "completed without hang" 的输出计入 hang 统计，本次运行日志
中不存在这类第三态，即所有非挂起的运行都正确通过了内部校验）。

## 核心结果：grid-scan 挂起率统计（M=30/grid，4 个变体 × grid∈{30,170}）

沿用与 R2-B 相同的"worst case"无预热单次启动 harness（`host_test_single.cpp`，5 秒超时）。
矩阵相对 R2-B 缩减（grid 只取 {30,170} 两个极值点、M=30 而非 50、超时 5s 而非 6s）——
明确是受时间预算限制的缩减，非默认选择，如实标注。

| 变体 | grid=30 | grid=170 |
|---|---|---|
| `backoff64`（64 步延迟） | 挂起 25/30（83%） | 挂起 30/30（100%） |
| `backoff256`（256 步延迟） | 挂起 22/30（73%） | 挂起 30/30（100%） |
| `backoff1024`（1024 步延迟） | 挂起 16/30（53%） | 挂起 30/30（100%） |
| `relaxed_backoff256`（R2-B relaxed+acquire ⊕ 256 步延迟组合） | 挂起 26/30（87%） | 挂起 30/30（100%） |

对照 R2-B 的 baseline（M=50，不同 M，仅供量级参考）：`baseline_acquire_in_loop` grid=30
挂起 39/50（78%），grid=170 挂起 50/50（100%）。

原始命令与逐次输出（完整记录见 `/tmp/r2c_gridscan.log`，由 `grid_scan.sh` 生成）：
```
$ ./grid_scan.sh /tmp/r2c_gridscan.log 5 30
...
SUMMARY backoff64 grid=30: 挂起 25/30 次
SUMMARY backoff64 grid=170: 挂起 30/30 次
SUMMARY backoff256 grid=30: 挂起 22/30 次
SUMMARY backoff256 grid=170: 挂起 30/30 次
SUMMARY backoff1024 grid=30: 挂起 16/30 次
SUMMARY backoff1024 grid=170: 挂起 30/30 次
SUMMARY relaxed_backoff256 grid=30: 挂起 26/30 次
SUMMARY relaxed_backoff256 grid=170: 挂起 30/30 次
ALL DONE
```

**结论（✅ 已验证，M=30 per cell，明确标注样本量小于 R2-B 的 M=50、结论强度相应打折扣）：**

1. grid=30 时，退避强度（延迟步数）与挂起率之间存在**单调、量级明显的负相关**
   （83%→73%→53%，backoff64 到 backoff1024 几乎腰斩），这是退避方案本轮观察到的唯一
   正面信号——降低轮询频率确实能部分缓解某种与轮询频率相关的竞争/资源争用效应。
2. 但即便是最强档位（1024 步延迟）也只把 grid=30 的挂起率降到 53%，仍然是"多数情况下会
   挂起"，远不是一个可用的修复。
3. grid=170（硬件驻留容量边界，R2-D 已确认）时，**退避完全无效**——4 个变体一致
   100% 挂起，与档位、与是否叠加 R2-B 的 relaxed+acquire 无关。这与 R2-A/R2-F 的预测
   一致：如果根因是 `BAR.SYNC.DEFER_BLOCKING` 在极限占用（grid=170 恰好是硬件能同时
   驻留的全部 block 数、零富余度）下的收敛/公平性问题，那么无论轮询多稀疏，只要 block
   还没有机会调度上 SM 执行完，这个根因就依然存在；退避只能缓解"轮询过于频繁导致的某种
   次要争用"，不能缓解"硬件资源占满导致的调度/收敛问题"。
4. **relaxed+acquire 与退避组合（relaxed_backoff256）没有产生叠加正向收益**，在 grid=30
   反而比单独 backoff256 略差（87% vs 73%）——这提示两种修复思路可能不是简单可加的
   （⚠️ documented-but-unverified：M=30 时这个差异是否是统计噪声未做显著性检验，
   不排除只是运行间方差）。

## 涉及文件

- `spin_wait_backoff64.mlir` / `spin_wait_backoff256.mlir` / `spin_wait_backoff1024.mlir` —
  三档 LCG 退避变体（1MB 数据，对应 E4 baseline 的 262144 float）。
- `spin_wait_relaxed_backoff256.mlir` — R2-B relaxed+acquire 与 256 档退避的组合变体。
- `backoff64.sass` / `backoff256.sass` / `backoff1024.sass` — 反汇编产物，用于验证延迟循环
  未被折叠、轮询频率不受影响。
- `host_test_single.cpp` — 与 R2-B 完全相同的单次启动、无预热 harness（从
  `../R2_B_relaxed_plus_acquire/` 拷贝，未修改）。
- `grid_scan.sh` — 4 变体 × grid∈{30,170} × 30 次的挂起率统计脚本。
- `/tmp/r2c_gridscan.log` — 完整逐次记录（本机 `/tmp`，未拷贝进实验目录，摘要已录入上方）。

## 对 megakernel 可行性的意义

退避是 R2-B 之后唯一还没测过的候选修复方向，本实验的结果是**部分正面但整体仍是负面结论**：
它证明轮询频率与挂起率之间确实存在某种量级相关性（不是完全无关的两件事），这是一个新的、
之前任何轮次都没有的定量证据，为"根因与轮询/竞争频率有关"这一假说增加了一点点支持；但
它在最重要的场景（grid 达到硬件容量边界）完全无效，且即便在能观察到缓解效果的 grid=30 也
无法把挂起率降到可接受水平。**结合 R2-A/R2-B/R2-F 的证据链，round 2 目前为止测试过的所有
候选修复方案（relaxed 内存序、退避、二者组合）均未能解决 megakernel 场景下跨 tile block
同步在大 grid 规模下必然挂起的问题**，根因指向自旋循环之外的 `BAR.SYNC.DEFER_BLOCKING`
收敛机制这一假说依然只有间接支持，未被最终坐实。

## 状态：R2-C 完成 ✅（核心 grid-scan 统计与 SASS 验证均已完成；"用 cuda-gdb 精确定位挂起
现场 PC 并与 baseline/R2-B 比对"这一步定点验证，与 R2-B 同样标记为可选的后续加强项，
未在本轮完成）
