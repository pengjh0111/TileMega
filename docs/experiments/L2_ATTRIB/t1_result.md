# T1：事件路径的独立干预

❌ **T1 的性能门槛失败，按用户 §6 停止后续性能扩展。** 冻结源码的 S=128
两级通知没有显著压低 notify，反而在四格全部显著变慢。所有新开关保留、默认关闭。
这不是完成全部验收的成功交付。

## 实现与开关

| 子项 | 宏（前缀 TILEMEGA_EVENT_） | 实现与状态 |
|---|---|---|
| T1.1 | LOAD_POLL | `EventSync.cuh:13`，device-scope relaxed atomic load；保留 nanosleep |
| T1.2 | SPLIT_LINES | `ModelRuntime.h:234`，arrivals/epoch 各 128 B，总 256 B；尺寸/对齐断言 |
| T1.3-A | SHARDED、SHARDS | `ModelHarness.cuh:425` 两级通知，`:1111` 主机按实际 producer 归属计算每片目标 |
| T1.3-B | CLUSTER_FANIN、CLUSTER_RESERVE | `ModelHarness.cuh:580` 初始化/保存 DSM 计数器，`ClusterSync.cuh` 映射 cluster peer |
| T1.4 | RELEASE_STORE | 关闭；若强行启用编译报错，未实现未经证明的 fence 删除 |

T1.1/T1.2 在 ModelHarness 与 GeneratedLlamaRuntime 的重复定义均同步；另外发现
`lib/Codegen/Codegen.cpp` 也生成 WAIT 宏，已让它调用同一 EventPoll，否则只改两个头文件不能覆盖实际生成路径。

A 的默认 S 为不超过 `TargetSpec::Res::num_sms` 的最大 2 次幂；4090 为 128。
它限制每片争用而使二级扇入不超过 SM 数，是待扫描检验的选择，不预先声称最优。
每事件使用 min(S, members) 个分片，物理 CTA 对该数取模；只计非空分片的二级触发数。
S=1 直接走原单级算法。aggregate 和 κ>0 fine group 共用 ArriveEvent。

B 由 Caps/ClusterSync 能力选择，未写架构号比较。task 队列中插入逐 task 的
cluster barrier 会要求不同 CTA 以相同顺序到达，可能死锁，因此使用非阻塞 DSM
到达计数；每簇最后一个 producer 再到全局计数。cluster.sync 只在 kernel 首尾
用于 DSM 初始化和生命周期闭合。一级计数跨 launch 保存到全局，不在迭代间清零。
当前预留 TargetSpec 的动态 SMEM 上限（扣去静态 shared），再计算 cluster occupancy；
可能降低驻留，不能隐藏这项成本。CLUSTER_RESERVE 可单独打开做同资源预算对照。
计数器超出预算时明确拒绝，不切回单级。

⚠️ B 目前仅 sm_120 交叉编译通过；最终交叉编译二进制的 L2 静态 SASS 中有
4 条 UCGABAR（两对 ARV/WAIT，对应首尾闭合），不是运行时执行次数。人工脚本
`../CLUSTER/run_notify_cluster_sm120.sh` 扫 1/2/4/8，统计 L2 的 UCGABAR、四臂 notify
及正确性；未在 5090 执行，不能声称收益或并发正确性已验证。

## 顺序语义

T1.1 选择 `cuda::atomic_ref<ull, cuda::thread_scope_device>::load(relaxed)`，而非
普通 volatile C++ load。等待成功后的既有 CTA 屏障与每线程 device fence 保留；
通知前每个 writer 的 fence 和 CTA 屏障也保留。成功的 relaxed atomic read 与
后续 acquire fence、发布端 release fence + atomic modification 构成 fence
synchronization；不能把“任意普通 load 后加 fence”当作同样的证明。
依据：[CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)。

两级 needed 分别是 shard 的真实 producer 数 × (iteration+1)、非空 shard 数 ×
(iteration+1)。胜出的一级到达者在二级 RMW 前 fence，二级胜出者发布 epoch 前 fence。
epoch 始终发布 iteration+1；没有迭代重置。GPU 进程内普通串行 kernel launch
之间保存状态，不在本改动中声称实现了跨 launch 的并发执行。

T1.4 没有启用：仅 tid0 的 release store 要覆盖 CTA 写入以及两级到达者的传递性，
不能仅引用 cumulativity 一词便删掉所有 writer fence。未完成该替换的逐边内存模型
论证及小 tile 50 进程压力测试，故保留原 fence；不是测出收益为零。

## 已完成的独立 T1.1 / T1.2 实验

✅ `raw_t1_initial/correctness.tsv`：2 模型 × seq={4,128} × 4 开关状态 ×
50 个全新进程，共 800/800。不是完整 seq×past 1500 矩阵。
✅ `raw_t1_shard_k4/correctness.tsv`：S=16、load+split、κ=4，四个模型/seq 单元
各 50，全新进程共 200/200，覆盖非单例 fine group。

四臂各 25 轮，轮内轮转，20,000 次配对 bootstrap；Wilcoxon 使用去零、结点修正
和连续性修正的双侧正态近似。每格 median/95% CI 见 `t1_initial/summary.tsv`，
相对 base 的配对差与 p 值见 `t1_initial/paired.tsv`。时间单位 ms：

| model / seq | base notify | load only | split only | load + split |
|---|---:|---:|---:|---:|
| gqa2 / 4 | .060416 | .060256 | .060416 | .060416 |
| gqa2 / 128 | .070368 | .070656 | .069856 | .069632 |
| mha4 / 4 | .112640 | .109888 | .113632 | .110592 |
| mha4 / 128 | .160544 | .160768 | .161568 | .161792 |

✅ 这些干预的 notify 配对差区间全部包含 0；不能宣称显著下降。
四臂 notify=nowait−neither，两臂都没有 wait，因此 wait 对 notify 的争用放大
在这个分解中被计入 full−nowait 的交互项，而非 notify 项。这限制因果解释，
但不修改用户要求的四臂定义或通过阈值。

分析时曾在 TSV 尚未写完时启动汇总，缺臂导致异常；已增加完整配对轮数检查，
并在全部 1600 行四臂数据写完后重新汇总，未用部分轮结果作结论。

## 分片探索扫描

✅ `raw_t1_shard_scan/correctness.tsv` 的八个开关状态、四个模型/seq 单元，
各 50 个全新进程，1600/1600。κ=1。

以下是 25 轮内配对 L2/L1 中位数，完整区间与相对 base 的配对差见
`t1_shard_scan/{summary,paired}.tsv`。shardN 同时打开 load+split，fanin64 只开分片。

| 状态 | gqa2 / 4 | gqa2 / 128 | mha4 / 4 | mha4 / 128 |
|---|---:|---:|---:|---:|
| base | 1.072139 | 1.071226 | 1.080201 | 1.087574 |
| load+split | 1.069825 | 1.064182 | 1.073698 | 1.087422 |
| S=1 | 1.091369 | 1.081962 | 1.096142 | 1.098039 |
| S=4 | 1.104738 | 1.094959 | 1.108449 | 1.114110 |
| S=16 | 1.109726 | 1.094053 | 1.109710 | 1.114871 |
| S=64 | 1.109255 | 1.095071 | 1.109296 | 1.115160 |
| S=128 | 1.107232 | 1.090747 | 1.109448 | 1.114769 |
| fanin64 only | 1.114677 | 1.094643 | 1.116191 | 1.119745 |

⚠️ 这次探索扫描的编译与 B 的实现修改有时间重叠，虽 B 关闭，不把它当作最终
冻结源码的单变量因果证据。`snapshot.json` 是完成后的源码/二进制快照，不能冒充
逐二进制的编译时源码证明。停止判据使用之后冻结运行时代码、重新编译 base 与
S=128 的独立四臂复测。以后 run_t1.py 会在编译前保存 build_manifest.json。

补充核对：两模型的扫描版/最终版 L2 SASS 分别逐字节一致，base 摘要
`3c579d8a351a12b3ecadf36451abdd3df5de23b1c3ed8c5f76256763d49cfba7`，S=128 摘要
`5d2d2b37ffde7e3e65662cf0861a7932b350c7beddb9fdb3d0ea6e7304d361a5`。
取 cuobjdump 中 Function 名含 tilemega_l2_kernel 的段做 SHA256。这是设备代码
一致性的证据，不把它扩大成所有 host 初始化代码的等价证明。

S=1 的运行算法直接退化为单级，不保证编译器消除所有计划/分支/寄存器成本；
不能把 S=1 与 base 的差值误称为额外原子次数导致。S=128、mha4/128 的一级计数器
为 8844 个，占 1,132,032 B（每个 128 B），尚未做存储压缩。

## 冻结源码四臂：停止门槛触发

✅ `raw_t1_final/attrib.tsv`：2 模型 × 2 seq × 2 状态 × 4 臂 × 25 轮，800 行。
其中 full 200/200；额外 `correctness.tsv` 200/200。每个模型/seq/状态均有
25 个四臂 full 进程 + 25 个额外正确性进程 = **50/50 全新进程**，总计 400/400。
该冻结版本不是完整 seq×past 矩阵（仍只有 seq=4/128、past=3）。

以下分解为轮内差值的中位数，单位 µs。各格 median/95% CI 与配对 Wilcoxon
见 `t1_final/{summary,paired}.tsv`；不能要求各项中位数相加仍严格等于总项中位数。

| model / seq | 状态 | L1 | L2 | notify | wait | barrier | loop |
|---|---|---:|---:|---:|---:|---:|---:|
| gqa2 / 4 | base | 412.512 | 441.184 | 61.312 | 24.704 | 46.272 | -10.240 |
| | S=128 | 411.840 | 455.680 | 76.800 | 23.488 | 46.080 | -9.376 |
| gqa2 / 128 | base | 571.552 | 612.352 | 68.768 | 25.792 | 41.120 | -14.144 |
| | S=128 | 573.440 | 627.776 | 83.040 | 26.784 | 41.184 | -14.336 |
| mha4 / 4 | base | 804.896 | 874.496 | 113.920 | 45.056 | 74.752 | -16.160 |
| | S=128 | 807.872 | 900.960 | 142.336 | 43.776 | 76.640 | -16.224 |
| mha4 / 128 | base | 1144.832 | 1248.128 | 167.008 | 45.952 | 81.120 | -27.648 |
| | S=128 | 1143.968 | 1279.968 | 193.536 | 50.176 | 82.048 | -26.496 |

| model / seq | notify 配对增量 µs | 配对 95% CI µs | Wilcoxon p |
|---|---:|---|---:|
| gqa2 / 4 | +15.552 | [+14.336, +17.344] | 1.305e-5 |
| gqa2 / 128 | +14.112 | [+11.456, +15.456] | 1.307e-5 |
| mha4 / 4 | +27.648 | [+25.344, +30.432] | 1.307e-5 |
| mha4 / 128 | +25.568 | [+22.432, +28.768] | 1.306e-5 |

✅ 每轮 `L2 = neither + notify + wait` 的闭合误差为 0.000000 µs；这是四臂定义的
代数恒等式，不能独自验证因果归因。notify 仍为最大正项，裸队列 loop 仍为负。

❌ 在该实现与硬件上，“拆线 + load + 分片足以显著压低主项”的干预预测被否定。
它不证明原子争用不存在；额外一级 RMW、目标/计划加载、分支和寄存器代价均可能
抵消争用减少，但本轮没有进一步把这些候选原因隔离，不将候选写成已证实的解释。
按停止指令不继续调参，不启用这些默认开关，也不靠 T1.4 删 fence 追逐验收。

⚠️ 完整 1500 seq×past 矩阵、seq=512 E2E、κ 全性能扫描未执行，因 T1 性能门槛
停止；B 硬件实测仍留给人工脚本。T2 有独立的归因失败门槛；T3 已独立完成。

✅ `cmake --build build-portable --target check-policy` 通过；该构建的 ctest
24/24 通过（含 target_audit）。当前完整构建目录为 build-portable，而非旧 build。

复现入口：`T1_VARIANTS=base,shard128 docs/experiments/L2_ATTRIB/run.sh`。
`T1_VARIANTS=base,load,lines,load_lines` 复现前两项独立对照；两者均重新生成源码。
