# R13 sm120 — 最终验收与 E6 分析（带限制封存）

状态：verified，可执行项目及 r7 补采集已完成并终验；195/195 可用性能记录收齐，不是 297/297 全规格通过。
PR 页 trace 的两份重编、四项采集及 E6 更新全部 exit0；没有待运行实验，未通过/不可用项按规格保留。
本报告的性能来自未改变的 E4 数据；本轮不追加实验、不实施 R14 方案。所有提交仅本地保存，不 push。

## 基本信息与规格偏离

- verified：开工 HEAD `36f17e6ee66404660bccf7df40946c3cdbe208b5`；在 sm89 Phase D 尚未完成时并行启动（用户授权）。
- verified：sm89 最终 `9aebaf6553247ec83c79bc8f101e61ad4ce564fd` 已合并；上游设备/编译器源码相对开工检查点未变。
- verified：本阶段测量整理前 HEAD `9759e14e3`；PR trace 采集修正提交 `7f8cfb98a`。
- verified：最终验收开工 HEAD `4c0946989ee7db4323c88da123ed4d2af45d7fc9`；离线统计/证据修正代码 HEAD `fef76d4cc`。
- prompt：`/root/Prompt/TileMega_R13_sm120_prompt.md`，本机保存的 LF 文本 SHA256：
  `05f3b54517180636152a0423e63d8ab110efa41410d86f957421acf7d8ec5009`。
- 用户覆盖：保持已安装 vLLM 0.29.0，不切换为 sm89 的 0.30.0。两机版本不同，不能将跨机差异全部归因于架构。
- verified：27 个已冻结性能二进制 SHA、九节标定印章、SL-5 特征/solver 配置与最终基线对齐；不复用 sm89 .so。
- verified：本机模型 config/权重 SHA 与本机此前固定 revisions 一致；sm89 完整 SHA 未提供，跨机同权重未验证。
- verified：所有 GPU 测量经过继承的 scheduler/guard/lock；设备恢复后重新六采样，空闲功耗中位数 29.95 W，其余阈值不变。
- 明确偏离：MB-1c 十二个已知非法指令组合隔离为 unavailable；不再次触发已观察的设备故障，不冒充通过。
- E4e cluster 端到端与 E5e ncu 按可选项省略；不扩大实验空间，不调优、不修改求解器/代价模型。

## 代码改动与首次执行

| 改动 | 提交 / 验证 |
|---|---|
| native-only target 读写/准入，不误拒 optional legacy=0 | `17860b184`；实机全九节标定与 CPU round-trip 通过，无同步改动 |
| 仅 host 隔离 MB-1c 的 TN128/TK64/method5 | `c65d02b91`；204 点照原 kernels 执行，12 点 unavailable，不称为 PTX 修复 |
| 本机 anchor policy / NVML 退出后采样判断 | `ca91ce0b0` / `40c36af17`；守卫阈值未放宽，无设备同步改动 |
| 自动依赖 / 最终 CPU 检查器回放 / 缺失锚定轮恢复 | `31070cd45` / `7d804043c` / `fbc8c373c`；原始记录保留 |
| PR trace 补 PAGE_TRACE，并隔离输出、clean-guard 后接纳 | `7f8cfb98a`；两份原源码/几何重编、四项 clean guard 采集通过，性能二进制未变 |
| 离线页统计区分累积次数、实际 past、最后一次时间戳 | `fef76d4cc`；CPU 38/38；原计数/测量归档，不改 GPU 源码，不新增计时 |

verified：没有修改 PDL/bulk/TMA/cluster 的设备同步源码，因此没有伪造“修正后第五组 50/50”。
实际首次执行同步检验为 Llama B16 非分页 PDL 50/50 与 L1 loop 50/50；合计 100 个不同新 PID。
其覆盖不延伸到分页、cluster 或所有模型/几何。页 trace 插桩本身也不建立新的同步正确性结论。
verified（CPU）：最终 CFG 检查器回放既有 sm120 PTX，trigger 0/1 合计 8/8；旧误判保留，GPU 二进制未改。

## S1 环境与能力

verified：RTX 5090，170 SM，32607 MiB；驱动 580.82.07；TM nvcc 12.8.93。
共享内存 SM 102400 B、block 默认 49152 B/opt-in 101376 B；L2 100663296 B；cluster 支持，portable 上限 8。
Python 路径与硬件完整读数见 [env_sm120.json](env_sm120.json)、[caps_sm120.json](caps_sm120.json)；torch 2.13.0+cu130，transformers 5.17.0。
Caps 的 cp.async/mbarrier/tx/bulk/TMA/bulk-prefetch/PDL/cluster 均启用；tcgen05、BF16 collective builder 均为 0。
verified：本机 CUTLASS/CuTe 头文件宏为 4.8.0，version.h SHA 见 S1_vendor.tsv；vendor 目录非独立 Git checkout，不拿主仓库 HEAD 冒充其版本。

- inferred（源码/manifest）：固定 tiled 权重的 GEMM loader 为 LoadTile→PublishBulk；不是 GEMM Tensor2D。
- verified（编译）：分页 attention KV 有 Tensor2D/tensor map；非分页 pg=l2 有 bulk-prefetch。
- verified（编译）：CuTe/CUTLASS 共享 SM80-class BF16 mma/cp.async 实现可生成 sm120 HMMA/LDGSTS；不等于使用 SM120 collective builder。
- 固定 serving 物理 cluster=1；κ 不是 cluster 大小。B0 的 L1+L2-prefetch 平坦 grid 限制保留。
- 指令存在是编译证据，不是动态分支计数；不支持“所有 loader 都执行 TMA”或“serving 已使用 cluster”的结论。

## S2 标定、S3 微基准

| 项目 | verified 实测 |
|---|---|
| E1 TL-2 DRAM / 理论值 | 1691.251596 GB/s / 1792.128 GB/s = 94.371% |
| 五个新进程间极差 | 1.033770 GB/s，0.061125%；均无污染标记 |
| MB-1a 表观最大值 / E1 | 1866.030856 GB/s / 1691.251596 = 1.103343 |
| MB-1b bulk、1 loader warp、只加载最好点 | 1700.827307 GB/s（16 KiB×5 页），/MB-1a = 0.911468 |
| MB-1c | 204 执行点，12 隔离点；最高 1698.09882 GB/s |
| MB-1d cluster | 12 请求点中仅 2 执行，10 unsupported；170 CTA / background 时 85 CTA 不整除大多数 cluster 大小 |
| MB-1e | 普通三臂、PDL 四臂执行；冷 stream 扣减多数为负，不能当作物理步边界延迟 |

MB-1a 是重复复制的表观速率，不是超过物理 DRAM 带宽的证据；保留 TL-2 sustained ceiling。
cluster 实测两点均为 cluster=2、无 background；单进程结果不足以给同步/竞态结论。
原 MB-1c 非法指令后曾无 owner 持续满载；由用户恢复，本轮没有 agent reset。根因未明确，不猜改 PTX。
S2/S3 同时保留 sm89 参考列与各自 MB-1a 归一化值。

## S4 锚定（固定 B0，不是 R13F）

verified：每格同组配对三轮；TPOT=(E2E−TTFT)/1023。加速比列为逐轮 vLLM/TM 的中位数。

| 模型 / batch | B0 TPOT ms | vLLM TPOT ms | TPOT 加速比 | E2E 加速比 | B0 / vLLM TTFT ms |
|---|---:|---:|---:|---:|---:|
| Llama B1 | 1.7762 | 2.0322 | 1.1441 | 1.1462 | 2.81 / 6.96 |
| Llama B16 | 1.9626 | 2.8287 | 1.4414 | 1.3966 | 78.71 / 20.04 |
| Qwen3 B1 | 2.6861 | 3.0339 | 1.1295 | 1.1318 | 4.41 / 11.57 |
| Qwen3 B16 | 3.6265 | 4.3677 | 1.2059 | 1.1704 | 130.10 / 28.68 |

B0 相对 native 中间 past 下界为 1.2063/1.1967/1.2954/1.3636（表中顺序）。
B16 的 TM TTFT 明显更差，不只报告 decode 优势；prefill 沿用固定几何，本轮不调优。
Llama B1 B0h 的 C-1 失败，见 S9；这些性能数不能当作所有几何均通过参考值验证。

## S5 机制跨架构（TPOT 变化；负值更快）

verified：每项只取同组、同格、相同三个轮次；原始样本、极差、sm89 列见 S5.tsv。

| 对比 | Llama B1 | Llama B16 | Qwen3 B1 | Qwen3 B16 |
|---|---:|---:|---:|---:|
| NL2e / B0 | +12.284% | +11.692% | +9.462% | +6.352% |
| NL2g / B0 | +11.078% | +7.341% | +9.311% | +7.875% |
| PR_L1 / B0h | +1.982% | n/a | +2.562% | n/a |
| PS_L1 / PR_L1 | +7.143% | n/a | +12.136% | n/a |
| PSA_L2 / PS_L2 | +1.188% | n/a | +0.731% | n/a |
| B0l / B0-noev | −0.631% | −0.296% | +1.986% | +0.289% |

NL2e/g 均与 sm89 同号（更慢）；PSA/PS 的 B1 两格由 sm89 改善变为本机退化；Qwen loop 也变号。
这是实测方向对照，不是已隔离全部混杂因素的架构因果证明。
所有分页 loop 效应 n/a：B16 分页构建缺失导致必要 50 进程组不可用，不越过门禁。

## S6 阶段账本、S7 边界/滞后诊断

verified：可用 B0/PR 的 TR-1 past=64/575/1000 各 16 launch，NL2e/g TRACE_V2，以及 16 步 TR-3 均采集。
verified：B1 PR 漏编的 PAGE_TRACE 已补齐；逐文件核验 16 次阶段/步骤记录、两个 L2 各 16 个 page step×170 worker，旧 trace 保留。
B16 PR/PS/PSA 页 trace仍无构建；导出器在多 past 间覆盖同一页文件，TR-1 页记录仅保留 past1000，past64/575 的阶段账本完整但页明细缺失。
verified（离线修正）：TR-1 页计数累积 16 launch、begin/end 被最后一次覆盖；S7_pages 将等待除以 16，并保留 raw 值，重复臂的 loader 占比不报单次比例。
S7 页下界使用实际 past 的 B0 request-floor（575–1086 内线性插值为 inferred），不再把 past1000 当作575；不将它称为硬件 DRAM 字节计数。
插桩诊断：PR_L2 的 dependency/page-full/二者重叠 worker 均值的 step 中位数，Llama B1 为 706/1833/610 µs，Qwen B1 为 980/3035/808 µs；重叠不能重复相加。
B0 past575 的主要 native 下界超出量（按类别汇总、launch 中位数，µs）：
Llama B1 o=162.46、attention=109.96、merge=93.23；Qwen B1 attention=266.19、o=240.38。
Llama B16 o=236.48、down=105.66；Qwen B16 attention=582.75、o=337.57。
原 TR-3 B0-noev 边界中位数约 4.80–5.38 µs，提前 PDL trace 约 7.52–7.94 µs。
这些插桩数不是不插桩 E4 的性能；PR past575 L1 mean-step 为 2.764/4.368 ms，L2 为 3.200/4.540 ms，口径与 E4 不同。
没有匹配的同 past 无插桩控制，≤2% overhead 尚未验证，不能把差值全归因于插桩；账本仅用于定位。
L1 未填充的 token/KV lag 字段为零不能解释为实际依赖等待为零；MB-1e 的负残差同样不作为负延迟。
S6_cross_arch/S7_cross_arch 按同阶段种类/执行方式并列最终 sm89 T4/T6；前者各用自身 TL-2 ceiling，后者独立取中位数，不混算跨机时间戳。
阶段负 excess 是与 DRAM 下界近似/预取重叠的残差，不是负执行时间；两机插桩门槛限制保留。大型原始表在证据归档中保留。

## S8 选择、S9 正确性、S10 预测

- verified：原固定/trace 47/55 构建成功，r7 新增两份 trace 重编 2/2；八个 B16 分页构建仍被 `StageFlowModel.cpp:205` 的 negative counterfactual traffic 检查拒绝。
- verified：R13F Llama/Qwen3 在 pages-B1 同样拒绝；build 内耗时 6125.12/6855.86 s，超过 2×1800 s 后仍未主动中止。
- inferred：MainStart 减去平均预取量后的负数说明流量账本存在不平衡；已定位拒绝点，尚未验证其更深根因。不钳位、不修改 solver/cost model。
- S8 保留 10 个冻结的局部第一层候选，不冒充完成 SL-5 最终 pg×executor×loop 联合选择。
- verified：43 个成功 decode plan 的 64 步冒烟 43/43；C-2 162/162 条记录通过，其中非自身对照 120/120，未发现 token mismatch。
- verified：C-1 vLLM 四格 4/4；B0h 可用三格 2/3。Llama B1 gap≤0.5 比例 0.983398<0.99，max gap 10.5625>3，参考/容差未改。
- R13F C-1 四格 n/a；Qwen B16 B0h 原规格无此臂；bulk/seed+FX21 两个 50 进程组 unavailable，不记 PASS。
- CTest 合并原全套与定向恢复为 92/97，不是一次新全套通过；五项历史 fixture/API/schema/attention 数值问题未解决。
- verified（CPU）：本次 sm120 38/38、R13 40/40；native fingerprint 与九节标定印章再次核对；这些不是 GPU 同步通过证据。
- S10 共 40 个预测分项：15 在注册范围内，17 不在范围内，8 缺失/残差无效；这是主 matches 字段，saving_us 附加条件另列。
- DRAM 理论比例命中；MB-1a/E1、bulk≥0.97 未命中；Llama B1 有效带宽相对峰值跨机低 4.011 个百分点命中。
- R13F 的 Llama B1 1.06–1.13 预测未测到，不能用固定 B0 替代它。

## 四个问题

1. **verified（方向）**：L2e/g 退化结论同号；B1 PSA/PS、Qwen L1 loop 变号。**inferred**：默认策略应按架构/模型区分，但本轮不改默认。
2. **verified（可用 B1）**：bulk 分页未胜过同几何非分页 L1，PR_L1 比 B0h 慢 1.982%/2.562%；B16 不可用，Llama B1 另有 C-1 限定。
3. **verified（固定 B0）**：B1 吞吐优势约 14.4%/12.9%，比 sm89 固定 B0 对照更大；**stated（预测未验证）**：R13F 的规划区间没有最终选择结果；版本/权重 SHA 限定使架构归因仅为 inferred。
4. **verified（E4c 条件化）**：提前 PDL 的 L1 每步回收量，按 Llama B1/B16、Qwen B1/B16 为 +4.78/+35.36/+6.41/−10.08 µs；
   L2 提前触发为 −12.21/−6.59/−9.46/+43.14 µs。晚触发 L1 四格均更慢（+0.398–0.675%）。
   分页 PDL B1 为 −0.81/+3.42 µs，缺分页 50 进程覆盖；不据此做同步正确性结论。sm89 PDL 均 n/a。

## R14 方案（不实施）

- 对负流量拒绝增加按阶段/worker 的生产、预取、消费守恒诊断，区分账本重复扣减与真实路径问题；保留非负不变量，先证明再修。
- 分离架构/模型的 PDL 触发与 loop 默认；Llama B16 L1 early、Qwen B16 L2 early 是候选，其余先保留 off；先补相应几何同步/正确性覆盖，不在本轮改默认。
- 分页/B16 可构建后补齐 bulk/FX21 50 进程、页账本与完整 SL-5，检查 loader 算术与 LA 临界链；本轮不优化。
- attention/merge、低 SM 覆盖 o/down、B16 prefill 及 SM120 专用 CuTe/CUTLASS 路径列为后续定位点；先核实指令合法性和 trace overhead，不直接拿插桩账本定价。
- 页导出改为每 past 独立文件、每 launch 独立/清零计数；这是后续方案，本轮只修离线账本并记录未保留的页明细。
- 固定两机 vLLM 版本与模型 config/权重 SHA 后，才做更强的跨架构因果判断；cluster 用合法整除 grid 并补 50 新进程。

## 证据与封存状态

[results/README.md](results/README.md) 索引 S1–S10；[baseline_alignment.json](baseline_alignment.json) 记录共同基线。
本次原始 E4/E5/E6 与旧阶段报告：`raw/acceptance_04/original_e4_e5_e6.tar.xz`，2761 文件，84614296 B。
SHA256：`19058e454dece28b55f987fc2f1120fdaa1a0b20d67e3ebac052199cb7433f13`；完整文件 SHA 索引见 `original_files.json`。
原 r5 74 节点为 70 done/4 failed；一个锚定入口失败由 r6 补齐，三个 Qwen B16 E4d 节点没有可用臂，失败不清空。
验收域为 195/195 可用记录已收集，规格总域 297 中 102 明确 unavailable；不是 297/297 通过。
verified：r7 六节点均 done/exit0；195 条测量以及 S4/S5/S9/S10 的 SHA 与补测前完全一致，没有 pending/running。
补采原始与修正前页统计：`raw/acceptance_05/unreviewed_r7_evidence.tar.xz`，SHA256 `328d32d4949c5ee302be4832ebd647934a09f6e7632f0ed293ff5923f863061f`。
最终核验与离线修正证据见 `raw/acceptance_05/{review.json,completion.json,cpu_checks.json,final_analysis.tar.json}`；最终表归档的逐文件 SHA 一并保存。
29 份 SO/sidecar/生成源码已持久化至 `/root/shared-nvme/junhuipeng/TileMega_R13_SM120/artifacts/r13_sm120_frozen_binaries.tar.xz`，不依赖 tmpfs 寿命。
该 bundle SHA256 `ef0c8df08eebc3bd31d8b287d800aac026c7fbd9219dd1153be9b9d1fee6dbbf`；本机模型 config/权重仍在共享存储，跨机 SHA 限定不变。
verified：确认所有已注册节点终态、无子进程后，原空闲调度器 PID4488 已用 SIGTERM 正常退出；无 GPU reset、不清空历史失败/状态。
本轮按允许范围带限制封存；不可用/未通过项与 R14 方案留待后续授权，不以追加实验或 solver workaround 填成 PASS。
