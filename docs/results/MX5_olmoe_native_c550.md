# MX5 OLMoE native 与完整 decode 编译基线

证据：数值均为 verified；推断单独说明。日期2026-10-30。
Prompt SHA256：`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`。
本轮起点 `d391f5589619a46b2a4c30e9c8a36b48dac3f541`，native计时构建为干净
`e9fc6f256d4d57aed93a1e3cdefe8ed00dd6ffb1`；最终提交/推送见分支历史。
MACA Version.txt=3.8.2.6；cucc=mxcc1.0.0(890c70f3aa)，安装LLVM目录maca-3.8.2。

## 当前结论

OLMoE已支持C550静态prompt/decode并完成五格运行、512步C1/C2及双轮计时。
这些是手工开发对照，尚未完成成本表/top4选择，且B1/B16路由门失败；不称为最终验收，不启用默认。
Mistral与OLMoE为用户后续授权的新增覆盖，原prompt没有这两模型的独立性能预测。
Mistral的五格TileMega及三套基线此前已齐，本轮保持原数据。
Qwen与OLMoE各五格的新完整decode编译基线已齐；旧部分编译与失败行全部保留。

## 实现与sm90参照

开工fetch后，origin/sm90r4-merge=`29de9aa8a`已含OLMoE；参照
`15b86caf6`整行norm、`01c329afc`后端norm、`163113e78`原始routing概率、
`1e4f1e4fd`的16head/Q1处理，按C550静态体结构适配。
OLMoE官方完整16层，H2048/I1024/E64/top8/hkv16/D128/Q1，随机BF16权重种子20260925。
Q/K先在整个2048投影行上RMS再reshape；routing对全部64专家softmax，选8后不重归一。
保持本机HF5.12的BF16路由权重舍入点，没有照搬sm90旧FP32边界。
前端导出/匹配、native模型分析、外部权重recipe、全行norm、topk分派与scratch均实现此语义。
不支持的attention bias/clip_qkv及错误几何显式拒绝；legacy不能承接这些新语义。
prompt与decode共享166个外部缓冲；97stage/步，prompt共6145个屏障。

## OLMoE TPOT（ms）

协议：ISL64/OSL1024、greedy、BF16；每轮新进程warmup1＋timed2，两轮中位数。
TPOT=(E2E−TTFT)/1023。所有系统共享同一checkpoint与64-token输入。
TC旧=experts=eager部分编译；TC新=HF完整decode编译，实际decode专家为batched_mm。

|B|TileMega开发|vLLM|SGLang|TC旧|TC新|最快基线/TileMega|
|---:|---:|---:|---:|---:|---:|---:|
|1|3.502505|3.653538|3.156181|45.753506|8.798108|0.9011×|
|2|4.517575|4.762878|4.341845|78.633358|13.839850|0.9611×|
|4|6.492231|6.215642|5.817549|128.368666|22.909596|0.8961×|
|8|9.200200|8.259111|7.907548|197.038825|41.199638|0.8595×|
|16|12.237924|10.395362|10.015969|262.257277|77.068176|0.8184×|

所有五格均慢于SGLang；不据此启用默认或宣称达成1.1×目标。
各项指标分别取两轮中位数，output tok/s=B×1024/E2E，不从中位TPOT反推。

|B|TileMega TTFT ms|E2E ms|output tok/s|TPOT轮间极差ms|C1 max gap|
|---:|---:|---:|---:|---:|---:|
|1|231.040728|3814.102962|268.477354|0.003592|0.015625|
|2|288.227840|4909.707276|417.133020|0.006819|0.03125|
|4|409.117458|7050.670029|580.937697|0.000572|0.046875|
|8|556.189613|9967.993771|821.830381|0.002716|0.03125|
|16|716.416634|13235.813321|1237.853679|0.003200|0.046875|

每格C1 gap≤0.5比例=1.0；C2两种L1执行模式逐位一致，全部512步。

## 路由门与停止位置

|B|位置数|top8集合相同|允许边界平局|非平局失败|通过|
|---:|---:|---:|---:|---:|---|
|1|1024|941|67|16|否|
|16|16384|15267|933|184|否|

位置=(64个decode步，16层，batch token)。HF参考用用户批准的experts=eager。
严格保留HF第8/9名差≤2BF16ULP的平局规则；其余集合差异都是失败。
C1输出logit容差、C2模式一致和路由集合是独立的门。上述数字不是最终token错误率。
`python/tilemega/moe/native_routing_check.py:138`按实际失败退出1；两份诊断二进制/guard保留。
按prompt§0.2.1停止该项验收；没有换路由、修改参考或放宽判据。
具体数值来源仍待定位，不能把所有失败归因为RMS求和次序或上游平局传播。
此前Qwen的28/472个非平局失败同样未关闭。

## 资源与prompt差异

DT-W仍记录576；DT-R=fence_atomicAdd，DT-P=P-B，DT-X=X-A，DT-KV=block_fragments。
沿用既有决策证据，本轮没有重测或改写五表结论；P-B局部LDS-only同步使用已获用户批准。
新OLMoE计划实际为512线程。576 cap168/frame384B，cap220/frame128B；
cap227实际MT222/frame0，但设备查询resident_ctas=0；cap248诊断同样MT222/frame0。
同几何512为MT222/frame0且实际resident1；各B的normal/watchdog/prompt均检查了真实资源。
576十个query-only产物未执行。此处不能声称体本身超过227寄存器；这是实测occupancy与简化资源公式的冲突，
具体硬件分配原因未证实。512作为显式开发适配记录，不冒充prompt§4.5规定的超227分支已满足。
512执行产物全部MT≤255/frame0/static-smem0，稳态MIR无4160、generic访存或非内联调用。
仅新增既有体内的全行norm路径；不声称本轮完成新的5进程×2000次同步litmus。

手工几何全B相同：qkv(32,1,1,TileRange)、o(32,1,1,StreamK)、router(16,1,1,StreamK)、
gate_up(64,2,1,TileRange)、down(64,2,1,TileRange)、head(32,8,1,TileRange)；
括号顺序TN/WN/D/Part。b_min4、attn_prefetch0；P_w为1/1/1/0/1/1，R_next=-1。
这是开发对照，未按§4.5资源成本公式和§5/6.9完成OLMoE最终选择。
`tools/commands/mx_moe_v2.cpp:64`的solver入口仍需该模型实际geometry-outcome表，OLMoE全域实测及top4尚缺。
程序级资源/总字节已入CSV；逐体指令与字节仍是null，编译耗时仍在runs receipts而未独立入资源CSV。
原始dense25/MoE5完整验收、OLMoE成本表/top4、Qwen约1.1×目标及DNN25格仍未关闭。

## 编译基线修复与完整性

缺失的MetaX grouped_mm内核由显式opt-in的独立Triton provider补齐，只替换HF矩阵乘法入口，
不调用TileMega运行时。HF5.12在prefill使用grouped_mm，随后自动切换batched_mm做decode，结束后恢复。
`generation/utils.py:2101`的切换、`:2784`的prefill和`:2797`的decode上下文经安装源文件和实际探针核实。
所以新增数据应称完整decode编译＋自定义grouped_mm prefill，不能称原生MetaX grouped_mm decode。
冻结baseline六个源文件SHA均与e9fc6f256对应文件逐字节一致；冻结记录的起始HEAD=d391f5589，不能单以该HEAD代替实际源哈希。
模式reduce-overhead/fullgraph=True，compiled_call_used=True，每个标准进程unique_graphs1且无graph_break。
预填充没有纳入完整forward编译；fullgraph只描述decode forward，与原baseline compile_path一致。
新编译两模型共20个成功标准进程，全部在900s内；四个B1初次attempt因冻结目录缺guard-policy文件失败，保留失败空值行，修复布局后新进程重测。
OLMoE native另10个标准进程，共30个成功计时进程均复核guard/PID/六次生成/token形状/公式。
native两轮全部生成序列逐位一致；所有MX4三份固定CSV与22a352242字节一致。
新行append-only，未回填旧状态。Qwen repair CSV旧checkpoint_identity_sha256实际为config哈希，沿用并明确区分权重校验。

## 验证与数据入口

CPU相关回归57/57；后续phase/structure元数据测试8/8。
C++模型测试positive65/negative480/emitted120/config-negative240/buffer-negative120，含全B实际OLMoE bridge。
GPU全行attention28组全部通过：B1/B16、L1/31/32/33/65/257/1088、b_min1/8，含KV历史/确定性。
GPU分派6组通过BF16完整softmax权重及确定性；grouped-mm五种ragged/empty/skew/真实形状检查和fullgraph通过。
旧auditor fixture八例仍因过期header哈希closure失败，另52例通过；不改期望掩盖失败。
新基线Qwen及OLMoE各自短8token teacher forcing max_gap0，通过；短诊断不作为标准性能。

- [逐轮新增模型数据](MX5_additional_llm_c550.csv)、[Qwen编译修复数据](MX5_torchcompile_repair_c550.csv)。
- [native计时](MX5_serving_c550.csv)、[资源](MX5_resources_c550.csv)，manual pins在configs/pins/mx5/。
- raw：runs/mx5/olmoe_native_20261030_v1/；FINAL_VALIDATED_COLLECTION.json保留输入哈希与复核统计。
- native_current/B*/verified_execution_v2/为成功C1/C2/双轮；原控制器local_bytes字段读取错误单独保留，实际CUDA字段localSizeBytes复核后才执行。
- 本轮新数据追加additional32/torchcompile17/serving15/resources30行；没有添加OLMoE solver-selected记录。

## Mistral五格（已完成，原数据不重测）

|B|TileMega|vLLM|SGLang|torch.compile|
|---:|---:|---:|---:|---:|
|1|10.080156|11.148076|10.606020|18.437016|
|2|10.473901|12.525795|12.029260|25.129721|
|4|10.926902|12.676094|12.184568|35.886293|
|8|11.376112|12.986500|12.631397|57.645759|
|16|12.664796|13.668050|13.536814|100.390065|

Mistral已完成原始top4实测选择与512步C1/C2；历史说明见[前轮报告](MX5_collection_c550.md)。

本轮guard作业共58项，单卡墙钟求和上界2.358611 GPU小时，累计128.752778小时；不是硬件busy计数。
累计60GPU小时预算已获用户放宽，最多4卡并行，仅物理7/6/5/4、每卡单作业；本轮计时均≤900秒。

## 与prompt推断或验收预期不同的实测

576线程即使MT222≤227且frame0，仍被实际occupancy查询判resident0；不能仅按寄存器总量估算可驻留。
OLMoE开发计划五格均慢于最快基线；C1/C2通过也没有保证内部top8路由通过严格非平局门。
新Qwen完整decode编译可在900s内完成全部B，旧B8/B16超时属于旧eager专家/部分编译实现，不能推广为编译系统固有上限或OOM。
