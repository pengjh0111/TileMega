# MX5 本轮 LLM 数据采集收口

本页保留前轮收口数据（当时OLMoE native不支持、TC为部分编译）。最新OLMoE native与新完整decode编译数据见 [本轮报告](MX5_olmoe_native_c550.md)；旧表不是当前能力状态。

证据标记：下列数值均为 verified；推断单独说明。
本轮 Mistral native 最终构建 HEAD：`7c9e3522ef6e1d10a35828ea213a5a780dff3b0b`。
外部基线沿用各CSV记录的冻结实现；不同轮次不冒充同一构建。
Prompt SHA256：`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`。

## 范围与协议

Mistral-7B-v0.3、OLMoE-1B-7B-0924 保留完整官方架构，使用已记录种子20260925的随机BF16权重。
各模型内部，TileMega与基线共享同一 checkpoint、64-token 输入；greedy，OSL1024。
每轮独立进程，warmup1＋timed2；两轮取中位数。TPOT=(E2E−TTFT)/1023。
Mistral 每个B枚举原始预测top4，四个候选均检查C1/C2(512步)，各测两轮，再按1%规则选择。
Mistral decode 为 native静态程序；dense prefill 沿用MX4同形状几何，重新构建watchdog0/1及fragment KV写入。
没有替换外部基线或旧测量行，没有启用全局默认值。

## 新增模型 TPOT（ms）

|模型|B|TileMega|vLLM|SGLang|torch.compile|最快基线/TileMega|
|---|---:|---:|---:|---:|---:|---:|
|Mistral-7B-v0.3|1|10.080156|11.148076|10.606020|18.437016|1.0522×|
|Mistral-7B-v0.3|2|10.473901|12.525795|12.029260|25.129721|1.1485×|
|Mistral-7B-v0.3|4|10.926902|12.676094|12.184568|35.886293|1.1151×|
|Mistral-7B-v0.3|8|11.376112|12.986500|12.631397|57.645759|1.1103×|
|Mistral-7B-v0.3|16|12.664796|13.668050|13.536814|100.390065|1.0689×|
|OLMoE-1B-7B|1|不支持，未计时|3.653538|3.156181|45.753506|—|
|OLMoE-1B-7B|2|不支持，未计时|4.762878|4.341845|78.633358|—|
|OLMoE-1B-7B|4|不支持，未计时|6.215642|5.817549|128.368666|—|
|OLMoE-1B-7B|8|不支持，未计时|8.259111|7.907548|197.038825|—|
|OLMoE-1B-7B|16|不支持，未计时|10.395362|10.015969|262.257277|—|

外部基线两模型共60个标准新进程全部成功；Mistral native top4 共40个标准计时新进程完成。
OLMoE的torch.compile与下表Qwen MoE均明确使用experts=eager的部分编译边界，不能标为完整grouped_mm编译。
早期Mistral B1手工开发结果10.130998ms保留；上表使用完成top4选择后的结果。

## Mistral 选中计划的端到端指标

|B|候选ID|TTFT ms|E2E ms|output tok/s|
|---:|---:|---:|---:|---:|
|1|0|42.746508|10354.745971|98.891853|
|2|0|71.343731|10786.144349|189.873229|
|4|0|120.185236|11298.406003|362.529085|
|8|0|326.060317|11963.822746|684.731042|
|16|0|694.571180|13650.657196|1200.235488|

Mistral各格均为512线程，每步161个stage；实际各类(TN, WN, D, Part)、b_min与P_w见对应pin及serving CSV。
四个候选的预测/实测与配置在 `MX5_selection_c550.csv`，选中计划在 `configs/pins/mx5/mistral_7b_v03_B<b>_selected.json`。
源形状表覆盖每格203/203合法键；新N32768 head的100个键均为3个新进程实测。相同非head形状仅复用已核实源字节相同的原始测量。

## Qwen3-30B-A3B-L24 torch.compile

|B|TPOT ms|TTFT ms|E2E ms|output tok/s|单作业上限 s|
|---:|---:|---:|---:|---:|---:|
|1|71.823444|230.192118|73705.575797|13.895101|900|
|2|122.764771|341.366953|125929.727268|16.263905|900|
|4|215.622438|543.906912|221125.661254|18.523448|900|
|8|364.138602|840.315042|373354.105390|21.942870|1800|
|16|554.719642|936.596331|568414.789823|28.824300|2700|

用户在本轮明确允许仅B8/B16分别延长至1800/2700秒，其他测量协议保持不变。
原900秒下B8/B16各两轮超时仍保留在CSV；新数据使用extended轮次标记，不回填旧失败行。
成功完成完整生成进一步支持耗时超限的诊断；原日志没有OOM异常，不能将旧超时写成OOM。
Qwen修复CSV沿用的checkpoint_identity_sha256列实际为config.json的SHA256，并非完整权重分片哈希；不据此宣称权重逐字节校验。
实现为decode周边编译＋原HF eager专家，fullgraph=False，24层的Dynamo重编译预算32；专家算术不改。

## 28/472个非平局路由差异的含义

|B|核对位置数|完全相同集合|允许的边界平局|非平局差异|比例|
|---:|---:|---:|---:|---:|---:|
|1|1536|1376|132|28|1.8229%|
|16|24576|21788|2316|472|1.9206%|

一次核对位置是(前64个decode步之一，24层之一，一个batch token)，比较top8专家集合，不比较集合内排序。
若集合不同，且HF第8/9名logit差距超过2个BF16 ULP，不能按平局豁免，计为非平局失败。
这不是最终输出token错误率；C1检查输出logit容差，C2检查L1两种执行模式的token逐位一致，三项门不同。
真实例子：step26/token0，layer0在1ULP边界选出不同专家76/105，属于允许平局；layer1边界差10ULP，故计失败。
24/28、415/472个失败之前，在相同步/token的较低层已有边界平局差异。上游专家分支可传播激活差异是推断，未证明全部因果。
HF eager参考与2ULP规则未改变，MoE路由验收仍失败，不能据C1/C2通过宣称MoE全面验收。

## 缺项、差异与构建记录

前轮OLMoE TileMega在 `python/tilemega/serving/export.py:22` 的架构检查处拒绝导出，CPU实际尝试记录在 `olmoe_native_capability_check/RESULT.json`，未启动GPU模型。
`lib/Codegen/MxMoeModel.cpp:89,98` 仅接受本轮Qwen的专家与注意力几何。
其64专家、全投影Q/K RMS及不重归一top8概率不符合当前Qwen native路径；不是OOM，也没有使用错误架构或legacy替代。
五个B的CSV状态均为capability_status_not_a_measurement，所有latency字段留空。
Mistral旧计划的源哈希受前一轮Python修复影响，故按原几何重建；B1/B2还重建了同一原始512候选，以匹配normal/watchdog的HEAD元数据。
cap227的仅查询诊断观测到occupancy0及小栈帧；首次启动器误要求诊断也resident1后停止。恢复保留真实零值，诊断二进制从未执行。
真正计时计划仍全部满足512线程MT<=255、栈帧0、静态smem0、MIR检查和实际resident1查询，没有放宽执行门。
solver TSV只做已关闭文件的无损gzip归档，完整解压SHA核对，保留RAM可读原路径；失败日志与二进制不删除。
历史MX4三份固定CSV与22a352242字节一致。全局原始MX5 G-LLM/MoE验收未宣称关闭。

## 数据与复核入口

- `MX5_additional_llm_c550.csv`：新增模型基线、手工/solver native及不支持状态，逐轮标注实现。
- `MX5_torchcompile_repair_c550.csv`：原900秒失败与新延长进程完整数据。
- `MX5_serving_c550.csv`、`MX5_selection_c550.csv`、`MX5_resources_c550.csv`：native执行、选择和资源。
- raw：`runs/mx5/extra_models_20261030_v1/mistral_complete_top4/`、`torchcompile_repair_20261030_v1/extended_B8_B16/`。
- 本轮数据经现有strict results consumer、实际进程/guard、token文件和指标公式复核，旧CSV前缀保持不变。

GPU用量为guard作业墙钟区间的单卡求和上界，并非硬件busy计数：本次修复/扩展campaign 13.291 GPU小时；累计 126.394 GPU小时。
累计60 GPU小时上限已由用户放宽；仅用物理7/6/5/4，各卡单作业，已批准最多4卡并行。

## 与原prompt推断不一致或未满足之处

新增Mistral求解使用runs内每格203键的scoped实测表，尚未纳入configs/targets全局表；不将本轮选择称为全局成本表验收。
编译时长保存在原始构建receipts中，当前资源CSV没有独立时长列，仍与原prompt的入库格式要求有差异。
Qwen MoE torch.compile完整协议在原900秒限制内无法完成B8/B16，需本轮显式批准的时限例外。
MoE内部路由仍有28/472个非平局差异，尽管C1/C2通过；用户期望的约1.1×MoE收益尚未满足。
前轮OLMoE只有外部基线可运行；本轮已补native开发数据，路由门及最终选择仍未通过，详见新报告。
