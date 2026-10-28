# MX-5 要求重审与本次授权边界（2026-10-27）

原规格：`/root/Prompt/TileMega_MX5_prompt.md`。
SHA256：`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`。
本次用户补充授权覆盖局部 LDS 同步；其余原文要求保留。未修改原 prompt。

## 目标、顺序与管线

- 顺序：阶段0探针 → dense → MoE native → LLM 30格验收 → DNN native → 总验收。
- dense：五模型×B={1,2,4,8,16}=25格；MoE-L24同5个B；DNN五模型×同5个B=25格。
- LLM优化指标：BF16 greedy、ISL64/OSL1024的TPOT；TTFT照报。
- DNN指标：单次forward端到端时延；structure_only随机权重、synthetic检查。
- 正常dense路线：checkpoint/bridge → ModelPlan分析 → 同源阶段成本 → exact top4
  → constexpr StageProgram → cucc/身份/资源/MIR → 权重打包与共享KV
  → ServingEngine L1/separate或loop → 正确性及端到端选择。
- MoE和DNN native尚未实现，不能把现存legacy路线当本轮交付。

## 固定实现要求

- 保留静态stage类型、按值绑定、动态smem、104CTA及逐计划线程/占用率验证。
- dense：kStore qkv、attention侧q/k-norm/RoPE/KV写、GEMM/head v2、LA v2、embed并入layer0。
- dense每步5L+1屏障；MoE每步6L+1。无运行时stage描述符或解释式回退。
- dense prefill沿MX4几何，仅计时watchdog0；检查用watchdog1。
- MoE需router/top-k/dispatch、grouped GEMM、combine、prompt64及路由核对。
- DNN需原计划对应native bodies/epilogue/布局、阶段表、top4求解和time命令。
- 原实测DT保持W576、role fence+atomic、P-B、X-A、block-fragment KV。
- 576/512按原逐计划寄存器重建规则及实际occupancy判定；不全局强制512。
- 不能由stagebench576拒绝推断实际wholeprogram576拒绝；现有167MT/frame0反例保留。

## 本次唯一新增的协议授权

目的（inferred）：减少不必要的全局加载等待，恢复局部合并与下一段预取的重叠，争取降低TPOT。
两个候选的端到端P-PAIR均未显示可分辨收益（+0.146%、−0.300%）。

用户原话：若目的是更好TPOT，“我可以放宽你的限制，然后你只需要如实记录就可以了”。
按该条件已明确成立的范围，首先仅替换attention wave结果的首个纯LDS发布点：
LDS写 → 编译器屏障 → arrive_bsmcnt(0) → barrier_inst → 编译器屏障 → LDS读。
所有wave（含无数据wave和576同步wave）仍参加。

全局DT-P仍为P-B；LA入口、global release/acquire及Run尾标准同步保留。
第二候选另将段尾纯LDS工作区复用点改用同一原语；独立验证，未提升默认值。
这不是P-A的新判定，也不宣称完整跨段重叠已修复。
其他结构、公式、舍入、判据及GPU规则没有随此授权放宽。

## 正确性、测量和终版选择

- 新同步：5新进程、每进程至少2000迭代、104CTA、0违例；新实现需独立验证。
- C-1：512步teacher forcing；C-2：separate/loop逐位一致。
- attention原文必需形状/Norm为336组合；stock PK012得到的1008是额外压力超集。
  每个必需项仍需绑定真实规则/合法配置，不能跳过失败项后称完整通过。
- LLM每轮warm1+timed2，两新进程；TPOT=(E2E−TTFT)/1023。
- P-PAIR同卡交替，每臂2新进程；可分辨阈值max(1%,轮间极差)。
- DNN保留原50-forward设备事件协议及25格synthetic检查。
- 每阶段成本3新进程，冷工作集至少256MiB；attention五L点线性插值公式不变。
- 每终版格exact top4各2新进程E2E；按原1%规则选，报告pred/meas。
- 微基准、MIR、trace只诊断；只有无插桩端到端结果决定默认值。
- 既有576 GEMM单元素FP32参考失败保留，不改oracle/容差，也不称已修复。

## 旧成本192/400与新候选的关系

- 192/400是上一检查点严格消费的冻结2678/A19cc122实现数据，与本次授权无关。
- 400指该版本资源合法的512线程attention键，不是全部LLM/DNN测量数量。
- 缺208的直接原因是guard75；本轮在同一冻结源码下补齐400/400。
  与同源GEMM/head合成4545键成本池，25个512成本scope严格校验通过；不是整计划512批准。
- 新LDS实现拥有不同源码指纹，旧192行不能换身份作为新实现的成本。
- 原行、失败、实际guard/argv、轮次、实现、卡号保留；不回填。
- 历史7格dense选择属cc29；Row2678 B2手动对照仍慢于固定MX4，不能当新候选验收。

## 资源、基线与交付

- 卡序7/6/5/4，最多2张、每卡1作业；全部经过原mx_gpu_job/guard。
- 作业≤1800s，计时≤900s；MoE NEEDS_MIB40960；不改时钟/功耗/驱动或他人进程。
- 总预算60 GPU小时；54.901小时为此前检查点，当前值见主报告与持续记账；仅DT-F可因预算裁剪。
- 固定MX4三CSV逐字节保持；不重测外部基线。
- 同源码与实际执行坐标才可复用证据。新候选未验收，不提升默认值。
- 报告保留verified/inferred/stated、失败和未知；runs不入git。
- 主报告≤260行、LLM报告≤200行；最终列起止HEAD/提交/推送、DT、全矩阵、GPU和差异。

## 2026-10-28 继续推进前的核对（verified）

- 两处局部LDS替换的新源码通过5新进程×336必需attention组合、每进程15项
  MX4变换逐位比较，以及5新进程×6项×2000迭代×104CTA实际helper litmus。
- 新源码400/400 attention键各3新进程冷测量完成，与同源G/H合成4545键，
  25个成本scope全部严格重放；旧源码400键与新源码400键分别保留。
- 五个开发格20个canonical576原始计划及最多3次共享预算的预取缩减，
  共46次实际零launch资源查询完成。16项获得逐计划512历史，4项保留576路线；
  4项均属Llama1B，不能用其stagebench失败批准整计划512。
- Qwen0.6B B1/B16各四候选已完成C-1/C2及两新进程计时，选中TPOT分别
  1.976266/3.524260ms，较固定MX4分别−0.367%（不可分辨）/−22.323%。
  默认值不变。Llama8B B1另完成四候选验收，10.585899ms（−1.210%）；
  B16也完成四候选验收，13.150235ms（−9.566%），新源码共4/25格。
  其余20格80个原始CPU编译完成，实际资源查询已启动；非数值运行。
- Llama1B的576整计划167MT/frame0/实际resident1，而按规则PK2的同源运行时
  stagebench有44–48B栈帧。该成本项仍停止，未以PK0成本或强制512替代。
- 旧调度器的Llama提示文件名错误发生在GPU提交前；修正路径后独立重启，
  失败记录与已完成Qwen结果保留。无外部基线重测、无MoE/DNN验收声明。

- 继续验收的Qwen0.6B B2四候选C1/C2及两新进程计时完成，TPOT2.301333ms，
  较固定MX4慢9.823%，G-LLM在该格失败。D7当前源码逐阶段账本/I1仍待补；默认值保留。
