#!/usr/bin/env python3
"""Rebuild the R9b report; --final refuses incomplete evidence.

report.py and verify.py independently reread raw evidence. This writer only
formats their outputs and the explicitly documented interpretation limits.
It never converts a failed numerical gate into a passed one.
"""
import argparse,csv,datetime,hashlib,json,math,pathlib,re,statistics,subprocess,sys

E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2];TABLES=E/'report_tables'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true');a=p.parse_args()
subprocess.run([sys.executable,str(E/'report.py')],cwd=ROOT,check=True)
with (E/'verify_report.log').open('w') as output:
    verdict=subprocess.run([sys.executable,str(E/'verify.py')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
verify=(E/'verify_report.log').read_text()
pending=json.loads((TABLES/'incomplete.json').read_text())
for model in ('llama','qwen3'):
    for family in ('validation','validation_colocated'):
        path=E/family/model/'exit.json'
        if not path.exists() or json.loads(path.read_text()).get('exit')!=0:pending.append(f'{family}/{model} incomplete')
    logs=list((E/'colocation'/model).glob('process_*.log'))
    if len(logs)<50:pending.append(f'colocation/{model}: {len(logs)}/50')
for model in ('gqa2','mha4'):
    for seq in (4,128):
        path=E/'reference'/f'{model}_s{seq}'/'selected.cu.measurements.json'
        if not path.exists():pending.append(f'reference/{model}_s{seq} incomplete')
if a.final and pending:raise RuntimeError('refuse final report with missing evidence: '+repr(pending))

def rows(name):
    with (TABLES/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def fmt(x):
    try:
        value=float(x)
        return f'{value:.7g}'
    except (ValueError,TypeError):return str(x).replace('|','\\|').replace('\n',' ')
def table(name,cols,select=lambda r:True):
    data=[r for r in rows(name) if select(r)]
    if not data:return f'待完成；原始产物未齐。预期表：`report_tables/{name}`。\n'
    out=['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
    out += ['| '+' | '.join(fmt(r.get(k,'')) for k in cols)+' |' for r in data]
    return '\n'.join(out)+f'\n\n完整字段与证据路径：[原始数据重算表](report_tables/{name})。\n'
def raw_table(path,cols):
    with path.open() as f:data=list(csv.DictReader(f,delimiter='\t'))
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+
        ['| '+' | '.join(fmt(r[k]) for k in cols)+' |' for r in data])

base=json.loads((E/'baseline.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
prompt=pathlib.Path(base['prompt_path']);sha=hashlib.sha256(prompt.read_bytes()).hexdigest()
assert sha==base['prompt_sha256']
commits=subprocess.check_output(['git','log','--reverse','--format=%h %s',base['baseline']+'..HEAD'],cwd=ROOT,text=True)
parts=[f'# TileMega R9b：物理模型、评估与放置\n\n状态：**{"测量与核验已收集，未达门项见下文" if a.final else "进行中；不是收尾报告"}**。生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。\n']
if pending:parts.append('仍缺以下证据；不以部分矩阵判定研究门：\n\n'+'\n'.join('- '+x for x in pending)+'\n')
parts.append(f'## 1. 基线、规格与提交\n\n基线 `{base["baseline"]}`；报告所见 HEAD `{head}`。规格 `{prompt}`，SHA256 `{sha}`。TODO 更新与冻结来自 `{base["todo_update_commit"]}`、`{base["r9_stop_commit"]}`。\n\n```text\n{commits}```\n')
parts.append(f'## 2. verify.py 完整输出\n\n命令：`python3 docs/experiments/SOLVER_R9B/verify.py`；退出码 `{verdict.returncode}`。每个门均执行，缺证据也输出 FAIL。\n\n```text\n{verify}```\n')
gates=[]
for line in verify.splitlines():
    m=re.match(r'(G-\d+) (PASS|FAIL) (.*)',line)
    if m:gates.append(m.groups())
parts.append('## 3. 逐门结果\n\n| 门 | 结果 | 实测与证据 |\n| --- | --- | --- |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in gates)+'\n')
parts.append('''## 4. 停摆、降级与偏离

| 项目 | 影响与现状 | 解锁或下一步 | 估计工作量（推断） |
| --- | --- | --- | --- |
| SV-9(e), G-3 | 静态 FP64 为零的门失败；不停止独立求解与计时。既有 RoPE `sinf/cosf` 的 CUDA 慢路径含 FP64，MIDPOINT_REFINE=0 | 需另行授权修改 TaskBody/math 语义，证明 RoPE 参数范围并验证替代范围约减；本轮禁改 | 定位已完成；实现及数值覆盖约 1–3 天 |
| SV-11(d) | 物理固定段均值误差下降、中位误差与回放排序变差，按 §7.3 保留旧固定段；新拟合仍独立输出 | 按有效行数、tile 与 split 分组校准固定段，保持留出集独立 | 约 1–2 天 |
| G-5 | BF16 排序硬门仍失败；关闭选项、FP32 开启的 binary64 不变门通过。开启物理流量与 stages，关闭劣化的 fixed 分量，降级支持下游实测 | 先重建 F-116 与当前目标文件/TaskBody 回放之间的差异，再标定 `TaskPriceParts.cpp` 固定/循环权重；不得通过换门线宣称通过 | 约 1–3 天 |
| SV-12(a), P-9 | CG 的 Coarsen 释放端点与执行器保守窗口并不总相同：参考抽样 472/2064 不同。Level 1 保留规格的 CG 释放，最终物化和 FIFO 模拟验证真实事件窗 | 下一版 Level 1 同时计 `max(CG endpoint, runtime event endpoint)` 与真实 masks，或另轮收紧执行窗口；前者需新搜索与 V2 | 约 1–2 天及重测 |
| G-7 | seq64 预热求值超过 10 ms；完整域搜索已有格超过 30 min。保留完整域继续计时，不宣称预算通过 | `FlowPreparation.cpp` 的逐边排序表准备、`PrepareSymbolicProblem` 精确关系准备占主导；缓存实际改变的 space/边并减少整模型重建；`EvaluateFlow` 的 pending 初始化与事件数另计 | 约 2–4 天及全矩阵复测 |

临时停止的旧 General Oracle 审计已解锁：全参数 `lexmax` 的分片爆炸改为精确当前纤维的最大线性序号，集合测试通过；已完成样本前缀保留，RNG 按原序续跑。这不是用包围盒近似。

其余明确偏离：embedding 唯一行取决于 token 值；trace 的 fixed/mainloop 合并；旧 prompt 的链深与真实 CG 不符，使用所测 CG 的 D；theta 2/8/32 的第一起点取下一已有 legacy 点，第二起点仍为当前 theta 的完整 uniform 全扫。详见 [逐项规格、实际做法与理由](deviations.md)。
''')
perf=rows('performance.tsv');gm=[]
for arm in ('legacy','skeleton'):
    values=[float(r['L2_over_floor']) for r in perf if r['arm']==arm and int(r['seq'])<=16]
    gm.append(f'{arm}: n={len(values)}/6'+(f'，几何平均={math.exp(statistics.mean(map(math.log,values))):.8f}' if len(values)==6 else '，未齐，不计算整门结果'))
parts.append('## 5. 绝对位置\n\n所有时间 ms；b 为 ns/链节。'+ '；'.join(gm)+'。\n\n'+table('performance.tsv',['cell','arm','T_dram_ms','T_compute_ms','T_floor_ms','l05_ms','l1_ms','l2_ms','L2_over_floor','L2_over_legacy','D','bubble_ns']))
parts.append('''## 6. 缺口分解与 trace

以下四项来自同一物理模型的反事实求值，和为 **预测 T − T_floor**，并非实测 L2 − T_floor 的直接测量分解。`flow_over_measured` 显示仍未被模型解释的部分，不能把它藏入某个已测相位。负争用项原样保留。

'''+table('decomposition.tsv',['cell','sync_ns','fixed_ns','contention_ns','chain_ns','closure_ns','flow_over_measured','fluid_over_measured']))
parts.append('四格 trace 的固定段与主循环合并报告；等待/hop、发布、屏障、空闲保持独立。逐节非重叠墙钟路径可求和；各 space 跨度相互重叠，不可相加当作关键路径。\n')
for cell in ('llama_s1','llama_s64','qwen3_s1','qwen3_s64'):
    path=E/'trace_analysis'/cell/'chain_categories.tsv'
    parts.append(f'\n{cell}：[逐节](trace_analysis/{cell}/chain_links.tsv)、[逐 space](trace_analysis/{cell}/task_spaces.tsv)、[按类别汇总](trace_analysis/{cell}/chain_categories.tsv)。\n')
    if path.exists():
        with path.open() as f:cols=next(csv.reader(f,delimiter='\t'))
        parts.append(raw_table(path,cols)+'\n')
parts.append('§1.2 的强结论不能完整证实：seq1 的等待与发布确实占据很大一部分已实现路径，但 trace 未区分 task 固定与主循环，不能把合并段全部归为固定开销。\n')
parts.append('''## 7. 代价模型

GB 级无生产者权重流量从 SDCM 约 0.5 的 DRAM 比例改为实测服务曲线给出的 1；有生产者中间量独立按 live footprint 定价。回放参考模型位于 L2 knee 以下，不能用其排序证明 GB 侧缓存规则。单元测试独立检查 2 GiB 的 df_np=1。

490 条观测拟合：λ=1.1593900607537728，r_sm=42.488827019004475 bytes/ns；保留旧 loop_wait/fixed 字段，生产路径不启用劣化的 physical_fixed。下表 nominal/physical 字节仅指注明的观测样本总体，不冒充完整真实模型总流量。

'''+table('fit_errors.tsv',list(rows('fit_errors.tsv')[0]) if rows('fit_errors.tsv') else [])+
    table('observed_traffic.tsv',['cell','population','nominal_operand_bytes','physical_operand_bytes','physical_over_nominal'])+
    '\n整图 task 访存量（按分片计数，包含 task 间重复读取；不是唯一 DRAM 字节或设备实测流量；标量路径原先已用物理域，两列保持相同）：\n\n'+table('model_traffic.tsv',['cell','arm','spaces','tasks','nominal_read_bytes','nominal_write_bytes','physical_read_bytes','physical_write_bytes','physical_over_nominal'])+
    table('replay.tsv',['arm','model','n','spearman','MAPE','best_actual_rank_in_predicted_top1','best_actual_rank_in_predicted_top3','best_actual_rank_in_predicted_top10']))
parts.append('## 8. Level 1 与逐 tile 模拟\n\n预热速度原始证据 `flow_final/`、`flow_arena/`；完整矩阵的求解阶段见 §11。随机样本只有完成 100 组且进程正常结束才标 complete。关闭流体模式的参考 24 计划对照共 102,312 行 binary64 输出与基线逐字节一致，见 `simulator_identity/`。\n\n'+
    table('consistency.tsv',['family','model','n','complete','spearman','ratio_p10','ratio_p50','ratio_p90','flow_ms_max','fluid_ms_max','nonprefix_edges_max','varying_spaces_max'])+
    table('legacy_flow.tsv',['cell','interpretation','T','T_floor','actual_legacy_ms','flow_over_actual_legacy']))
parts.append('## 9. 所选配置\n\n算子名到类的完整映射见表的 operators 字段。原始 `top3.tsv` 与 `metrics.tsv` 的 `floor_ns` 是历史放置下界字段，**不是**本轮物理 T_floor；本报告仅使用 `tmexec.dram_floor`/`*.flow.tsv` 的物理下界。\n\n'+table('configurations.tsv',['cell','operator_class','tile_m','tile_n','tile_k','stages','split_k','kappa','residency','distinct_variants','legacy_uniform']))
parts.append('## 10. 放置、消融与同步检验\n\n原有 affinity/home/spread_other 是互斥分类，affinity 与真实 home 可重叠。离开 home 的比例用独立计数，旧产物用同配置的纯模板 A worker 表逐 tile 重算；不能使用 1−home/placed。\n\n'+
    table('placements.tsv',['cell','grid','residency','flow_ns','simulated_ns','placed','moved_count','moved_fraction','interleaving','evidence'])+
    table('additional_measurements.tsv',['cell','arm','l05_ms','l1_ms','l2_ms','L2_over_floor','flow_ns','simulated_ns','residency','actual_limit','moved_fraction'])+
    '每模型 ≥50 个新进程与实际事件等待省略计数由 G-2 核查；不能以结构共置边数代替执行器真正省去的等待数。\n'+table('resources.tsv',['cell','rank','estimated','actual','re_solved','residency','flow_ns','simulated_ns']))
parts.append('## 11. 求解耗时\n\n阶段单位 ms；部分诊断计时嵌套，不能把所有行直接求和当作 total。每格 budget 按完整求解墙钟减最终 megakernel 编译核算；变体编译仍计入。R9 30–45 h 与旧 legacy 1.3–5.5 h 是历史对照，不是本轮重新求解时间。\n\n'+table('phases.tsv',['cell','phase','count','total_ms']))
parts.append('## 12. θ 网格\n\n同一模型只导入一次，改变 θ 绑定；两起点、完整域。只展示有 completed.tsv 的点；不能把尚未结束的当前最小值记成最终最优。变化区间只由相邻已完成点限定，不宣称区间内的全局最优证明。\n\n'+table('theta.tsv',list(rows('theta.tsv')[0]) if rows('theta.tsv') else []))
parts.append('## 13. R9 早期信号\n\n使用冻结 R9 库重解两个已选几何，实际驻留度 3；512-worker 原分数不归给 384-worker 二进制。两格并未胜过重新计时的 legacy，不能用 per-class 的潜力解释成已得收益。\n\n'+table('additional_measurements.tsv',['cell','arm','l05_ms','l1_ms','l2_ms','L2_over_floor'],lambda r:r['arm']=='early_R9'))
parts.append('''## 14. 未达门的定位与下一步

G-3、G-5、G-7 和额外 runtime-release 审计的原因及处理见 §4。研究门按完整八格/六格固定门线判定，未齐时不判成功。

模型低估实测时，先核对 `FlowPreparation.cpp` 的释放律与 `SkeletonFinalize.cpp` 的 requested-event 掩码差异，再对关键链 GEMM/标量的固定项做分组回放。即使 V2 排序通过，也只能证明两种预测相互一致，不能证明其绝对价格正确。`StageFlowModel.cpp` 的 T_floor 断言证明流量守恒下界，不证明上层模型贴近实测。

TaskBody、同步协议、W=1、分页、预取、融合、静态 batch、element_chunk、legacy 六启发式、CUTLASS 均未为本轮优化而修改；K-13 核实受保护源码。新流体模拟与 regime_a 均有关闭路径的逐位对照。无通过缩小矩阵、降低正确性次数或改变门线取得的通过项。
''')
parts.append('''## 15. R10 输入与 θ 通用性的边界

PG 上界按每格 `T − max(T_floor,T_np0)` 计算，属于模型上界。协议常数一栏是已识别关键链上的 wait+publication+hop 之和；与 T−T_s 的反事实差值可能不同，因为去掉同步会改变调度和关键链，不把二者等同。

'''+table('decomposition.tsv',['cell','pg_upper_ns','protocol_on_chain_ns','attention_chain_ns','attention_over_flow','sync_ns','fixed_ns','contention_ns','chain_ns'])+
    table('fusion_chain.tsv',['cell','category','links','potentially_fusible','wait_ns','publication_ns','hop_ns','fixed_ns']))
parts.append('''可融合节数是所列类别的候选机会，不是融合合法性证明；已折叠的 residual add 不重复计入。attention 的已实现路径份额不包括它可能造成的所有队列后果，不能仅凭这一个比例排除 AT-1。

下一步顺序：先补 Level 1 的真实事件窗口及绝对误差，随后按完成矩阵的 PG/Fuse/协议项上界分配 R10 工程投入；AT-1、静态 batch 与完整请求执行仍按用户规定范围推进。硬件后端或执行器变化需要独立正确性与性能门。

闭式读写计数、分片与 Oracle 接收 θ 绑定，未发现新求解价格/放置按某个 seq 数值或模型名挑分支。仍有两个接入 batch 前必须处理的边界：`ModelDims`（`include/tilemega/Solver/ModelDescription.h`）的生产接口目前显式承载 seq/past/total，没有 batch 字段；embedding 的精确唯一读像依赖 token 值，需要额外 distinct-token/间接像契约，不能称为仅 θ 的闭式量。实验脚本的 seq2/8/32 legacy 起点选择是唯一已声明的 seq 特判，未进入生产定价或放置代码。
''')
(E/'summary.md').write_text('\n'.join(parts))
print(f'R9B_SUMMARY final={a.final} pending={len(pending)} verifier_exit={verdict.returncode} path={E/"summary.md"}')
