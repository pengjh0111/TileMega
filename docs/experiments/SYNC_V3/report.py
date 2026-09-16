#!/usr/bin/env python3
"""Render the R4 review artifact; gates themselves remain raw-evidence audits."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
import sys
import metrics

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE='ee905036d2ec0c9dc880df604097981552423e53'


def rows(path):return list(csv.DictReader(path.open(),delimiter='\t'))

def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in data])+'\n'

def num(value,scale=1):return f'{float(value)*scale:.4f}'


def render():
    ablation=rows(HERE/'ablation.tsv')
    research=rows(HERE/'research.tsv')
    selection=json.loads((HERE/'best_configuration.json').read_text())
    positions=rows(HERE/'target_positions/positions.tsv')
    chain=rows(REPO/'docs/experiments/CHAIN2/final/measurements.tsv')
    fence=rows(REPO/'docs/experiments/FENCE/raw/measurements.tsv')
    audit=HERE/'sass_identity/verification.log'
    verification=audit.read_text() if audit.exists() else Path('/tmp/r4-verify-progress.log').read_text()
    gates=[]
    for line in verification.splitlines():
        fields=line.split('\t',3)
        if len(fields)==4 and fields[1] in ('hard','report','research'):
            gates.append(fields)
    parts=['# TileMega R4：目标同步协议完成与剩余差距\n']
    parts += ['## 1. 基线、提示词与提交顺序\n',
        f'基线：`{BASE}`。分支：`tilemega`。提示词保存在仓库外 '
        '`/root/Prompt/TileMega_R4_prompt.md`，SHA256：'
        f'`{hashlib.sha256(Path("/root/Prompt/TileMega_R4_prompt.md").read_bytes()).hexdigest()}`。\n',
        '恢复后的严格依赖顺序为 A `23c5d618` → target 单独冻结 `03053089` → '
        'B 五臂数据 `204df980` → C1 `e6dd44bc`。这三处先决顺序成立。'
        '必须保留的历史偏离：早先 `66ae000e` 曾在 B 数据完成前提交 C1，随后 '
        '`8299ccaa` 回退；该历史违规没有被抹去。恢复阶段依用户最新指令完成局部修复，'
        '不能把原始历史称为完全满足 H4。\n',
        '最终 SASS 证据采用可审计的父子提交：在全部代码/文档提交完成后编译，'
        '`sass_identity/manifest.json` 记录被编译的 HEAD；最后一个提交只携带该目录的证据，'
        '验证器要求当前 HEAD 等于被测提交，或是其仅修改证据的直接子提交。'
        '提交无法在自身内容中存储自身 hash；最终两次提交的实际 hash 见交付消息与 `git log`。\n',
        '```text\n'+subprocess.check_output(['git','log','--reverse','--format=%h %s',BASE+'..HEAD'],cwd=REPO,text=True)+'```\n']
    parts += ['## 2. 逐门结果与完整自检输出\n',
        table(['门','类型','结果','原始证据重算结果'],gates),
        '证据路径随以下完整输出逐门给出。研究门失败不会被转换为硬门通过；'
        'sm_120 的执行结果仍为未运行。最终封板后的完整输出位于 '
        '[sass_identity/verification.log](sass_identity/verification.log)。\n',
        '```text\n'+verification+'\n```\n']
    parts += ['## 3. 唯一研究门的位置\n',
        '默认放置始终为 `TILEMEGA_PLACEMENT=0`，每格 25 轮同会话配对、轮转臂序、'
        '全新进程。区间为配对比值中位数的 10,000 次 bootstrap 95% CI（seed=167）。'
        '七组配置均在完整四格上评估；至少一个配置达成三格才算研究门通过。代表配置优先选择达门格数最多者，'
        '再以四格协议/barrier 中位比的几何平均打破平局。最快端到端配置另行记录；窗口仍仅显式开启。'
        '差距缩减比例为 `(历史倍数−本轮倍数)/(历史倍数−1)`，允许负值。\n',
        table(['cell','配置','历史倍数','本轮倍数','95% CI','缩掉历史差距','达门'],
            [(r['model']+' s'+r['seq'],r['configuration'],r['historical_ratio'],num(r['ratio']),
              f'[{num(r["ci_low"])}, {num(r["ci_high"])}]',f'{float(r["historical_gap_closed"]):.2%}',
              'PASS' if r['median_pass']=='1' else 'FAIL') for r in research]),
        f'达成 **{sum(r["median_pass"]=="1" for r in research)}/4** 格，要求至少 3 格。'
        f'默认放置四格端到端耗时几何平均最小的配置为 **{selection["fastest_end_to_end"]}**。\n',
        table(['配置','达门格数','协议/barrier 几何平均','full L2 几何平均 ms'],
            [(c,f'{selection["achieved_cells"][c]}/4',num(selection['protocol_scores'][c]),
              num(selection['l2_scores'][c])) for c in metrics.CONFIGS]),
        '若研究门代表配置与端到端最快配置不同，两者分别报告；窗口配置均为显式实验设置，'
        '没有据此改变 W=1 默认值。\n']
    parts += ['## 4. 逐机制消融\n',
        '单位均为 ms。`baseline` 是 R3 B（E3-0..3 全开）；`c1` 增加单发布者 fence；'
        '`c2` 再增加 warp 发布；`windowN` 为 C1+C2、W=N；`localN` 再开 shared 完成标志。'
        '`l1no` 是 l1nosync 臂的 L1 耗时，其他探针列为 L2；差分中位数并不要求等于各中位数之差。'
        '完整 CI 在 [ablation.tsv](ablation.tsv)。\n',
        table(['配置','cell','放置','full','nofence','nowait','neither','l1no','fence','wait','notify','barrier','协议/barrier','L2/L1'],
            [(r['configuration'],r['model']+' s'+r['seq'],r['placement'],
              *[num(r[k]) for k in ('l2_ms','nofence_ms','nowait_ms','neither_ms','l1nosync_ms',
                 'fence_ms','wait_ms','notify_ms','barrier_ms','protocol_over_barrier','l2_over_l1')]) for r in ablation])]
    parts += ['## 5. Fence 定价与 C 的实际设计依据\n',
        table(['cell','放置','fence us','notify us','full−neither us','fence/notify','fence/(full−neither)'],
            [(r['model']+' s'+r['seq'],r['placement'],*[num(r[k],1000) for k in ('fence_ms','notify_ms','protocol_ms')],
              num(r['fence_over_notify']),num(r['fence_over_protocol'])) for r in fence]),
        '默认放置 fence 仅占协议差分约 21–33%，rotate 约 6–7%；因此没有把 C1 的 '
        '128→1 个线程参与误当作整个协议 128 倍缩减。C2 着重让同 warp 的两条独立发布并行，'
        'C3(a) 测 shared 状态对已有窗口执行器的实际影响。rotate 的 fence/notify 可大于 1：'
        'fence 在 wait 开启时测量，notify 在 wait 关闭时定价，二者存在交互，未截断比值。\n']
    parts += ['## 6. 协议实现、正确性与作用域\n',
        'C1：NotifyTask 先 CTA barrier，再由 thread 0 fence 后发布。新 litmus 共 1,800 '
        '个进程，覆盖 grid=64/128/256、tile=1024/4096；cache 与 delayed-writer 两个独立套件 '
        '分别使 no-fence、no-barrier 负对照在所有格 50/50 失败，正对照全部 50/50 通过。'
        '旧的 3,600 进程盲区复核保留在 `litmus_recheck/`。§8.5 在新复核通过后才追加解封注记。'
        '两模型静态 L2 MEMBAR.SC.GPU 为 2→2；变化是发布 fence 的参与线程 128→1 '
        '（常规 128-thread CTA 中发出该指令的 warp 从 4 个降到 1 个），不是静态指令站点消失。'
        '`c1/sass/` 与 `../FENCE/raw/sass/` 保存 BARRIERS 段和相邻谓词证据。\n',
        'C2：warp 0 的 lane 0 fence 后经 syncwarp 把顺序传给发布 lanes；lane 0/1 '
        '分别处理细粒度/聚合事件，其他 warp 可进入下一个 slot 的等待。发布 lanes 在自身发布前 '
        '不进入下一个依赖等待，因此即使该等待依赖本 task 的未发布事件，也不会形成发布者等待自身 '
        '事件的环。下一次 RunTask 前的 CTA 汇合保留。`c2_dependency/trace` 重建 κ=2 '
        '且 σ 省略未生效的真实相邻 slot 依赖，正确性原始格由验证器逐个检查。未引入 Prefetch。\n',
        'C3(a)：窗口内不能省略的同 worker 边在旧实现中已用 `slot_local_deps` 与寄存器 '
        '`done_mask` 处理，并不存在等待移除的全局 poll。本轮按目标改为独立于 TaskSmem '
        '的 shared 完成标志，并在 W=2/4 做对应对照。TaskSmem union 生命周期不变；'
        '无出事件的 task 也对 shared 标志提供 CTA 汇合。\n',
        'C3(b)：只把 DSMEM 的 cluster 内 fan-in 到达降为 cluster scope；最后一个本地 '
        '到达向跨 cluster 消费者发布时仍保留 GPU fence。修复 RED 与 shard 的组合后，'
        '消费者等待非空 shard 数，而非原始 writer 数。sm_89 的组合正确性、caps=false '
        'SASS 退化及 sm_120 PTX/完整模型编译有证据；sm_120 硬件正确性与性能未运行，'
        '不能据 sm_89 推定通过。\n']
    counts=[]
    for c in metrics.CONFIGS:
        root=REPO/'docs/experiments/FENCE/raw' if c=='baseline' else HERE/c
        for r in rows(root/'sass/census.tsv'):
            counts.append((c,r['model'],r['placement'],r['membar_sc_gpu'],r['bar_sync']))
    parts += [table(['配置','模型','放置','静态 MEMBAR.SC.GPU','静态 BAR.SYNC'],counts)]
    parts += ['## 7. 修正归因、旧界与窗口回退\n',
        (REPO/'docs/experiments/TRACE_V2/r4_rebuild/report.md').read_text().replace('# R4 task-DAG reconstruction audit','### 归因审计原始重算'),
        '补充：上述 32 个历史 dump 只有 rotate/chain；其余四候选的新 A trace 已由 '
        '`targets_raw/` 补齐后完成 24 项冻结。\n']
    gains=[]
    for m in ('gqa2','mha4'):
        for seq in (4,128):
            for placement in (0,5):
                cell=f'{m}_s{seq}_p{placement}'
                for w in (2,4):
                    control=metrics.samples(HERE/'ablation'/f'window{w}'/'paired'/cell)
                    local=metrics.samples(HERE/'ablation'/f'local{w}'/'paired'/cell)
                    fifo=metrics.samples(HERE/'ablation'/'c2'/'paired'/cell)
                    ratio,lo,hi=metrics.interval([a['l2_ms']/b['l2_ms'] for a,b in zip(local,control)])
                    net,nlo,nhi=metrics.interval([a['l2_ms']/b['l2_ms'] for a,b in zip(local,fifo)])
                    full_delta=statistics.median(a['l2_ms']-b['l2_ms'] for a,b in zip(local,control))*1000
                    neither_delta=statistics.median(a['neither_ms']-b['neither_ms'] for a,b in zip(local,control))*1000
                    gains.append((cell,w,num(ratio),f'[{num(lo)}, {num(hi)}]',f'{1-ratio:.2%}',num(full_delta),num(neither_delta),
                        f'{num(net)} [{num(nlo)}, {num(nhi)}]'))
    parts += ['本轮 shared 标志对窗口的直接改善（差值均为 local−control）：\n',
        table(['cell','W','local/control','95% CI','耗时改善','full 差 us','neither 差 us','local/C2(W=1) 与 CI'],gains),
        '必须区分完整 kernel 的改善与协议差分的改变：shared 完成状态即使在 neither 臂也需要 '
        'CTA 汇合，否则其他 warp 可能读取尚未更新的 head/done。W=2 的 neither 原始 SASS '
        '在两模型上均为 register 6 个、shared 8 个 BAR.SYNC 位点（`local_probe_sass/`）。'
        '这些本地同步成本保留在 neither 中；full−neither 因而不能把本地状态管理也计作 '
        '全局 wait/notify。研究门仍原样计算，但其差距缩减不能全部解释为端到端加速，'
        '上表独立展示两臂的实际变化。参考模型的原始 E2E_RESOURCE 同时显示，'
        'register/shared 窗口均为 2 CTA/SM、grid=256；寄存器数 218→220，'
        '静态 shared memory 0→16 bytes，TaskSmem 仍为 24576 bytes。没有通过改变 '
        '驻留 grid 来获得这组 local/control 差异。\n']
    parts += ['## 8. 冻结 target 与本轮位置\n',
        '完整冻结表：[targets.tsv](targets.tsv)，原始 trace 与 SHA 在 `targets_raw/`。'
        '每项 floor、measured_A 和 target 均从该候选配置 A 的同一观测上下文得到；'
        '24/24 target 不低于自己的 floor。floor 使用有 trace 的 task 时间，端到端位置使用 '
        '全新无 trace 进程，因此低于观测 floor 不等于突破硬件理论下限。'
        '下表每个候选只比较冻结的原 Plan，新增 Chain2 不借用旧 chain 的 floor。\n',
        table(['候选','cell','floor ms','measured_A ms','target ms','最优 W=1 协议','本轮 L2 ms','相对 target ms'],
            [(r['candidate'],r['model']+' s'+r['seq'],num(r['floor_ms']),
              next(num(t['measured_A_ms']) for t in rows(HERE/'targets.tsv') if (t['candidate'],t['model'],t['seq'])==(r['candidate'],r['model'],r['seq'])),
              num(r['target_ms']),r['configuration'],num(r['l2_ms']),num(r['delta_target_ms'])) for r in positions])]
    parts += ['## 9. 代价感知链化\n',
        (REPO/'docs/experiments/CHAIN2/README.md').read_text().replace('# R4 cost-aware extension and queue placement','### 实现与对照'),
        table(['cell','臂','L2 ms','对 rotate 配对比','95% CI'],
            [(r['model']+' s'+r['seq'],r['arm'],num(r['l2_ms']),num(r['ratio']),
              f'[{num(r["ci_low"])}, {num(r["ci_high"])}]') for r in chain])]
    diagnostic=rows(REPO/'docs/experiments/CHAIN2/diagnostic/minimal/path/mha4_s128_chain_path.tsv')
    parts += [f'价格测试单独诊断（保留同样四轮反馈）：mha4 s128 路径为 '
              f'{sum(r["edge_to_next"]=="h" for r in diagnostic)} 跳，模拟 makespan '
              f'{float(diagnostic[0]["end_ns"])/1000:.3f} us。它尚未满足 D-b，不能作为交付方案；'
              '最终队列放置修正与反馈组合的原始路径在下表独立给出。\n']
    pathrows=[]
    for m in ('gqa2','mha4','real'):
        for seq in (4,128):
            for c in ('legacy_grid_stride','balanced','rotate','eft','wavefront','chain'):
                rr=rows(REPO/'docs/experiments/CHAIN2/final/replay/path'/f'{m}_s{seq}_{c}_path.tsv')
                pathrows.append((m+' s'+str(seq),c,sum(r['edge_to_next']=='h' for r in rr),
                    sum(r['edge_to_next']=='q' for r in rr),len(rr),
                    f'{sum(float(r["task_ns"]) for r in rr)/1000:.3f}',num(rr[0]['end_ns'],.001)))
    parts += [table(['cell','候选','critical path hops','queue edges','路径节点','路径 task us','模拟 makespan us'],pathrows)]
    reject={}
    for r in rows(REPO/'docs/experiments/CHAIN2/final/on/rejected_extensions.tsv'):
        if r['source']!='cost_model':continue
        key=(r['model'],r['seq']);v=reject.setdefault(key,[0,float('inf'),0,float('inf'),0])
        v[0]+=int(r['count']);v[1]=min(v[1],float(r['hop_ns']));v[2]=max(v[2],float(r['hop_ns']))
        v[3]=min(v[3],float(r['queue_ns']));v[4]=max(v[4],float(r['queue_ns']))
    parts += [table(['cell','拒绝的唯一边数','hop ns 范围','queue ns 范围'],
        [(m+' s'+s,v[0],f'{v[1]:.3f}–{v[2]:.3f}',f'{v[3]:.3f}–{v[4]:.3f}') for (m,s),v in reject.items()])]
    parts += ['## 10. 偏离与理由\n',
        '1. 原 H4 提前提交保留历史，恢复阶段重新按 A→E→B→C 排序，详见 §1。\n'
        '2. dump 的 waits 已经丢失被省略的 DAG 边，因此读取匹配生成源的 `kDependencies0` '
        '补齐图，不能仅凭三个物化表逆推出不存在的信息。\n'
        '3. R3 RED 实际是 relaxed 原子，序关系仍来自 NotifyTask fence；PTX 原始审计保留。\n'
        '4. 原 no-barrier 探针对部分几何不敏感；固定 delayed-writer 与独立 cache 套件重新验证，'
        '不改期望结果，不把盲区当作同步冗余。\n'
        '5. C1 改变动态线程参与而非静态 MEMBAR 条数；按真实 SASS 报告。\n'
        '6. R3 B 已允许非发布 warp 前进；C2 实现实际发布 warp 特化与两条独立发布，'
        '没有重复删除已不存在的 barrier。\n'
        '7. W>1 本地依赖已在寄存器完成跟踪，本轮测量 shared 替换的实际成本。\n'
        '8. cluster 实验需要先修复 RED/shard 组合；保留 GPU 范围的跨 cluster 转发。\n'
        '9. D 的价格测试不足以通过跳数门，增加同 SM 队列代价、关键路径排序与跳数平局处理，'
        '使用既有反馈四轮；匹配对照及价格测试单独诊断保留。\n'
        '10. 窗口 no-wait 探针原先仍做 ProbeTaskDependencies 轮询；本轮修复后保留未完成的'
        '旧矩阵，并把全部七组配置重新放入一个新会话。正常窗口的完整 SASS 位同，未改变门口径。\n'
        '11. final SASS 用被测父提交与仅证据子提交绑定，避免要求提交存储自身 hash 的循环。\n']
    parts += ['## 11. 排除项与不变量\n',
        'EX-E4 预取、EX-E5、EX-S1c、EX-S3、EX-S4、EX-S5、EX-V1 均未实现；'
        '未修改 ChainDP、Plan dialect 语义或 TaskSmem union 生命周期；'
        'L-a/L-b/L-c/L-e 仍为硬校验。所有新机制默认关闭，W 默认仍为 1，epoch 单调且不在 '
        '并发迭代间清零。skeleton 仅在 §5.5.1、§5.5.2、§8.5 追加 v2.1 注记。'
        '原有 PLACE_EFT2 summary 与 SYNC_V2 SASS meta 的工作区改动未纳入本轮提交。\n']
    parts += ['## 12. 未达项定位与下一步\n',
        '发布 fence 只覆盖 B 协议差分的一部分。C1 的 `NotifyTask` 仍把 device fence '
        '放在 CTA barrier 之后、全局计数更新之前，减少参与 warp 并没有删除发布关键路径上的 '
        '这个 fence 位点。应先用发布阶段时间戳给 barrier 后的完成延迟定价，再验证显式 '
        'global-space release 原子能否合并该序关系；须保留敏感 litmus，避免再次把 '
        'STRONG.GPU 当作 release 的证明。完成 C1 后，'
        '`WaitTaskDependencies` 仍在非空 wait 后由所有线程执行 acquire fence，'
        '`EventPoll` 仍读取全局事件行；`ArriveEvent` 仍需要跨 CTA/跨 cluster 可见的 '
        '全局计数更新。当前 EVENT_LOAD_POLL=0 仍以 atomicAdd(ev,0) 做 RMW 轮询；'
        '其既有 load 开关须在 WAIT_POLICY=1 的真实调用路径下重新消融（F-162），'
        '先用 SASS 确认指令不同，再验证正确性。SOLO 的直接 epoch 路径也被 RED 组合屏蔽，'
        '单成员 arrivals 的直接发布仍是可明确验证的后续方案。这些成本随 Plan 的跨 worker 边、event fan-in 与等待数量变化，'
        '不能靠继续缩减已经剩一个线程的发布 fence 清除。下一步先给 acquire fence 单独定价，'
        '再为协作 acquire 构造带缓存复用与逐 warp 消费的敏感 litmus；不直接把 acquire '
        '也改成 thread 0。并在联合搜索中同时定价 κ 粗化减少事件数与增加就绪等待的代价。\n',
        '窗口的 `ProbeTaskDependencies` 每探一个候选仍做一次 CTA-wide '
        '`__syncthreads_and`，`WindowAcquireSlot` 还要扫描 wait 列表。shared 完成标志 '
        '不会消除这些扫描；若 local/control 未改善，就应按窗口探测次数与非阻塞轮询 '
        '指令数给扫描定价，再设计批量候选 ready 归约，保持 L-a/L-b/L-c/L-e 与汇合不变量。\n',
        '链化的成本测试只比较一个 hop 与后继 task 的直接队列成本，尚未完整定价 '
        '跨多个同 SM 队列的反向阻塞。原始路径显示 gqa2 s128 的 queue edges 为 10→20，'
        'mha4 s128 为 32→55，而 hops 分别仍为 17/35；real s4 的路径 task 时间则从 '
        '3718.801 增至 4722.158 us。`free_ns[w] = est_end[node]` 只累计 task 权重，'
        '没有单独给仍执行的 NotifyTask 发布成本定价；其对参考格差距的贡献需要下一次 phase trace 验证。'
        '`SchedulePass::finish_on` 的 sibling stretch '
        '是近似；对仍慢于 rotate 的格，下一步以模拟器的实际阻塞传播为增量代价，'
        '在延长时比较 makespan 增量并回传关键路径边，避免用单节点时间代替整个新增队列边链。'
        'sm_120 的更小 hop（约 448 ns）必须在目标机器重新校准和测量，不能由 4090 推定收益。\n',
        'sm_120 cluster 到达仍待目标硬件运行 `SYNC_V3/run_sm120.sh` 的 '
        'cluster_off/on 配对；脚本会在目标上重新求解驻留 Plan。编译通过与 CPU 自检 '
        '没有替代该正确性/性能验证。\n']
    parts += ['## 13. 下一轮优先级\n',
        '1. **EX-S1c + EX-S3 联合搜索优先**：同时优化 κ、跨 worker 跳数和新增队列延迟。'
        '六步实现后剩下的 wait/notify 主要仍是全局事件可见性、acquire、计数更新与 '
        'Plan 引入的依赖距离；减少真正必要的全局事件数量比继续改发布线程数更直接。\n'
        '2. **EX-S5 ISL 参数化放置其次**：用于表达并搜索 co-location/跨 stage 归属，'
        '但必须把物化队列与同 SM 竞争纳入代价，不能重演“被替换的队列边免费”。\n'
        '3. **EX-E4 预取再次**：本轮未加入预取。先由新五臂数据确认剩余 task-only '
        '部分与可重叠范围，再衔接 C2 的非发布 warp；必须单独处理 TaskSmem union '
        '的缓存生命周期，不能把当前等待阶段直接改成未验证的预取。\n']
    (HERE/'summary.md').write_text('\n'.join(parts))


if __name__=='__main__':render()
