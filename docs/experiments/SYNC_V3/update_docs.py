#!/usr/bin/env python3
"""Append final R4 findings and v2.1 ledger annotations from measured tables."""
import csv
import re
from pathlib import Path
import metrics

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def rows(p):return list(csv.DictReader(p.open(),delimiter='\t'))


def table(head,data):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in data])+'\n'


def main():
    research=rows(HERE/'research.tsv')
    chain=rows(REPO/'docs/experiments/CHAIN2/final/measurements.tsv')
    positions=rows(HERE/'target_positions/positions.tsv')
    findings=REPO/'docs/FINDINGS.md'
    text=findings.read_text()
    start='## F-176 — Cost-aware extraction must also price the queue placement it creates'
    if start in text:
        if max(map(int,re.findall(r'^## F-(\d+)',text,re.M)))!=179:
            raise ValueError('new FINDINGS appeared after the generated R4 block')
        text=text[:text.index(start)].rstrip()+'\n'
    if max(map(int,re.findall(r'^## F-(\d+)',text,re.M)))!=175:
        raise ValueError('FINDINGS tail changed; do not overwrite another author')
    additions=[start+'\n',
        '✅ Verified in R4: the strict extension test uses the calibrated hop curve and the '
        'successor task weight from the same cost-model source as extraction. Equality rejects; '
        'capacity formulas and hard Plan legality checks remain. The default-off implementation '
        'also ranks ready tasks by remaining critical work, prices sibling-SM sharing in finish_on, '
        'and breaks equal finishes toward fewer path hops. Four existing feedback rounds are fixed '
        'for the selected recipe, with a cost-off four-round matched control and the original '
        'zero-feedback chain retained. See CHAIN2/design.json and README.md.\n',
        '✅ The isolated price-only variant with the same four feedback rounds still gives '
        '40 hops and 597673.9223 ns on mha4 s128. The completed recipe gives 35 hops and '
        '446326.144 ns, against rotate at 35 hops and 437123.7068 ns. These are simulator '
        'path-record results, independently counted from CHAIN2/final/replay/path and '
        'CHAIN2/diagnostic/minimal/path. Replayed materialized chain sources match the '
        'measured Plans byte for byte. Four reference correctness cells pass 50/50.\n',
        '✅ Fresh-process paired GPU measurements (25 rotated rounds, configuration A):\n',
        table(['cell','arm','L2 ms','ratio to rotate','95% CI'],
              [(r['model']+' s'+r['seq'],r['arm'],f'{float(r["l2_ms"]):.6f}',f'{float(r["ratio"]):.5f}',
                f'[{float(r["ci_low"]):.5f}, {float(r["ci_high"]):.5f}]') for r in chain]),
        '✅ Unique excluded cost-model DAG edges: gqa2 s4/s128 412/538124, '
        'mha4 s4/s128 908/2127388, real s4/s128 25952/34394336. Exact hop/queue prices '
        'and multiplicities are retained in final/on/rejected_extensions.tsv. These count '
        'unique edges in the selected pass, not repeated DP visits. Reference worker-SM '
        'maps come from the existing traced calibration; real-width uses the modulo model.\n',
        '✅ The replayed critical path still adds queue edges: gqa2 s128 10→20 and '
        'mha4 s128 32→55, despite unchanged hop counts 17/35. Real s4 path task '
        'weight rises from 3718.801 to 4722.158 us. These are direct node/edge records.\n',
        '⚠️ Inferred next correction for slower cells: finish_on still approximates sharing '
        'and does not propagate all downstream queue blocking. free_ns tracks task '
        'weights without separately pricing NotifyTask work still executed on those '
        'queues; a fresh phase trace must quantify that contribution. Price extension by the '
        'simulator makespan increment with affected queue edges, rather than only the '
        'successor weight. The hop test remains part of that redesign; reverting it is '
        'not the outcome. No ChainDP change is included.\n',
        '## F-177 — Completed-protocol gains and shared-window costs are measured together\n',
        '✅ Verified from SYNC_V3/ablation: seven configurations, two placements, four '
        'reference cells and five probes, 25 rotated rounds in one session (7000 fresh '
        'processes). The required real-width seq=4 legacy matrix adds five configurations '
        'and 625 fresh processes. The superseded partial matrix with the window probe '
        'coverage hole is excluded. All full processes pass.\n']
    gains=[]
    for model in ('gqa2','mha4'):
        for seq in (4,128):
            for placement in (0,5):
                cell=f'{model}_s{seq}_p{placement}'
                for name,reference in (('c1','baseline'),('c2','c1'),('local2','window2'),('local4','window4'),('local2','c2'),('local4','c2')):
                    cur=metrics.samples(HERE/'ablation'/name/'paired'/cell)
                    ref=metrics.samples(HERE/'ablation'/reference/'paired'/cell)
                    ratio,lo,hi=metrics.interval([a['l2_ms']/b['l2_ms'] for a,b in zip(cur,ref)])
                    gains.append((cell,name+'/'+reference,f'{ratio:.5f}',f'[{lo:.5f}, {hi:.5f}]',f'{1-ratio:.2%}'))
    additions += [table(['cell','paired contrast','L2 ratio','95% CI','latency reduction'],gains),
        '✅ Static instruction counts remain separate from these measured gains: C1 '
        'changes notify-fence participation from 128 threads to one while static L2 '
        'MEMBAR.SC.GPU remains two. Window variants have three MEMBAR sites; shared '
        'variants have ten BAR.SYNC sites versus eight for the register-state window '
        'control. W remains disabled by default.\n',
        '✅ The neither probe also retains local completion convergence: its W=2 '
        'L2 SASS has six BAR.SYNC sites in the register control and eight in the '
        'shared variant, for both models (local_probe_sass/). The report lists '
        'paired full and neither changes separately. The registered full-minus-neither '
        'marginal excludes local state-management work retained by neither; a smaller '
        'research ratio must not be presented as the same percentage of full-kernel '
        'speedup. The gate equations are unchanged.\n',
        '⚠️ Inferred remaining window cost is localized in ProbeTaskDependencies '
        'and WindowAcquireSlot: candidate wait scans and a CTA reduction per probe '
        'remain even after local completion is shared. The next design should batch '
        'candidate readiness reductions and retain all four Plan legality checks, '
        'rather than assume that shared flags remove pre-existing global polls.\n',
        '## F-178 — The registered default-placement protocol target remains the research test\n',
        '✅ Verified from the fresh paired matrix. Every configuration is evaluated on '
        'all four default-placement cells; one configuration must achieve at least three. '
        'The representative maximizes achieved cells and then minimizes geometric mean '
        'protocol/barrier, with fastest end-to-end time reported separately. The ratio '
        'threshold 1, coverage, 25 rounds and bootstrap rule were not relaxed. '
        'research_rule.json was committed before the corrected matrix session.\n',
        table(['cell','configuration','historical ratio','current ratio','95% CI','historical excess gap closed'],
              [(r['model']+' s'+r['seq'],r['configuration'],r['historical_ratio'],f'{float(r["ratio"]):.5f}',
                f'[{float(r["ci_low"]):.5f}, {float(r["ci_high"]):.5f}]',f'{float(r["historical_gap_closed"]):.2%}') for r in research]),
        f'✅ Achieved cells: {sum(r["median_pass"]=="1" for r in research)}/4; required: 3/4. '
        'The full five-arm and L1 ratios, including other configurations, remain in '
        'ablation.tsv and research_all.tsv. Negative gap closure is retained.\n',
        '⚠️ Inferred residual causes: WaitTaskDependencies still issues the acquire '
        'fence from all threads when waits exist; EventPoll still uses atomicAdd(ev,0) '
        'with EVENT_LOAD_POLL=0; ArriveEvent still updates globally visible counters. '
        'The next concrete steps are an acquire-fence price probe and sensitive '
        'cooperative-acquire litmus, plus the existing load-poll ablation under '
        'WAIT_POLICY=1 with a nonempty SASS diff (F-162). The SOLO direct-epoch path '
        'is also disabled in the RED combination; single-member arrivals publication '
        'needs its own monotonicity/visibility check. Joint kappa/placement search '
        'must price both fewer global events and added readiness/queue delay. '
        'The unsafe full-minus-neither contrast includes dependency serialization '
        'as well as synchronization instructions; it is not a sum of fence opcodes.\n',
        '## F-179 — Frozen ceilings stay attached to the original candidate Plans\n',
        '✅ All 24 frozen targets reproduce from their candidate-specific configuration-A '
        'traces and none is below its own floor. The three W=1 protocols are measured '
        'on every unchanged frozen candidate; target_positions/positions.tsv records '
        'their positions. Balanced uses placement 4, matching the frozen trace; its '
        'earlier unused placement-0 compile is archived and was not measured. '
        'The new Chain2 Plan is reported separately and does not inherit the old '
        'chain floor. No target entry changes after commit 03053089.\n',
        f'✅ {sum(float(r["delta_target_ms"])<=0 for r in positions)}/24 fixed candidate/cells '
        'are at or below their frozen target in the best measured W=1 protocol. '
        'These are fresh untraced timings against trace-observed floors, not proof '
        'of a target-independent hardware lower bound.\n']
    # Adjacent string literals in the long prose above remain ordinary Python;
    # output is regenerated only after all raw-dependent tables are complete.
    findings.write_text(text+'\n'+'\n'.join(additions))
    ratios='/'.join(f'{float(r["ratio"]):.4f}' for r in research)
    achieved=sum(r['median_pass']=='1' for r in research)
    block=f'''<!-- R4_FINAL_BEGIN -->
（⚠️ v2.1 第四轮恢复完成，2026-09-16：此前 2026-09-15 的停止状态保留为历史。
C1/C2 各四参考格 50/50、SEQSCAN 子集各 300/300；新的双套件敏感 litmus
1800 个进程全部符合正/负对照预期，§8.5 已在复核后追加解封注记。C2 的 κ=2
相邻 slot 见证为 68/68/108/140 条，四格 50/50。C3(a) 在 W=2/4 各四格
50/50，窗口对照也全过；RED/shard 组合四格 50/50。cluster 作用域已实现、
sm_89 退化 SASS 与 sm_120 编译/脚本自检有证据，sm_120 硬件验证仍待运行。
G4/EX-E3 的研究门保持默认放置 wait+notify≤barrier，代表配置
{research[0]['configuration']} 四格倍数为 {ratios}，达成 {achieved}/4（要求 3/4）；
差距缩减与全部五臂见 SYNC_V3/summary.md、F-177/F-178，不能由实现完成推定达门。
G5/EX-E2：修正 HOL 表明 W=4 回收了等待，但分析器修复消除原始计时回退的比例为
0%，窗口默认仍为 1；本轮 shared/control 净差异另由新配对测量给出（F-173/F-177）。
G7/EX-D1：32 个放置 dump 加 12 个窗口 dump 的界与 split 共 44/44 通过，
旧口径以 _legacy 保留，24 个候选自身 target 在新 A trace 后单独冻结且不再改动。
EX-S2c 的价格测试与队列放置修正已测，mha4 s128 保持 35 跳，参考正确性四格
50/50；六格 chain/rotate、匹配反馈对照和原链对照全部 25 轮（F-176）。
窗口 no-wait 探针的漏覆盖在本轮修复，未完成旧矩阵保留、最终矩阵全新重跑（F-175）。
后续优先 EX-S1c+EX-S3，再 EX-S5，最后 EX-E4；本轮未实现这些排除项。）
<!-- R4_FINAL_END -->
'''
    for name in ('STATUS','TODO'):
        p=REPO/f'docs/{name}.md';s=p.read_text()
        s=re.sub(r'<!-- R4_FINAL_BEGIN -->.*?<!-- R4_FINAL_END -->\n?', '', s, flags=re.S)
        marker='### 1.5.3 ' if name=='STATUS' else '### 1.3 '
        at=s.index(marker);s=s[:at]+block+'\n'+s[at:]
        if name=='TODO':
            lines=s.splitlines()
            for i,line in enumerate(lines):
                if line.startswith('| EX-E3 |'):
                    fields=line.split('|');fields[4]=' 六步已实现；第 1–3 步沿用 R3，C1/C2/C3(a) 已验证，cluster 待 sm_120 ';lines[i]='|'.join(fields)
                if line.startswith('| EX-D1 |'):
                    fields=line.split('|');fields[4]=' 已验证（含 R4 task DAG 归因维护） ';lines[i]='|'.join(fields)
            s='\n'.join(lines)+'\n'
        p.write_text(s)


if __name__=='__main__':main()
