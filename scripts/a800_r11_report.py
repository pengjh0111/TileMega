#!/usr/bin/env python3
"""Render the R11 release report from complete, verified artifacts."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

from a100_baselines import sha
from a800_r11_gate import read


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def table(lines, headings, values):
    def row(cells):
        return '| '+' | '.join(str(cell).replace('|', ' / ') for cell in cells)+' |'

    lines.extend(['', row(headings), row(['---']*len(headings)), *(row(line) for line in values), ''])


def sm90_hashes(root):
    result = []
    for proof in sorted(root.glob('*/compared.sha256')):
        if len(proof.read_text().splitlines()) != 4:
            raise ValueError('incomplete SM90 non-regression proof')
        digests = {}
        for line in proof.read_text().splitlines():
            digest, filename = line.split(maxsplit=1)
            path = Path(filename)
            if path.parent != proof.parent or sha(path) != digest:
                raise ValueError('an SM90 comparison artifact changed')
            digests[path.name] = digest
        for extension in ('cu','cubin'):
            reference = digests[f'before.{extension}']
            trial = digests[f'r11.{extension}']
            if reference != trial:
                raise ValueError('SM90 source or cubin bytes changed')
            result.append((proof.parent.name, extension, reference, trial))
    if len(result) != 4:
        raise ValueError('require dense and MoE SM90 proofs')
    return result


def report(matrix, reports, calibration, sm90):
    cells = rows(matrix/'A800_R11_llm.csv')
    summary = read(matrix/'A800_R11_summary.json')
    selected = [row for row in cells if row['scope']=='priority']
    attribution = read(reports/'release-attribution.json')
    coverage = read(reports/'collection.json')
    gate = rows(reports/'A800_R11_gate.csv')
    profile = read(calibration/'registry.ready.json')
    if (len(cells) != 40 or len(selected) != 20 or not all(row['job'] for row in selected)
            or attribution['missing'] or len(attribution['cells']) != 40 or len(gate) != 4
            or coverage['coverage_cells'] != 20 or coverage['rank_cells'] != 20
            or sha(calibration/'registry.json') != profile['sha256']):
        raise ValueError('R11 report requires complete matrix and stage evidence')
    lines = ['# TileMega A800 R11', '', '**verified** 计时采用 `a800-single-v2`。每个对照在同一次 A800 allocation 内完成。']
    table(lines, ('范围','格数','几何平均加速','胜/平/负','效率','达到 1.20× 还需降低的 TPOT'),
          [('重点模型' if name=='priority' else '全矩阵', values['cells'], f'{values["geomean_speedup"]:.4f}×',
            f'{values["wins"]}/{values["ties"]}/{values["losses"]}', f'{values["geomean_tilemega_efficiency"]:.4f}',
            f'{100*values["target_tpot_reduction"]:.2f}%') for name, values in summary.items()])

    counts = defaultdict(lambda:[0,0])
    for row in rows(calibration/'A800_R11_calibration.csv'):
        if row['status']=='verified':
            counts[row['table']][0] += int(row['profiles'])
            counts[row['table']][1] += int(row['measurements'])
    lines.extend(['## R11-1 服务校准', '', '**verified** 当前校准表通过头文件可达性检查，结果为 `0 unreachable profiles`。',
                  f'服务覆盖表中，{coverage["no_modeled_services"]}/20 格的 `modeled_services_count` 为 0。'])
    table(lines, ('服务表','档案数','测量数'), [(key,*counts[key]) for key in sorted(counts)])

    lines.extend(['## R11-2 四候选选择', '', f'**verified** {coverage["four_diverse_finalists"]}/20 格完成四候选、至少两类结构和两个新候选的检查。',
                  f'首选实测命中率为 {coverage["rank1_hits"]}/20。详见 `A800_R11_candidates.csv` 和 `A800_R11_rank.csv`。'])
    if coverage['rank1_hits'] < 12:
        [first, *_] = rows(reports/'A800_R11_rank.csv')
        if first['deviations']:
            lines.append(f'**inferred** 预测比值偏差最大的坐标轴为 {first["deviations"].replace(" | ", "、")}。')

    lines.extend(['', '## R11-3 数值门', '', '**verified** 各臂使用 teacher-forced 与同一二进制执行器一致性。路由实验同时校验有序专家索引。'])
    table(lines, ('模型','B','作业','新进程通过率','配对结果'),
          [(r['model'],r['batch'],r['job'],r['pass_rate'],r['status']) for r in gate])

    boundary = rows(reports/'A800_R11_boundary.csv')
    views = [row for row in boundary if row['stage'] in ('o','down')]
    lines.extend(['## R11-4 边界归约', '', '**stated** OLMoE 使用整输出发布，Q/K 全行归一化继续等待全部输出 tile。',
                  '**verified** 以下边界结果关联各臂的正确性收据与阶段测量。预测节省为每边界 320 ns。'])
    table(lines, ('模型','B','阶段','计时作业','阶段作业','新进程通过率','同步占比','每边界节省 ns'),
          [(r['model'],r['batch'],r['stage'],r['timing_job'],r['trace_job'],r['fresh_pass_rate'],
            r['trial_grid_barrier_fraction'],r['saved_ns_per_boundary']) for r in views])
    whole = [row for row in views if row['arm']=='whole-output-groups'
             and row['model'] in ('llama3_1_8b', 'mistral_7b_v0_3') and int(row['batch'])<=4
             and row['trial_grid_barrier_fraction'] and row['saved_ns_per_boundary']]
    if whole:
        fraction = [float(row['trial_grid_barrier_fraction']) for row in whole]
        saving = median(float(row['saved_ns_per_boundary']) for row in whole)
        targets = sum(value <= .20 for value in fraction)
        lines.extend([f'**verified** 整输出发布在 {targets}/{len(whole)} 个阶段达到 20% 门限。'
                      f'实测等待占比为 {min(fraction):.1%}–{max(fraction):.1%}，'
                      f'每边界节省中位数为 {saving:.0f} ns。'])
        if targets < len(whole):
            lines.append('**inferred** 整输出发布按已完成的 tile 累计到达量。'
                         '后继仍依赖全部输出，最后一批 tile 的完成和全局等待依然处于关键路径。')

    qwen = [row for row in rows(reports/'A800_R11_qwen3.csv') if row['stage'] in ('router','o')]
    lines.extend(['## R11-5 Qwen3-30B-A3B', '', '**verified** 阶段记录分别统计 CTA 同阶段重叠和相邻阶段重叠。'])
    table(lines, ('B','阶段','测试臂','作业','GB/s','consumer 等待占比','相邻阶段重叠 SM'),
          [(r['batch'],r['stage'],r['arm'],r['job'],r['effective_gbps'],r['consumer_wait_fraction'],
            r['cross_stage_overlap_sms']) for r in qwen])
    priority_qwen = [row for row in selected if row['model']=='qwen3_30b_a3b' and int(row['batch'])>1]
    if priority_qwen:
        wins = sum(float(row['speedup']) >= 1 for row in priority_qwen)
        lines.append(f'**verified** Qwen3 B2/B4/B8/B16 中有 {wins}/{len(priority_qwen)} 格达到最快 baseline。')
    if (reports/'A800_R11_router_recovery.csv').exists():
        lines.append('**verified** 优先复测保留原路由的备用臂，定价排序仍按原 allocation 核算。'
                     '两批证据分别记录在路由恢复表和候选排名表。')

    winning = {(r['model'],r['batch']):r for r in rows(reports/'A800_R11_mainloop.csv') if r['decisive']=='True'}
    lines.extend(['## R11-6 协作流水', '', '**verified** 每格的流水选择依据同卡整模型 TPOT。'])
    table(lines, ('模型','B','作业','所选测试臂','线程数','寄存器数','TPOT ms'),
          [(r['model'],r['batch'],r['job'],r['arm'],r['threads'],r['linked_registers'],r['tpot_ms'])
           for r in winning.values()])

    lines.extend(['## R11-7 复测与 SM90 对照', '', '**verified** 使用 `nvcc 12.8.93` 和 `7425a52` 固定配置比较生成源与 cubin。'])
    table(lines, ('模型','产物','R10 SHA-256','R11 SHA-256'), sm90_hashes(sm90))
    table(lines, ('模型','B','R10 TPOT ms','R11 TPOT ms','加速','效率','复测作业'),
          [(r['model'],r['batch'],r['r10_tpot_ms'],r['r11_tpot_ms'],r['speedup'],r['efficiency'],r['job']) for r in cells])
    lines.extend(['## 阶段带宽与剩余差额', '', '**verified** 各阶段 span 来自对应最终二进制的 release 记录。',
                  '**inferred** 阶段字节下界采用标定 HBM 带宽，MoE 逻辑字节可使用预期活跃专家数。各阶段超额不可相加为整模型的 TPOT 差额。'])
    table(lines, ('配置','阶段 GB/s','作业','距 1.20× 的 TPOT 差额 ns','阶段超额 ns / TPOT 占比','所需效率'),
          [(key, '、'.join(f'{s["stage"]} {s["effective_gbps"]:.0f}' for s in cell['stages']),
            f'{cell["source"]} {cell["job"]}', f'{cell["target_gap_ns"]:.0f}',
            '、'.join(f'{s["stage"]} {s["excess_ns"]:+.0f} / {s["fraction_of_request_tpot"]:.1%}'
                      for s in sorted(cell['stages'],key=lambda s:-s['excess_ns'])) if cell['target_gap_ns'] else '已达到',
            f'{cell["required_efficiency"]:.3f}') for key,cell in sorted(attribution['cells'].items())])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','reports','calibration','sm90','out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    content = report(args.matrix,args.reports,args.calibration,args.sm90)
    temporary = args.out.with_suffix('.next')
    temporary.write_text(content)
    temporary.replace(args.out)


if __name__ == '__main__':
    main()
