#!/usr/bin/env python3
"""Publish all accepted mechanism-removal processes without ranking filters."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from sm90d1_common import ROOT, read, sha, write
from sm90_decode_ae_collect import ORDER


STEM = 'SM90_decode_mechanism_AE_qwen3_1_7b_b1'


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def process_row(arm, slot, path, binary, reused, inventory):
    m = read(path)
    values = {}
    for n in (1, 1024):
        runs = [r for r in m['generation_runs'] if r['N'] == n]
        assert sum(r['warmup'] for r in runs) == 1
        values[n] = [r['wall_seconds'] for r in runs if not r['warmup']]
        assert len(values[n]) == 3
    tpot = (statistics.median(values[1024])-statistics.median(values[1]))/1023*1000
    assert math.isclose(tpot, m['tpot_seconds']*1000, rel_tol=1e-12)
    telemetry = [json.loads(line)['telemetry'] for line in
        (path.parent / 'guard.jsonl').read_text().splitlines()]
    return dict(arm=arm, slot=slot, pid=m['pid'], tpot_ms=tpot,
        new_or_reused='reused' if reused else 'new', binary_sha256=binary,
        n1_wall_seconds=json.dumps(values[1]), n1024_wall_seconds=json.dumps(values[1024]),
        generation_runs=json.dumps(m['generation_runs']), gpu=m['gpu_index'], inventory=inventory,
        sm_clocks=json.dumps(sorted({s['clocks.sm'] for s in telemetry})),
        mem_clocks=json.dumps(sorted({s['clocks.mem'] for s in telemetry})),
        acceptance=m['acceptance'], ownership_verified=m['ownership_verified'],
        model_config_sha256=m['model_config_sha256'], prompt_ids_sha256=m['prompt_ids_sha256'],
        checkpoint_sha256=m['checkpoint']['checkpoint_sha256'],
        versions=json.dumps(m['versions'], sort_keys=True),
        source=str(path), source_sha256=sha(path))


def plots(out, rows, ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    colors = ['#6e7785', '#237ab8', '#32a18d', '#184f38', '#ad7047']
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for i, r in enumerate(rows):
        values = json.loads(r['three_process_tpot'])
        bar = ax.bar(i, r['tpot_ms'], color=colors[i], width=.65,
            hatch='//' if r['arm'] == 'D' else None)
        ax.scatter([i-.08, i, i+.08], values, color='black', s=17, zorder=3)
        ax.bar_label(bar, fmt='%.4f', padding=7)
    ax.set_xticks(range(5), ['A\nS1 basic', 'B\nS16 basic', 'C\nS1 TileMega',
        'D*\nS16 TileMega', 'E\nS1 basic graph'])
    ax.set_ylabel('Request-wall TPOT (ms); lower is faster')
    ax.set_title('Qwen3-1.7B / H800 / BF16 / BS1 / ISL64 OSL1024')
    ax.set_ylim(0, max(r['tpot_ms'] for r in rows)*1.16)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, -.015, 'Dots: three process values. *D: historical reused main experiment.',
        ha='center', fontsize=9)
    for ext in ('svg', 'png'):fig.savefig(out / f'{STEM}_tpot.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
    labels = [r['ratio'] for r in ratios]
    bars = ax.bar(range(len(ratios)), [r['value'] for r in ratios], color='#237ab8')
    bars[-1].set_color('#8d659b')
    ax.bar_label(bars, fmt='%.4f', padding=5)
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(1, color='#555555', linewidth=1, linestyle='--')
    ax.set_ylim(0, max(r['value'] for r in ratios)*1.16)
    ax.set_ylabel('TPOT ratio (x); I is the multiplicative interaction')
    ax.set_title('Mechanism-removal comparisons (D is historical)')
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ('svg', 'png'):fig.savefig(out / f'{STEM}_ratios.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def report(run, historical):
    state = read(run / 'timing/state.json');spec = read(run / 'spec.json')
    assert state['status'] == 'complete' and len(state['items']) == 12
    assert tuple(v['arm'] for v in state['items']) == ORDER
    assert state['spec_sha256'] == sha(run / 'spec.json')
    assert spec['consistency_sha256'] == sha(run / 'consistency.json')
    manifest = read(ROOT / 'configs/serving/sm90_ae/qwen3_1_7b_b1/D.manifest.json')
    processes = []
    for item in state['items']:
        assert item['status'] == 'done'
        path = Path(item['result']);assert sha(path) == item['result_sha256']
        guard = read(Path(item['attempts'][-1]['directory']) / 'guard_result.json')
        assert guard['code'] == 0
        m = read(path);arm = item['arm']
        binary = spec['arms'][arm]['decode_sha256']
        assert m['configuration']['decode_sha256'] == binary
        assert sha(Path(spec['arms'][arm]['decode'])) == binary
        assert not any(m['configuration']['checks'].values())
        assert all(e['decode_loop_used'] for e in m['configuration']['executions'] if e['tokens']==1024)
        processes.append(process_row(arm, item['slot'], path, binary, False, state['inventory']))
    for item in read(historical)['rows']:
        path = Path(item['tokens']).parent / 'B1/measurements.json'
        row = process_row('D', item['job'], path, manifest['binary_sha256'], True, 'historical H800')
        assert math.isclose(row['tpot_ms'], item['tpot_ms'], rel_tol=1e-12)
        processes.append(row)
    assert len({r['pid'] for r in processes if r['new_or_reused'] == 'new'}) == 12
    rows = []
    for arm in 'ABCDE':
        group = [r for r in processes if r['arm'] == arm]
        assert len(group) == 3
        values = [r['tpot_ms'] for r in group]
        t = manifest['measured']['tpot_ms'] if arm == 'D' else statistics.median(values)
        if arm == 'D':assert t == 1.6754311115689438
        rows.append(dict(arm=arm,
            decomposition='Operator-local S1' if arm in 'ACE' else 'TileMega-selected S16',
            plan='Basic: aggregate QKV release; embedding stage' if arm in 'ABE' else
                'TileMega: KV-group release; fused Head-to-embedding',
            execution='Typed-stage kernels + CUDA Graph' if arm == 'E' else 'Megakernel',
            tpot_ms=t, three_process_tpot=json.dumps(values),
            process_range_pct=(max(values)-min(values))/t*100,
            new_or_reused='reused' if arm == 'D' else 'new', binary_sha256=group[0]['binary_sha256']))
    times = {r['arm']:r['tpot_ms'] for r in rows}
    comparisons = [('A','B','decomposition gain under basic planner'),
        ('C','D','decomposition gain under TileMega planner'),
        ('A','C','planner gain under operator-local decomposition'),
        ('B','D','planner gain under TileMega decomposition'),
        ('A','D','combined decomposition and planner gain'),
        ('E','A','megakernel execution gain'), ('E','D','total baseline to full TileMega gain')]
    ratios = [dict(ratio=a+'/'+b, value=times[a]/times[b], interpretation=label,
        uses_historical_D=b=='D') for a,b,label in comparisons]
    ratios.append(dict(ratio='I', value=times['B']*times['C']/(times['A']*times['D']),
        interpretation='multiplicative interaction: (B*C)/(A*D)', uses_historical_D=True))
    attempts = [dict(arm=i['arm'], **a) for i in state['items'] for a in i['attempts'] if a['code']]
    environment = {k:len({r[k] for r in processes}) == 1 for k in
        ('versions','model_config_sha256','prompt_ids_sha256','checkpoint_sha256','sm_clocks','mem_clocks')}
    receipt = dict(evidence='verified performance, compilation and static path audit only',
        accepted_new_processes=12, reused_processes=3, D_remeasured=False,
        rows=rows, ratios=ratios, all_attempts=str(run / 'timing/state.json'),
        state_sha256=sha(run / 'timing/state.json'), rejected_attempts=attempts,
        spec=spec, consistency=read(run / 'consistency.json'),
        new_and_historical_environment_match=environment,
        measured_order=sorted(times, key=times.get),
        acceptance=sorted({r['acceptance'] for r in processes}),
        historical_result=str(historical), historical_result_sha256=sha(historical))
    out = ROOT / 'docs/results';out.mkdir(parents=True, exist_ok=True)
    csv_write(out / f'{STEM}.csv', rows)
    csv_write(out / f'{STEM}_processes.csv', processes)
    csv_write(out / f'{STEM}_ratios.csv', ratios)
    write(out / f'{STEM}_provenance.json', receipt)
    plots(out, rows, ratios)
    text = ('# SM90 mechanism-removal A-E: Qwen3-1.7B\n\n'
        'Verified performance-only scope: H800, TP1, BF16, BS1, ISL/OSL 64/1024. '
        'Branch `sm90-decode-mechanism-ae-v1` starts at '
        '`4a84be39988967cb0a3c8f30c532bd25a6587960`. D is the unchanged historical '
        'main-experiment value 1.6754311115689438 ms, originally built from '
        '`4b9d1362e648e2ee2cc7f96da9422f97a6d36c45`; no D job was submitted.\n\n'
        'Operator-local here means the prescribed unsplit S1 attention baseline, '
        'not a newly searched or measured local optimum. Only attention.chunks, '
        'group_wait and emb differ from D. GEMM geometries, MMA, warp layout, '
        'slots, register stages, residency, cluster and task-body source remain '
        'fixed. S1 has 8 attention tasks per layer and writes final context '
        'directly; S16 has 128 tasks and uses the existing partial-result LA merge.\n\n'
        'Basic disables both KV-group early release at QKV→attention and fused '
        'Head→embedding handoff. It retains aggregate stage barriers, the same '
        'cyclic/grid-stride ownership, typed order and folded argmax. TileMega '
        'restores those two coordination mechanisms. This experiment does not '
        'isolate placement or order search.\n\n'
        'E.json is byte-identical to A.json; their emitted body aliases and '
        'NativeProgram prefix are byte-identical. E uses the inherited '
        'preserve-body sequence branch: its graph contains embedding, the '
        'five typed stages per layer and folded Head in order, with '
        'kernel completion carrying cross-stage order. The captured graph is '
        'replayed for each decode step; capture is outside timed generation. '
        'kEmbed=false keeps the embedding stage every step; kPostHead=false '
        'avoids a separate argmax kernel. No task body or persistent path '
        'needed changes for this matrix, and no artificial work was added.\n\n'
        'Verified protocol: A B C E / B C E A / C E A B on one physical H800, '
        'three independent processes per new arm. Each process has one warmup '
        'and three timed requests at N1 and N1024. TPOT = '
        '(median wall N1024 − median wall N1)/1023; the table uses the median '
        'of the three process TPOTs. Range = (max−min)/median × 100%. '
        'Historical prefill, name-seeded checkpoint algorithm/seed0, prompt '
        'and L1 decode-loop protocol are retained. No solve, calibration, '
        'microbenchmark sweep, C1/C2, HF, token/logit comparisons, sanitizer, '
        'stress tests or profiling were run. The existing driver checks output '
        'shape only.\n\n'
        '| Arm | Decomposition | Plan | Execution | TPOT ms | Three process TPOTs ms | Range % | New/reused |\n'
        '|---|---|---|---|---:|---|---:|---|\n')
    for r in rows:
        text += (f'| {r["arm"]} | {r["decomposition"]} | {r["plan"]} | {r["execution"]} | '
            f'{r["tpot_ms"]:.9f} | {r["three_process_tpot"]} | {r["process_range_pct"]:.6f} | '
            f'{r["new_or_reused"]} |\n')
    text += '\n| Ratio | Value | Interpretation |\n|---|---:|---|\n'
    for r in ratios:text += f'| {r["ratio"]} | {r["value"]:.9f} | {r["interpretation"]} |\n'
    text += ('\nVerified observed ordering, fastest first: `' + ' < '.join(receipt['measured_order']) +
        '`. No process was selected or excluded for its measured speed. '
        f'There were {len(attempts)} guard/driver-rejected attempts, retained with reason and logs. '
        'Interpretation is limited to these two decomposition settings and the '
        'two bundled planner mechanisms. Comparisons containing D are cross-session '
        'and inherit historical variability. No numerical correctness claim is '
        'made from this performance-only run.\n\n'
        'New GPU: `' + state['inventory'] + '`. Exact environment matching across '
        'all new and historical process records: `' + json.dumps(environment,sort_keys=True) +
        '`. Full clock sets and versions are in the process CSV. Existing namespace-limited '
        'GPU guard acceptance is retained: `' + ', '.join(receipt['acceptance']) +
        '`; unresolvable host PIDs are not asserted to have verified ownership. '
        'No GPU clock or power setting was changed.\n\n'
        f'![TPOT]({STEM}_tpot.svg)\n\n![Ratios]({STEM}_ratios.svg)\n\n'
        'Reproduction: `configs/serving/sm90_mechanism_ae/README.md`. Full precision '
        'binary hashes and process values are in the CSVs; static execution-path '
        'audit and artifact hashes are in the provenance JSON. Raw binaries, generated '
        'CUDA, compile logs, all guard attempts and every request measurement are '
        f'retained at `{run}` (not committed).\n')
    (out / f'{STEM}.md').write_text(text)
    write(run / 'final-summary.json', receipt)
    print(json.dumps(dict(times=times, ratios=ratios), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--historical-result', type=Path, required=True)
    args = parser.parse_args();report(args.run.resolve(), args.historical_result.resolve())
