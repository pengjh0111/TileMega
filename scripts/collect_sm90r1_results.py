#!/usr/bin/env python3
"""Collect complete SM90R1 cells without treating queued work as results."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from r16_measurement_validation import completed_guard, file_sha, validate_dnn, validate_llm
from run_sm90r1_final_cell import validate_numeric, validate_pair


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def table(name):
    with (ROOT / 'docs/results' / name).open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def aggregate_ratios(llm, llm_ratios, dnn, dnn_ratios):
    rows = []

    def append(scope, baseline, metric, values):
        if values:
            assert all(math.isfinite(v) and v > 0 for v in values)
            rows.append(dict(scope=scope, baseline=baseline, metric=metric,
                             cells=len(values), geometric_mean=math.exp(
                                 sum(math.log(v) for v in values) / len(values)),
                             minimum=min(values), maximum=max(values)))

    for scope in ('dense', 'moe', 'all_llm'):
        selected = [r for r in llm_ratios if scope == 'all_llm' or
                    (r['model'] == 'qwen3_30b_a3b') == (scope == 'moe')]
        native = [r for r in llm if scope == 'all_llm' or
                  (r['model'] == 'qwen3_30b_a3b') == (scope == 'moe')]
        for metric in ('tpot', 'ttft', 'e2e', 'throughput'):
            for baseline in sorted({r['baseline'] for r in selected}):
                append(scope, baseline, metric,
                       [r['speedup_' + metric] for r in selected if r['baseline'] == baseline])
            engines = {}
            for r in selected:
                if r['baseline'] in ('vllm', 'sglang', 'tensorrt-llm'):
                    engines.setdefault((r['model'], r['batch']), []).append(r['speedup_' + metric])
            append(scope, 'strongest_engine_per_metric', metric,
                   [min(values) for values in engines.values()])
        for metric in ('tpot', 'e2e'):
            append(scope, 'TM-R16', metric,
                   [r['speedup_' + metric + '_vs_tm_r16'] for r in native])
    for metric in ('median', 'p90'):
        for baseline in sorted({(r['baseline'], r['precision'], r['compile_mode']) for r in dnn_ratios}):
            label = '/'.join(x for x in baseline if x)
            append('dnn', label, metric,
                   [r['speedup' if metric == 'median' else 'speedup_p90'] for r in dnn_ratios
                    if (r['baseline'], r['precision'], r['compile_mode']) == baseline])
        fastest = {}
        for r in dnn_ratios:
            if r['same_precision']:
                fastest.setdefault(r['model'], []).append(r['speedup' if metric == 'median' else 'speedup_p90'])
        append('dnn', 'strongest_bf16_per_metric', metric,
               [min(values) for values in fastest.values()])
    append('dnn', 'TM-R16', 'median', [r['speedup_vs_tm_r16'] for r in dnn])
    return rows


def llm_cells(raw, states):
    progress = read(raw / 'final-llm/progress.json')
    rows = []
    for key, cell in progress['settled'].items():
        if cell['status'] != 'completed':
            continue
        config_path = Path(cell['config'])
        config = read(config_path)
        identities = validate_pair(config)
        certificate = Path(cell['numeric_receipt'])
        batch = config['batch']
        validate_numeric(certificate, file_sha(config_path), identities, batch,
                         256 if config['model'] == 'qwen3_30b_a3b' else 1024)
        reference = read(read(certificate)['reference_tokens']['path'])
        rounds = []
        for recorded in cell['rows']:
            job = states[recorded['job']]
            directory = Path(job['out'])
            guard = completed_guard(directory, job)
            path = directory / f'measurements/B{batch}/measurements.json'
            value, _, ownership = validate_llm(path, guard, ROOT)
            selection = read(directory / 'measurements/selected-config.json')
            assert selection['config_sha256'] == file_sha(config_path)
            assert {p: i['artifact_id'] for p, i in value['configuration']['identities'].items()} == {
                p: i['artifact_id'] for p, i in identities.items()}
            for run in value['generation_runs']:
                if run['N'] == 1024:
                    assert read(path.parent / run['tokens_file']) == reference
            rounds.append((value, ownership))
        assert len(rounds) == 3 and len({v['pid'] for v, _ in rounds}) == 3
        assert {v['round_index'] for v, _ in rounds} == {0, 1, 2}
        assert len({v['checkpoint']['checkpoint_sha256'] for v, _ in rounds}) == 1
        row = dict(model=config['model'], batch=batch, fresh_processes=3,
                   c1_pass=True, c2_pass=True, config=str(config_path.relative_to(ROOT)),
                   checkpoint_sha256=rounds[0][0]['checkpoint']['checkpoint_sha256'],
                   acceptance=';'.join(sorted({o['acceptance'] for _, o in rounds})))
        for source, target, scale in [('tpot_seconds', 'tpot_ms', 1000),
                                      ('ttft_seconds', 'ttft_ms', 1000),
                                      ('e2e_seconds', 'e2e_ms', 1000),
                                      ('output_tokens_per_second', 'tokens_per_second', 1)]:
            values = [v[source] * scale for v, _ in rounds]
            row[target] = statistics.median(values)
            row[target + '_range_pct'] = 100 * (max(values) - min(values)) / row[target]
        rows.append(row)
    return rows


def dnn_cells(raw, states):
    progress = read(raw / 'final-dnn/progress.json')
    rows = []
    for model, cell in progress['settled'].items():
        if cell['status'] != 'final_three_process_measurement_queued':
            continue
        if not all(states.get(name, {}).get('status') == 'done' for name in cell['jobs']):
            continue
        config_path = ROOT / f'configs/dnn/sm90r1/{model}_b1.json'
        config = read(config_path)
        assert file_sha(config_path) == progress['published'][model]['config_sha256']
        gate = read(cell['gate'])
        assert gate['passed'] and not gate['synthetic_weights']
        assert {1, 2} <= {v['mode'] for v in gate['cases']}
        rounds = []
        for index, name in enumerate(cell['jobs']):
            state = states[name]
            directory = Path(state['out'])
            value, ownership = validate_dnn(directory / 'measurements/measurements.json',
                                            completed_guard(directory, state))
            assert value['artifact_id'] == config['artifact_id'] == gate['artifact_id']
            assert value['input_sha256'] == gate['input_tensors_sha256']
            assert value['source_checkpoint_sha256'] == gate['source_checkpoint_sha256']
            assert value['numerical_gate_sha256'] == file_sha(cell['gate'])
            assert value['executor'] == config['executor'] and value['batch'] == 1
            # The first BERT jobs omitted round-index; names and distinct PIDs
            # preserve the actual three-process protocol without rewriting data.
            assert value['round_index'] == index or model == 'bert'
            rounds.append((value, ownership))
        assert len(rounds) == 3 and len({v['pid'] for v, _ in rounds}) == 3
        row = dict(model=model, batch=1, precision='bf16', fresh_processes=3,
                   bf16_gate_pass=True, artifact_id=config['artifact_id'],
                   config=str(config_path.relative_to(ROOT)),
                   acceptance=';'.join(sorted({o['acceptance'] for _, o in rounds})),
                   source_checkpoint_sha256=gate['source_checkpoint_sha256'],
                   exported_program_sha256=gate['archive_sha256'],
                   round_metadata='job names; BERT raw indices are all 0' if model == 'bert' else '0,1,2')
        for metric in ('median_ms', 'p90_ms'):
            values = [v[metric] for v, _ in rounds]
            row[metric] = statistics.median(values)
            row[metric + '_range_pct'] = 100 * (max(values) - min(values)) / row[metric]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'runs/sm90r1')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    states = {}
    for gpu in range(8):
        states.update(read(args.raw / f'scheduler/gpu{gpu}/state.json'))
    llm, dnn = llm_cells(args.raw, states), dnn_cells(args.raw, states)
    native_llm = {(v['model'], int(v['batch'])): v for v in table('R16_final_llm.csv')}
    native_dnn = {v['model']: v for v in table('R16_final_dnn.csv')}
    baselines = {(v['model'], int(v['batch']), v['system']): v
                 for v in table('R16_final_baselines.csv')}
    for value in table('R16_moe_paired_external.csv'):
        system = 'tensorrt-llm' if value['system'] == 'trtllm' else value['system']
        baselines[value['model'], int(value['batch']), system] = value
    llm_ratios, dnn_ratios = [], []
    for row in llm:
        previous = native_llm[row['model'], row['batch']]
        row['speedup_tpot_vs_tm_r16'] = float(previous['tpot_ms']) / row['tpot_ms']
        row['speedup_e2e_vs_tm_r16'] = float(previous['e2e_ms']) / row['e2e_ms']
        for (model, batch, system), baseline in baselines.items():
            if (model, batch) != (row['model'], row['batch']):
                continue
            if baseline.get('checkpoint_sha256'):
                assert baseline['checkpoint_sha256'] == row['checkpoint_sha256']
            comparison = dict(model=model, batch=batch, baseline=system,
                              checkpoint_paired=bool(baseline.get('checkpoint_sha256')),
                              acceptance=row['acceptance'])
            for metric in ('tpot_ms', 'ttft_ms', 'e2e_ms'):
                comparison['speedup_' + metric.removesuffix('_ms')] = float(baseline[metric]) / row[metric]
            comparison['speedup_throughput'] = row['tokens_per_second'] / float(baseline['tokens_per_second'])
            llm_ratios.append(comparison)
    for row in dnn:
        row['speedup_vs_tm_r16'] = float(native_dnn[row['model']]['median_ms']) / row['median_ms']
        for baseline in table('R16_final_dnn_baselines.csv'):
            if baseline['model'] != row['model']:
                continue
            if baseline['source_checkpoint_sha256']:
                assert baseline['source_checkpoint_sha256'] == row['source_checkpoint_sha256']
            assert baseline['exported_program_sha256'] == row['exported_program_sha256']
            dnn_ratios.append(dict(model=row['model'], baseline=baseline['system'],
                                   precision=baseline['precision'], compile_mode=baseline['compile_mode'],
                                   same_precision=baseline['precision'] == 'bf16',
                                   checkpoint_scope=baseline['checkpoint_scope'],
                                   speedup=float(baseline['median_ms']) / row['median_ms'],
                                   speedup_p90=float(baseline['p90_ms']) / row['p90_ms']))
    if args.require_complete:
        assert len(llm) == 30 and len(dnn) == 5, 'final matrix is incomplete'
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix, rows in [('llm', llm), ('dnn', dnn), ('llm_ratios', llm_ratios), ('dnn_ratios', dnn_ratios)]:
        write(args.out / f'SM90R1_final_{suffix}.csv', rows)
    write(args.out / 'SM90R1_summary.csv', aggregate_ratios(llm, llm_ratios, dnn, dnn_ratios))
    print(json.dumps(dict(llm_complete=len(llm), dnn_complete=len(dnn),
                          final_complete=len(llm) == 30 and len(dnn) == 5)))


if __name__ == '__main__':
    main()
