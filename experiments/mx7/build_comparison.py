"""Combine the once-collected baselines and actual selected TP measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from baseline_results import checked, frozen_baselines

SYSTEMS = ('vllm', 'sglang', 'torchcompile')
METRICS = ('tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_native(path):
    result = {}
    for row in csv.DictReader(path.open()):
        if row['record_type'] == 'final_two_fresh_processes':
            result[int(row['tp']), int(row['batch'])] = row
    return result


def build(reference, native_path, development_path):
    baselines = frozen_baselines(reference)
    config = json.loads(reference.read_text())
    native = read_native(native_path)
    development = {(int(r['tp']), int(r['batch'])): r for r in csv.DictReader(development_path.open())}
    hashes = dict(baseline_reference_sha256=sha(reference), frozen_baseline_csv_sha256=config['reference_csv']['sha256'],
                  native_csv_sha256=sha(native_path), development_csv_sha256=sha(development_path), generator_sha256=sha(__file__))
    snapshot = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    rows = []
    for tp in (2, 4):
        for batch in (1, 2, 4, 8, 16):
            actual = native.get((tp, batch))
            row = dict(record_type='derived_comparison_not_a_measurement', evidence='verified', model='qwen3_1_7b',
                       tp=tp, batch=batch, contract='bf16_greedy', isl=64, osl=1024,
                       tilemega_status='selected_final' if actual else 'pending_selected_final')
            for metric in METRICS:
                row['tilemega_' + metric] = actual[metric] if actual else ''
            groups = {}
            for system in SYSTEMS:
                baseline = baselines[tp, batch, system]
                assert baseline['model'] == row['model'] and baseline['contract'] == row['contract']
                assert baseline['isl'] == '64' and baseline['osl'] == '1024'
                records = json.loads(baseline['measurements_json'])
                assert len(records) == 2
                physical = set()
                for record in records:
                    metrics = json.loads(checked(record['metrics']).read_text())
                    assert metrics['tp'] == tp and metrics['batch'] == batch
                    physical.add(tuple(metrics['gpu_indices']))
                assert len(physical) == 1
                groups[system] = list(physical.pop())
                for metric in METRICS:
                    row[system + '_' + metric] = baseline[metric]
                row['speedup_vs_' + system] = float(baseline['tpot_ms']) / float(actual['tpot_ms']) if actual else ''
                row[system + '_tp1_frozen_tpot_ms'] = baseline['tp1_frozen_tpot_ms']
                row[system + '_scaling_efficiency'] = baseline['scaling_efficiency']
            fastest = min(SYSTEMS, key=lambda system: float(row[system + '_tpot_ms']))
            row.update(fastest_baseline_system=fastest, fastest_baseline_tpot_ms=row[fastest + '_tpot_ms'],
                       speedup_vs_fastest=row['speedup_vs_' + fastest], native_gpu_indices=actual['gpu_indices'] if actual else '',
                       baseline_gpu_indices_json=json.dumps(groups, sort_keys=True),
                       tilemega_tp1_frozen_tpot_ms=baselines[tp, batch, 'vllm']['tp1_tilemega_frozen_tpot_ms'],
                       tilemega_scaling_efficiency=actual['scaling_efficiency'] if actual else '',
                       native_artifact_sha256=actual['artifact_sha256'] if actual else '',
                       native_pin=actual['pin'] if actual else '', native_selection_results=actual['selection_results'] if actual else '')
            dev = development.get((tp, batch))
            for metric in METRICS:
                row['tilemega_development_' + metric] = dev[metric] if dev else ''
            row['development_artifact'] = dev['artifact'] if dev else ''
            row.update(**hashes, snapshot_sha256=snapshot)
            rows.append(row)
    return rows


def markdown(rows):
    fmt = lambda value: f'{float(value):.3f}' if value != '' else '—'
    lines = ['| TP | BS | vLLM ms | SGLang ms | torch.compile ms | TileMega ms | vLLM/TM | SGLang/TM | compile/TM |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    columns = ['vllm_tpot_ms', 'sglang_tpot_ms', 'torchcompile_tpot_ms', 'tilemega_tpot_ms',
               'speedup_vs_vllm', 'speedup_vs_sglang', 'speedup_vs_torchcompile']
    for row in rows:
        lines.append(f"| {row['tp']} | {row['batch']} | " + ' | '.join(fmt(row[key]) for key in columns) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('configs/bench/mx7_frozen_baselines.json'))
    parser.add_argument('--native', type=Path, default=Path('docs/results/MX7_tp_final_c550.csv'))
    parser.add_argument('--development', type=Path, default=Path('docs/results/MX7_tp_development_c550.csv'))
    parser.add_argument('--out', type=Path, default=Path('docs/results/MX7_tp_comparison_c550.csv'))
    args = parser.parse_args()
    rows = build(args.reference, args.native, args.development)
    old = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    snapshots = {r['snapshot_sha256'] for r in old}
    if rows[0]['snapshot_sha256'] not in snapshots:
        with args.out.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            if not old:
                writer.writeheader()
            writer.writerows(rows)
    print(markdown(rows))
    print('Selected TileMega cells:', sum(r['tilemega_status'] == 'selected_final' for r in rows), '/10')
    print('Snapshot:', rows[0]['snapshot_sha256'])


if __name__ == '__main__':
    main()
