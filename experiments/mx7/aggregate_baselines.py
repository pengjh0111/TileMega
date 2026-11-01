"""Append completed two-process TP baseline cells without rerunning TP1."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def source_record(folder):
    source = folder / 'source_identity.json'
    if source.exists():
        identity = json.loads(source.read_text())
        archive = folder / 'source_snapshot'
    else:
        source = Path('runs/mx7/sync/baseline_initial_source/identity.json')
        identity = json.loads(source.read_text())
        archive = source.parent
    for name, expected in identity['source_files'].items():
        if sha(archive / name) != expected:
            raise ValueError('baseline archived source differs: ' + name)
    return ref(source)


def aggregate(root, tp):
    frozen = {int(row['batch']): row for row in csv.DictReader(
        open('docs/results/MX5_llm_performance_snapshot_c550.csv')) if row['model'] == 'qwen3_1_7b'}
    rows = []
    for system in ('vllm', 'sglang', 'torchcompile'):
        for batch in (1, 2, 4, 8, 16):
            folders = [root / f'tp{tp}' / system / 'qwen3_1_7b_rand' / f'B{batch}' / f'round{r}'
                       for r in (1, 2)]
            if any(not (folder / 'exit_code.txt').exists() for folder in folders):
                continue
            codes = [int((folder / 'exit_code.txt').read_text()) for folder in folders]
            guards = [json.loads((folder / 'guard_result.json').read_text()) for folder in folders]
            if any(guard['code'] == 75 or guard['code'] == 124 for guard in guards):
                print('guard remedy required', tp, system, batch)
                continue
            if any('mxkwCreateQueueBlock' in (folder / 'run.log').read_text(errors='replace') or
                   'mxc_queue_acquire failed' in (folder / 'run.log').read_text(errors='replace')
                   for folder in folders):
                print('platform queue failure; no qualified baseline', tp, system, batch)
                continue
            records, errors = [], []
            for folder, code in zip(folders, codes):
                if code:
                    lines = (folder / 'run.log').read_text(errors='replace').splitlines()
                    error = next((line for line in lines if 'ValueError:' in line or 'RuntimeError:' in line),
                                 next((line for line in lines if 'Error:' in line), 'child exit ' + str(code)))
                    errors.append(dict(returncode=code, original_error=error,
                                       log=ref(folder / 'run.log'), guard=ref(folder / 'guard_result.json'),
                                       source=source_record(folder)))
                    continue
                metrics = folder / f'B{batch}' / 'measurements.json'
                value = json.loads(metrics.read_text())
                if value['tp'] != tp or value['batch'] != batch or value['max_tokens'] != 1024:
                    raise ValueError('baseline result scope differs')
                raw = value['generation_runs']
                for count in (1024, 1):
                    samples = [row for row in raw if row['N'] == count]
                    if len(samples) != 3 or sum(not row['warmup'] for row in samples) != 2:
                        raise ValueError('baseline must have warm1 + timed2')
                physical = list(map(int, guards[len(records)]['gpu_indices'])) if not errors else value['gpu_indices']
                if physical not in ([4, 5], [6, 7], [4, 5, 6, 7]) or len(physical) != tp:
                    raise ValueError('invalid baseline GPU group')
                records.append(dict(metrics=ref(metrics), source=source_record(folder),
                                    tpot_ms=value['tpot_seconds'] * 1000,
                                    ttft_ms=value['ttft_seconds'] * 1000,
                                    e2e_ms=value['e2e_seconds'] * 1000,
                                    tokens_per_s=value['output_tokens_per_second']))
            row = dict(record_type='final_two_fresh_processes', evidence='verified', model='qwen3_1_7b',
                       system=system, tp=tp, batch=batch, status='N/A' if errors else 'pass',
                       contract='bf16_greedy', isl=64, osl=1024, warmup=1, repeats=2,
                       fresh_processes=2, tpot_ms='', ttft_ms='', e2e_ms='', tokens_per_s='',
                       tp1_frozen_tpot_ms=frozen[batch][system + '_tpot_ms'],
                       tp1_tilemega_frozen_tpot_ms=frozen[batch]['tpot_ms'], scaling_efficiency='',
                       round1_tpot_ms=records[0]['tpot_ms'] if len(records) == 2 else '',
                       round2_tpot_ms=records[1]['tpot_ms'] if len(records) == 2 else '',
                       measurements_json=json.dumps(records), errors_json=json.dumps(errors))
            if not errors:
                for key in ('tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s'):
                    row[key] = statistics.median(record[key] for record in records)
                row['scaling_efficiency'] = float(row['tp1_frozen_tpot_ms']) / (tp * row['tpot_ms'])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/mx7/baselines'))
    parser.add_argument('--tp', type=int, choices=(2, 4), required=True)
    parser.add_argument('--out', type=Path, default=Path('docs/results/MX7_tp_baselines_c550.csv'))
    args = parser.parse_args()
    rows = aggregate(args.root, args.tp)
    old = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    existing = {(row['system'], int(row['tp']), int(row['batch'])) for row in old
                if row['record_type'] == 'final_two_fresh_processes'}
    fresh = [row for row in rows if (row['system'], row['tp'], row['batch']) not in existing]
    if fresh:
        with args.out.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fresh[0]), lineterminator='\n')
            if not old:
                writer.writeheader()
            writer.writerows(fresh)
    for row in rows:
        print(row['system'], 'TP'+str(row['tp']), 'B'+str(row['batch']), row['status'], row['tpot_ms'])


if __name__ == '__main__':
    main()
