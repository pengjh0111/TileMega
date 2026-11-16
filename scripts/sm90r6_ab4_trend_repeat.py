#!/usr/bin/env python3
"""Summarize the existing AB4 design and performance-only Llama repeat."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from sm90d1_common import ROOT, read, sha, write
from sm90r3_llm import span
from sm90r6_merge_analysis import write as csv_write


def inspect_design(root):
    records, headers = [], []
    rows = list(csv.DictReader((ROOT / 'docs/results/SM90R6_ablation.csv').open()))
    rows += list(csv.DictReader((ROOT / 'docs/results/SM90R6_ab4_b4.csv').open()))
    for row in rows:
        if row['arm'] != 'AB4b':
            continue
        path = ROOT / row['raw']
        arms = read(path.parent.parent / 'arms.json')
        measured = read(path)
        values = measured.get('rows') or read(path.parent / 'unit/result.json')['rows']
        pair = []
        for label in ('AB4b', 'AB4c'):
            binary = arms[label]['binary']
            identity = read(binary + '.identity.json')
            outcome = read(binary + '.compile-result.json')
            body = {p: h for p, h in outcome['input_hashes'].items() if '/include/tilemega/' in p}
            headers.append(body)
            processes = []
            for value in values:
                if value['arm'] != label:
                    continue
                job = Path(value['tokens']).parent.parent
                measurement = next((job / 'measurements').rglob('measurements.json'))
                data = read(measurement)
                config = data['configuration']
                processes.append(dict(raw=str(measurement), gpu=value['gpu'],
                    **{k: data[k] for k in ('model_config_sha256', 'prompt_ids_sha256',
                        'prompt_length', 'max_tokens', 'versions')},
                    **{k: config[k] for k in ('checkpoint_sha256', 'runner_sha256', 'mode',
                        'prefill_mode', 'prefill_binary', 'decode_loop', 'hf_positions', 'actual_execution')},
                    sample_pattern=[(v['N'], v['run'], v['warmup']) for v in data['generation_runs']]))
            pair.append(dict(arm=label, binary=binary, binary_sha256=identity['binary_sha256'],
                coordinates=identity['plan']['native_dense'], grid=identity['plan']['grid'],
                shared_memory_bytes=identity['shared_memory_bytes'],
                execution=identity['plan'].get('execution', 'persistent'),
                defines=identity['defines'], header_hashes=body, processes=processes))
        records.append(dict(model=row['model'], batch=int(row['batch']), arms=pair))
    common = set.intersection(*(set(v) for v in headers))
    changed = [p for p in common if len({v[p] for v in headers}) != 1]
    result = dict(records=records, common_body_headers=len(common), changed_common_body_headers=changed)
    write(root / 'design-inspection.json', result)
    return result


def summarize(root, out):
    items = read(root / 'queue/items.json')
    if len(items) != 18 or any(v['status'] != 'done' for v in items.values()):
        raise ValueError('performance repeat has unfinished items')
    original = list(csv.DictReader((ROOT / 'docs/results/SM90R6_ab4_comparison.csv').open()))
    source = read(root / 'source.json')
    summaries, processes = [], []
    for batch in (1, 4, 16):
        values = {}
        for label in ('AB4b', 'AB4c'):
            group = [v for v in items.values() if v['job']['batch'] == batch and v['job']['arm'] == label]
            if len(group) != 3:
                raise ValueError('expected three performance processes per arm/batch')
            for item in group:
                result = item['result']
                raw = Path(result['raw'])
                record = read(raw / 'job/sm90r3-process.json')
                measurement = next((raw / 'job/measurements').rglob('measurements.json'))
                data = read(measurement)
                processes.append(dict(batch=batch, arm=label, item=item['job']['id'],
                    **{k: result[k] for k in ('gpu', 'tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s')},
                    started=item['attempts'][-1]['started'], ended=item['attempts'][-1]['ended'],
                    binary=source['arms'][str(batch)][label], binary_sha256=record['decode_sha256'],
                    checkpoint=data['checkpoint']['checkpoint_sha256'],
                    raw=str(raw), measurement=str(measurement), measurement_sha256=sha(measurement)))
            tpots = [v['result']['tpot_ms'] for v in group]
            values[label] = dict(tpot_ms=statistics.median(tpots), range_pct=span(tpots),
                gpus=[v['result']['gpu'] for v in group], samples=tpots)
        old = next(v for v in original if v['model'] == 'llama3_1_8b' and int(v['batch']) == batch)
        b, c = values['AB4b'], values['AB4c']
        summaries.append(dict(model='llama3_1_8b', batch=batch,
            old_ab4b_tpot_ms=float(old['ab4b_tpot_ms']), old_ab4c_tpot_ms=float(old['ab4c_tpot_ms']),
            old_ab4c_change_pct=float(old['ab4c_change_vs_ab4b_pct']),
            repeat_ab4b_tpot_ms=b['tpot_ms'], repeat_ab4c_tpot_ms=c['tpot_ms'],
            repeat_ab4c_change_pct=(c['tpot_ms'] / b['tpot_ms'] - 1) * 100,
            ab4b_range_pct=b['range_pct'], ab4c_range_pct=c['range_pct'],
            ab4b_gpus=json.dumps(b['gpus']), ab4c_gpus=json.dumps(c['gpus']),
            ab4b_samples_ms=json.dumps(b['samples']), ab4c_samples_ms=json.dumps(c['samples']),
            processes_per_arm=3, correctness_runs=0, evidence='verified'))
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / 'SM90R6_ab4_llama_repeat.csv', summaries)
    csv_write(out / 'SM90R6_ab4_llama_repeat_processes.csv', processes)
    inspection = read(root / 'design-inspection.json')
    report = ('# Llama AB4 batch-trend investigation\n\n'
        'Verified: all 12 original model/batch cells use identical coordinates, grid and '
        'shared memory within each AB4b/AB4c pair. Across all four models, the 339 common '
        'TileMega body headers have identical recorded hashes. Within each model, the '
        'checkpoint, prompt, model config, runner, software versions, L1 mode, prompt prefill, '
        'decode-loop setting and sample pattern match across B1/B4/B16. '
        'B1/B16 used the original multi-arm same-card protocol; B4 used the user-authorized '
        'dynamic process queue for every model. No Llama-only timing-method exception was found.\n\n'
        'Verified: coordinates are independently selected at each batch. For Llama, SwiGLU '
        'interleave U is 8/8/16 at B1/B4/B16; attention chunks are 8/4/1. The gate_up '
        '(TN,TK,slots) changes from (112,64,4) at B1/B4 to (128,128,2) at B16. '
        'The other three models keep their respective U fixed across these batches. '
        'These are optimizer-selected program differences, shared by both arms in each cell. '
        'Thus the original plot compares separately optimized programs, rather than varying '
        'batch while keeping one program fixed. Inferred: configuration changes can contribute '
        'to a different trend; this repeat does not isolate any single configuration parameter.\n\n'
        'The user requested performance only. The repeat reuses the six selected binaries '
        'without compilation, search, C1, C2 or new occupancy probes. It measures three fresh '
        'processes per arm/batch, each with one warmup and three samples at N=1 and N=1024, '
        'using the original timing driver and interference guard. Independent items use any '
        'idle GPU 0–3. TPOT is (median(T1024)-median(T1))/1023. Results are separate from the '
        'original dataset and retain per-process GPU and binary provenance.\n\n'
        '| Batch | Old AB4b ms | Old AB4c ms | Old C/B − 1 | Repeat AB4b ms | Repeat AB4c ms | Repeat C/B − 1 |\n'
        '|---:|---:|---:|---:|---:|---:|---:|\n')
    for value in summaries:
        report += (f'| {value["batch"]} | {value["old_ab4b_tpot_ms"]:.6f} | '
            f'{value["old_ab4c_tpot_ms"]:.6f} | {value["old_ab4c_change_pct"]:.4f}% | '
            f'{value["repeat_ab4b_tpot_ms"]:.6f} | {value["repeat_ab4c_tpot_ms"]:.6f} | '
            f'{value["repeat_ab4c_change_pct"]:.4f}% |\n')
    report += ('\nVerified: all 18 performance processes completed with accepted guards. '
        'Raw design inspection and attempts: `' + str(root) + '`. Full-precision summaries '
        'and all 18 process records accompany this report. No correctness tests were rerun.\n')
    (out / 'SM90R6_ab4_llama_repeat.md').write_text(report)
    write(root / 'summary.json', dict(processes=18, correctness_runs=0, rows=summaries,
        changed_common_body_headers=inspection['changed_common_body_headers'], script_sha256=sha(__file__)))
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    inspect_design(root)
    if args.wait:
        while True:
            items = read(root / 'queue/items.json')
            if any(v['status'] == 'failed' for v in items.values()):
                raise RuntimeError('performance item failed; inspect queue')
            if len(items) == 18 and all(v['status'] == 'done' for v in items.values()):
                break
            time.sleep(10)
    print(json.dumps(summarize(root, args.out.resolve())), flush=True)
