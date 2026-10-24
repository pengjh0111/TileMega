#!/usr/bin/env python3
"""Collect SM90R3 native constants from guarded fresh processes (prompt §4 L0-1).

Each metric is the median over processes of the per-process median; p99 is the
median of per-process p99. Writes the T5 CSV and the compact
`sm90_native_constants` section of the H800 target. No fitting is performed.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess

from sm90d1_common import ROOT, read, sha

HEADERS = ['include/tilemega/Backend/Sm90DenseEpilogue.h', 'include/tilemega/Backend/Sm90DenseToken.cuh',
           'include/tilemega/Backend/Sm90DenseAttention.h', 'include/tilemega/Backend/Sm90DenseAttentionStream.cuh',
           'include/tilemega/Backend/Sm90DenseGemm.h', 'include/tilemega/Codegen/executor/Async.cuh']
ATTENTION_FIELDS = ('batch', 'head_dim', 'queries', 'qk_norm', 'chunks', 'block_keys', 'slots',
                    'new_row_work', 'past', 'grid')


def attempt_rows(log):
    text = log.read_text().rsplit('\nATTEMPT ', 1)[-1]
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(('name,', 'metric,')))
    width = len(next(csv.reader([lines[header]])))
    body = [lines[header]] + [l for l in lines[header + 1:] if l and len(next(csv.reader([l]))) == width]
    return list(csv.DictReader(io.StringIO('\n'.join(body))))


def process_rows(state, name, processes):
    result = []
    for fresh in range(processes):
        job = f'{name}-f{fresh}'
        row = state[job]
        if row['status'] != 'done':
            raise ValueError(f'{job} is {row["status"]}')
        folder = Path(row['out'])
        if read(folder / 'guard_result.json')['code']:
            raise ValueError(f'{job} guard rejected')
        logs = sorted(Path(row['out']).parent.parent.parent.glob(f'{job}.log'))
        log = logs[0] if logs else Path(row['log'])
        result.append(attempt_rows(log))
    return result


def merge(processes, key):
    groups = {}
    for rows in processes:
        for row in rows:
            groups.setdefault(key(row), []).append(row)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--constants', nargs='+', required=True, help='job names of constants probes')
    p.add_argument('--attention', nargs='*', default=[], help='job names of attention gap probes')
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--section-only', type=Path, help='write the section JSON here instead of the target')
    a = p.parse_args()
    state = read(a.state)
    table = []
    for name in a.constants:
        groups = merge(process_rows(state, name, a.processes), lambda r: (r['name'], r['key']))
        for (metric, key), rows in groups.items():
            if len(rows) != a.processes:
                raise ValueError(f'{metric} {key}: {len(rows)} processes')
            table.append(dict(name=metric, key=key, median=statistics.median(float(r['median']) for r in rows),
                              p99=statistics.median(float(r['p99']) for r in rows), unit=rows[0]['unit'],
                              processes=len(rows), method=name))
    for name in a.attention:
        processes = process_rows(state, name, a.processes)
        processes = [[r for r in rows if r['metric'] == 'attention_critical_before_barrier' and r['past'] == '575']
                     for rows in processes]
        groups = merge(processes, lambda r: tuple(r[f] for f in ATTENTION_FIELDS))
        for fields, rows in groups.items():
            if len(rows) != a.processes:
                raise ValueError(f'attention {fields}: {len(rows)} processes')
            key = ':'.join(f'{f}={v}' for f, v in zip(ATTENTION_FIELDS, fields))
            table.append(dict(name='attention_stage', key=key,
                              median=statistics.median(float(r['median']) for r in rows),
                              p99=statistics.median(float(r['p99']) for r in rows), unit='ns',
                              processes=len(rows), method=name))
    seen = set()
    for row in table:
        if (row['name'], row['key']) in seen:
            raise ValueError(f'duplicate constant {row["name"]} {row["key"]}')
        seen.add((row['name'], row['key']))
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'key', 'median', 'p99', 'unit', 'processes', 'method'],
                                lineterminator='\n')
        writer.writeheader()
        for row in sorted(table, key=lambda r: (r['name'], r['key'])):
            writer.writerow(dict(row, median=f"{row['median']:.6g}", p99=f"{row['p99']:.6g}"))
    value = {(r['name'], r['key']): r['median'] for r in table}

    def single(metric, prefix=''):
        found = [v for (n, k), v in value.items() if n == metric and k.startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f'{metric} {prefix}: {len(found)} values')
        return round(found[0], 2)

    epilogue, embedding, argmax = {}, {}, {}
    for (metric, key), v in value.items():
        fields = dict(part.split('=') for part in key.split(':') if '=' in part)
        if metric == 't_ep':
            family = key.split(':')[0]
            if family == 'lm_head' and fields['par'] != '1':
                continue
            node = epilogue.setdefault(family, {}).setdefault('vs' + fields['vs'], {}).setdefault('k' + fields['k'], {})
            node.setdefault(fields['tn'], {})[fields['b']] = round(v, 1)
        elif metric == 't_emb':
            embedding.setdefault('g' + fields['g'], {}).setdefault('w' + fields['w'], {})[fields['b']] = round(v, 1)
        elif metric == 't_argmax':
            argmax.setdefault('g' + fields['g'], {}).setdefault('p' + fields['parts'], {})[fields['b']] = round(v, 1)
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    section = dict(
        schema=1,
        method='scripts/sm90r3_constants.cu and scripts/sm90r2_attention_calibration.cu (SM90R3 gap keys); '
               'median of five fresh guarded processes; no fitting',
        csv=str(a.csv.relative_to(ROOT)),
        source_commit=head, header_sha256={h: sha(ROOT / h) for h in HEADERS},
        c_bar_ns={g: single('c_bar', f'g={g}') for g in ('128', '132', '264')},
        l2_latency_ns=single('L_l2_loaded'), dram_latency_ns=single('L_dram_loaded'),
        l2_latency_single_ns=single('L_l2_single'), dram_latency_single_ns=single('L_dram_single'),
        dram_gbps=single('BW_dram'), l2_gbps=single('BW_l2'), sm_gbps=single('BW_sm'),
        activation_cp_async_gbps=20.0,
        activation_cp_async_source='stated: SM90R2_activation_stream_diagnostic.csv per-CTA ceiling (F9)',
        epilogue_ns=epilogue, embedding_ns=embedding, argmax_ns=argmax,
        attention_tables=['docs/results/SM90R2_attention_calibration.csv', str(a.csv.relative_to(ROOT))],
        attention_metric='attention_critical_before_barrier at past 575; stage = c_bar + table')
    if a.section_only:
        a.section_only.write_text(json.dumps(section, indent=1) + '\n')
    else:
        target = read(a.target)
        target['sm90_native_constants'] = section
        a.target.write_text(json.dumps(target, indent=2) + '\n')
    print(json.dumps(dict(rows=len(table), section_bytes=len(json.dumps(section, separators=(',', ':'))))))


if __name__ == '__main__':
    main()
