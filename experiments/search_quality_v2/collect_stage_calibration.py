"""Reduce successful phase-1 probe samples without using full-program TPOT."""

import csv
import io
import json
import math
from pathlib import Path
import statistics

from calibration_queue import CALIB, CASE
from guarded import ROOT, read, sha, write


METRICS = {
    'gemm': ('head_entry_ns', 'head_body_ns', 'steady_ns', 'stage_entry_ns', 'stage_body_ns',
             'kernel_gpu_event_ns', 'cta_weight_bytes_per_ns'),
    'attention': ('stage_entry_ns', 'stage_body_ns', 'merge_ns', 'combined_body_ns',
                  'kernel_gpu_event_ns'),
}
CONTEXTS = (64, 320, 575, 831, 1086)


def csv_rows(path):
    lines = Path(path).read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('geometry_id,'))
    width = len(next(csv.reader([lines[header]])))
    return list(csv.DictReader(io.StringIO('\n'.join(
        line for line in lines[header:] if len(next(csv.reader([line]), [])) == width
    ))))


def metric_median(rows, metric):
    values = [float(row[metric]) for row in rows]
    if all(math.isnan(v) for v in values) and metric == 'steady_ns':
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f'bad value for {metric}: {values}')
    return statistics.median(values)


def epilogue_cohort():
    state = read(CALIB / 'legacy/dense-measurement/state.json')
    cohorts = {}
    for repeat in range(5):
        result = state[f'constants-epilogue-f{repeat}']
        assert read(Path(result['out']) / 'guard_result.json')['code'] == 0
        path = Path(result['log'])
        with path.open() as handle:
            rows = [line for line in handle if line.startswith('t_ep,')]
        for row in csv.reader(rows):
            _, key, _, median, _, unit = row
            assert unit == 'ns' and float(median) > 0
            cohorts.setdefault(key, []).append(float(median))
    assert cohorts and all(len(rows) == 5 for rows in cohorts.values())
    return cohorts


def collect(kind, binding, epilogue):
    folder = CALIB / kind
    jobs = read(folder / 'queue.json')
    state = read(folder / 'measurement/state.json')
    desc = {r['id']: r for r in read(folder / 'geometries.json')}
    cohort = {}
    for job in jobs:
        result = state[job['id']]
        assert result['status'] == 'done'
        assert read(Path(result['out']) / 'guard_result.json')['code'] == 0
        receipt = read(Path(job['build_attempt']) / 'build.json')
        assert receipt['binary_sha256'] == job['binary_sha256'] == sha(job['command'][0])
        assert all(receipt['header_sha256'][path] == value for path, value in binding.items())
        rows = csv_rows(result['log'])
        grouped = {}
        for row in rows:
            key = (row['geometry_id'], int(row['past']) if kind == 'attention' else None)
            grouped.setdefault(key, []).append(row)
            assert int(row['grid']) == 132 * desc[key[0]]['residency']
        contexts = CONTEXTS if kind == 'attention' else (None,)
        assert set(grouped) == {(g, past) for g in job['geometry_ids'] for past in contexts}
        for key, values in grouped.items():
            assert sorted(int(row['repeat']) for row in values) == list(range(5, 17))
            record = {metric: metric_median(values, metric) for metric in METRICS[kind]}
            cohort.setdefault(key, []).append({
                'job': job['id'], 'process_round': job['process_round'], 'uuid': result['gpu_uuid'],
                'source_log_sha256': sha(result['log']), **record,
            })

    header_key = ('include/tilemega/Backend/Sm90DenseGemm.h' if kind == 'gemm'
                  else 'include/tilemega/Backend/Sm90DenseAttentionBody.cuh')
    table = []
    for (key, past), process in sorted(cohort.items()):
        assert sorted(r['process_round'] for r in process) == [0, 1, 2]
        r = desc[key]
        row = {'geometry_id': key, 'model': r['model'], 'residency': r['residency']}
        if kind == 'gemm':
            row.update({k: r[k] for k in ('family', 'n', 'k', 'tile_k', 'swiglu_u')})
            row.update(r['geometry'])
            family = 'residual' if r['family'] in ('o', 'down') else r['family']
            ep_key = f'{family}:tn={row["tile_n"]}:b=1:wk={row["warps_k"]}'
            ep_samples = epilogue[ep_key]
            row['epilogue_ns'] = statistics.median(ep_samples)
            row['epilogue_ns_process_range'] = max(ep_samples) - min(ep_samples)
            row['epilogue_process_medians'] = json.dumps(ep_samples, separators=(',', ':'))
            row['epilogue_header_sha256'] = binding['include/tilemega/Backend/Sm90DenseEpilogue.h']
        else:
            row.update({k: r[k] for k in ('hkv', 'head_dim', 'queries', 'norm')})
            row.update(r['attention'], past=past)
        for metric in METRICS[kind]:
            samples = [p[metric] for p in process]
            if all(v is None for v in samples):
                row[metric] = None
                row[metric + '_process_range'] = None
            else:
                assert all(v is not None for v in samples), (key, metric)
                row[metric] = statistics.median(samples)
                row[metric + '_process_range'] = max(samples) - min(samples)
        row['header_sha256'] = binding[header_key]
        row['process_medians'] = json.dumps(process, separators=(',', ':'))
        table.append(row)
    assert len(table) == len(desc) * (len(CONTEXTS) if kind == 'attention' else 1)
    output = CASE / f'calib_{kind}.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    return output, len(table)


def main():
    bound = read(CALIB / 'body_binding.json')['header_sha256']
    immutable = read(CALIB / 'production_source_binding.json')['production_headers']
    for path, expected in immutable.items():
        assert sha(ROOT / path) == expected, path
    source = ROOT / 'configs/targets/sm_90_h800.json'
    target = read(source)
    constants = read(CALIB / 'legacy/dense-measurement/sm90_native_constants.json')
    for path, expected in constants['header_sha256'].items():
        assert sha(ROOT / path) == expected, path
    epilogue = epilogue_cohort()
    table = dict(collect(kind, bound, epilogue) for kind in METRICS)
    constants['method'] = (
        'Current SM90R3 dense constants probes. Five fresh guarded processes per suite. '
        'SQ2 GEMM and attention geometry rates are supplied by the SHA-bound stage tables.'
    )
    target['sm90_native_constants'] = constants
    target['sm90_sq2_calibration'] = {
        'scope': ['qwen3_1_7b B1 TP1', 'llama3_1_8b B1 TP1'],
        'header_sha256': bound, 'base_target_sha256': sha(source),
        'tables': {path.name: {'sha256': sha(path), 'rows': size} for path, size in table.items()},
    }
    destination = ROOT / 'configs/targets/sm_90_h800_sq2.json'
    write(destination, target)
    gemm = list(csv.DictReader((CASE / 'calib_gemm.csv').open()))
    catalog = read(CALIB / 'gemm/geometries.json')
    check = {}
    for q in ('q079', 'q135'):
        matches = [d['id'] for d in catalog if d['model'] == 'qwen3_1_7b' and d['family'] == 'o'
                   and q in d['sources']]
        assert len(matches) == 1, (q, matches)
        row = next(r for r in gemm if r['geometry_id'] == matches[0])
        check[q] = {k: row[k] for k in ('geometry_id', 'tile_n', 'tile_k', 'warps_n', 'warps_k',
                                       'stage_body_ns', 'stage_body_ns_process_range')}
    check['agrees_with_v1_tpot_direction'] = (
        float(check['q135']['stage_body_ns']) < float(check['q079']['stage_body_ns']))
    write(CALIB / 'collection.json', {'tables': target['sm90_sq2_calibration']['tables'],
                                     'target': str(destination), 'target_sha256': sha(destination),
                                     'o_projection_sanity': check})
    progress = CASE / 'progress.md'
    text = progress.read_text().replace('Phase 1 — in progress', 'Phase 1 — complete')
    text += (
        f'\n- Accepted {table[CASE / "calib_gemm.csv"]} GEMM geometry rows and '
        f'{table[CASE / "calib_attention.csv"]} attention geometry/context rows, '
        'each from three fresh guarded processes. The dense legacy constants use five.\n'
        f'- o_proj makespan for q079 and q135 was {check["q079"]["stage_body_ns"]} ns '
        f'and {check["q135"]["stage_body_ns"]} ns. Agreement with v1 TPOT direction: '
        f'{check["agrees_with_v1_tpot_direction"]}. This is diagnostic only.\n'
    )
    progress.write_text(text)


if __name__ == '__main__':
    main()
