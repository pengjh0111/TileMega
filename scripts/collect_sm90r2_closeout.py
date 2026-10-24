#!/usr/bin/env python3
"""Collect guarded TileMega-only closeout runs and frozen-baseline ratios."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from sm90d1_common import ROOT, read, sha, write
from r16_measurement_validation import completed_guard, telemetry, ownership, validate_native_repetitions
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.serving.baseline_common import summarize, check_tokens


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def span(values):
    return (max(values) - min(values)) / statistics.median(values) * 100


def sample(record):
    folder = Path(record['out'])
    guard = completed_guard(folder, record)
    matches = list((folder / 'job/measurements').rglob('measurements.json'))
    if len(matches) != 1:
        raise ValueError('each closeout process must produce exactly one measurement')
    path = matches[0]; value = read(path)
    if value.get('evidence') != 'verified':
        raise ValueError('measurement lacks execution evidence')
    samples = telemetry(Path(value['guard_log']))
    own = ownership(guard, samples)
    identity = read(folder / 'job/physical-device-identity.json')
    if identity['gpu_index'] != value['gpu_index'] or identity['gpu_index'] in (5, 6):
        raise ValueError('physical GPU identity differs or timing card is excluded')
    selected = read(folder / 'job/selected-config.json')
    if selected['stage'] != 'measure':
        raise ValueError('only uninstrumented E2E measurements are eligible')
    if 'generation_runs' in value:
        batch = value['batch']; rows = value['generation_runs']
        if value['max_tokens'] != 1024 or value['prompt_length'] != 64 or len(rows) != 8:
            raise ValueError('LLM workload is not ISL64/OSL1024')
        for count in (1, 1024):
            group = [r for r in rows if r['N'] == count]
            if len(group) != 4 or {r['run'] for r in group} != {0, 1, 2, 3}:
                raise ValueError('one warmup and three repetitions are required')
            for r in group:
                if r['warmup'] is not (r['run'] == 0) or not math.isfinite(r['wall_seconds']) or r['wall_seconds'] <= 0:
                    raise ValueError('invalid warmup order or duration')
                tokens = path.parent / r['tokens_file']
                if tokens.resolve().parent != path.parent.resolve():
                    raise ValueError('token evidence escapes its measurement directory')
                check_tokens(read(tokens), batch, count)
                for boundary in ('before', 'after'):
                    if not any(s['label'] == f"N{count}-run{r['run']}-{boundary}" for s in samples):
                        raise ValueError('request boundary telemetry is missing')
        # These files already exist from timed requests; this adds no GPU test.
        validate_native_repetitions(path, value)
        recomputed = summarize(rows, batch, 1024)
        if any(not math.isclose(value[k], v, rel_tol=1e-12) for k, v in recomputed.items()):
            raise ValueError('summary differs from timed requests')
        if value['configuration']['decode_sha256'] != selected['binary_sha256']['decode']:
            raise ValueError('timed decode differs from the frozen artifact')
        key = ('llm', selected['model'], batch)
    else:
        if (value['batch'] != 1 or value['precision'] != 'bf16'
                or value['measurement']['warmup'] != 50 or value['measurement']['iterations'] != 200):
            raise ValueError('DNN workload is not BF16 B1 warm50/event200')
        values = value['samples_ms']
        if len(values) != 200 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('invalid CUDA event samples')
        if (not math.isclose(value['median_ms'], statistics.median(values), rel_tol=1e-12)
                or not math.isclose(value['p90_ms'], sorted(values)[179], rel_tol=1e-12)):
            raise ValueError('DNN summary differs from its CUDA events')
        key = ('dnn', value['model'], 1)
    return key, dict(path=str(path), measurement=value, selected=selected,
                     physical=identity, ownership=own, guard_sha256=sha(folder / 'guard_result.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    grouped, failures = {}, []
    for state in args.state:
        for name, record in read(state).items():
            try:
                key, value = sample(record)
                grouped.setdefault(key, []).append(value)
            except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
                failures.append(dict(state=str(state), job=name, status=record['status'], error=str(error)))
    llm, dnn, details = [], [], []
    for (kind, model, batch), values in sorted(grouped.items()):
        required = 5 if model == 'llama3_1_8b' else 3
        rounds = [v['measurement']['round_index'] for v in values]
        if len(values) != required or set(rounds) != set(range(required)):
            failures.append(dict(state='', job=f'{model}-B{batch}', status='incomplete',
                                 error=f'expected {required} distinct fresh processes, got {rounds}'))
            continue
        if len({v['measurement']['pid'] for v in values}) != required:
            raise ValueError('fresh process count is not supported by distinct worker PIDs')
        uuids = {v['physical']['cuda_uuid'] for v in values}
        if len(uuids) != 1:
            raise ValueError('fresh processes for a cell must use the same physical card')
        common = dict(model=model, batch=batch, fresh_processes=required,
                      gpu=values[0]['physical']['gpu_index'], gpu_uuid=next(iter(uuids)),
                      ownership_verified=all(v['ownership']['ownership_verified'] for v in values),
                      acceptance='provisional_container_pid_namespace' if any(
                          not v['ownership']['ownership_verified'] for v in values) else 'strict',
                      correctness='not_performed', e3_validated=False,
                      concurrency='user-authorized parallel cohort; consistency not measured')
        measurements = [v['measurement'] for v in values]
        if kind == 'llm':
            if (len({json.dumps(v['selected']['binary_sha256'], sort_keys=True) for v in values}) != 1
                    or len({v['model_config_sha256'] for v in measurements}) != 1
                    or len({v['prompt_ids_sha256'] for v in measurements}) != 1
                    or len({v.get('checkpoint', {}).get('checkpoint_sha256') for v in measurements}) != 1):
                raise ValueError('fresh processes used different artifacts or model inputs')
            times = [v['tpot_seconds'] * 1000 for v in measurements]
            row = dict(**common, tpot_ms=statistics.median(times), process_range_pct=span(times),
                       ttft_ms=statistics.median(v['ttft_seconds'] * 1000 for v in measurements),
                       e2e_ms=statistics.median(v['e2e_seconds'] * 1000 for v in measurements),
                       tokens_per_second=statistics.median(v['output_tokens_per_second'] for v in measurements),
                       selection=values[0]['selected']['selection'],
                       decode_sha256=values[0]['selected']['binary_sha256']['decode'],
                       prefill_sha256=values[0]['selected']['binary_sha256']['prefill'],
                       source_heads=json.dumps(values[0]['selected']['source_heads'], sort_keys=True),
                       manifest_sha256=values[0]['selected']['manifest_sha256'])
            llm.append(row)
        else:
            if len({v['artifact_id'] for v in measurements}) != 1 or len({v['input_sha256'] for v in measurements}) != 1:
                raise ValueError('DNN fresh processes used different artifacts or inputs')
            times = [v['median_ms'] for v in measurements]
            dnn.append(dict(**common, median_ms=statistics.median(times), process_range_pct=span(times),
                            p90_ms=statistics.median(v['p90_ms'] for v in measurements),
                            artifact_id=measurements[0]['artifact_id'],
                            source_head=measurements[0]['identity']['source']['head']))
        details.append(dict(kind=kind, model=model, batch=batch, processes=values))
    baseline_path = ROOT / 'docs/results/R16_final_baselines.csv'
    dnn_baseline_path = ROOT / 'docs/results/R16_final_dnn_baselines.csv'
    bases, dbases = table(baseline_path), table(dnn_baseline_path)
    ratios, dratios = [], []
    for current in llm:
        matched = [b for b in bases if b['model'] == current['model'] and int(b['batch']) == current['batch']]
        engines = [b for b in matched if b['system'] in ('vllm', 'sglang', 'tensorrt-llm')]
        comparisons = [(b['system'], b) for b in matched]
        if engines:
            comparisons.append(('fastest_engine_tpot', min(engines, key=lambda b: float(b['tpot_ms']))))
        for system, base in comparisons:
            ratios.append(dict(model=current['model'], batch=current['batch'], system=system,
                reference_system=base['system'],
                tpot_speedup=float(base['tpot_ms']) / current['tpot_ms'],
                ttft_speedup=float(base['ttft_ms']) / current['ttft_ms'],
                e2e_speedup=float(base['e2e_ms']) / current['e2e_ms'],
                throughput_ratio=current['tokens_per_second'] / float(base['tokens_per_second']),
                reference='frozen historical baseline; no same-session anchor',
                baseline_file_sha256=sha(baseline_path)))
    for current in dnn:
        matched = [b for b in dbases if b['model'] == current['model'] and int(b['batch']) == 1]
        comparisons = [(b['system'], b) for b in matched]
        bf16 = [b for b in matched if b['precision'] == 'bf16']
        if bf16:
            comparisons.append(('fastest_bf16', min(bf16, key=lambda b: float(b['median_ms']))))
        for system, base in comparisons:
            dratios.append(dict(model=current['model'], batch=1, system=system,
                reference_system=base['system'],
                precision=base['precision'], compile_mode=base.get('compile_mode', ''),
                median_speedup=float(base['median_ms']) / current['median_ms'],
                p90_speedup=float(base['p90_ms']) / current['p90_ms'],
                reference='frozen historical baseline; no same-session anchor',
                baseline_file_sha256=sha(dnn_baseline_path)))
    geomeans = []
    for scope in ('dense25', 'moe5', 'all30', *sorted({r['model'] for r in llm})):
        for system in sorted({r['system'] for r in ratios}):
            values = [r for r in ratios if r['system'] == system and (
                scope == 'all30' or scope == r['model'] or
                scope == 'dense25' and r['model'] != 'qwen3_30b_a3b' or
                scope == 'moe5' and r['model'] == 'qwen3_30b_a3b')]
            if not values:
                continue
            expected = {'dense25': 25, 'moe5': 5, 'all30': 30}.get(scope, 5)
            row = dict(scope=scope, system=system, cells=len(values), expected_cells=expected,
                       complete=len(values) == expected,
                       min_tpot_speedup=min(r['tpot_speedup'] for r in values))
            for metric in ('tpot_speedup', 'ttft_speedup', 'e2e_speedup', 'throughput_ratio'):
                row[metric + '_geomean'] = math.exp(statistics.mean(math.log(r[metric]) for r in values))
            geomeans.append(row)
    save_csv(args.out / 'SM90R2_closeout_llm.csv', llm)
    save_csv(args.out / 'SM90R2_closeout_dnn.csv', dnn)
    save_csv(args.out / 'SM90R2_closeout_llm_ratios.csv', ratios)
    save_csv(args.out / 'SM90R2_closeout_dnn_ratios.csv', dratios)
    save_csv(args.out / 'SM90R2_closeout_geomeans.csv', geomeans)
    receipt = dict(evidence='verified collection of completed executions only', llm_cells=len(llm),
                   dnn_cells=len(dnn), e3_validated=False, solver_final_claimed=False,
                   external_baselines_remeasured=False, states=[dict(path=str(s), sha256=sha(s)) for s in args.state],
                   collector_sha256=sha(__file__), failures=failures, processes=details)
    write(args.out / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('processes', 'failures')}))
    print(json.dumps(dict(failures=failures)))
    return 0 if len(llm) == 30 and len(dnn) == 5 and not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
