"""Compact native timelines and independent route diagnostics into final tables."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

from moe_profile_metrics import interval_union
from moe_profile_matrix import ROOT

RAW = ROOT / 'runs/moe_profile'


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'no measured rows for {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accepted(folder, artifact):
    paths = [path for path in sorted(folder.glob('attempt*'))
             if (path / 'guard/guard_result.json').exists()
             and read(path / 'guard/guard_result.json')['code'] == 0
             and (path / artifact).exists()
             and not (path / 'superseded_for_missing_kv_metadata.json').exists()]
    return paths[-1] if paths else None


def key(row):
    return tuple(str(row[name]) for name in ('kernel', 'grid', 'block', 'registers_per_thread',
                                            'static_shared_bytes', 'dynamic_shared_bytes'))


def common_stage(stages):
    if len(stages) == 1:
        return next(iter(stages))
    if stages <= {'expert_gate_up_gemm', 'expert_down_gemm', 'expert_gemm_other'}:
        return 'expert_gemm_gate_up_or_down'
    if stages <= {'qkv_projection', 'output_projection', 'router_projection', 'lm_head'}:
        return 'dense_projection_subkind_unresolved'
    return 'unresolved'


def timelines(out):
    sources = {}
    for folder in (RAW / 'attribution').glob('*_b*_i*_*'):
        attempt = accepted(folder, 'export/kernel_metrics.csv')
        if attempt:
            sources[folder.name] = (attempt, csv_rows(attempt / 'export/kernel_metrics.csv'))
    kernel_rows, stage_rows, windows = [], [], []
    for folder in sorted((RAW / 'timeline').glob('*_b*_i*_*')):
        match = re.fullmatch(r'(vllm|sglang|trtllm)_b(\d+)_i(\d+)_(prefill|decode)', folder.name)
        if not match:
            continue
        framework, batch, length, phase = match.groups()
        attempt = accepted(folder, 'export/kernel_metrics.csv')
        if not attempt:
            continue
        exact = collections.defaultdict(list)
        symbols = collections.defaultdict(list)
        for name, (source, rows) in sources.items():
            if not name.startswith(framework + '_'):
                continue
            for row in rows:
                if row['stage'] == 'unresolved':
                    continue
                proof = dict(stage=row['stage'], source=str(source), evidence=row['source_evidence'])
                symbols[row['kernel']].append(proof)
                if name == folder.name:
                    exact[key(row)].append(proof)
        mappings = {}
        measured = csv_rows(attempt / 'export/kernel_metrics.csv')
        context = dict(framework=framework, batch=int(batch), input_tokens=int(length), phase=phase,
                       physical_gpu=read(attempt / 'worker/result.json')['cuda_visible_devices'])
        for row in measured:
            proofs = exact[key(row)]
            mode = 'matching kernel+launch resources in same-case source-NVTX diagnostic'
            if not proofs:
                proofs = symbols[row['kernel']]
                mode = 'kernel symbol observed in source-NVTX diagnostics;subkind combined when ambiguous'
            stage = common_stage({proof['stage'] for proof in proofs}) if proofs else 'unresolved'
            mappings[key(row)] = stage
            evidence = next((proof for proof in proofs if proof['stage'] == stage), proofs[0] if proofs else {})
            row.update(stage=stage, source_evidence=evidence.get('evidence', ''),
                       source_stage_candidates=';'.join(sorted({proof['stage'] for proof in proofs})),
                       attribution='evidence_supported_inference_from_separate_source_trace' if stage != 'unresolved' else 'unresolved',
                       attribution_rule=mode if proofs else '', source_trace=evidence.get('source', ''),
                       native_trace=str(attempt / 'trace.nsys-rep'))
            kernel_rows.append(dict(**context, **row))
        connection = sqlite3.connect(f'file:{attempt / "trace.sqlite"}?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        strings = dict(connection.execute('SELECT id,value FROM StringIds'))
        intervals, totals, calls = collections.defaultdict(list), collections.Counter(), collections.Counter()
        for item in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            row = dict(item)
            resource = dict(kernel=strings[row['demangledName']],
                            grid=str(tuple(row[name] for name in ('gridX', 'gridY', 'gridZ'))),
                            block=str(tuple(row[name] for name in ('blockX', 'blockY', 'blockZ'))),
                            registers_per_thread=row['registersPerThread'], static_shared_bytes=row['staticSharedMemory'],
                            dynamic_shared_bytes=row['dynamicSharedMemory'])
            stage = mappings[key(resource)]
            intervals[stage].append((row['start'], row['end']))
            totals[stage] += row['end'] - row['start']
            calls[stage] += 1
        connection.close()
        summed = sum(totals.values())
        for stage in sorted(totals):
            stage_rows.append(dict(**context, stage=stage, kernel_calls=calls[stage],
                                   kernel_sum_ns=totals[stage], stage_kernel_busy_union_ns=interval_union(intervals[stage]),
                                   share_of_kernel_sum=totals[stage]/summed,
                                   evidence='Kernel durations verified;stage transfer is evidence-supported inference',
                                   scope='Native graph/fusion configuration;stage unions can overlap,not E2E wall time. '
                                         'Nsight node tracing perturbs scheduling;see separate overhead control.'))
        activity = read(attempt / 'export/gpu_activity.json')
        for device, row in activity['devices'].items():
            windows.append(dict(**context, cuda_device=device, **row,
                                unresolved_kernel_sum_fraction=totals['unresolved']/summed,
                                raw_evidence=str(attempt)))
    write_csv(out / 'kernel_metrics.csv', kernel_rows)
    write_csv(out / 'stage_breakdown.csv', stage_rows)
    write_csv(out / 'gpu_windows.csv', windows)


def routing(out):
    import torch
    rows, cases, checks, shapes = [], [], [], []
    destination = out / 'routing_stats'
    destination.mkdir(exist_ok=True)
    for folder in sorted((RAW / 'routing').glob('*_b*_i*_decode')):
        match = re.fullmatch(r'(vllm|sglang|trtllm)_b(\d+)_i(\d+)_decode', folder.name)
        if not match:
            continue
        framework, batch, length = match.groups()
        artifact = 'worker/routing_stats/' + ('helper_metrics/routing.json' if framework == 'sglang' else 'routing.json')
        attempt = accepted(folder, artifact)
        if not attempt:
            continue
        data = read(attempt / artifact)
        samples = data['route_samples']
        if len(samples) != 48 * 17 or {row['layer'] for row in samples} != set(range(48)):
            raise ValueError(f'incomplete per-layer routing sample: {folder}')
        metadata, native = None, None
        if framework == 'sglang':
            meta = list((attempt / 'worker/routing_stats').glob('forward_metadata*.pt'))
            dumps = list((attempt / 'worker/routing_stats').glob('expert_distribution_recorder*.pt'))
            if len(meta) != 1 or len(dumps) != 1:
                raise ValueError('missing native SGLang count or forward-metadata dump')
            metadata = torch.load(meta[0], map_location='cpu', weights_only=True)['records']
            native = torch.load(dumps[0], map_location='cpu', weights_only=True)['logical_count']
        groups = collections.defaultdict(list)
        for row in samples:
            if sum(row['n_e']) != int(batch) * 8 or row['tokens'] != int(batch):
                raise ValueError('actual Decode cohort/expert assignment count changed')
            if framework == 'sglang':
                index = row['metadata']['forward_index']
                if not metadata[index]['is_decode'] or native[index, row['layer']].tolist() != row['n_e']:
                    raise ValueError('native SGLang counts/forward modes disagree with consumed top-k IDs')
                kv = metadata[index]['seq_lens'].tolist()
            else:
                kv = row['metadata'].get('kv_seq_lens', row['metadata'].get('seq_lens'))
            if not kv or min(kv) < int(length):
                raise ValueError('actual KV length missing;query length must not be substituted')
            row['kv_lengths'] = kv
            groups[row['layer']].append(row)
        for layer, group in sorted(groups.items()):
            mean = [statistics.mean(row['n_e'][expert] for row in group) for expert in range(128)]
            maximum = [max(row['n_e'][expert] for row in group) for expert in range(128)]
            hist = collections.Counter(value for row in group for value in row['n_e'])
            gemms = [row for row in data['gemm_samples'] if row['layer'] == layer]
            pad = [row['actual_padded_rows'] for row in gemms]
            rows.append(dict(framework=framework, batch=int(batch), input_tokens=int(length), layer=layer,
                decode_samples=len(group), decode_ordinals='32..48,pure-Decode forwards',
                route_assignments_per_step=int(batch)*8,
                active_experts_mean=statistics.mean(row['active_experts'] for row in group),
                empty_experts_mean=statistics.mean(128-row['active_experts'] for row in group),
                active_experts_min=min(row['active_experts'] for row in group),
                active_experts_max=max(row['active_experts'] for row in group),
                max_n_e=max(row['max_n_e'] for row in group),
                max_to_active_mean=statistics.mean(row['max_n_e']/(int(batch)*8/row['active_experts']) for row in group),
                kv_len_min=min(min(row['kv_lengths']) for row in group),
                kv_len_max=max(max(row['kv_lengths']) for row in group),
                mean_n_e_by_expert=json.dumps(mean, separators=(',', ':')),
                max_n_e_by_expert=json.dumps(maximum, separators=(',', ':')),
                pooled_n_e_histogram=json.dumps(dict(sorted(hist.items())), separators=(',', ':')),
                actual_padded_rows_mean=statistics.mean(pad) if pad else '',
                padded_row_status='GPU num_tokens_post_padded sampled' if pad else 'unavailable:opaque CUTLASS operator',
                gemm_configs=json.dumps(sorted({json.dumps(row['config'],sort_keys=True) for row in gemms})),
                gate_up_weight_shape='128x1536x2048', down_weight_shape='128x2048x768',
                useful_expert_gemm_flops_per_step=2*int(batch)*8*(1536*2048+2048*768),
                source=str(attempt / artifact), evidence='verified actual consumed routes;no synthetic routing'))
        checks.append(dict(framework=framework, batch=int(batch), input_tokens=int(length),
                           layers=48, iterations_per_layer=17, samples=len(samples),
                           actual_cohort_verified=True, native_stat_crosscheck=framework == 'sglang', source=str(attempt)))
        cases.append(dict(framework=framework, batch=int(batch), input_tokens=int(length),
                          active_experts_mean=statistics.mean(row['active_experts'] for row in samples),
                          empty_experts_mean=statistics.mean(128-row['active_experts'] for row in samples),
                          max_n_e=max(row['max_n_e'] for row in samples),
                          actual_padded_rows_mean=statistics.mean(row['actual_padded_rows'] for row in data['gemm_samples'])
                          if data['gemm_samples'] else '',
                          kv_len_min=min(min(row['kv_lengths']) for row in samples),
                          kv_len_max=max(max(row['kv_lengths']) for row in samples),
                          source=str(attempt)))
        unique = collections.defaultdict(list)
        for item in data['gemm_samples']:
            identity = json.dumps({name: item.get(name) for name in
                ('a_shape', 'b_shape', 'c_shape', 'config', 'top_k', 'sorted_token_ids_elements', 'flags')}, sort_keys=True)
            unique[identity].append(item['actual_padded_rows'])
        for identity, pads in sorted(unique.items()):
            item = json.loads(identity)
            shapes.append(dict(framework=framework, batch=int(batch), input_tokens=int(length),
                a_shape=json.dumps(item['a_shape']), b_shape=json.dumps(item['b_shape']),
                c_shape=json.dumps(item['c_shape']), config=json.dumps(item['config'], sort_keys=True),
                top_k=item['top_k'], sorted_token_ids_elements=item['sorted_token_ids_elements'],
                flags=json.dumps(item['flags'], sort_keys=True), observed_calls=len(pads),
                actual_padded_rows_min=min(pads), actual_padded_rows_max=max(pads),
                source=str(attempt / artifact),
                scope='Actual consumed helper tensors;per-expert M=n_e,not tensor A rows or padded total. '
                      'Gate-up N=1536,K=2048;down N=2048,K=768.'))
    write_csv(destination / 'per_layer.csv', rows)
    write_csv(destination / 'summary.csv', cases)
    write_csv(destination / 'gemm_shapes.csv', shapes)
    (destination / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')


def controls(out):
    baseline = csv_rows(out / 'benchmark.csv')
    plain = next(float(row['tpot_median_ms']) for row in baseline
                 if row['framework'] == 'vllm' and row['batch'] == '1' and row['input_tokens'] == '128')
    traces = []
    for mode, path in [('node', 'timeline/vllm_b1_i128_decode'),
                       ('graph', 'timeline_graph/vllm_b1_i128_decode')]:
        attempt = accepted(RAW / path, 'export/gpu_activity.json')
        if attempt is None:
            raise ValueError('profiler control has no accepted export')
        activity = read(attempt / 'export/gpu_activity.json')['devices']['0']
        traces.append(dict(mode=mode, graph_launches=activity['graph_launches'],
            cadence_median_ms=activity['graph_launch_cadence_median_ns']/1e6,
            graph_span_median_ms=activity['graph_span_median_ns']/1e6,
            baseline_host_tpot_ms=plain,
            cadence_to_baseline_ratio=activity['graph_launch_cadence_median_ns']/1e6/plain,
            scope=activity['graph_scope'], source=str(attempt)))
    replays = []
    for batch in (1, 4):
        path = RAW / f'replay_graph/b{batch}/worker/replay.json'
        if read(path.parents[1] / 'guard/guard_result.json')['code'] != 0:
            raise ValueError('replay control did not exit successfully')
        data = read(path)
        first = [row['median_ms'] for row in data['results'] if row['variant'] == 0]
        second = [row['median_ms'] for row in data['results'] if row['variant'] == 1]
        replays.append(dict(batch=batch, order=data['order'], changed_factor=data['changed_factor'],
            input_identity=data['input_identity'], actual_padded_rows=data['actual_padded_rows'],
            static_safe_bound=data['static_safe_bound'],
            results=[{key: value for key, value in row.items() if key != 'samples_ms'} for row in data['results']],
            nominal_reduction_fraction=1-statistics.mean(second)/statistics.mean(first),
            largest_within_variant_order_spread_ms=max(max(first)-min(first), max(second)-min(second)),
            all_bitwise_equal=all(row['bitwise_equal'] for row in data['results']),
            conclusion='No reliable benefit established;nominal difference below observed order drift.',
            scope=data['scope'], source=str(path)))
    (out / 'controls.json').write_text(json.dumps(dict(
        evidence='verified diagnostic observations;not accepted production optimizations',
        profiler_granularity=traces, static_bound_replay=replays), indent=2) + '\n')


def coverage(out):
    benchmark = csv_rows(out / 'benchmark.csv')
    windows = csv_rows(out / 'gpu_windows.csv')
    checks = read(out / 'routing_stats/validation.json')
    if len(benchmark) != 36 or any(row['status'] != 'ok' or int(row['rounds']) < 3 for row in benchmark):
        raise ValueError('ordinary matrix incomplete')
    if len(windows) != 18 or len(checks) != 9 or any(row['samples'] != 816 for row in checks):
        raise ValueError('representative diagnostics incomplete')
    failures = []
    for path in sorted(RAW.glob('**/guard/guard_result.json')):
        guard = read(path)
        if guard['code'] != 0:
            worker = read(path.parents[1] / 'worker/result.json') if (path.parents[1] / 'worker/result.json').exists() else {}
            failures.append(dict(attempt=str(path.parents[1]), code=guard['code'],
                reason=worker.get('error') or guard.get('reason', 'See retained worker/guard logs'),
                scope='Excluded attempt;not an accepted baseline or final diagnostic'))
    if failures:
        write_csv(out / 'excluded_attempts.csv', failures)
    else:
        with (out / 'excluded_attempts.csv').open('w', newline='') as stream:
            csv.writer(stream).writerow(['attempt', 'code', 'reason', 'scope'])
    value = dict(evidence='verified consolidated accepted artifacts', benchmark_cells=len(benchmark),
        benchmark_rounds=sum(int(row['rounds']) for row in benchmark),
        capacity_limited_cohort_cells=sum(row['full_resident_batch_fit'] == 'no' for row in benchmark),
        native_timeline_windows=len(windows), source_attribution_cases=sum(
            accepted(path, 'export/kernel_metrics.csv') is not None for path in (RAW / 'attribution').glob('*')),
        route_cases=len(checks), route_layers_per_case=48, route_iterations_per_layer=17,
        route_samples=sum(row['samples'] for row in checks),
        hardware_counters='unavailable:ERR_NVGPUCTRPERM',
        online_serving_and_tp2='not run;closed-cohort TP1 execution study,not arrival/EP/communication conclusions',
        excluded_attempts=len(failures))
    (out / 'coverage.json').write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results/moe_profile')
    parser.add_argument('--part', choices=['all', 'timelines', 'routing'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part in ('all', 'timelines'):
        timelines(args.out)
    if args.part in ('all', 'routing'):
        routing(args.out)
    if args.part == 'all':
        controls(args.out)
        coverage(args.out)


if __name__ == '__main__':
    main()
