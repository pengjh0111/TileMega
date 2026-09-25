#!/usr/bin/env python3
"""Rebuild report tables from raw measurements and solver outputs.

Safe to run while experiments are active: missing cells are recorded explicitly
and no incomplete top-three selection is promoted to a measured winner. This
does not generate acceptance verdicts; verify.py independently checks the gates.
"""
import csv
import json
import math
import pathlib
import re
import statistics

E = pathlib.Path(__file__).resolve().parent
OUT = E / 'report_tables'
CELLS = [(m, s) for m in ('llama', 'qwen3') for s in (1, 4, 16, 64)]
missing = []


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write(name, data, fields=None):
    data = list(data)
    if not fields:
        fields = list(dict.fromkeys(k for row in data for k in row))
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)


def measurements(directory, required=10):
    logs = sorted(directory.glob('process_*.log'))
    if len(logs) != required:
        raise ValueError(f'{directory.relative_to(E)}: {len(logs)}/{required} processes')
    values = []
    for log in logs:
        text = log.read_text()
        hashes = re.search(r'E2E_HASH l05=(\w+) l1=(\w+) l2=(\w+)', text)
        if not hashes or len(set(hashes.groups())) != 1 or not all(
                re.search(k + r'=0\b', text) for k in
                ('l1_vs_l05_mismatch', 'l2_vs_l1_mismatch')):
            raise ValueError(f'internal mismatch or missing hashes: {log}')
        match = re.search(r'E2E_TIME l05_ms=([\d.]+) l1_ms=([\d.]+).*? l2_ms=([\d.]+)', text)
        if not match:
            raise ValueError(f'missing timing: {log}')
        values.append(tuple(map(float, match.groups())))
    return dict(zip(('l05_ms', 'l1_ms', 'l2_ms'), map(statistics.median, zip(*values))))


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(order)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for index in order[i:j]:
            result[index] = (i + j - 1) / 2
        i = j
    return result


def quantile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def floor(cell):
    return {k: float(v) for k, v in re.findall(
        r'(\w+)=([\d.eE+-]+)', (E / 'floor' / (cell + '.log')).read_text())}


def selected(cell):
    directory = E / 'matrix' / cell
    candidates = rows(directory / 'selected.cu.top3.tsv')
    if len(candidates) != 3:
        raise ValueError(f'{cell}: expected three shortlisted candidates')
    measured = [(measurements(pathlib.Path(c['source'] + '.measurement')), c) for c in candidates]
    return min(measured, key=lambda pair: pair[0]['l2_ms'])


def performance():
    result, configs, decompositions, fusion, placements, resources, phases = [], [], [], [], [], [], []
    for model, seq in CELLS:
        cell = f'{model}_s{seq}'
        bound = floor(cell)
        control = measurements(E / 'controls' / cell)
        baseline_flow = rows(E / 'flow_final' / cell / 'home.flow.tsv')[0]
        arms = [('legacy', control, int(float(baseline_flow['chain_depth'])), None)]
        try:
            measured, candidate = selected(cell)
            directory = E / 'matrix' / cell
            # Match by the full per-class key. The A/B records have identical
            # Level-1 physics, but different FIFO simulation and placement.
            matching = [(p, r) for p in directory.glob('*.flow.tsv') for r in rows(p)
                        if r['key'] == candidate['key']]
            if not matching:
                raise ValueError(f'{cell}: selected configuration has no raw flow solve')
            path, raw = matching[0]
            x = {k: float(v) for k, v in raw.items() if k != 'key'}
            arms.append(('skeleton', measured, int(x['chain_depth']), candidate))
            sync = x['T'] - x['T_s']
            fixed = x['T_s'] - x['T_sf']
            contention = x['T_sf'] - max(x['T_floor'], x['T_sf_infinite'])
            chain = max(0., x['T_sf_infinite'] - x['T_floor'])
            links = rows(path.with_name(path.name.replace('.flow.tsv', '.flow_chain.tsv')))
            protocol = sum(float(link[k]) for link in links for k in ('wait_ns', 'publication_ns', 'hop_ns'))
            attention = sum(float(link['end_ns']) - float(link['start_ns'])
                            for link in links if link['category'] == 'attention')
            decompositions.append(dict(cell=cell, flow_ns=x['T'], floor_ns=x['T_floor'],
                sync_ns=sync, fixed_ns=fixed, contention_ns=contention, chain_ns=chain,
                closure_ns=sync + fixed + contention + chain - (x['T'] - x['T_floor']),
                pg_upper_ns=x['T'] - max(x['T_floor'], x['T_np0']),
                protocol_on_chain_ns=protocol, attention_chain_ns=attention,
                attention_over_flow=attention / x['T'], chain_depth=x['chain_depth'],
                flow_over_measured=x['T'] / (measured['l2_ms'] * 1e6),
                fluid_over_measured=float(candidate['predicted_ns']) / (measured['l2_ms'] * 1e6),
                colocated_edges=x['colocated_edges'], sync_omitted_edges=x['sync_omitted_edges'], evidence=str(path.relative_to(E))))
            for category in sorted({link['category'] for link in links}):
                group = [link for link in links if link['category'] == category]
                fusion.append(dict(cell=cell, category=category, links=len(group),
                    potentially_fusible=category in ('rmsnorm', 'combine', 'add', 'rope', 'append', 'silu', 'activation'),
                    **{k: sum(float(link[k]) for link in group) for k in ('wait_ns', 'publication_ns', 'hop_ns', 'fixed_ns')},
                    evidence=str(path.relative_to(E))))
            shapes = re.findall(r'(\d+)x(\d+)x(\d+)s(\d+)k(\d+)', candidate['key'])
            old = (E.parent / 'SOLVER_V2/legacy_r8_domain' / cell / 'selected.mlir').read_text()
            runtime = re.search(r'tilemega.gemm_runtime = \[(.*?)\]', old, re.S)[1]
            first = re.search(r'\{(.*?)\}', runtime)[1]
            legacy = dict(re.findall(r'(tile_m|tile_n|tile_k|stages|split_k) = (\d+)', first))
            for index, shape in enumerate(shapes):
                configs.append(dict(cell=cell, operator_class=index, **dict(zip(('tile_m', 'tile_n', 'tile_k', 'stages', 'split_k'), shape)),
                    legacy_uniform=json.dumps(legacy, sort_keys=True), kappa=candidate['kappa'], residency=candidate['residency'],
                    distinct_variants=len({s[:4] for s in shapes}), key=candidate['key']))
            for p, r in matching:
                metric_path = p.with_name(p.name.replace('.flow.tsv', '.metrics.tsv'))
                metric = rows(metric_path)[0]
                placements.append(dict(cell=cell, **metric, evidence=str(metric_path.relative_to(E))))
        except (OSError, ValueError, KeyError, IndexError) as error:
            missing.append(str(error))
        for arm, measured, depth, candidate in arms:
            result.append(dict(cell=cell, model=model, seq=seq, arm=arm, **measured,
                T_dram_ms=bound['dram_ns'] / 1e6, T_compute_ms=bound['compute_ns'] / 1e6,
                T_floor_ms=bound['floor_ns'] / 1e6, L2_over_floor=measured['l2_ms'] * 1e6 / bound['floor_ns'],
                L2_over_legacy=measured['l2_ms'] / control['l2_ms'], L2_over_L1=measured['l2_ms'] / measured['l1_ms'],
                L2_over_L05=measured['l2_ms'] / measured['l05_ms'], D=depth,
                bubble_ns=(measured['l2_ms'] * 1e6 - bound['floor_ns']) / depth))
        for name, accumulator in [('resources', resources), ('timing', phases)]:
            path = E / 'matrix' / cell / f'selected.cu.{name}.tsv'
            if path.exists():
                accumulator.extend(dict(cell=cell, **r) for r in rows(path))
    for name, data in [('performance', result), ('configurations', configs), ('decomposition', decompositions),
                       ('fusion_chain', fusion), ('placements', placements), ('resources', resources), ('phases', phases)]:
        write(name + '.tsv', data)


def consistency():
    result = []
    for family in ('validation', 'validation_colocated'):
        for model in ('llama', 'qwen3'):
            path = E / family / model / 'samples.tsv'
            data = rows(path) if path.exists() else []
            if len(data) < 2:
                missing.append(f'{family}/{model}: insufficient samples')
                continue
            a, b = [float(r['flow_ns']) for r in data], [float(r['fluid_ns']) for r in data]
            ratios = [x / y for x, y in zip(a, b)]
            phase_values = [float(r['total_ms']) for p in path.parent.glob('sample_*.timing.tsv')
                            for r in rows(p) if r['phase'] == 'simulate']
            result.append(dict(family=family, model=model, n=len(data), complete=(path.parent / 'exit.json').exists(),
                spearman=statistics.correlation(ranks(a), ranks(b)),
                **{f'ratio_p{q}': quantile(ratios, q / 100) for q in (0, 10, 50, 90, 100)},
                flow_ms_max=max(float(r['flow_ms']) for r in data), fluid_ms_max=max(phase_values, default=math.nan),
                nonprefix_edges_min=min(int(r['nonprefix_edges']) for r in data),
                nonprefix_edges_max=max(int(r['nonprefix_edges']) for r in data),
                varying_spaces_max=max(int(r['varying_spaces']) for r in data), evidence=str(path.relative_to(E))))
    write('consistency.tsv', result)


def replays():
    result = []
    for arm in ('baseline_bf16', 'physical_bf16', 'stages_bf16', 'fixed_bf16', 'all_bf16', 'selected_bf16', 'historical_target_bf16'):
        for model in ('gqa2', 'mha4'):
            path = E / 'replay' / arm / f'predictions_{model}.tsv'
            if not path.exists():
                missing.append(str(path.relative_to(E)))
                continue
            data = rows(path)
            actual = [float(r['measured_ms']) for r in data]
            predicted = [float(r['model_ms']) for r in data]
            rank = ranks(actual)
            chosen = sorted(range(len(data)), key=predicted.__getitem__)
            result.append(dict(arm=arm, model=model, n=len(data), spearman=statistics.correlation(ranks(predicted), rank),
                MAPE=statistics.mean(abs(p / a - 1) for p, a in zip(predicted, actual)),
                **{f'best_actual_rank_in_predicted_top{k}': 1 + min(rank[i] for i in chosen[:k]) for k in (1, 3, 10)},
                evidence=str(path.relative_to(E))))
    write('replay.tsv', result)


def theta():
    result = []
    for model in ('llama', 'qwen3'):
        previous = None
        for seq in (1, 2, 4, 8, 16, 32, 64):
            directory = E / 'theta' / f'{model}_s{seq}'
            if not (directory / 'completed.tsv').exists():
                missing.append(f'theta/{model}_s{seq}: incomplete')
                continue
            done = rows(directory / 'completed.tsv')[0]
            bound = rows(directory / 'selected.cu.floor.tsv')[0]
            result.append(dict(model=model, **done, **bound,
                flow_over_floor=float(done['flow_ns']) / float(bound['floor_ns']),
                changed_since_previous=previous is not None and previous[1] != done['key'],
                previous_seq=previous[0] if previous else ''))
            previous = (seq, done['key'])
    write('theta.tsv', result)


def fits_and_traffic():
    data = rows(E / 'fit/observations.tsv')
    result = []
    for stages in (0, 2, 3):
        group = [r for r in data if stages == 0 or int(r['stages']) == stages]
        for component, observed in [('loop', 'observed_loop_ns'), ('fixed', 'fixed_ns')]:
            for version in ('old', 'new'):
                errors = [abs(float(r[f'{version}_{component}_ns']) / float(r[observed]) - 1) for r in group]
                result.append(dict(stages=stages or 'all', component=component, version=version,
                                   n=len(errors), median=statistics.median(errors), mean=statistics.mean(errors)))
    write('fit_errors.tsv', result)
    # These are the same observed task population, weighted by the number of
    # samples. They are not a whole-model unique-DRAM byte count (DramFloor is).
    traffic = []
    for cell in sorted({r['cell'] for r in data}):
        group = [r for r in data if r['cell'] == cell]
        nominal = sum(2 * float(r['operand_elements']) * float(r['iterations']) * int(r['samples']) for r in group)
        physical = sum(float(r['physical_stage_bytes']) * float(r['iterations']) * int(r['samples']) for r in group)
        traffic.append(dict(cell=cell, population='490 phase observations, weighted by raw sample count',
            nominal_operand_bytes=nominal, physical_operand_bytes=physical, physical_over_nominal=physical / nominal,
            nominal_write_elements=sum(float(r['nominal_writes']) * int(r['samples']) for r in group),
            physical_write_elements=sum(float(r['physical_writes']) * int(r['samples']) for r in group)))
    write('observed_traffic.tsv', traffic)


def additional_measurements():
    result = []
    paths = [(E / 'ablations' / f'{m}_s4' / arm, m, 4, arm) for m in ('llama', 'qwen3')
             for arm in ('template', 'eft', 'wide')]
    paths += [(E / 'stages' / f'llama_s4_S{s}', 'llama', 4, f'S{s}') for s in (2, 3, 4)]
    paths += [(E / 'early_resident' / f'llama_s{s}.measurement', 'llama', s, 'early_R9') for s in (1, 4)]
    for path, model, seq, arm in paths:
        try:
            measured = measurements(path)
            record = dict(cell=f'{model}_s{seq}', arm=arm, **measured,
                          L2_over_floor=measured['l2_ms'] * 1e6 / floor(f'{model}_s{seq}')['floor_ns'])
            metrics = path / 'materialized/selected.metrics.tsv'
            if metrics.exists():
                row = rows(metrics)[0]
                record.update(flow_ns=row['flow_ns'], simulated_ns=row['simulated_ns'], residency=row['residency'],
                              moved_fraction=1 - float(row['home']) / float(row['placed']))
            occupancy = path / 'occupancy/query.log'
            if occupancy.exists():
                record['actual_limit'] = json.loads(occupancy.read_text())['resident']
            result.append(record)
        except (OSError, ValueError, KeyError) as error:
            missing.append(str(error))
    write('additional_measurements.tsv', result)
    legacy_flow = []
    for model, seq in CELLS:
        cell = f'{model}_s{seq}'
        row = rows(E / 'flow_final' / cell / 'home.flow.tsv')[0]
        measured = measurements(E / 'controls' / cell)
        legacy_flow.append(dict(cell=cell, interpretation='legacy uniform geometry; pure home, not actual legacy placement',
            **row, actual_legacy_ms=measured['l2_ms'], flow_over_actual_legacy=float(row['T']) / (measured['l2_ms'] * 1e6)))
    write('legacy_flow.tsv', legacy_flow)


def main():
    OUT.mkdir(exist_ok=True)
    for operation in (performance, consistency, replays, theta, fits_and_traffic, additional_measurements):
        operation()
    (OUT / 'incomplete.json').write_text(json.dumps(missing, indent=2) + '\n')
    print(f'R9B_REPORT tables={OUT} incomplete_items={len(missing)}')


if __name__ == '__main__':
    main()
