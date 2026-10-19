#!/usr/bin/env python3
"""Collect complete SM90R1 cells without treating queued work as results."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
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
                   [r['speedup_' + metric + '_vs_tm_r16'] for r in native
                    if r['tm_r16_reference_valid']])
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
    cells = dict(progress['settled'])
    revisions = raw / 'final-llm/revisions.json'
    if revisions.exists():
        for key, cell in read(revisions)['settled'].items():
            assert key in cells and cell['status'] == 'completed'
            assert (cell['model'], cell['batch']) == (cells[key]['model'], cells[key]['batch'])
            cells[key] = cell
    rows = []
    for key, cell in cells.items():
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
                   gpus=';'.join(str(i) for i in sorted({v['gpu_index'] for v, _ in rounds})),
                   physical_gpu_uuids=';'.join(sorted({v['physical_device_identity']['nvml_uuid']
                                                      for v, _ in rounds})),
                   acceptance=';'.join(sorted({o['acceptance'] for _, o in rounds})))
        decode, prefill = (identities[phase]['plan'] for phase in ('decode', 'prefill'))
        row.update(decode_executor=config['execution']['decode'], decode_loop=config['execution']['loop'],
                   prefill_executor=config['execution']['prefill'], decode_pg=decode['pg'],
                   attention_impl=decode['attention_impl'], attention_kv_block=decode['attention_kv_block'],
                   residency_requested=decode['requested_residency'], cluster_dim=decode['cluster_dim'],
                   deferred_norm=decode['deferred_norm'], nonpaged_la=decode['nonpaged_la'],
                   dm_wgmma_inline=decode['dm_wgmma_inline'],
                   stage_descriptor_prefetch=decode.get('stage_descriptor_prefetch', 0),
                   task_prefix_prefetch=decode.get('task_prefix_prefetch', 0),
                   qkv_epilogue_decode=decode['qkv_epilogue'], qkv_epilogue_prefill=prefill['qkv_epilogue'])
        fields = ['tile_m', 'tile_n', 'tile_k', 'stages', 'split_k', 'impl']
        tables = {phase: [[g[field] for field in fields] for g in identity['plan']['gemms']]
                  for phase, identity in identities.items()}
        geometry = ROOT / 'configs/serving/sm90r1/geometry' / config_path.name
        if geometry.exists():
            assert read(geometry)['tables'] == tables
        else:
            geometry.write_text(json.dumps(dict(schema=1, fields=fields,
                index='GEMM ordinal in the full plan', tables=tables), separators=(',', ':')) + '\n')
        row['geometry_table'] = str(geometry.relative_to(ROOT))
        for phase in ('decode', 'prefill'):
            source = ROOT / (config['libraries'][phase]['path'] + '.cu')
            generated = source.read_text()
            match = re.search(r'kStages,\s*(\d+)u', generated)
            assert match, 'generated stage count is unavailable'
            row[phase + '_declared_stages'] = int(match[1])
            block = generated.split('constexpr StageDesc kStages[] = {', 1)[1].split('\n};', 1)[0]
            entries = [line.strip() for line in block.splitlines() if '{TaskKind::' in line]
            gemms = [int(value) for value in re.findall(r'\{TaskKind::kGemm, (\d+)u,', block)]
            unelided = all(re.search(r', kNoOperand, false(?:, \d+u)*},$', line)
                           for line in entries)
            if (unelided and sorted(gemms) == list(range(len(tables[phase]))) and
                    '{TaskKind::kAttention,' not in block):
                row[phase + '_runtime_stages_inferred'] = len(entries) + sum(
                    gemm['split_k'] > 1 for gemm in identities[phase]['plan']['gemms'])
            else:
                row[phase + '_runtime_stages_inferred'] = ''
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
    cells, published = dict(progress['settled']), dict(progress['published'])
    revisions = raw / 'final-dnn/revisions.json'
    if revisions.exists():
        revision = read(revisions)
        assert set(revision['settled']) <= set(cells)
        cells.update(revision['settled'])
        published.update(revision['published'])
    rows = []
    for model, cell in cells.items():
        if cell['status'] != 'final_three_process_measurement_queued':
            continue
        if not all(states.get(name, {}).get('status') == 'done' for name in cell['jobs']):
            continue
        config_path = ROOT / cell.get('config', f'configs/dnn/sm90r1/{model}_b1.json')
        config = read(config_path)
        assert file_sha(config_path) == published[model]['config_sha256']
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
                   gpus=';'.join(str(i) for i in sorted({v['gpu_index'] for v, _ in rounds})),
                   physical_gpu_uuids=';'.join(sorted({v['physical_device_identity']['nvml_uuid']
                                                      for v, _ in rounds})),
                   acceptance=';'.join(sorted({o['acceptance'] for _, o in rounds})),
                   source_checkpoint_sha256=gate['source_checkpoint_sha256'],
                   exported_program_sha256=gate['archive_sha256'],
                   round_metadata='job names; BERT raw indices are all 0' if model == 'bert' else '0,1,2')
        execution = rounds[0][0]['execution_identity']
        row.update(executor=config['executor'],
                   shared_memory_bytes=execution['shared_memory_bytes'], spill=execution['spill'],
                   residency_requested=config['coordinates']['residency'])
        for metric in ('median_ms', 'p90_ms'):
            values = [v[metric] for v, _ in rounds]
            row[metric] = statistics.median(values)
            row[metric + '_range_pct'] = 100 * (max(values) - min(values)) / row[metric]
        rows.append(row)
    return rows


def anchor_rows(states):
    frozen = {int(v['batch']): float(v['tpot_ms'])
              for v in table('R16_final_baselines.csv')
              if v['model'] == 'llama3_2_1b' and v['system'] == 'vllm'
              and int(v['batch']) in (1, 16)}
    rows, starts = [], {}
    for phase in ('start', 'end'):
        for batch in (1, 16):
            prefix = f'sm90r1_M0_{phase}_B{batch}_gpu'
            matches = [value for name, value in states.items()
                       if name.startswith(prefix) and value.get('status') == 'done']
            if not matches:
                continue
            assert len(matches) == 1, 'ambiguous M0 anchor'
            record = matches[0]
            directory = Path(record['out'])
            path = directory / f'measurements/B{batch}/measurements.json'
            measured, _, ownership = validate_llm(path, completed_guard(directory, record), ROOT)
            assert measured['system'] == 'vllm' and measured['batch'] == batch
            elapsed = measured['tpot_seconds'] * 1000
            if phase == 'start':
                starts[batch] = (elapsed, measured['gpu_index'])
            rows.append(dict(phase=phase, batch=batch, gpu=measured['gpu_index'],
                             tpot_ms=elapsed, frozen_tpot_ms=frozen[batch],
                             drift_vs_frozen_pct=100 * (elapsed / frozen[batch] - 1),
                             drift_vs_start_pct=100 * (elapsed / starts[batch][0] - 1),
                             same_card_as_start=measured['gpu_index'] == starts[batch][1],
                             acceptance=ownership['acceptance'], measurement=str(path.relative_to(ROOT))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'runs/sm90r1')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    selection_revision = args.raw / 'final-llm/selection-revision.json'
    selection_pending = selection_revision.exists() and read(selection_revision)['status'] == 'pending'
    states = {}
    for gpu in range(8):
        states.update(read(args.raw / f'scheduler/gpu{gpu}/state.json'))
    llm, dnn = llm_cells(args.raw, states), dnn_cells(args.raw, states)
    anchors = anchor_rows(states)
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
        # The audit also identifies intermittent B8 errors despite its CSV flag.
        valid_reference = previous['valid'] == 'True' and not (
            row['model'] == 'qwen3_30b_a3b' and row['batch'] >= 4)
        row['tm_r16_reference_valid'] = valid_reference
        row['tm_r16_reference_status'] = ('valid frozen native reference' if valid_reference else
                                         'excluded: known R16 correctness failure')
        row['speedup_tpot_vs_tm_r16'] = (float(previous['tpot_ms']) / row['tpot_ms']
                                       if valid_reference else '')
        row['speedup_e2e_vs_tm_r16'] = (float(previous['e2e_ms']) / row['e2e_ms']
                                      if valid_reference else '')
        for (model, batch, system), baseline in baselines.items():
            if (model, batch) != (row['model'], row['batch']):
                continue
            checkpoint = baseline.get('checkpoint_sha256') or baseline.get('source_checkpoint_sha256')
            if baseline.get('checkpoint_sha256') or baseline.get('checkpoint_identity_shared') == 'True':
                assert checkpoint == row['checkpoint_sha256']
            comparison = dict(model=model, batch=batch, baseline=system,
                              checkpoint_paired=bool(checkpoint and checkpoint == row['checkpoint_sha256']),
                              checkpoint_scope=('verified matching checkpoint' if checkpoint == row['checkpoint_sha256']
                                                else 'different frozen checkpoint' if checkpoint
                                                else 'frozen checkpoint identity unavailable'),
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
    anchor_note = ('frozen ratios unadjusted; M0 drift exceeds 2%' if any(
        abs(row['drift_vs_frozen_pct']) > 2 or abs(row['drift_vs_start_pct']) > 2
        for row in anchors) else 'frozen ratios unadjusted; M0 drift within 2%')
    if any(not row['same_card_as_start'] for row in anchors):
        anchor_note += '; M0 start/end use different GPUs'
    for row in llm_ratios:
        row['anchor_status'] = anchor_note if len(anchors) == 4 else anchor_note + '; end anchors pending'
    complete = len(llm) == 30 and len(dnn) == 5 and not selection_pending and len(anchors) == 4
    if args.require_complete:
        assert complete, 'final matrix, selection or M0 anchors are incomplete'
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix, rows in [('llm', llm), ('dnn', dnn), ('llm_ratios', llm_ratios), ('dnn_ratios', dnn_ratios)]:
        write(args.out / f'SM90R1_final_{suffix}.csv', rows)
    write(args.out / 'SM90R1_summary.csv', aggregate_ratios(llm, llm_ratios, dnn, dnn_ratios))
    write(args.out / 'SM90R1_anchors.csv', anchors)
    if complete:
        index = dict(schema=1, llm={}, dnn={})
        for row in llm:
            index['llm'].setdefault(row['model'], {})[str(row['batch'])] = row['config']
        for row in dnn:
            index['dnn'][row['model']] = row['config']
        path = ROOT / 'configs/serving/sm90r1/selected.json'
        path.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(dict(llm_complete=len(llm), dnn_complete=len(dnn),
                          selection_revision_pending=selection_pending,
                          anchor_complete=len(anchors) == 4, final_complete=complete)))


if __name__ == '__main__':
    main()
