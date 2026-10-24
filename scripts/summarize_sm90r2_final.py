#!/usr/bin/env python3
"""Recollect final TileMega pairs and compare with frozen external CSVs offline."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile

import run_sm90r2_final_pair as paired
from sm90d1_common import ROOT, read, sha, write

MODELS = paired.final.MODELS
BATCHES = (1, 2, 4, 8, 16)
SYSTEMS = ('torch.compile', 'vllm', 'sglang', 'tensorrt-llm')
ENGINES = SYSTEMS[1:]
FROZEN_BASELINE_SHA256 = 'd3bded2e5acbb9302ccba6a42a7274801af1888d0cc3c3661a6aa1d74299f253'
METRICS = ('tpot', 'ttft', 'e2e', 'throughput')


def positive(value):
    if isinstance(value, bool):
        raise ValueError('positive finite timing and throughput values are required')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError('positive finite timing and throughput values are required')
    return number


def baseline_table(rows):
    table = {}
    for row in rows:
        if row['model'] not in MODELS:
            continue
        key = (row['model'], int(row['batch']), row['system'])
        if key in table or key[1] not in BATCHES or key[2] not in SYSTEMS:
            raise ValueError('duplicate or unsupported frozen baseline cell')
        if row['protocol_complete'] != 'True' or int(row['fresh_processes']) < 3:
            raise ValueError('frozen dense baseline protocol is incomplete')
        table[key] = dict(tpot=positive(row['tpot_ms']), ttft=positive(row['ttft_ms']),
                          e2e=positive(row['e2e_ms']), throughput=positive(row['tokens_per_second']),
                          acceptance=row['acceptance'], physical_uuid=row['physical_gpu_uuid'])
    expected = {(m, b, s) for m in MODELS for b in BATCHES for s in SYSTEMS}
    if set(table) != expected:
        raise ValueError('all 100 frozen dense baseline cells are required')
    return table


def candidate_metrics(value):
    return dict(tpot=positive(value['tpot_seconds']) * 1000,
                ttft=positive(value['ttft_seconds']) * 1000,
                e2e=positive(value['e2e_seconds']) * 1000,
                throughput=positive(value['output_tokens_per_second']))


def summarize(records, baselines):
    """Arithmetic on recollected pairs; this helper does not authenticate files."""
    cells, heads = {}, set()
    for record in records:
        result = record['comparison']; key = (result['model'], result['batch'])
        if key in cells or key[0] not in MODELS or type(key[1]) is not int or key[1] not in BATCHES:
            raise ValueError('duplicate or unsupported TileMega final cell')
        if (result['kind'] != 'sm90r2_final_tilemega_pair' or result['schema'] != 1
                or result['user_final_e2e_amendment'] != paired.final.AMENDMENT):
            raise ValueError('collected final TileMega pairs with the final-test amendment are required')
        pairs, blanks = result['paired_tpot_seconds'], result['reference_blank_tpot_seconds']
        if (len(pairs) != result['processes_per_arm']
                or len(pairs) < (5 if key[0] == 'llama3_1_8b' else 3)):
            raise ValueError('final fresh-process cohort is incomplete')
        decision = paired.paired_decision(pairs, blanks)
        if any(result.get(k) != v for k, v in decision.items()):
            raise ValueError('collected paired decision differs from the raw paired times')
        metrics = candidate_metrics(result['arm_metrics']['candidate'])
        if metrics['tpot'] != statistics.median(p[1] for p in pairs) * 1000:
            raise ValueError('candidate TPOT differs from the fresh-process median')
        head = record['source_head']
        if len(head) != 40 or any(c not in '0123456789abcdef' for c in head):
            raise ValueError('a full committed source HEAD is required')
        config = record['config_key']
        if len(config) != 64 or any(c not in '0123456789abcdef' for c in config):
            raise ValueError('a SHA256 coordinate configuration key is required')
        heads.add(head)
        cells[key] = dict(model=key[0], batch=key[1], source_head=head,
            config_key=record['config_key'], physical_uuid=result['physical_uuid'],
            processes=result['processes_per_arm'], ownership_verified=result['ownership_verified'],
            tpot_ms=metrics['tpot'], ttft_ms=metrics['ttft'], e2e_ms=metrics['e2e'],
            tokens_per_second=metrics['throughput'],
            process_range_pct=decision['candidate_process_range_pct'],
            paired_change_pct=decision['paired_change_pct'],
            decision_threshold_pct=decision['decision_threshold_pct'], decision=decision['decision'],
            c1='not_performed', c2='not_performed', metrics=metrics)
    if set(cells) != {(m, b) for m in MODELS for b in BATCHES} or len(heads) != 1:
        raise ValueError('all 25 final cells must come from a single source HEAD')
    ratios = []
    for key in sorted(cells):
        current = cells[key]
        best = min(ENGINES, key=lambda s: baselines[key + (s,)]['tpot'])
        for system in (*SYSTEMS, 'fastest_engine_tpot'):
            actual = best if system == 'fastest_engine_tpot' else system
            base = baselines[key + (actual,)]
            row = dict(model=key[0], batch=key[1], system=system, baseline_system=actual,
                baseline_acceptance=base['acceptance'], baseline_physical_uuid=base['physical_uuid'],
                tilemega_physical_uuid=current['physical_uuid'],
                tilemega_ownership_verified=current['ownership_verified'])
            row.update({metric + '_ratio': current['metrics'][metric] / base[metric]
                if metric == 'throughput' else base[metric] / current['metrics'][metric]
                for metric in METRICS})
            ratios.append(row)
    summaries = []
    for system in (*SYSTEMS, 'fastest_engine_tpot'):
        rows = [r for r in ratios if r['system'] == system]
        row = dict(system=system, cells=len(rows))
        for metric in METRICS:
            values = [r[metric + '_ratio'] for r in rows]
            row[metric + '_geomean'] = math.exp(statistics.mean(math.log(v) for v in values))
            row[metric + '_minimum'] = min(values)
        summaries.append(row)
    return dict(schema=1, kind='sm90r2_final_summary', source_head=heads.pop(),
        evidence='verified offline arithmetic on recollected guarded E2E receipts; ownership status preserved',
        cells=[{k: v for k, v in cells[key].items() if k != 'metrics'} for key in sorted(cells)],
        ratios=ratios, ratio_summary=summaries,
        g_perf_all_cells=all(c['decision'] == 'faster' for c in cells.values()),
        ownership_verified=all(c['ownership_verified'] for c in cells.values()),
        user_final_e2e_amendment=paired.final.AMENDMENT,
        ratio_convention='greater than one means TileMega is faster; latency baseline/TileMega, throughput TileMega/baseline',
        fastest_engine_convention='minimum frozen TPOT among vLLM/SGLang/TensorRT-LLM; all four ratios use that same engine',
        comparison_limitation='historical frozen baselines; no new anchors, matched checkpoint or cross-session drift correction',
        gpu_jobs_started=0, external_baselines_run=0)


def checked(record, base):
    return paired.final.checked_file(record, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); path = args.manifest.resolve(); manifest = read(path)
    if manifest.get('schema') != 1 or manifest.get('kind') != 'sm90r2_final_matrix':
        raise ValueError('a final matrix collection manifest is required')
    if len(manifest.get('cells', [])) != 25:
        raise ValueError('all 25 final matrix cells are required')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'runs/sm90r2') or out.exists():
        raise ValueError('retain new summary evidence under runs/sm90r2')
    baseline = checked(manifest['frozen_baseline'], path.parent)
    if sha(baseline) != FROZEN_BASELINE_SHA256:
        raise ValueError('external baseline differs from the frozen R16 CSV')
    with baseline.open() as stream:
        baselines = baseline_table(list(csv.DictReader(stream)))
    records, bindings = [], []
    # Reuse the guarded collector instead of trusting arithmetic copied into a CSV.
    with tempfile.TemporaryDirectory(prefix='.matrix-recollect-', dir=ROOT / 'runs/sm90r2') as temporary:
        for index, entry in enumerate(manifest['cells']):
            files = {name: checked(entry[name], path.parent) for name in
                     ('pair_manifest', 'preparation', 'queue', 'state', 'comparison')}
            preparation = files['preparation'].parent
            if (files['preparation'].name != 'preparation.json'
                    or files['queue'] != preparation / 'queue/queue_final_pair.json'):
                raise ValueError('matrix preparation/queue paths differ from the paired collector')
            bound = paired.validate_manifest(files['pair_manifest'])
            recollected = Path(temporary) / f'cell{index}.json'
            paired.collect(bound, preparation, files['state'], recollected)
            value = read(recollected)
            if value != read(files['comparison']):
                raise ValueError('final comparison differs from fresh offline guard/raw-token recollection')
            candidate = bound['candidate']['manifest']
            records.append(dict(source_head=candidate['source_head'],
                config_key=candidate['coordinates']['sha256'], comparison=value))
            bindings.append({name: dict(path=str(p), sha256=sha(p)) for name, p in files.items()})
    result = summarize(records, baselines)
    result.update(manifest_sha256=sha(path), summarizer_sha256=sha(__file__), inputs=bindings,
                  frozen_baseline=dict(path=str(baseline), sha256=sha(baseline)))
    out.mkdir(parents=True)
    for name, key in [('SM90R2_final.csv', 'cells'), ('SM90R2_ratios.csv', 'ratios'),
                      ('SM90R2_ratio_summary.csv', 'ratio_summary')]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader(); writer.writerows(result[key])
    write(out / 'summary.json', result)
    print(json.dumps(dict(cells=len(result['cells']), g_perf_all_cells=result['g_perf_all_cells'],
                         ownership_verified=result['ownership_verified'], gpu_jobs_started=0)))


if __name__ == '__main__':
    main()
