#!/usr/bin/env python3
"""Publish the fixed A-E matrix, retaining every accepted process value."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from sm90d1_common import ROOT, read, sha, write


STEM = 'SM90_decode_AE_qwen3_1_7b_b1'


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def report(run, publish):
    state = read(run / 'timing/state.json')
    if len(state['items']) != 12 or any(v['status'] != 'done' for v in state['items']):
        raise ValueError('the twelve fresh performance processes are not complete')
    spec = read(run / 'historical-spec.json')
    manifest = read(run / 'D.manifest.json')
    historical = read(run / 'D.result.json')
    processes, rows, groups = [], [], {}
    for value in state['items']:
        path = Path(value['result']);measurement = read(path)
        if sha(path) != value['result_sha256']:
            raise ValueError('accepted measurement changed')
        arm = value['arm'];attempt = Path(value['attempts'][-1]['directory'])
        if read(attempt / 'guard_result.json')['code']:
            raise ValueError('accepted attempt has a rejected guard')
        telemetry = [json.loads(line) for line in (path.parent / 'guard.jsonl').read_text().splitlines()]
        config = measurement['configuration']
        if config['checks']['c1'] or config['checks']['c2'] or config['checks']['hf']:
            raise ValueError('this collection is performance only')
        if config['decode_sha256'] != spec['arms'][arm]['decode_sha256']:
            raise ValueError('the collected binary differs from the frozen arm')
        row = dict(arm=arm, slot=value['slot'], pid=measurement['pid'], new_reused='new',
            tpot_ms=measurement['tpot_seconds'] * 1000,
            ttft_ms=measurement['ttft_seconds'] * 1000,
            e2e_ms=measurement['e2e_seconds'] * 1000,
            gpu=state['gpu'], gpu_inventory=state['inventory'],
            sm_clocks=json.dumps(sorted({s['telemetry']['clocks.sm'] for s in telemetry})),
            mem_clocks=json.dumps(sorted({s['telemetry']['clocks.mem'] for s in telemetry})),
            model_config_sha256=measurement['model_config_sha256'],
            prompt_ids_sha256=measurement['prompt_ids_sha256'],
            checkpoint_sha256=measurement['checkpoint']['checkpoint_sha256'],
            versions=json.dumps(measurement['versions'], sort_keys=True),
            decode_sha256=config['decode_sha256'], source=str(path), source_sha256=sha(path))
        processes.append(row);groups.setdefault(arm, []).append(row)
    for value in historical['rows']:
        original = Path(value['tokens']).parent / 'B1/measurements.json'
        original_measurement = read(original)
        processes.append(dict(arm='D', slot=value['job'], pid=original_measurement['pid'], new_reused='reused',
            tpot_ms=value['tpot_ms'], ttft_ms=value['ttft_ms'], e2e_ms=value['e2e_ms'],
            gpu=value['gpu'], gpu_inventory='historical H800 GPU0', sm_clocks='["1980"]',
            mem_clocks='["2619"]', model_config_sha256=original_measurement['model_config_sha256'],
            prompt_ids_sha256=original_measurement['prompt_ids_sha256'],
            checkpoint_sha256=original_measurement['checkpoint']['checkpoint_sha256'],
            versions=json.dumps(original_measurement['versions'], sort_keys=True),
            decode_sha256=manifest['binary_sha256'],
            source=str(original), source_sha256=sha(original)))
    for arm in 'ABCDE':
        if arm == 'D':
            values = [r['tpot_ms'] for r in historical['rows']]
            tpot = manifest['measured']['tpot_ms']
            spread = manifest['measured']['process_range_pct']
            source = 'configs/serving/sm90r3/qwen3_1_7b_b1.json; SM90R3 main P-FINAL'
        else:
            values = [r['tpot_ms'] for r in groups[arm]]
            if len(values) != 3:raise ValueError('expected three fresh processes per new arm')
            tpot = statistics.median(values)
            spread = (max(values) - min(values)) / tpot * 100
            source = json.dumps([r['source'] for r in groups[arm]])
        rows.append(dict(arm=arm,
            decomposition='operator-local L (=D)' if arm in 'AC' else 'TileMega-selected D',
            policy='basic static: aggregate grid barriers' if arm in 'AB' else
                'TileMega: QKV group release' if arm in 'CD' else 'kernel completion; D intra-stage bodies',
            execution='stage-granular CUDA Graph replay' if arm == 'E' else 'megakernel',
            tpot_ms=tpot, processes_ms=json.dumps(values), process_range_pct=spread,
            processes=3, new_reused='reused' if arm == 'D' else 'new', source=source,
            evidence='verified performance; no new correctness validation'))
    times = {r['arm']: r['tpot_ms'] for r in rows}
    ratios = [dict(ratio=f'T{a}/T{b}', value=times[a] / times[b], interpretation=label)
        for a, b, label in [('A', 'B', 'decomposition; L=D, so no decomposition treatment difference'),
            ('C', 'D', 'decomposition; L=D, with historical D denominator'),
            ('A', 'C', 'policy: QKV-to-attention release only'),
            ('B', 'D', 'policy with historical D denominator'),
            ('A', 'D', 'joint comparison with historical D denominator'),
            ('E', 'D', 'overall execution organization with historical D denominator'),
            ('E', 'B', 'same stage computations and aggregate inter-stage boundaries')]]
    rejected_driver_attempts = [dict(arm=v['arm'], directory=a['directory'],
        code=a['code'], classification=a.get('failure_classification'))
        for v in state['items'] for a in v['attempts'] if a.get('code') not in (0, 75)]
    matching_environment = {field: len({r[field] for r in processes}) == 1
        for field in ('versions', 'model_config_sha256', 'prompt_ids_sha256',
                      'checkpoint_sha256', 'sm_clocks', 'mem_clocks')}
    receipt = dict(evidence='verified configuration, compilation and performance only',
        rows=rows, ratios=ratios, accepted_new_processes=12, reused_processes=3,
        correctness_processes=0, D_remeasured=False, body_source_head=spec['body_source_head'],
        D_job_submitted=False, C_equals_D_configuration=True,
        D_remeasured_flag_meaning='No separate D-arm job; C uses the same configuration because L=D',
        rejected_driver_attempts=rejected_driver_attempts,
        new_and_historical_environment_match=matching_environment,
        body_source_resolution=spec['body_version_resolution'],
        historical_source_recovery=str(run / 'historical-source-recovery.json'),
        source_consistency=str(run / 'historical-consistency.json'),
        source_consistency_sha256=sha(run / 'historical-consistency.json'),
        consistency=read(run / 'historical-consistency.json'),
        spec=spec, spec_sha256=sha(run / 'historical-spec.json'),
        historical_source_recovery_sha256=sha(run / 'historical-source-recovery.json'),
        experiment_root=str(ROOT), timing_state=str(run / 'timing/state.json'),
        timing_state_sha256=sha(run / 'timing/state.json'))
    out = publish / 'docs/results';out.mkdir(parents=True, exist_ok=True)
    csv_write(out / f'{STEM}.csv', rows);csv_write(out / f'{STEM}_processes.csv', processes)
    csv_write(out / f'{STEM}_ratios.csv', ratios)
    (out / f'{STEM}_candidates.csv').write_bytes((run / 'historical-local/candidate_scores.csv').read_bytes())
    write(out / f'{STEM}_provenance.json', receipt)
    markdown = ('# Qwen3-1.7B SM90 native decode A-E ablation\n\n'
        'Verified scope: BS=1, BF16, ISL/OSL=64/1024. D reuses the SM90R3 main '
        '`measured.tpot_ms=1.6754311115689438`; the four other arms each have three '
        'accepted fresh timing processes on one physical H800. The D reference is '
        'held fixed; no solve or separate D-arm job was submitted.\n\n'
        'The frozen D artifact was built from `4b9d1362e648e2ee2cc7f96da9422f97a6d36c45`. '
        'All 606 recorded source files and the target JSON matched when recovered. '
        'The experiment uses branch `sm90-decode-ae-r3`, retaining D task-body headers '
        'and adding only a sequence execution adapter. The preliminary merged-version '
        'builds were not timed and are excluded.\n\n'
        'The fixed pipeline is export bridge -> BuildModelPlan -> imported '
        'semantics -> MaterializeSm90Dense with pinned coordinates and task '
        'ranges -> typed native Program -> nvcc/identity -> ServingEngine L1 '
        'decode-loop -> request-wall TPOT. --tmexec-config bypasses solve.\n\n'
        'Verified CPU selection: independently minimize existing Gemm(pre=false) and '
        'complete Attention costs over deduplicated FamilyCandidates TN and '
        'AttentionCandidates chunks plus D values, filtering with both policies\' '
        'Legal checks and available calibrated terms. All other D parameters stay '
        'fixed; slot_bytes follows 2*TN*TK_D. No Structure/derive, whole-program '
        'search or GPU candidate sweep is used for L. Exact ties prefer D. '
        'The historical calibration reports zero stale body headers.\n\n'
        '**L equals D:** TN=(32,16,48,16,128), chunks=16. A=B configurations, and '
        'C has the same configuration and byte-identical generated CUDA source as '
        'historical D. Thus this case has no decomposition treatment difference; '
        'measured A/B or C/D deviations cannot establish a decomposition gain.\n\n'
        'Basic static retains cyclic/grid-stride ownership, typed stage order, '
        'body-internal LA reductions, fusion and embedding handoff. Its only '
        'policy change is group_wait=false, replacing QKV-to-attention group '
        'release with the aggregate grid boundary. Ownership/order are not '
        'independent optimizations in this experiment.\n\n'
        'E retains the D tiles, chunks, LA attention merge, folded argmax and '
        'head-to-embedding body. It replays initial/steady CUDA graphs over '
        'ordinary typed-stage kernels with NoTail and streamed=false. Only absolute '
        'step 0 runs the embedding prologue; steady steps start with QKV. Final is '
        'conditional on kPostHead (false in frozen R3), so no ArgmaxStage is forced. '
        'Kernel completion orders stages; body-internal synchronization is retained. '
        'E/B preserve the same stage computations and aggregate boundaries, so TE/TB '
        'is also reported.\n\n'
        'Timing uses the historical prefill binary and checkpoint algorithm/seed 0, '
        'L1 decode-loop execution, one warmup plus three timed requests at each '
        'N=1 and N=1024. Per-process TPOT=(median wall N1024 - median wall N1)/1023; '
        'the arm statistic is the median of three fresh processes. Order is '
        'A B C E / B C E A / C E A B. Existing GPU guards reject interference '
        'and retain retry logs. The current recorded idle-power guard is reused '
        'without changing GPU clocks or collecting new calibration. No C1/C2, '
        'token/logit comparison, HF, sanitizer, synchronization stress or profiling '
        'was run. The common driver checks output shape only.\n\n'
        '| Arm | Decomposition | Policy | Execution | TPOT ms | Three process TPOTs ms | Range % | New/reused |\n'
        '|---|---|---|---|---:|---|---:|---|\n')
    for r in rows:
        markdown += (f'| {r["arm"]} | {r["decomposition"]} | {r["policy"]} | {r["execution"]} | '
            f'{r["tpot_ms"]:.9f} | {r["processes_ms"]} | {r["process_range_pct"]:.6f} | {r["new_reused"]} |\n')
    markdown += '\n| Ratio | Value | Interpretation |\n|---|---:|---|\n'
    for r in ratios:markdown += f'| {r["ratio"]} | {r["value"]:.9f} | {r["interpretation"]} |\n'
    markdown += ('\nD denominators refer to another timing session. Matching source and '
        'task bodies do not remove cross-session variability; per-process values, '
        'clock observations and full precision ratios are retained in the companion CSVs. '
        'Results are performance measurements, with correctness explicitly outside '
        'this collection. Two A driver failures were excluded: one guard API '
        'mismatch before timing and one missing metadata field after timing. '
        'Both attempts and their raw records are retained; exclusion was independent '
        'of observed performance. No separate D-arm job was submitted; the prescribed '
        'C arm has the same configuration as D because L=D. Raw provenance: `' + str(run) + '`.\n\n'
        'Reproduction commands are in `configs/serving/sm90_ae/README.md` on '
        '`sm90-decode-ae-r3`. Source recovery and build/identity receipts retain the '
        'complete original and failed-attempt evidence under runs/.\n')
    (out / f'{STEM}.md').write_text(markdown)
    write(run / 'final-summary.json', receipt)
    print(json.dumps(dict(times=times, ratios=ratios), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish-root', type=Path, default=ROOT)
    args = parser.parse_args();report(args.run.resolve(), args.publish_root.resolve())
