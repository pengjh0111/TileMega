"""Measure TP8-only attention V3, retain V2, and reuse every external baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

from tilemega.tp.sweep_tp8 import Campaign, checked, read, ref, save, summarize
from tilemega.tp.solve_tp import ROOT


def check_default_cohort(result):
    candidates = result['candidates']
    for correctness in [c['correctness'] for c in candidates] + [result['final_correctness']]:
        if not read(checked(correctness['C1']))['pass']:
            raise ValueError('C-1 did not pass')
        checks = [read(checked(item)) for item in correctness['C2']]
        expected = 5 if correctness is result['final_correctness'] else 2
        if len(checks) != expected or len({r['PID'] for r in checks}) != expected:
            raise ValueError('fresh C-2 cohort is incomplete')
        if not all(r['pass'] and r['rank_tokens_identical'] for r in checks):
            raise ValueError('C-2 or rank agreement failed')
        if len({r['token_sha256'] for r in checks}) != 1:
            raise ValueError('fresh processes generated different tokens')
    for timing in [c['timing'] for c in candidates] + [result['final_timing']]:
        pids = [read(checked(item))['PID'] for item in timing['processes']]
        if len(pids) != 2 or len(set(pids)) != 2:
            raise ValueError('two fresh timing processes are required')
        if read(checked(timing['guard']))['code']:
            raise ValueError('timing guard failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/mx7/tp8_opt/attention/run')
    parser.add_argument('--builds', type=Path, default=ROOT / 'runs/mx7/tp8_opt/attention/cost_build')
    parser.add_argument('--original', type=Path, default=ROOT / 'runs/mx7/tp8_queue')
    parser.add_argument('--model', type=Path, default=Path('/root/models/qwen3_1_7b_rand'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--poll-s', type=float, default=10)
    parser.add_argument('--idle-wait-s', type=float, default=-1)
    parser.add_argument('--cucc', default='/opt/maca/tools/cu-bridge/bin/cucc')
    parser.add_argument('--vllm-python', default='/root/venv_vllm/bin/python')
    parser.add_argument('--sglang-python', default=sys.executable)
    parser.add_argument('--torchcompile-python', default='/root/venv_mx7_tc/bin/python')
    args = parser.parse_args();args.costs = None
    campaign = Campaign(args)
    out = campaign.out;original = args.original.resolve();builds = args.builds.resolve()
    old_acceptance = read(original / 'ACCEPTED.json')
    old_costs = checked(old_acceptance['costs'])
    old_catalog = read(old_costs)
    frozen = ref(original / 'TP8_results.csv')
    export = Path(read(original / 'jobs/export/ACCEPTED.json')['folder']) / 'export'
    bridge = export / 'qwen3_1_7b_tp8_decode/bridge.json'
    save(out / 'FROZEN_BASELINES.json', dict(comparison=frozen,
        records=[ref(p) for p in (original / 'results').glob('*.json') if not p.name.startswith('tilemega_')],
        rule='no vLLM, SGLang, torch.compile or TP1 baseline jobs are launched'))
    frozen_gate = list(csv.DictReader((ROOT / 'docs/results/MX7_frozen_gate_c550.csv').open()))
    gates = [r for r in frozen_gate if r['record_type'] == 'tp8_attention_v3_optin']
    if len(gates) != 8 or any(r['status'] != 'pass' for r in gates):
        raise ValueError('all eight frozen paths must pass before continuing')
    keys = read(builds / 'KEYS_CPU.json')
    driver = builds / 'frozen/experiments/mx7/measure_costs.py'
    if any(r['key'].get('attn_version') != 3 or r['key']['tp'] != 8 for r in keys):
        raise ValueError('attention build scope differs')
    chunks = [keys[i:i + 32] for i in range(0, len(keys), 32)]

    def on_card(card):
        results = []
        for index in range(card, len(chunks), 8):
            key_path = out / 'keys' / f'{index:04d}.json'
            save(key_path, chunks[index]);records = []
            for number in (1, 2, 3):
                folder, _ = campaign.job(f'attention{index:04d}_r{number}', lambda p: [sys.executable,
                    driver, '--keys', key_path, '--out', p / 'raw.jsonl', '--round', number], str(card))
                records.append(dict(round=number, raw=ref(folder / 'raw.jsonl'),
                    process=ref(folder / 'PROCESS.json'), guard=ref(folder / 'guard/guard_result.json')))
            results.append((index, dict(keys=ref(key_path), records=records)))
        return results
    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = [item for group in pool.map(on_card, range(8)) for item in group]
    request = dict(source=ref(builds / 'SOURCE.json'), all_keys=ref(builds / 'KEYS_CPU.json'),
        driver=ref(driver), domain=ref(builds / 'DOMAIN.json'),
        CPU_resource_rejections=read(builds / 'REJECTIONS_CPU.json'), jobs=[r for _, r in sorted(measured)])
    request_path = out / 'REQUEST.json';save(request_path, request)
    raw, _ = campaign.job('aggregate', lambda p: [sys.executable, 'experiments/mx7/aggregate_costs.py',
        '--request', request_path, '--out', p / 'catalog.json'])
    constants = out / 'constants.json';save(constants, old_catalog['tp_constants'])
    qualified, _ = campaign.job('qualify', lambda p: [sys.executable, 'experiments/mx7/qualify_costs.py',
        '--request', request_path, '--catalog', raw / 'catalog.json', '--constants', constants,
        '--merge-tp8-costs', old_costs, '--card', '7', '--work', p / 'noise', '--out', p / 'costs.json'])
    costs = qualified / 'costs.json'
    save(out / 'COSTS_READY.json', dict(costs=ref(costs), qualification=ref(costs.with_suffix('.qualification.json'))))
    smoke = ROOT / 'runs/mx7/tp8_opt/attention/smoke/PASSED.json'
    while not smoke.exists():
        time.sleep(10)
    final_rows, selection_rows = [], []
    baseline_rows = list(csv.DictReader((original / 'TP8_results.csv').open()))
    for batch in (16, 1, 2, 4, 8):
        folder, _ = campaign.job(f'select_B{batch}', lambda p: [sys.executable, '-m', 'tilemega.tp.solve_tp',
            '--tp', '8', '--batch', batch, '--group', '0,1,2,3,4,5,6,7', '--costs', costs,
            '--bridge', bridge, '--model', args.model, '--out', p / 'selection'])
        result = read(folder / 'selection/results.json');check_default_cohort(result)
        old = read(original / f'jobs/select_B{batch}/ACCEPTED.json')
        old = read(Path(old['folder']) / 'selection/results.json')
        arms = dict(control=old['resources']['binary'], candidate=result['resources']['binary'])
        pairs = {name: [] for name in arms}
        for number in (1, 2):
            for name, binary in arms.items():
                measured, receipt = campaign.job(f'paired_B{batch}_{name}_{number}', lambda p, binary=binary: [
                    sys.executable, '-m', 'tilemega.serving.measure_tp', '--model', args.model,
                    '--decode-so', checked(binary), '--prompt-ids', ROOT / 'configs/prompts/qwen_isl64.json',
                    '--batch', batch, '--devices', '0,1,2,3,4,5,6,7', '--max-new-tokens', '1024',
                    '--warmup', '1', '--repeats', '2', '--policy', ROOT / 'configs/bench/measurement_policy_c550.json',
                    '--out', p / 'measure'], '0,1,2,3,4,5,6,7')
                metrics = measured / 'measure/measurements.json';value = read(metrics)
                if not value['timed_tokens_identical'] or not value['rank_tokens_identical']:
                    raise ValueError('paired timing token/rank consistency failed')
                pairs[name].append(dict(metrics=ref(metrics), process=receipt['process'],
                    guard=ref(measured / 'guard/guard_result.json'), tpot_ms=value['tpot_seconds'] * 1000))
        medians = {name: statistics.median(r['tpot_ms'] for r in rows) for name, rows in pairs.items()}
        threshold = max([.01] + [(max(r['tpot_ms'] for r in rows) - min(r['tpot_ms'] for r in rows)) / medians[name]
                                 for name, rows in pairs.items()])
        use_new = medians['candidate'] < medians['control'] * (1 - threshold)
        selected = result if use_new else old
        selected_arm = 'candidate' if use_new else 'control'
        pin_path = checked(selected['selected_pin']);config = read(pin_path)['mx_decode_v2']
        if batch in (1, 16):
            built, _ = campaign.job(f'delay_build_B{batch}', lambda p: ['env', f'PIN={pin_path}',
                f'EXPORT_ROOT={export}', f'CELLS={p / "plans"}', 'TILEMEGA_TP_LITMUS=1',
                'scripts/mx_tp_cell.sh', 'build', '8', batch])
            so = built / f'plans/tp8/B{batch}/plan.so'
            reference = Path(selected['final_correctness']['C2'][0]['path']).parent / 'tokens.json'
            for number in range(5):
                campaign.job(f'delay_B{batch}_p{number}', lambda p: [sys.executable,
                    '-m', 'tilemega.serving.check_tp', '--model', args.model, '--decode-so', so,
                    '--prompt-ids', ROOT / 'configs/prompts/qwen_isl64.json', '--batch', batch,
                    '--devices', '0,1,2,3,4,5,6,7', '--steps', '1024', '--reference', reference, '--out', p / 'check'],
                    '0,1,2,3,4,5,6,7')
        values = [read(r['metrics']['path']) for r in pairs[selected_arm]]
        sg = next(r for r in baseline_rows if r['system'] == 'sglang' and int(r['batch']) == batch)
        previous = next(r for r in baseline_rows if r['system'] == 'tilemega' and int(r['batch']) == batch)
        row = dict(batch=batch, tp=8, status='pass', selected_arm=selected_arm,
            attn_version=config['attn_version'], split=config['split'], attn_prefetch=config['attn_prefetch'],
            tpot_ms=medians[selected_arm], previous_tpot_ms=previous['tpot_ms'],
            ttft_ms=statistics.median(v['ttft_seconds'] for v in values) * 1000,
            e2e_ms=statistics.median(v['e2e_seconds'] for v in values) * 1000,
            tokens_per_s=statistics.median(v['output_tokens_per_second'] for v in values),
            sglang_tpot_ms=sg['tpot_ms'], speedup_vs_sglang=float(sg['tpot_ms']) / medians[selected_arm],
            paired_control_ms=medians['control'], paired_candidate_ms=medians['candidate'],
            paired_threshold=threshold, pin=selected['selected_pin'], result=ref(folder / 'selection/results.json'), pairs=pairs)
        save(out / f'B{batch}_ACCEPTED.json', row);final_rows.append(row)
        print('TP8 attention final', batch, config['attn_version'], config['split'], medians[selected_arm], row['speedup_vs_sglang'], flush=True)
    if ref(original / 'TP8_results.csv') != frozen:
        raise ValueError('frozen original TP8 table changed')
    campaign.check_source()
    save(out / 'ACCEPTED.json', dict(evidence='verified', results=sorted(final_rows, key=lambda r: r['batch']),
        costs=ref(costs), baseline_comparison=frozen, all_external_baselines='unchanged'))
    print('all five TP8 attention scopes accepted', flush=True)


if __name__ == '__main__':
    main()
