"""Qualify and publish TP8 refinement without launching an external baseline."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import os
import shutil
import statistics
import time

from tilemega.tp.solve_tp import Cell, ROOT, ref, save
from tilemega.tp.process_cohorts import seal_and_attach, validate_run, validate_checks


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'runs/mx7/tp8_opt/depth')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/mx7/tp8_opt/acceptance')
    args = parser.parse_args()
    inputs = args.inputs.resolve();out = args.out.resolve();out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'controller.py')
    original = ROOT / 'runs/mx7/tp8_queue'
    frozen = ref(original / 'TP8_results.csv')
    export = Path(read(original / 'jobs/export/ACCEPTED.json')['folder']) / 'export'
    cell = Cell.__new__(Cell)
    cell.a = SimpleNamespace(group='0,1,2,3,4,5,6,7', startup_hidden_mib=None)
    cell.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288',
        MX_GPU_WAIT_IDLE='1', MX_GPU_IDLE_WAIT_S='-1', MX_GPU_UNLIMITED_RETRIES='1',
        MX_GPU_JOB_LOG=str(out / 'gpu_jobs.log'), TILEMEGA_GPU_JOB_LOG=str(out / 'gpu_jobs.log'))
    cell.cpu_env = dict(cell.env, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
    seal_and_attach(out / 'process_cohorts.py', cell, ROOT)
    accepted = []
    for batch in (16, 1, 2, 4, 8):
        result_path = inputs / f'B{batch}/RESULT.json'
        while not result_path.exists():
            time.sleep(10)
        result = read(result_path)
        validate_checks(result['checks']['qualification'])
        selected = result['selected'];scope = out / f'B{batch}';scope.mkdir()
        query_inputs = Path(selected['build_folder']) / 'query_inputs.json'
        query, run = cell.output_job(scope, 'occupancy', scope / 'occupancy.json', lambda dst: [
            'python', '-m', 'tilemega.tp.plan_occupancy', '--inputs', str(query_inputs), '--out', str(dst)])
        if not read(query)['qualified']:
            raise RuntimeError('selected all-rank residency failed')
        delays = []
        if batch in (1, 16):
            build = cell.run_command(scope / 'delay_build', ['env', f'PIN={selected["pin"]["path"]}',
                f'EXPORT_ROOT={export}', f'CELLS={out / "delay_plans"}', 'WATCHDOG=0',
                'TILEMEGA_TP_LITMUS=1', 'scripts/mx_tp_cell.sh', 'build', '8', str(batch)])
            if build['returncode']:
                raise RuntimeError('selected delay build failed')
            so = out / f'delay_plans/tp8/B{batch}/plan.so'
            if not read(str(so) + '.plan.json').get('tp_litmus'):
                raise ValueError('delay instrumentation missing')
            reference = Path(result['checks']['C2'][0]['path']).parent / 'tokens.json'
            for number in range(5):
                target, run = cell.output_job(scope, f'delay{number}', scope / f'delay{number}', lambda dst: [
                    'python', '-m', 'tilemega.serving.check_tp', '--model', '/root/models/qwen3_1_7b_rand',
                    '--decode-so', str(so), '--prompt-ids', str(ROOT / 'configs/prompts/qwen_isl64.json'),
                    '--batch', str(batch), '--devices', cell.a.group, '--steps', '1024',
                    '--reference', str(reference), '--out', str(dst)])
                check = read(target / 'check.json')
                if not check['pass'] or not check['rank_tokens_identical']:
                    raise RuntimeError('selected delay check failed')
                delays.append(dict(check=ref(target / 'check.json'), run=run))
                print('refinement delay pass', batch, number, flush=True)
        save(scope / 'QUALIFIED.json', dict(result=ref(result_path), residency=ref(query), delays=delays))
        accepted.append(ref(scope / 'QUALIFIED.json'))
        print('refinement qualified', batch, flush=True)
    if ref(original / 'TP8_results.csv') != frozen:
        raise ValueError('original TP8 comparison changed')
    save(out / 'ACCEPTED.json', dict(qualified=accepted, original_comparison=frozen,
        all_external_baselines='reused unchanged', evidence='verified'))
    table = list(csv.DictReader((original / 'TP8_results.csv').open()))
    updated = []
    lines = ['# MX-7 TP8 depth refinement', '',
        '**verified** Qwen3-1.7B, BF16 greedy, ISL64/OSL1024. External baselines are reused unchanged.', '',
        'The diagnostic screen reuses immutable packed weights inside one process. It is not final performance evidence.',
        'Reported TileMega values use independent ordinary processes, warmup1/timed2, paired against the original pin.',
        'Only a gain above max(1%, the two arms\' process spread) enables a new configuration.', '',
        '| BS | previous TM ms | new TM ms | paired control ms | paired gain | SGLang ms | SGLang/new TM |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    pins = ROOT / 'configs/pins/mx7/tp8_refined';pins.mkdir(exist_ok=True)
    for batch in (1, 2, 4, 8, 16):
        result = read(inputs / f'B{batch}/RESULT.json')
        values = [read(row['measurement']['path']) for row in result['timing']]
        control = statistics.median(row['tpot_ms'] for row in result['paired']['control'])
        metrics = dict(tpot_ms=statistics.median(v['tpot_seconds'] for v in values) * 1000,
            ttft_ms=statistics.median(v['ttft_seconds'] for v in values) * 1000,
            e2e_ms=statistics.median(v['e2e_seconds'] for v in values) * 1000,
            tokens_per_s=statistics.median(v['output_tokens_per_second'] for v in values))
        old = next(row for row in table if row['system'] == 'tilemega' and int(row['batch']) == batch)
        sg = next(row for row in table if row['system'] == 'sglang' and int(row['batch']) == batch)
        updated.append(dict(tp=8, batch=batch, system='tilemega', status='pass', **metrics,
            previous_tpot_ms=old['tpot_ms'], paired_control_tpot_ms=control,
            paired_speedup=control / metrics['tpot_ms'], selected_variant=result['selected']['name'],
            speedup_vs_sglang=float(sg['tpot_ms']) / metrics['tpot_ms'],
            measurement_evidence=json.dumps(result['timing']), qualification=ref(out / f'B{batch}/QUALIFIED.json')['sha256']))
        source = Path(result['selected']['pin']['path'])
        shutil.copy2(source, pins / f'qwen3_1_7b_tp8_B{batch}_selected.json')
        lines.append(f'| {batch} | {float(old["tpot_ms"]):.6f} | {metrics["tpot_ms"]:.6f} | {control:.6f} | '
                     f'{control / metrics["tpot_ms"]:.3f}x | {float(sg["tpot_ms"]):.6f} | '
                     f'{float(sg["tpot_ms"]) / metrics["tpot_ms"]:.3f}x |')
    lines += ['', '**verified** Every selected pin passes C-1, five fresh 1024-token C-2 starts, all-rank residency,',
        'and rank token consistency. B1/B16 selected delay builds each pass five fresh processes.',
        'No arithmetic, LL8 format, peer primitive, barrier, TP1 or MoE body changed.',
        '', 'Raw attempts, failed/interfered runs, source/binary receipts, screen data and paired measurements:',
        '`runs/mx7/tp8_opt/`. Earlier measurements and baselines remain intact.']
    csv_path = ROOT / 'docs/results/MX7_tp8_refinement_c550.csv'
    if csv_path.exists():
        raise ValueError('preserve the previous published refinement')
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(updated[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(updated)
    report = ROOT / 'docs/results/MX7_tp8_refinement_c550.md'
    with report.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('TP8 refined results published locally', flush=True)


if __name__ == '__main__':
    main()
