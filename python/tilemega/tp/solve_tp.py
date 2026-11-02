"""Qualify dense TP k-best candidates, then select by four measured cohorts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tilemega.build.identity import verify
from tilemega.mx5_plan_occupancy import validate_inputs
from tilemega.mx5_resources import inspect
from tilemega.tp.selection_tp import k_best, pin, select_measured, stage_lists

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_qualification(path):
    qualification = json.loads(path.with_suffix('.qualification.json').read_text())
    if qualification.get('cost_table_sha256') != sha(path) or qualification.get('solver_eligible') is not True:
        raise ValueError('source-bound eligible TP costs are required')
    catalog = json.loads(path.read_text())
    if catalog.get('kind') != 'mx7_tp_measured_stage_costs' or catalog.get('complete_scope') is not True:
        raise ValueError('complete MX7 TP costs are required')
    if set(catalog.get('tp_constants', {})) != {'t_ar', 't_head_x'}:
        raise ValueError('measured AR and head exchange constants are required')
    return catalog, qualification


def check_constant_group(catalog, tp, devices):
    measured_group = catalog.get('tp_gpu_groups', {}).get(str(tp))
    if measured_group is not None and devices != measured_group:
        raise ValueError('TP constants were measured on a different physical GPU group')


class Cell:
    def __init__(self, args):
        self.a = args
        devices = list(map(int, args.group.split(',')))
        if len(devices) != args.tp or len(set(devices)) != args.tp or not set(devices) <= set(range(8)):
            raise ValueError('TP degree and physical GPU group differ')
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), NEEDS_MIB='12288',
                        TILEMEGA_GPU_JOB_LOG=str(ROOT / 'runs/mx7/gpu_jobs.log'))
        self.cpu_env = dict(self.env, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
        self.catalog, qualification = check_qualification(args.costs)
        check_constant_group(self.catalog, args.tp, devices)
        sealed = self.out / 'inputs'
        sealed.mkdir()
        for source in (args.costs, args.costs.with_suffix('.qualification.json'), args.bridge,
                       args.bridge.parent / 'manifest.json', args.compiler):
            target = sealed / source.name
            shutil.copy2(source, target)
            if sha(target) != sha(source):
                raise ValueError('sealed selection input differs from source')
        self.compiler = sealed / args.compiler.name
        self.bridge = sealed / args.bridge.name
        self.cost_sha = sha(args.costs)
        manifest = json.loads((sealed / 'manifest.json').read_text())
        if manifest.get('tp') != args.tp:
            raise ValueError('export manifest TP differs from --serving-tp')
        cohort_source = None
        if getattr(args, 'guard_each_process', False):
            cohort_source = sealed / 'process_cohorts.py'
            shutil.copy2(ROOT / 'python/tilemega/tp/process_cohorts.py', cohort_source)
        save(sealed / 'IDENTITY.json', dict(inputs=[ref(p) for p in sealed.iterdir()],
                                          qualification=qualification, runner=ref(__file__)))
        shutil.copy2(__file__, sealed / 'solve_tp.py')
        if cohort_source is not None:
            from tilemega.tp.process_cohorts import load_and_attach
            load_and_attach(cohort_source, self, ROOT)

    def run_command(self, folder, argv, guarded=False, cap=900):
        folder.mkdir(parents=True, exist_ok=False)
        if guarded:
            argv = [ROOT / 'scripts/mx_gpu_group_job.sh', self.a.group, str(cap), folder / 'guard', *argv]
        argv = list(map(str, argv))
        start = time.time()
        with (folder / 'stdout.log').open('x') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=self.env if guarded else self.cpu_env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        record = dict(argv=argv, PID=child.pid, start=start, end=time.time(), returncode=code)
        save(folder / 'PROCESS.json', record)
        return record

    def compile(self, folder, pin_path):
        so = folder / 'plan.so'
        argv = [self.compiler, 'compile', self.bridge, so, '--serving', 'decode', '--emit', 'serving',
                '--frontend', 'decoder', '--batch', self.a.batch, '--prompt-len', '64', '--capacity', '1088',
                '--past-range', '0:1086', '--serving-tp', self.a.tp, '--runtime-target', self.a.target,
                '--evaluate-configs', pin_path, '--selection', 'predicted', '--watchdog', '0']
        execution = self.run_command(folder, argv)
        if execution['returncode']:
            return dict(status='compile_or_resource_rejected', process=ref(folder / 'PROCESS.json'))
        identity = verify(so)
        resources = inspect(so, str(so) + '.ptxas.log', 512)
        save(folder / 'actual_resources.json', resources)
        if resources['violations']:
            return dict(status='resource_rejected', resources=resources)
        spec = dict(kind='mx5_plan_occupancy_inputs', schema_version=1, threads=512,
                    context_kind='native_identity', binary=ref(so), generated_source=ref(str(so) + '.cu'),
                    resources=ref(folder / 'actual_resources.json'), log=ref(str(so) + '.ptxas.log'),
                    context=ref(str(so) + '.identity.json'), source_snapshot=ref(str(so) + '.source.json'),
                    manifest=ref(str(so) + '.plan.json'))
        validate_inputs(spec)
        save(folder / 'query_inputs.json', spec)
        independent = getattr(self.a, 'guard_each_process', False)
        for attempt in range(2 if independent else 1):
            query_folder = folder / ('occupancy' if not attempt else 'occupancy_guard_retry1')
            query_result = query_folder / 'result.json'
            execution = self.run_command(query_folder, [sys.executable, '-m', 'tilemega.tp.plan_occupancy',
                '--inputs', folder / 'query_inputs.json', '--out', query_result], guarded=True, cap=300)
            if independent and execution['returncode'] == 75:
                if attempt:
                    raise RuntimeError('residency query interrupted twice; preserve both guard records')
                continue
            if execution['returncode'] or not json.loads(query_result.read_text())['qualified']:
                return dict(status='actual_residency_rejected', occupancy=ref(query_result))
            break
        return dict(status='qualified', binary=ref(so), artifact_id=identity['artifact_id'],
                    resources=ref(folder / 'actual_resources.json'), occupancy=ref(query_result))

    def correctness(self, folder, fresh=2):
        folder.mkdir()
        so = folder.parent / 'plan.so'
        devices = ','.join(map(str, range(self.a.tp)))
        check = [sys.executable, '-m', 'tilemega.serving.check_tp', '--model', self.a.model,
                 '--decode-so', so, '--prompt-ids', self.a.prompt_ids, '--batch', self.a.batch,
                 '--devices', devices]
        commands = [check + ['--steps', '512', '--out', folder / 'c1_tokens']]
        commands += [[sys.executable, '-m', 'tilemega.serving.hf_check', '--model', self.a.model,
                      '--prompt-ids', self.a.prompt_ids, '--generated', folder / 'c1_tokens/tokens.json',
                      '--out', folder / 'hf_check.json', '--skip-free-greedy']]
        for number in range(fresh):
            command = check + ['--steps', '1024', '--out', folder / f'c2_process{number}']
            if number:
                command += ['--reference', folder / 'c2_process0/tokens.json']
            commands.append(command)
        commands = [list(map(str, command)) for command in commands]
        driver = folder / 'driver.py'
        driver.write_text('import subprocess\nfor argv in ' + repr(commands) + ':\n subprocess.run(argv, check=True)\n')
        execution = self.run_command(folder / 'execution', [sys.executable, driver], guarded=True)
        if execution['returncode']:
            raise RuntimeError('correctness failure; retain records and diagnose once: ' + str(folder))
        hf = folder / 'hf_check.json'
        if not json.loads(hf.read_text())['pass']:
            raise RuntimeError('C-1 failure; diagnose BF16 AR under MX7 B6: ' + str(hf))
        return dict(C1=ref(hf), C2=[ref(folder / f'c2_process{i}/check.json') for i in range(fresh)])

    def measure(self, folder):
        values, records, commands = [], [], []
        for number in (1, 2):
            out = folder / f'timing_process{number}'
            commands.append(list(map(str, [sys.executable, '-m', 'tilemega.serving.measure_tp',
                '--model', self.a.model, '--decode-so', folder / 'plan.so', '--prompt-ids', self.a.prompt_ids,
                '--batch', self.a.batch, '--devices', ','.join(map(str, range(self.a.tp))),
                '--max-new-tokens', '1024', '--warmup', '1', '--repeats', '2',
                '--policy', ROOT / 'configs/bench/measurement_policy_c550.json', '--out', out / 'measure'])))
        cohort = folder / 'timing_execution'
        cohort.mkdir()
        driver = cohort / 'driver.py'
        driver.write_text('import json,pathlib,subprocess,time\n'
            'for argv in ' + repr(commands) + ':\n'
            ' folder=pathlib.Path(argv[-1]).parent;folder.mkdir()\n'
            ' start=time.time();child=subprocess.Popen(argv);code=child.wait()\n'
            ' (folder/"PROCESS.json").write_text(json.dumps(dict(argv=argv,PID=child.pid,start=start,end=time.time(),returncode=code))+"\\n")\n'
            ' if code:raise SystemExit(code)\n')
        # Each measurement uses a fresh process; only the guard preflight is shared.
        execution = self.run_command(cohort / 'execution', [sys.executable, driver], guarded=True)
        if execution['returncode']:
            raise RuntimeError('timing cohort failed; preserve and remedy once: ' + str(cohort))
        processes = []
        for number in (1, 2):
            out = folder / f'timing_process{number}'
            process = out / 'PROCESS.json'
            processes.append(ref(process))
            metrics = out / 'measure/measurements.json'
            value = json.loads(metrics.read_text())
            if not value['timed_tokens_identical'] or not value['rank_tokens_identical']:
                raise RuntimeError('C-2 / rank consistency failed during timing')
            values.append(value['tpot_seconds'] * 1e9)
            records.append(ref(metrics))
        if len({json.loads(Path(item['path']).read_text())['PID'] for item in processes}) != 2:
            raise RuntimeError('two distinct timing processes are required')
        return dict(fresh_process_ns=values, measurements=records, processes=processes,
                    guard=ref(cohort / 'execution/guard/guard_result.json'), driver=ref(driver))

    def run(self):
        model = f'qwen3_1_7b_tp{self.a.tp}'
        lists = stage_lists(self.catalog['rows'], model, self.a.batch, self.catalog['tp_constants'])
        qualified, rejected = [], []
        for choice in k_best(lists, 28):
            folder = self.out / f"rank{choice['candidate_rank']:07d}"
            folder.mkdir()
            save(folder / 'CHOICE.json', choice)
            pin_path = folder / 'resource_pin.json'
            save(pin_path, pin(choice, self.cost_sha, self.a.tp))
            result = self.compile(folder / 'build', pin_path)
            candidate = dict(choice=choice, build=result, folder=str(folder / 'build'), pin=ref(pin_path))
            save(folder / 'RESOURCE_RESULT.json', candidate)
            (qualified if result['status'] == 'qualified' else rejected).append(candidate)
            if len(qualified) == 4:
                break
        if len(qualified) != 4:
            raise RuntimeError('fewer than four qualified dense TP combinations')
        save(self.out / 'QUALIFIED.json', dict(qualified=qualified, rejected=rejected))
        for candidate in qualified:
            candidate['correctness'] = self.correctness(Path(candidate['folder']) / 'correctness')
        for candidate in qualified:
            candidate['timing'] = self.measure(Path(candidate['folder']))
            save(Path(candidate['folder']) / 'RESULT.json', candidate)
        selection = select_measured([c['choice']['predicted_ns'] for c in qualified],
                                    [c['timing']['fresh_process_ns'] for c in qualified])
        selected = qualified[selection['selected']]
        final_pin = pin(selected['choice'], self.cost_sha, self.a.tp, 'manual')
        final_pin['evidence'].update(selection=selection,
                                    candidates=[dict(candidate_rank=c['choice']['candidate_rank'],
                                        predicted_ns=c['choice']['predicted_ns'], **c['timing']) for c in qualified])
        selected_path = self.out / 'selected_pin.json'
        save(selected_path, final_pin)
        final = self.out / 'final'
        resources = self.compile(final, selected_path)
        if resources['status'] != 'qualified':
            raise RuntimeError('selected pin rebuild failed qualification')
        correctness = self.correctness(final / 'correctness', fresh=5)
        timing = self.measure(final)
        result = dict(tp=self.a.tp, batch=self.a.batch, group=self.a.group, selection=selection,
                      selected_pin=ref(selected_path), cost_table_sha256=self.cost_sha,
                      candidates=qualified, rejected=rejected, resources=resources,
                      final_correctness=correctness, final_timing=timing)
        save(self.out / 'results.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp', type=int, choices=(2, 4), required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--guard-each-process', action='store_true')
    for name in ('costs', 'bridge', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--compiler', type=Path, default=ROOT / 'build/tools/tilemega')
    parser.add_argument('--target', type=Path, default=ROOT / 'configs/targets/xcore1000.json')
    parser.add_argument('--model', type=Path, default=Path('/root/models/qwen3_1_7b_rand'))
    parser.add_argument('--prompt-ids', type=Path, default=ROOT / 'configs/prompts/qwen_isl64.json')
    args = parser.parse_args()
    print(json.dumps(Cell(args).run(), indent=2), flush=True)


if __name__ == '__main__':
    main()
