"""Collect Qwen3-1.7B TP8 once, with guarded scheduling and immutable receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BATCHES = (1, 2, 4, 8, 16)
SYSTEMS = ('tilemega', 'vllm', 'sglang', 'torchcompile')
GROUP = ','.join(map(str, range(8)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(value):
    path = Path(value['path'])
    if sha(path) != value['sha256']:
        raise ValueError('retained evidence changed: ' + str(path))
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def replace_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def validate_measurement(value, batch, system, directory):
    if (value['tp'], value['batch'], value['max_tokens']) != (8, batch, 1024):
        raise ValueError('measurement is outside the requested TP8 scope')
    if list(map(int, value['gpu_indices'])) != list(range(8)):
        raise ValueError('measurement did not use all eight GPUs')
    if system == 'tilemega':
        if not all(value.get(k) is True for k in ('timed_tokens_identical', 'rank_tokens_identical', 'timing_eligible')):
            raise ValueError('TileMega timing lacks token/rank qualification')
        runs = value['runs']
    else:
        runs = value['generation_runs']
    tokens = []
    for count in (1024, 1):
        samples = [r for r in runs if r['N'] == count]
        if len(samples) != 3 or [r['warmup'] for r in samples] != [True, False, False]:
            raise ValueError('each process must use warmup1 + timed2 for N1024 and N1')
        for sample in samples:
            path = Path(sample['tokens_file'])
            path = path if path.is_absolute() else directory / path
            rows = read(path)
            if len(rows) != batch or any(len(row) != count for row in rows):
                raise ValueError('measurement token count differs')
            tokens.append(ref(path))
    for key in ('tpot_seconds', 'ttft_seconds', 'e2e_seconds', 'output_tokens_per_second'):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('non-positive/non-finite metric: ' + key)
    return tokens


class Campaign:
    def __init__(self, args):
        self.a, self.out = args, args.out.resolve()
        self.a.model = args.model.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        # A second launcher must not remeasure a completed or running cell.
        self.lock = (self.out / 'campaign.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), MX_GPU_WAIT_IDLE='1',
                        MX_GPU_IDLE_WAIT_S=str(args.idle_wait_s), MX_GPU_POLL_S=str(args.poll_s),
                        MX_GPU_JOB_LOG=str(self.out / 'gpu_jobs.log'),
                        TILEMEGA_GPU_JOB_LOG=str(self.out / 'gpu_jobs.log'), NEEDS_MIB='12288')
        self.cpu_env = dict(self.env, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='')
        for key in ('TILEMEGA_DEVICE_GROUP', 'TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_GPU_LOCK'):
            self.env.pop(key, None)
            self.cpu_env.pop(key, None)
        config = dict(model=str(args.model.resolve()), group=GROUP, batch=list(BATCHES), tp=8,
                      systems=list(SYSTEMS), isl=64, osl=1024, fresh_processes=2, warmup=1, repeats=2,
                      python=dict(tilemega=sys.executable, vllm=args.vllm_python,
                                  sglang=args.sglang_python, torchcompile=args.torchcompile_python),
                      cucc=args.cucc, costs=ref(args.costs) if args.costs else None)
        identity = self.out / 'IDENTITY.json'
        if identity.exists():
            original = read(identity)
            if original['config'] != config:
                raise ValueError('campaign options changed; use another --out directory')
            for name, digest in original['source_files'].items():
                if sha(ROOT / name) != digest or sha(self.out / 'source' / name) != digest:
                    raise ValueError('campaign source changed; preserve this run and use a fresh directory: ' + name)
            for item in original['protected']:
                checked(item)
            self.source_files = original['source_files']
        else:
            names = []
            for directory in ('include', 'python/tilemega/serving', 'python/tilemega/tp',
                              'python/tilemega/bench', 'experiments/mx7', 'test/unit/mx'):
                names += [p for p in (ROOT / directory).rglob('*')
                          if p.is_file() and '__pycache__' not in p.parts]
            names += [p for p in (ROOT / 'scripts').glob('mx*') if p.is_file()]
            names += [ROOT / p for p in ('build/tools/tilemega', 'configs/targets/xcore1000.json',
                'configs/bench/guard_policy_c550.json', 'configs/bench/measurement_policy_c550.json',
                'configs/prompts/qwen_isl64.json')]
            files = {}
            for source in sorted(set(names)):
                name = str(source.relative_to(ROOT))
                target = self.out / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files[name] = sha(source)
            model_files = sorted(args.model.glob('*.json')) + sorted(args.model.glob('*.safetensors'))
            protected = [ref(p) for p in model_files]
            protected += [ref(ROOT / p) for p in ('docs/results/MX7_tp_baselines_frozen_c550.csv',
                'configs/bench/mx7_frozen_baselines.json', 'docs/results/MX6_final_moe_c550.csv')]
            save(identity, dict(config=config, source_files=files, protected=protected,
                                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                captured_at=time.time(), evidence='verified'))
            self.source_files = files

    def check_source(self):
        for name, expected in self.source_files.items():
            if sha(ROOT / name) != expected:
                raise ValueError('campaign source changed during collection: ' + name)

    def job(self, name, command, group=None, allow_failure=False):
        base = self.out / 'jobs' / name
        base.mkdir(parents=True, exist_ok=True)
        receipt = base / 'ACCEPTED.json'
        if receipt.exists():
            value = read(receipt)
            checked(value['process'])
            if value['command'] != list(map(str, command(Path(value['folder'])))):
                raise ValueError('retained command differs: ' + name)
            for item in value['artifacts']:
                checked(item)
            return Path(value['folder']), value
        for attempt in range(2):
            number = len(list(base.glob('attempt*')))
            folder = base / f'attempt{number:03d}'
            folder.mkdir()
            argv = list(map(str, command(folder)))
            actual = argv
            if group is not None:
                wrapper = 'mx_gpu_group_job.sh' if ',' in group else 'mx_gpu_job.sh'
                actual = [str(ROOT / 'scripts' / wrapper), group, '900', str(folder / 'guard'), *argv]
            print(name, 'attempt', number, flush=True)
            started = time.time()
            with (folder / 'stdout.log').open('x') as stream:
                child = subprocess.Popen(actual, cwd=ROOT, env=self.env if group else self.cpu_env,
                                         stdout=stream, stderr=subprocess.STDOUT)
                try:
                    code = child.wait()
                except BaseException:
                    from tilemega.bench.gpu_guard import stop
                    stop(child)
                    save(folder / 'INTERRUPTED.json', dict(PID=child.pid, time=time.time()))
                    raise
            process = dict(argv=actual, PID=child.pid, start=started, end=time.time(), returncode=code)
            if group is not None and ',' not in group:
                process['card'] = int(group)
            save(folder / 'PROCESS.json', process)
            guard_path = folder / 'guard/guard_result.json'
            if group is not None and code == 0 and (not guard_path.exists() or read(guard_path)['code']):
                raise RuntimeError('successful child lacks a passing guard')
            if code == 75:
                # An idle-window expiry or unlaunched preflight is pending, not an engine N/A.
                guard = read(guard_path) if guard_path.exists() else {}
                if not (folder / 'guard/pgid').exists():
                    raise RuntimeError('GPU group not idle; resume the same command later: ' + str(folder))
                if attempt == 0:
                    continue
                raise RuntimeError('guard interference repeated: ' + str(folder) + ' ' + str(guard))
            value = dict(folder=str(folder), command=argv, returncode=code,
                         process=ref(folder / 'PROCESS.json'), artifacts=[])
            if code and not allow_failure:
                raise RuntimeError('job failed; inspect preserved log: ' + str(folder / 'stdout.log'))
            if hasattr(self, 'source_files'):
                self.check_source()
            value['artifacts'] = [ref(p) for p in sorted(folder.rglob('*'))
                                  if p.is_file() and p.name != 'monotonic_duration.json']
            save(receipt, value)
            return folder, value
        raise AssertionError('unreachable')

    def prepare(self):
        export, _ = self.job('export', lambda p: ['env', f'MODEL={self.a.model}',
            f'EXPORT_ROOT={p / "export"}', 'scripts/mx_tp_cell.sh', 'export', '8', '1'])
        self.bridge = export / 'export/qwen3_1_7b_tp8_decode/bridge.json'
        probes, _ = self.job('build_p2p', lambda p: [self.a.cucc, '-std=c++17', '-O3',
            ROOT / 'experiments/mx7/p2p_probe.cu', '-o', p / 'probe'])
        self.probe = probes / 'probe'
        ll, _ = self.job('build_ll', lambda p: [sys.executable, 'experiments/mx7/build_ll_static.py',
            '--store', '0', '--load', '1', '--out', p / 'build'])
        self.ll = ll / 'build'
        tests, _ = self.job('build_tests', lambda p: [sys.executable, 'experiments/mx7/build_tp_tests.py',
            '--tp', '8', '--out', p / 'build'])
        self.tests = tests / 'build'
        save_path = self.out / 'PREPARED.json'
        if not save_path.exists():
            save(save_path, dict(bridge=ref(self.bridge), note='CPU preparation only; no TP8 performance claim'))

    def qualify_link(self):
        p2p, _ = self.job('p2p', lambda p: [self.probe], GROUP)
        rows = [json.loads(line) for line in (p2p / 'stdout.log').read_text().splitlines() if line.startswith('{')]
        if len(rows) != 56 or not all(row['pass'] for row in rows):
            raise RuntimeError('TP8 requires all 56 ordered P2P pairs to pass')
        for packets in (1024, 262144):
            self.job(f'll_{packets}', lambda p: [sys.executable, 'experiments/mx7/run_probe.py',
                '--exe', self.ll / 'll_litmus', '--source', self.ll / 'll_static.cu', '--out', p / 'cohort',
                '--', '8', packets, '2000', '1', '0', '1', '1'], GROUP)
        for kind in ('ar', 'head'):
            for batch in (1, 16):
                self.job(f'{kind}_B{batch}', lambda p, kind=kind, batch=batch: [sys.executable,
                    'test/python/test_mx7_tp.py', '--build', self.tests / f'{kind}_t8_b{batch}',
                    '--out', p / 'cohort'], GROUP)

    def collect_costs(self):
        if self.a.costs:
            from .solve_tp import check_qualification, check_constant_group
            catalog, _ = check_qualification(self.a.costs)
            check_constant_group(catalog, 8, list(range(8)))
            from .selection_tp import stage_lists
            for batch in BATCHES:
                stage_lists(catalog['rows'], 'qwen3_1_7b_tp8', batch, catalog['tp_constants'])
            return self.a.costs.resolve()
        costs, _ = self.job('build_costs', lambda p: [sys.executable, 'experiments/mx7/build_costs.py',
            '--tp', '8', '--workers', self.a.workers, '--out', p / 'build'])
        self.builds = costs / 'build'
        constants, _ = self.job('build_constants', lambda p: [sys.executable,
            'experiments/mx7/build_tp_constants.py', '--tp', '8', '--out', p / 'build'])
        self.constant_builds = constants / 'build'
        keys = read(self.builds / 'KEYS_CPU.json')
        driver = self.builds / 'frozen/experiments/mx7/measure_costs.py'
        chunks = [keys[i:i + 48] for i in range(0, len(keys), 48)]

        def card_jobs(card):
            completed = []
            for number in range(card, len(chunks), 8):
                path = self.out / 'keys' / f'{number:04d}.json'
                if not path.exists():
                    save(path, chunks[number])
                records = []
                for round_number in (1, 2, 3):
                    folder, _ = self.job(f'cost{number:04d}_r{round_number}', lambda p: [sys.executable,
                        driver, '--keys', path, '--out', p / 'raw.jsonl', '--round', round_number], str(card))
                    records.append(dict(round=round_number, raw=ref(folder / 'raw.jsonl'),
                        process=ref(folder / 'PROCESS.json'), guard=ref(folder / 'guard/guard_result.json')))
                completed.append((number, dict(keys=ref(path), records=records)))
            return completed
        with ThreadPoolExecutor(max_workers=8) as pool:
            completed = [item for result in pool.map(card_jobs, range(8)) for item in result]
        request = dict(source=ref(self.builds / 'SOURCE.json'), all_keys=ref(self.builds / 'KEYS_CPU.json'),
            driver=ref(driver), domain=ref(self.builds / 'DOMAIN.json'),
            CPU_resource_rejections=read(self.builds / 'REJECTIONS_CPU.json'),
            jobs=[row for _, row in sorted(completed)])
        path = self.out / 'COST_REQUEST.json'
        if not path.exists():
            save(path, request)
        if read(path) != request:
            raise ValueError('retained cost request differs')
        raw, _ = self.job('aggregate_costs', lambda p: [sys.executable, 'experiments/mx7/aggregate_costs.py',
            '--request', path, '--out', p / 'catalog.json'])
        measurements = self.out / 'constants'
        for batch in BATCHES:
            folder, _ = self.job(f'constant_B{batch}', lambda p, batch=batch: [sys.executable,
                'experiments/mx7/measure_tp_constants.py', '--build', self.constant_builds / f'tp8_B{batch}',
                '--out', p / 'cohort'], GROUP)
            destination = measurements / f'tp8_B{batch}'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.symlink_to(folder, target_is_directory=True)
        constants, _ = self.job('qualify_constants', lambda p: [sys.executable,
            'experiments/mx7/aggregate_tp_constants.py', '--measurements', measurements,
            '--builds', self.constant_builds, '--tp', '8', '--group', GROUP,
            '--work', p / 'work', '--out', p / 'constants.json'])
        qualified, _ = self.job('qualify_costs', lambda p: [sys.executable, 'experiments/mx7/qualify_costs.py',
            '--request', path, '--catalog', raw / 'catalog.json', '--constants', constants / 'constants.json',
            '--card', '7', '--work', p / 'work', '--out', p / 'costs.json'])
        return qualified / 'costs.json'

    def baseline_command(self, system, batch, folder):
        prefix = dict(vllm=[self.a.vllm_python, ROOT / 'python/tilemega/serving/vllm_baseline.py', '--tp', '8'],
            sglang=[self.a.sglang_python, ROOT / 'python/tilemega/serving/sglang_baseline.py', '--tp', '8'],
            torchcompile=[self.a.torchcompile_python, '-m', 'torch.distributed.run', '--standalone',
                          '--nproc-per-node', '8', ROOT / 'python/tilemega/serving/torchcompile_tp_baseline.py'])
        argv = prefix[system] + ['--model', self.a.model, '--prompt-ids', ROOT / 'configs/prompts/qwen_isl64.json',
            '--batch', batch, '--out', folder / 'measure', '--max-tokens', '1024', '--warmup', '1',
            '--repeats', '2', '--policy', ROOT / 'configs/bench/measurement_policy_c550.json']
        if system == 'vllm':
            argv = ['env', 'VLLM_CACHE_ROOT=/root/deps/vllm_src/vllm_cache', *argv]
        return argv

    def baseline_sweep(self):
        for round_number in (1, 2):
            for batch in BATCHES:
                for system in SYSTEMS[1:]:
                    path = self.out / 'results' / f'{system}_B{batch}_r{round_number}.json'
                    if path.exists():
                        for item in read(path)['evidence']:
                            checked(item)
                        continue
                    folder, process = self.job(f'{system}_B{batch}_r{round_number}',
                        lambda p, system=system, batch=batch: self.baseline_command(system, batch, p),
                        GROUP, allow_failure=True)
                    value = dict(system=system, batch=batch, round=round_number,
                                 status='N/A' if process['returncode'] else 'pass',
                                 evidence=[process['process'], ref(folder / 'stdout.log')])
                    if process['returncode'] == 0:
                        metrics = folder / f'measure/B{batch}/measurements.json'
                        try:
                            tokens = validate_measurement(read(metrics), batch, system, metrics.parent)
                        except (ValueError, KeyError, OSError) as error:
                            value.update(status='N/A', reason='invalid baseline output: ' + repr(error))
                        else:
                            value.update(metrics=ref(metrics), evidence=value['evidence'] + [ref(metrics), *tokens])
                    else:
                        value['reason'] = f'engine exited {process["returncode"]}; see {folder / "stdout.log"}'
                    save(path, value)
                    summarize(self.out)

    def tilemega_sweep(self, costs):
        for batch in BATCHES:
            folder, _ = self.job(f'select_B{batch}', lambda p, batch=batch: [sys.executable,
                '-m', 'tilemega.tp.solve_tp', '--tp', '8', '--batch', batch, '--group', GROUP,
                '--guard-each-process', '--costs', costs, '--bridge', self.bridge, '--model', self.a.model,
                '--out', p / 'selection'])
            selection = folder / 'selection/results.json'
            result = read(selection)
            final = result['resources']['binary']
            checked(final)
            records = result['final_timing']['measurements']
            for index, item in enumerate(records, 1):
                metrics = checked(item)
                tokens = validate_measurement(read(metrics), batch, 'tilemega', metrics.parent)
                path = self.out / 'results' / f'tilemega_B{batch}_r{index}.json'
                if not path.exists():
                    save(path, dict(system='tilemega', batch=batch, round=index, status='pass', metrics=item,
                        evidence=[ref(selection), final, item, *tokens], selected_pin=result['selected_pin']))
            if batch in (1, 16):
                pin = checked(result['selected_pin'])
                built, _ = self.job(f'build_delay_B{batch}', lambda p: ['env', 'TILEMEGA_TP_LITMUS=1',
                    f'MODEL={self.a.model}', f'PIN={pin}', f'EXPORT_ROOT={self.bridge.parent.parent}',
                    f'CELLS={p / "tm"}', 'scripts/mx_tp_cell.sh', 'build', '8', batch])
                delay_so = built / f'tm/tp8/B{batch}/plan.so'
                manifest = read(str(delay_so) + '.plan.json')
                if manifest.get('tp_litmus') is not True:
                    raise ValueError('delay binary did not enable TP litmus')
                from .process_cohorts import validate_checks
                proof = validate_checks(result['final_correctness']['qualification'])
                reference = checked(proof['C2'][0]).parent / 'tokens.json'
                for process in range(5):
                    self.job(f'delay_B{batch}_p{process}', lambda p: [sys.executable,
                        '-m', 'tilemega.serving.check_tp', '--model', self.a.model, '--decode-so', delay_so,
                        '--prompt-ids', ROOT / 'configs/prompts/qwen_isl64.json', '--batch', batch,
                        '--devices', GROUP, '--steps', '1024', '--reference', reference, '--out', p / 'check'], GROUP)
            summarize(self.out)


def summarize(out, write=True):
    rows = []
    for batch in BATCHES:
        for system in SYSTEMS:
            parts = [out / 'results' / f'{system}_B{batch}_r{r}.json' for r in (1, 2)]
            records = [read(path) for path in parts if path.exists()]
            status = ('N/A' if any(r['status'] == 'N/A' for r in records) else
                      'pass' if len(records) == 2 else 'pending')
            row = dict(tp=8, batch=batch, system=system, status=status, fresh_processes=len(records),
                       contract='bf16_greedy', isl=64, osl=1024, tpot_ms='', ttft_ms='', e2e_ms='',
                       tokens_per_s='', measurements_json=json.dumps([ref(p) for p in parts if p.exists()]))
            if status == 'pass':
                values = [read(checked(r['metrics'])) for r in records]
                for target, source in (('tpot_ms', 'tpot_seconds'), ('ttft_ms', 'ttft_seconds'),
                                       ('e2e_ms', 'e2e_seconds'), ('tokens_per_s', 'output_tokens_per_second')):
                    row[target] = statistics.median(v[source] for v in values) * (1 if target == 'tokens_per_s' else 1000)
            rows.append(row)
    if write:
        temporary = out / 'TP8_results.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out / 'TP8_results.csv')
    lines = ['Qwen3-1.7B, BF16 greedy, ISL/OSL=64/1024; TPOT ms.', '',
             '| BS | TileMega | vLLM | SGLang | torch.compile | fastest baseline / TM |',
             '|---:|---:|---:|---:|---:|---:|']
    for batch in BATCHES:
        cells = {r['system']: r for r in rows if r['batch'] == batch}
        values = [f'{cells[s]["tpot_ms"]:.3f}' if cells[s]['status'] == 'pass' else cells[s]['status'] for s in SYSTEMS]
        baselines = [r['tpot_ms'] for s, r in cells.items() if s != 'tilemega' and r['status'] == 'pass']
        ratio = f'{min(baselines) / cells["tilemega"]["tpot_ms"]:.3f}x' if baselines and cells['tilemega']['status'] == 'pass' else 'pending'
        lines.append('| ' + ' | '.join([str(batch), *values, ratio]) + ' |')
    if write:
        temporary = out / 'TP8_table.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(out / 'TP8_table.md')
    state = dict(cells=20, passed=sum(r['status'] == 'pass' for r in rows),
                 unavailable=sum(r['status'] == 'N/A' for r in rows), pending=sum(r['status'] == 'pending' for r in rows),
                 numerical_acceptance='see final ACCEPTED.json; metric availability alone is not qualification')
    if write:
        replace_json(out / 'STATUS.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'run', 'status'), nargs='?', default='plan')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/mx7/tp8')
    parser.add_argument('--model', type=Path, default=Path('/root/models/qwen3_1_7b_rand'))
    parser.add_argument('--costs', type=Path, help='reuse an eligible TP8 measured table; never reuse TP2/TP4 coordinates')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--poll-s', type=float, default=10)
    parser.add_argument('--idle-wait-s', type=float, default=21600)
    parser.add_argument('--cucc', default='/opt/maca/tools/cu-bridge/bin/cucc')
    parser.add_argument('--vllm-python', default='/root/venv_vllm/bin/python')
    parser.add_argument('--sglang-python', default=sys.executable)
    parser.add_argument('--torchcompile-python', default='/root/venv_mx7_tc/bin/python')
    args = parser.parse_args()
    if args.workers < 1 or args.poll_s <= 0 or args.idle_wait_s < 0:
        parser.error('invalid worker count or idle-wait interval')
    if args.command == 'plan':
        print(json.dumps(dict(tp=8, group=GROUP, model=str(args.model), batch=BATCHES, systems=SYSTEMS,
            measurement='two fresh processes, each warmup1 + timed2; BF16 greedy; ISL64/OSL1024',
            stages=['CPU export/build', 'baseline rounds', '56 P2P pairs', 'LL8/AR/head5-process checks',
                    'TP8 measured costs and communication constants', 'k-best top4/C-1/C-2/selection',
                    'selected delay cohorts5/5', 'summary and immutable acceptance'],
            out=str(args.out)), indent=2))
        return 0
    if args.command == 'status':
        if not args.out.is_dir():
            parser.error('campaign output does not exist')
        print(json.dumps(summarize(args.out, write=False), indent=2))
        table = args.out / 'TP8_table.md'
        if table.exists():
            print(table.read_text())
        return 0
    campaign = Campaign(args)
    try:
        campaign.prepare()
        if args.command == 'prepare':
            print('CPU preparation complete; GPU tests have not started.', flush=True)
            return 0
        campaign.baseline_sweep()
        campaign.qualify_link()
        costs = campaign.collect_costs()
        campaign.tilemega_sweep(costs)
        status = summarize(campaign.out)
        campaign.check_source()
        for item in read(campaign.out / 'IDENTITY.json')['protected']:
            checked(item)
        acceptance = campaign.out / 'ACCEPTED.json'
        if not acceptance.exists():
            save(acceptance, dict(status, evidence='verified', time=time.time(), costs=ref(costs),
                scope='TP8 only; external TP2/TP4 baselines and MoE data preserved',
                table=ref(campaign.out / 'TP8_results.csv'), identity=ref(campaign.out / 'IDENTITY.json')))
        print((campaign.out / 'TP8_table.md').read_text(), flush=True)
        return 0 if status['passed'] == 20 else 2
    except BaseException as error:
        summarize(campaign.out)
        replace_json(campaign.out / 'LAST_STOP.json', dict(time=time.time(), error=repr(error),
            note='preserve all attempts; rerun the same command to reuse accepted processes'))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
