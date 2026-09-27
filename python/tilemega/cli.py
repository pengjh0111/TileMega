"""End-to-end orchestration; compilation and execution remain in their own tools."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

from .cache import atomic_json, export_key, file_sha, key, locked, plan_key, record_outputs, valid_record
from .fingerprint import ROOT, calibration_stamps, source_fingerprint

DEFAULTS = {
    'workload': dict(batch=[1, 16], prompt_len=64, max_new_tokens=1024,
                     prompts='docs/experiments/SERVING_R10/prompts/passages.jsonl'),
    'device': dict(index=0, cache_dir='~/.cache/tilemega'),
    'solver': dict(passes=2, top_m=8, measure_top=3, jobs=3, mode='auto', pruning=True, time_budget_s=600),
    'features': dict(pg='auto', handoff='auto', sync='calibrated', arch_paths='auto', pdl='auto', weight_layout='row'),
    'test': dict(warmup=1, repeats=3, hf_check=True, mode_check=True, guard=True, vllm=False,
                 vllm_python='/root/venv_vllm/bin/python', policy_file=None),
    'output': dict(dir='runs/{model}-{timestamp}'),
}


def read_config(path: Path) -> dict:
    if path.suffix == '.json':
        config = json.loads(path.read_text())
    else:
        if sys.version_info < (3, 11):
            raise ValueError('TOML requires Python >=3.11; this interpreter accepts equivalent JSON')
        import tomllib
        config = tomllib.loads(path.read_text())
    for section, values in DEFAULTS.items():
        config[section] = dict(values, **config.get(section, {}))
    if not config.get('model', {}).get('path'):
        raise ValueError('[model].path is required')
    if config['workload']['prompt_len'] != 64:
        raise ValueError('the serving exporter currently supports prompt_len=64')
    if config['solver']['measure_top'] != 3:
        raise ValueError('the serving compiler currently measures exactly top-3')
    if config['solver']['jobs'] < 1:
        raise ValueError('solver.jobs must be positive')
    if not config['workload']['batch'] or any(not 1 <= b <= 16 for b in config['workload']['batch']):
        raise ValueError('static serving batches must lie in [1,16]')
    for name, allowed in dict(pg=['off', 'l2', 'pages', 'auto'], handoff=['off', 'auto'],
                              sync=['calibrated', 'legacy'], arch_paths=['auto', 'sm80'],
                              pdl=['auto', 'off'], weight_layout=['row', 'tiled']).items():
        if config['features'][name] not in allowed:
            raise ValueError(f'invalid features.{name}')
    if config['solver']['mode'] not in ('auto', 'L1', 'L2'):
        raise ValueError('solver.mode must be auto, L1 or L2')
    if config['test']['repeats'] < 1 or config['test']['warmup'] < 0:
        raise ValueError('invalid measurement repeat counts')
    return config


class Run:
    def __init__(self, config: dict, binary: str | None):
        self.config = config
        self.model = Path(config['model']['path']).expanduser().resolve()
        self.name = config['model'].get('name', self.model.name)
        self.cache = Path(config['device']['cache_dir']).expanduser().resolve()
        self.gpu_lock = Path(os.getenv('TILEMEGA_GPU_LOCK', str(self.cache / 'gpu.lock')))
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.out = Path(config['output']['dir'].format(model=self.name, timestamp=stamp)).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        candidate = binary or os.getenv('TILEMEGA_BIN') or shutil.which('tilemega')
        if not candidate:
            candidate = str(ROOT / 'build/tools/tilemega')
        self.binary = str(Path(candidate).resolve())
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / 'python') + os.pathsep + os.getenv('PYTHONPATH', ''),
                        CUDA_VISIBLE_DEVICES=str(config['device']['index']),
                        TILEMEGA_DEVICE_INDEX=str(config['device']['index']),
                        TILEMEGA_GPU_LOCK=str(self.gpu_lock))
        self.events = []
        self.version = self.capture([self.binary, 'version', '--json'])
        self.device = None
        atomic_json(self.out / 'config.json', config)

    def command(self, argv, label, *, gpu=False, env_extra=None):
        argv = list(map(str, argv))
        environment = dict(self.env, **(env_extra or {}))
        folder = self.out / 'commands' / label
        folder.mkdir(parents=True, exist_ok=True)
        with (self.out / 'commands.sh').open('a') as script:
            exported={name:environment[name] for name in ('TILEMEGA_DEVICE_INDEX','PYTHONPATH',
                                                           'CUDA_VISIBLE_DEVICES','TILEMEGA_GPU_LOCK')}
            exported.update(env_extra or {})
            script.write(' '.join(name+'='+shlex.quote(value) for name,value in exported.items())+
                         ' '+shlex.join(argv)+'\n')
        start = time.monotonic()
        def execute():
            with (folder / 'stdout.txt').open('w') as stdout, (folder / 'stderr.txt').open('w') as stderr:
                return subprocess.run(argv, env=environment, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
        if gpu:
            with locked(self.gpu_lock):
                status = execute()
        else:
            status = execute()
        atomic_json(folder / 'command.json', dict(argv=argv, environment=env_extra or {},
                    returncode=status, seconds=time.monotonic() - start))
        if status:
            raise RuntimeError(f'{label} failed ({status}); see {folder / "stderr.txt"}')
        return folder

    def capture(self, argv):
        folder = self.command(argv, 'query-' + str(time.time_ns()))
        return json.loads((folder / 'stdout.txt').read_text())

    def event(self, layer, hit, reason, **extra):
        row = dict(layer=layer, hit=hit, reason=reason, **extra)
        self.events.append(row)
        atomic_json(self.out / 'cache.json', self.events)
        print(json.dumps(row), flush=True)

    def doctor(self):
        self.device = self.capture([self.binary, 'probe', 'device'])
        name = re.sub(r'[^A-Za-z0-9._-]+', '-', self.device['name'])
        self.device_key = (f'{name}-{self.device["arch_tag"].replace("_", "")}-'
                           f'{self.device["resources"]["num_sms"]}-drv{self.device["driver_version"] // 1000}')
        self.target = self.cache / 'targets' / (self.device_key + '.json')
        clocks = subprocess.check_output(['nvidia-smi', '-i', str(self.config['device']['index']),
            '--query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader,nounits'], text=True)
        owners = subprocess.check_output(['nvidia-smi', '-i', str(self.config['device']['index']),
            '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
        self.idle_power = float(clocks.split(',')[0]) if not owners.strip() else None
        data = dict(device=self.device, device_key=self.device_key, tool=self.version,
                    target=str(self.target), target_exists=self.target.exists(),
                    idle_power_w=self.idle_power, gpu_owners=owners.strip(), clocks=clocks.strip(),
                    cache_dirs={name: str(self.cache / name) for name in ('targets', 'exports', 'plans', 'artifacts')})
        if not (self.model / 'config.json').is_file():
            data['missing_model'] = str(self.model)
            data['download_command'] = shlex.join([sys.executable, '-m', 'tilemega.serving.download_models',
                '--model', self.name if self.name in ('llama', 'qwen3') else 'all', '--model-root', str(self.model.parent)])
        atomic_json(self.out / 'doctor.json', data)
        print(json.dumps(data, indent=2))
        return data

    def calibrate(self):
        if self.device is None:
            self.doctor()
        expected = self.capture([self.binary, 'calibrate', '--stamps'])
        if expected != calibration_stamps():
            raise RuntimeError('calibration sources differ from the built tool; rebuild tilemega')
        self.target.parent.mkdir(parents=True, exist_ok=True)
        with locked(Path(str(self.target) + '.lock')):
            before = json.loads(self.target.read_text()) if self.target.exists() else {}
            actual = before.get('calibration_sections', {})
            missing = [name for name, stamp in expected.items() if actual.get(name, {}).get('stamp') != stamp]
            # Replacing a pipeline profile also discards its dependent fits.
            if 'base' in missing or 'bf16' in missing:
                missing = list(expected)
            if 'wait' in missing:
                missing = list(set(missing) | {'events', 'hop'})
            self.event('calibration', not missing, 'all section stamps match' if not missing else 'missing or changed sections', sections=sorted(missing))
            if missing:
                base = self.target if self.target.exists() else ROOT / 'configs/targets' / (self.device['arch_tag'] + '.json')
                self.command([self.binary, 'calibrate', '--suite', 'serving', '--base', base,
                              '--out', self.target, '--sections', ','.join(sorted(missing))], 'calibrate', gpu=True)
        return json.loads(self.target.read_text())

    def export(self, phase):
        workload = self.config['workload']
        capacity = workload['prompt_len'] + workload['max_new_tokens']
        exporter = ROOT / 'python/tilemega/serving/export.py'
        digest = export_key(self.model / 'config.json', exporter, phase, 1 if phase == 'decode' else workload['prompt_len'], capacity)
        directory = self.cache / 'exports' / digest
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / 'record.json'
        with locked(directory / '.lock'):
            hit = valid_record(marker)
            self.event('export', hit, 'content key and outputs match' if hit else 'export input changed or missing output', phase=phase, key=digest)
            if not hit:
                model_type = json.loads((self.model / 'config.json').read_text())['model_type']
                self.command([sys.executable, '-m', 'tilemega.serving.export', '--config', self.model / 'config.json',
                    '--model', model_type, '--phase', phase, '--capacity', capacity, '--out', directory], 'export-' + phase)
                self.command([sys.executable, '-m', 'tilemega.export_bridge', directory / 'exported_program.pt2',
                    '--out', directory / 'bridge.json'], 'bridge-' + phase)
                record_outputs(marker, [directory / 'exported_program.pt2', directory / 'bridge.json', directory / 'manifest.json'])
        return digest, directory

    def build(self):
        settings = self.config['solver']; features = self.config['features']; workload = self.config['workload']
        # Weight prepacking is still optional research work. A decode handoff
        # is selected by the compiler; prefill has no paged handoff path.
        if features['weight_layout'] != 'row':
            raise RuntimeError('features.weight_layout=tiled is not implemented')
        target = self.calibrate()
        if self.version['source_sha256'] != source_fingerprint():
            raise RuntimeError('compiler fingerprint differs from sources; rebuild tilemega')
        result = {}
        for phase in ('prefill', 'decode'):
            export, directory = self.export(phase)
            previous = None
            for batch in sorted(workload['batch']):
                interval = (0, 0) if phase == 'prefill' else (workload['prompt_len'], workload['prompt_len'] + workload['max_new_tokens'] - 2)
                target_inputs = dict(target.get('calibration_sections', {}), target_sha256=file_sha(self.target))
                digest = plan_key(export, target_inputs, self.version['source_sha256'], settings, features, batch, interval)
                plan = self.cache / 'plans' / digest;plan.mkdir(parents=True, exist_ok=True)
                marker = plan / 'record.json'; library = plan / 'plan.so'; manifest = Path(str(library) + '.plan.json')
                with locked(plan / '.lock'):
                    hit = valid_record(marker)
                    self.event('plan', hit, 'plan key and outputs match' if hit else 'inputs changed or incomplete output', phase=phase, batch=batch, key=digest)
                    if not hit:
                        options = [str(directory / 'bridge.json'), str(library), '--serving', phase,
                            '--batch', str(batch), '--past-range', f'{interval[0]}:{interval[1]}',
                            '--capacity', str(workload['prompt_len'] + workload['max_new_tokens']),
                            '--solver', 'skeleton', '--solve', str(self.target), '--emit', 'serving',
                            '--search-passes', str(settings['passes']), '--top-m', str(settings['top_m']),
                            '--search-jobs', str(settings['jobs']),
                            '--serving-pruning', str(int(settings['pruning'])), '--incremental-prepare', '1',
                            '--variant-cache', str(self.cache / 'variants' / self.device_key),
                            '--artifact-cache', str(self.cache / 'artifacts'), '--dump-cg', str(plan / 'selected.mlir'),
                            '--measure-cmd', shlex.join([sys.executable, '-m', 'tilemega.serving.measure_candidate', '--model', str(self.model)])]
                        for name, value in features.items():
                            if name == 'weight_layout':
                                continue
                            if name == 'handoff' and phase == 'prefill':
                                value = 'off'
                            if name == 'pg' and phase == 'prefill' and value == 'pages':
                                value = 'l2'
                            options += ['--' + name.replace('_', '-'), str(value)]
                        if previous:
                            options += ['--serving-warm-start', str(previous)]
                        atomic_json(plan / 'options.json', options)
                        start = time.monotonic()
                        self.command([self.binary, 'compile', '--options', plan / 'options.json'], f'build-{phase}-B{batch}')
                        seconds = time.monotonic() - start
                        self.command([self.binary, 'audit', 'sass', library, '--out', plan / 'sass.json'], f'sass-{phase}-B{batch}')
                        self.command([self.binary, 'inspect', 'request-floor', plan / 'selected.mlir', self.target,
                            batch, *interval, plan / 'floor.json', plan / 'floor.tsv'], f'floor-{phase}-B{batch}')
                        record_outputs(marker, [library, manifest, plan / 'selected.mlir', plan / 'floor.json', plan / 'floor.tsv'],
                            solve_seconds=seconds, budget_s=settings['time_budget_s'], budget_pass=seconds <= settings['time_budget_s'])
                previous = manifest
                result.setdefault(str(batch), {})[phase] = str(library)
        atomic_json(self.out / 'plans.json', result)
        return result

    def prompts(self):
        workload = self.config['workload']; source = workload['prompts']; destination = self.out / 'prompt_ids.json'
        if source.startswith('random:'):
            config = json.loads((self.model / 'config.json').read_text())
            rng = random.Random(int(source.split(':', 1)[1]))
            atomic_json(destination, [[rng.randrange(config['vocab_size']) for _ in range(workload['prompt_len'])] for _ in range(16)])
        else:
            path = Path(source).expanduser().resolve()
            if path.suffix == '.json':
                shutil.copy2(path, destination)
            else:
                self.command([sys.executable, '-m', 'tilemega.serving.prepare_prompts', '--passages', path,
                              '--model', self.model, '--ids-out', destination], 'prompts')
        return destination

    def bench(self, plans):
        prompts = self.prompts();settings = self.config['test']
        if settings['policy_file']:
            # EV-2 uses one threshold frozen before the alternating model/B
            # matrix. Do not let a later hot doctor sample change it.
            source = Path(settings['policy_file']).expanduser().resolve()
            policy = json.loads(source.read_text())
            if bool(policy.get('guard')) != bool(settings['guard']):
                raise ValueError('the predeclared guard and test.guard differ')
            if settings['guard'] and (float(policy['idle_power_w']) <= 0 or
                                      float(policy['power_margin_w']) <= 0):
                raise ValueError('invalid predeclared GPU power threshold')
            policy['source_file'] = str(source)
        else:
            # build() captured the idle baseline before calibration heated the
            # device. Keep it fixed for all arms of this run.
            if self.device is None:self.doctor()
            if settings['guard'] and self.idle_power is None:
                raise RuntimeError('GPU is not idle; doctor could not establish the pollution threshold')
            policy = dict(idle_power_w=self.idle_power, power_margin_w=30,
                          cooldown_seconds=30, retries=3, guard=settings['guard'])
        atomic_json(self.out / 'measurement_policy.json', policy)
        for index, batch in enumerate(sorted(self.config['workload']['batch'])):
            cell = self.out / f'B{batch}';cell.mkdir(exist_ok=True)
            pair = plans[str(batch)]
            common = ['--model', self.model, '--prompt-ids', prompts, '--batch', batch]
            tilemega = [sys.executable, '-m', 'tilemega.serving.measure', *common,
                '--prefill-so', pair['prefill'], '--decode-so', pair['decode'], '--out', cell / 'tilemega',
                '--mode', self.config['solver']['mode'], '--max-new-tokens', self.config['workload']['max_new_tokens'],
                '--warmup', settings['warmup'], '--repeats', settings['repeats'], '--policy', self.out / 'measurement_policy.json']
            vllm = [settings['vllm_python'], str(ROOT / 'python/tilemega/serving/vllm_baseline.py'), *common,
                    '--out', cell / 'vllm', '--max-tokens', self.config['workload']['max_new_tokens'],
                    '--policy', self.out / 'measurement_policy.json', '--warmup', settings['warmup'], '--repeats', settings['repeats']]
            arms = [('tilemega', tilemega)]
            if settings['vllm']:
                arms.append(('vllm', vllm))
                if index % 2: arms.reverse()
            for label, command in arms:
                extra={}
                if label=='vllm':
                    # Some vLLM wheels depend on a CUDA runtime packaged in
                    # their own venv rather than the system toolkit path.
                    venv=Path(settings['vllm_python']).expanduser().absolute().parents[1]
                    runtimes=list(venv.glob('lib/python*/site-packages/nvidia/cu*/lib/libcudart.so.*'))
                    if runtimes:
                        paths=list(dict.fromkeys(str(path.parent) for path in runtimes))
                        extra['LD_LIBRARY_PATH']=os.pathsep.join([*paths,os.getenv('LD_LIBRARY_PATH','')])
                self.command(command, f'bench-{label}-B{batch}', gpu=True, env_extra=extra)
        return prompts

    def check(self, plans, prompts):
        for batch in self.config['workload']['batch']:
            cell = self.out / f'B{batch}';pair = plans[str(batch)]
            common = ['--model', self.model, '--prompt-ids', prompts, '--batch', batch]
            if self.config['test']['mode_check']:
                self.command([sys.executable, '-m', 'tilemega.serving.check_modes', *common,
                    '--prefill-so', pair['prefill'], '--decode-so', pair['decode'],
                    '--steps', str(self.config['workload']['max_new_tokens']), '--out', cell / 'mode'],
                    f'mode-B{batch}', gpu=True)
            if self.config['test']['hf_check']:
                warmup = self.config['test']['warmup'];count = self.config['workload']['max_new_tokens']
                extra = []
                if self.config['test']['vllm']:
                    self.command([sys.executable, '-m', 'tilemega.serving.hf_check', '--model', self.model,
                        '--prompt-ids', prompts, '--generated', cell / 'vllm' / f'B{batch}' / f'tokens_N{count}_run{warmup}.json',
                        '--out', cell / 'hf_vllm.json'], f'hf-vllm-B{batch}', gpu=True)
                    extra = ['--vllm-metrics', cell / 'hf_vllm.json']
                self.command([sys.executable, '-m', 'tilemega.serving.hf_check', '--model', self.model,
                    '--prompt-ids', prompts, '--generated', cell / 'tilemega' / f'tokens_N{count}_run{warmup}.json',
                    '--out', cell / 'hf_tilemega.json', *extra], f'hf-tilemega-B{batch}', gpu=True)

    def report(self, plans):
        rows = []
        for batch in self.config['workload']['batch']:
            cell = self.out / f'B{batch}';pair = plans[str(batch)]
            tm = json.loads((cell / 'tilemega/measurements.json').read_text())
            floors = {phase: json.loads((Path(so).parent / 'floor.json').read_text()) for phase, so in pair.items()}
            total = sum(f['sum_floor_seconds'] for f in floors.values())
            row = dict(batch=batch, tilemega=tm, floors=floors, sum_floor_seconds=total,
                       e2e_over_floor=tm['e2e_seconds'] / total)
            for name in ('hf_tilemega', 'hf_vllm'):
                if (cell / f'{name}.json').exists():row[name] = json.loads((cell / f'{name}.json').read_text())
            if (cell / 'mode/mode_check.json').exists():row['c2'] = json.loads((cell / 'mode/mode_check.json').read_text())
            if self.config['test']['vllm']:
                baseline = json.loads((cell / 'vllm' / f'B{batch}/measurements.json').read_text())
                row.update(vllm=baseline, throughput_ratio=tm['output_tokens_per_second'] / baseline['output_tokens_per_second'],
                           vllm_e2e_over_floor=baseline['e2e_seconds'] / total)
            rows.append(row)
        atomic_json(self.out / 'report.json', dict(model=str(self.model), cells=rows, cache=self.events))
        lines = ['| B | TTFT ms | TPOT mean/p50/p90 ms | E2E s | token/s | E2E/ΣT_floor | TM/vLLM |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            t=r['tilemega'];ratio=r.get('throughput_ratio')
            tpot='/'.join(f'{1000*t[name]:.3f}' for name in
                          ('tpot_mean_seconds', 'tpot_p50_seconds', 'tpot_p90_seconds'))
            lines.append(f'| {r["batch"]} | {1000*t["ttft_seconds"]:.3f} | {tpot} | {t["e2e_seconds"]:.4f} | '
                         f'{t["output_tokens_per_second"]:.2f} | {r["e2e_over_floor"]:.3f} | {ratio if ratio is not None else "not measured"} |')
        (self.out / 'report.md').write_text('\n'.join(lines)+'\n')
        print(self.out / 'report.md')

    def hwcheck(self):
        """Exercise the native page path; higher-arch claims need that device."""
        device = self.doctor()['device']
        build_dir = Path(self.binary).parent.parent
        targets = ('page_ring_test', 'paged_gemm_test', 'serving_attention_cases_test',
                   'independent_attention_test', 'paged_attention_test')
        self.command(['cmake', '--build', build_dir, '--target', *targets], 'hwcheck-build')
        self.command(['ctest', '--test-dir', build_dir, '--output-on-failure',
                      '-R', '^(page_ring|paged_gemm|serving_attention_cases|'
                            'independent_attention|paged_attention)$'], 'hwcheck-unit', gpu=True)
        # Hardware validation is a smoke check, not a coordinate-descent
        # benchmark. Materialize one explicit legal Llama geometry through the
        # same solver and codegen path instead of searching thousands of
        # candidates before testing 64 tokens on a new device.
        if self.name != 'llama':
            raise ValueError('native hwcheck currently uses the Llama B=1 smoke workload')
        self.calibrate()
        if self.version['source_sha256'] != source_fingerprint():
            raise RuntimeError('native hwcheck requires a compiler built from the current sources')
        shape = dict(tile_m=16, tile_n=128, tile_k=128, stages=2, split_k=1)
        smoke = self.out / 'smoke'
        smoke.mkdir(exist_ok=True)
        domain = smoke / 'domain.json'
        cases = smoke / 'cases.json'
        atomic_json(domain, {'geometries': [shape]})
        atomic_json(cases, {'cases': [dict(geometries=[shape] * 6, kappa=1, residency=1)]})
        pair = {}
        prompt_len = self.config['workload']['prompt_len']
        capacity = prompt_len + self.config['workload']['max_new_tokens']
        for phase in ('prefill', 'decode'):
            export_digest, exported = self.export(phase)
            interval = (0, 0) if phase == 'prefill' else (prompt_len, capacity - 2)
            library = smoke / (phase + '.so')
            smoke_key = key(dict(export=export_digest, target=file_sha(self.target),
                                 source=self.version['source_sha256'], phase=phase,
                                 geometry=shape, capacity=capacity, past_range=interval,
                                 sync=self.config['features']['sync'],
                                 arch_paths=self.config['features']['arch_paths'],
                                 pdl=self.config['features']['pdl'],
                                 device=self.device_key))
            marker = smoke / (phase + '.' + smoke_key + '.record.json')
            with locked(smoke / (phase + '.lock')):
                hit = valid_record(marker)
                self.event('hwcheck_plan', hit, 'native smoke outputs match' if hit else
                           'missing or changed smoke input', phase=phase, key=smoke_key)
                if not hit:
                    self.command([self.binary, 'compile', exported / 'bridge.json', library,
                                  '--serving', phase, '--batch', '1', '--past-range',
                                  f'{interval[0]}:{interval[1]}', '--capacity', str(capacity),
                                  '--solver', 'skeleton', '--solve', self.target,
                                  '--emit', 'serving', '--search-passes', '1', '--top-m', '1',
                                  '--dump-cg', str(library) + '.selected.mlir',
                                  '--search-domain', domain, '--evaluate-configs', cases,
                                  '--measure-cmd', shlex.join([sys.executable, '-m',
                                      'tilemega.serving.measure_candidate', '--model', str(self.model)]),
                                  '--runtime-target', self.target,
                                  '--pg', 'pages' if phase == 'decode' else 'l2',
                                  '--page-bytes', '16384', '--sync', self.config['features']['sync'],
                                  '--arch-paths', self.config['features']['arch_paths'],
                                  '--pdl', self.config['features']['pdl'],
                                  '--artifact-cache', self.cache / 'artifacts',
                                  '--variant-cache', self.cache / 'variants' / self.device_key],
                                 f'hwcheck-{phase}-seed')
                    record_outputs(marker, [library, Path(str(library) + '.plan.json'),
                                           Path(str(library) + '.selected.mlir'),
                                           Path(str(library) + '.build_command.txt')])
            pair[phase] = str(library)
        sass_file = smoke / 'sass.json'
        self.command([self.binary, 'audit', 'sass', pair['prefill'], pair['decode'],
                      '--out', sass_file], 'hwcheck-sass')
        sass = json.loads(sass_file.read_text())
        if sass['fp64_total'] != 0:
            raise RuntimeError('native serving smoke plan contains FP64 instructions')
        prompts = self.prompts()
        manifest = json.loads(Path(pair['decode'] + '.plan.json').read_text())
        command_file = Path(pair['decode'] + '.build_command.txt')
        native_arch = device['arch_tag']
        command = command_file.read_text() if command_file.exists() else ''
        if f'-arch={native_arch}' not in shlex.split(command):
            raise RuntimeError(f'plan was not compiled for native {native_arch}')
        self.command([sys.executable, '-m', 'tilemega.serving.check_modes',
                      '--model', self.model, '--prefill-so', pair['prefill'],
                      '--decode-so', pair['decode'], '--prompt-ids', prompts,
                      '--batch', '1', '--steps', '64', '--out', self.out / 'hwcheck-mode'],
                     'hwcheck-generation', gpu=True)
        pdl = {'applicable': bool(device['caps'].get('pdl')), 'pass': None}
        if pdl['applicable']:
            plain = self.out / 'hwcheck-no-pdl.so'
            page_bytes = manifest.get('pages', {}).get('page_bytes')
            if not page_bytes:
                raise RuntimeError('PDL hardware check needs a paged decode plan')
            self.command([self.binary, 'compile', pair['decode'] + '.selected.mlir', plain,
                          '--serving', 'decode', '--emit', 'serving', '--batch', '1',
                          '--past-range', f'{prompt_len}:{capacity - 2}', '--capacity', str(capacity),
                          '--pg', 'pages', '--page-bytes', page_bytes,
                          '--pdl', 'off', '--sync', self.config['features']['sync'],
                          '--runtime-target', self.target], 'hwcheck-pdl-off-build')
            self.command([sys.executable, '-m', 'tilemega.serving.compare_pdl',
                          '--model', self.model, '--prefill-so', pair['prefill'],
                          '--pdl-so', pair['decode'], '--plain-so', plain,
                          '--prompt-ids', prompts, '--out', self.out / 'hwcheck-pdl.json'],
                         'hwcheck-pdl-smoke', gpu=True)
            pdl['pass'] = json.loads((self.out / 'hwcheck-pdl.json').read_text())['pass']
        report = {'pass': True, 'native_arch': native_arch,
                  'compute_path': 'SM80-class mma.sync', 'transport_path':
                  'TMA/bulk' if device['caps'].get('tma') else 'cp.async',
                  'unit_cases': list(targets), 'generation_steps': 64,
                  'plan_selection': 'one explicit seed geometry, no coordinate descent',
                  'sass_fp64_total': sass['fp64_total'],
                  'pdl': pdl,
                  'decode_so': pair['decode']}
        atomic_json(self.out / 'hwcheck.json', report)
        print(json.dumps(report, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor','calibrate','export','build','bench','check','report','run'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--tilemega')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--hwcheck', action='store_true', help='native page and PDL smoke checks')
    args=parser.parse_args()
    if args.hwcheck and args.command != 'doctor':
        parser.error('--hwcheck is only valid with doctor')
    if not args.config and not args.hwcheck:
        parser.error('--config is required')
    config=read_config(args.config or ROOT / 'configs/e2e/llama_b1.json')
    if args.hwcheck:
        # This checks transport and protocol on a known Llama workload. TF-1
        # is separately tested by EV-2; do not disguise an unimplemented
        # handoff as a checked architecture path.
        config['features']['handoff']='off'
        config['features']['weight_layout']='row'
        config['workload']['batch']=[1]
    if args.run_dir:config['output']['dir']=str(args.run_dir)
    run=Run(config,args.tilemega)
    if args.command=='doctor':
        run.hwcheck() if args.hwcheck else run.doctor()
        return
    if args.command=='calibrate':run.calibrate();return
    if args.command=='export':
        for phase in ('prefill','decode'):run.export(phase)
        return
    if args.command in ('build','run'):plans=run.build()
    else:plans=json.loads((run.out/'plans.json').read_text())
    if args.command=='build':return
    if args.command in ('bench','run'):prompts=run.bench(plans)
    else:prompts=run.out/'prompt_ids.json'
    if args.command in ('check','run'):run.check(plans,prompts)
    if args.command in ('report','run'):run.report(plans)


if __name__=='__main__':
    main()
