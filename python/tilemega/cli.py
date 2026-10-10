"""End-to-end orchestration; compilation and execution remain in their own tools."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import statistics

from .serving.integrated_selection import successive_halving, SelectionBudgetExhausted, admit_pilot, first_stage_budget_ms, coverage_order
from .serving.attention_selection import variants as attention_variants, matches as matches_attention, label as attention_label, pinned_geometry, compile_options as attention_compile_options
from .build.budget import run_until
from .cache import atomic_json, export_key, file_sha, key, locked, plan_key, record_outputs, valid_record
from .fingerprint import ROOT, calibration_stamps, source_fingerprint
from .serving.execution import compiler_features, execution_combinations, select_execution, write_execution, pin_prefill, phase_pg_choices, prefill_combinations

DEFAULTS = {
    'workload': dict(batch=[1, 16], prompt_len=64, max_new_tokens=1024,
                     prompts='docs/experiments/SERVING_R10/prompts/passages.jsonl'),
    'device': dict(index=0, cache_dir='~/.cache/tilemega'),
    'solver': dict(passes=2, top_m=8, measure_top=6, jobs=3, mode='L2', candidate_loop=0, pruning=True, time_budget_s=600, candidate_guard_wait_s=300, prefill_pins={}, exclude_l1_loop=False, resident2_batches=[]),
    'features': dict(pg='auto', handoff='off', sync='calibrated', arch_paths='auto', pdl='auto', weight_layout='tiled', kphase_mask=31, lookahead_bytes=-1, v3_poll_ns=0, watchdog=0, mma_reg_pipe=0, nonpaged_la=0, attention_frontier=0, parallel_argmax=0, ep_direct=0, attention_noinline=0, attention_buffers=2, attention_impl="mma16", l2_slim=0, page_loop_split=0, nonpaged_weight_layout='row', evict_first=0, evict_last=1, decode_executor="L2", decode_loop=1, prefill_executor="L1"),
    'test': dict(warmup=1, repeats=3, hf_check=True, mode_check=True, guard=True, vllm=False,
                 vllm_python='/root/venv_vllm/bin/python', policy_file=None),
    'output': dict(dir='runs/{model}-{timestamp}'),
}


def missing_serving_fit_sections(target):
    """Native measurements are independent of the optional legacy phase fit."""
    body = target.get('calibration_by_dtype', {}).get('bf16', {}).get('task_body', {})
    serving, paged = body.get('serving', {}), body.get('serving_paged', {})
    gemms = ('gemm_store', 'gemm_residual', 'gemm_swiglu', 'gemm_argmax_partial')
    kinds = (*gemms, 'embedding', 'rmsnorm', 'argmax_reduce', 'attention_merge',
             'fused_attention_decode_d64', 'fused_attention_decode_d128',
             'fused_attention_prefill_d64', 'fused_attention_prefill_d128')
    def valid(entry, fields):
        return isinstance(entry, dict) and all(
            isinstance(entry.get(k), (int, float)) and not isinstance(entry[k], bool)
            and math.isfinite(entry[k]) and entry[k] >= 0 for k in fields)
    missing = []
    if any(not valid(serving.get(k), ('fixed_ns', 'byte_ns', 'flop_ns', 'samples'))
           or serving[k]['samples'] < 1 for k in kinds):
        missing.append('task_bodies')
    paged_kinds = [f'{g}_n{n}_k{k}' for g in gemms
                   for n, k in ((128, 64), (32, 64), (64, 128), (64, 64))]
    paged_kinds += ['attention_decode_d64', 'attention_decode_d128']
    loader = paged.get('loader', {})
    if (any(not valid(paged.get(k), ('fixed_ns', 'iter_ns', 'median_relative_error'))
            for k in paged_kinds) or not valid(loader, ('stream_gbps_per_sm', 'aggregate_gbps'))
            or loader.get('stream_gbps_per_sm', 0) <= 0 or loader.get('aggregate_gbps', 0) <= 0):
        missing.append('task_bodies_paged')
    return missing


def aggregate_paired_runs(cell: Path, batch: int, count: int, repeats: int,
                          model: Path) -> None:
    """Keep measured generations adjacent across engines, even after reloads."""
    def quantile(values, q):
        values = sorted(values)
        at = q * (len(values) - 1)
        lo = int(at)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)

    for name in ('tilemega', 'vllm'):
        parent = cell / name
        target = parent if name == 'tilemega' else parent / f'B{batch}'
        target.mkdir(parents=True, exist_ok=True)
        samples = []
        for repeat in range(repeats):
            directory = parent / f'round{repeat}'
            if name == 'vllm':
                directory /= f'B{batch}'
            samples.append(json.loads((directory / 'measurements.json').read_text()))
            for tokens in (1, count):
                shutil.copy2(directory / f'tokens_N{tokens}_run1.json',
                             target / f'tokens_N{tokens}_run{repeat + 1}.json')
        full = [sample['e2e_seconds'] for sample in samples]
        first = [sample['ttft_seconds'] for sample in samples]
        e2e, ttft = statistics.median(full), statistics.median(first)
        rows = []
        for repeat, sample in enumerate(samples):
            for row in sample.get('runs', sample.get('generation_runs', [])):
                if not row['warmup']:
                    rows.append(dict(row, run=repeat + 1))
        result = dict(samples_seconds=full, ttft_samples_seconds=first,
                      e2e_seconds=e2e, ttft_seconds=ttft,
                      tpot_seconds=(e2e - ttft) / (count - 1),
                      output_tokens_per_second=batch * count / e2e,
                      measurement_policy=samples[0]['measurement_policy'],
                      batch=batch, max_tokens=count, paired_rounds=repeats)
        if name == 'tilemega':
            tokens = [row['tokens'] for row in rows if row['N'] == count]
            if any(item != tokens[0] for item in tokens[1:]):
                raise AssertionError('C-2: paired TileMega generations produced different tokens')
            steps = [ms / 1000 for row in rows if row['N'] == count
                     for ms in row['gpu_step_ms'][1:]]
            result.update(runs=rows, timed_tokens_identical=True,
                          tpot_mean_seconds=statistics.mean(steps),
                          tpot_p50_seconds=quantile(steps, .5),
                          tpot_p90_seconds=quantile(steps, .9))
            with (parent / 'step_times.tsv').open('w') as output:
                output.write('run\tstep\tgpu_ms\n')
                for row in rows:
                    if row['N'] == count:
                        for step, ms in enumerate(row['gpu_step_ms']):
                            output.write(f'{row["run"]}\t{step}\t{ms:.9g}\n')
        else:
            result.update(model=str(model), vllm_version=samples[0]['vllm_version'],
                          generation_runs=rows)
        atomic_json(target / 'measurements.json', result)


def batch_features(config, batch, phase):
    values=dict(config['features'])
    if phase=='decode':values.update(config.get('features_by_batch',{}).get(str(batch),{}))
    return values


def plan_build_choices(pg_choices, settings, batch, phase):
    """Keep the ordinary family alongside a conditionally required resident-2 family."""
    rows=[]
    for pg in pg_choices:
        rows.append((pg,pg,{}))
        if phase=='decode' and pg=='l2' and batch in settings['resident2_batches']:
            rows.append(('l2_resident2','l2',{'attention_buffers':1}))
    return rows


def read_config(path: Path) -> dict:
    if path.suffix == '.json':
        config = json.loads(path.read_text())
    else:
        if sys.version_info < (3, 11):
            raise ValueError('TOML requires Python >=3.11; this interpreter accepts equivalent JSON')
        import tomllib
        config = tomllib.loads(path.read_text())
    explicit_nonpaged_la = 'nonpaged_la' in config.get('features', {})
    config.setdefault('features', {}).setdefault('prefill_pg', 'l2')
    for section, values in DEFAULTS.items():
        config[section] = dict(values, **config.get(section, {}))
    if not config.get('model', {}).get('path'):
        raise ValueError('[model].path is required')
    model_config = Path(config['model']['path']).expanduser() / 'config.json'
    if not explicit_nonpaged_la and model_config.is_file() and \
            json.loads(model_config.read_text()).get('model_type') == 'qwen3_moe':
        config['features']['nonpaged_la'] = 1
    routing_profile = config['features'].get('routing_profile')
    if routing_profile is not None:
        if not isinstance(routing_profile, str) or not routing_profile:
            raise ValueError('features.routing_profile must be a nonempty path')
        profile = Path(routing_profile).expanduser().resolve()
        if not profile.is_file():
            raise ValueError('features.routing_profile does not exist: ' + str(profile))
        if not model_config.is_file() or json.loads(model_config.read_text()).get('model_type') != 'qwen3_moe':
            raise ValueError('features.routing_profile requires qwen3_moe')
        config['features']['routing_profile'] = str(profile)
    if 'moe_profile_layer' in config['features']:
        layer = config['features']['moe_profile_layer']
        if type(layer) is not int or layer < 0 or routing_profile is None:
            raise ValueError('features.moe_profile_layer requires a profile and nonnegative integer')
    if config['workload']['prompt_len'] != 64:
        raise ValueError('the serving exporter currently supports prompt_len=64')
    if not 1 <= config['solver']['measure_top'] <= 8:
        raise ValueError('solver.measure_top must be in [1,8]')
    if config['solver']['jobs'] < 1:
        raise ValueError('solver.jobs must be positive')
    if not config['workload']['batch'] or any(not 1 <= b <= 16 for b in config['workload']['batch']):
        raise ValueError('static serving batches must lie in [1,16]')
    for name, allowed in dict(pg=['off', 'l2', 'pages', 'auto', 'measure'], handoff=['off', 'auto'],
                              sync=['calibrated', 'legacy'], arch_paths=['auto', 'sm80'],
                              mma_reg_pipe=[0,1], nonpaged_la=[0,1], attention_frontier=[0,1], parallel_argmax=[0,1], ep_direct=[0,1], attention_noinline=[0,1], attention_buffers=[1,2], attention_impl=["mma16","pvswap"], l2_slim=[0,1], page_loop_split=[0,1], pdl=['auto', 'off'], weight_layout=['row', 'tiled'], nonpaged_weight_layout=['row','tiled'], evict_first=[0,1], evict_last=[0,1],
                              decode_executor=['L1', 'L2', 'measure'],
                              decode_loop=[0, 1, 'measure'], prefill_executor=['L1', 'L2', 'measure'],
                              prefill_pg=['l2', 'measure']).items():
        if config['features'][name] not in allowed:
            raise ValueError(f'invalid features.{name}')
    if config['solver']['candidate_loop'] not in (0,1):
        raise ValueError('solver.candidate_loop must be 0 or 1')
    if not isinstance(config['solver']['exclude_l1_loop'],bool):
        raise ValueError('solver.exclude_l1_loop must be boolean')
    resident2=config['solver']['resident2_batches']
    if not isinstance(resident2,list) or any(type(b) is not int or b not in config['workload']['batch'] for b in resident2):
        raise ValueError('solver.resident2_batches must name configured batches')
    if resident2 and config['features']['pg'] not in ('l2','measure'):
        raise ValueError('resident-2 family requires a nonpaged decode build')
    if config['solver']['mode'] == 'auto':
        config['solver']['mode'] = 'L2'
    if config['solver']['mode'] not in ('L1', 'L2'):
        raise ValueError('solver.mode must be L1 or L2')
    if not 0 <= int(config['features']['kphase_mask']) <= 31:
        raise ValueError('features.kphase_mask must be in [0,31]')
    if int(config['features']['lookahead_bytes']) not in (-1, 0, 65536, 131072):
        raise ValueError('invalid features.lookahead_bytes')
    if int(config['features']['watchdog']) not in (0,1):
        raise ValueError('features.watchdog must be 0 or 1')
    if int(config['features']['v3_poll_ns']) < 0:
        raise ValueError('features.v3_poll_ns must be nonnegative')
    if int(config['solver']['candidate_guard_wait_s']) < 0:
        raise ValueError('candidate_guard_wait_s must be nonnegative')
    if config['test']['repeats'] < 1 or config['test']['warmup'] < 0:
        raise ValueError('invalid measurement repeat counts')
    # Conditional R14 gates are per cell. Do not silently enable a B16-only
    # residency experiment on the B1 requests of the same model.
    overrides=config.get('features_by_batch',{})
    if not isinstance(overrides,dict):raise ValueError('features_by_batch must be an object')
    allowed={'attention_buffers':(1,2),'ep_direct':(0,1),'attention_frontier':(0,1)}
    for batch,values in overrides.items():
        if batch not in {str(b) for b in config['workload']['batch']} or not isinstance(values,dict):
            raise ValueError('features_by_batch must name a configured batch')
        for name,value in values.items():
            if name not in allowed or value not in allowed[name]:
                raise ValueError(f'invalid features_by_batch.{batch}.{name}')
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

    def command(self, argv, label, *, gpu=False, env_extra=None, deadline=None):
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
                return run_until(argv, deadline=deadline, env=environment, cwd=ROOT, stdout=stdout, stderr=stderr)
        if gpu and os.environ.get("TILEMEGA_GPU_LOCK_HELD") != "1":
            with locked(self.gpu_lock):
                self.preflight_gpu(folder / 'guard_preflight.json')
                environment['TILEMEGA_GPU_LOCK_HELD']='1'
                status = execute()
        else:
            status = execute()
        atomic_json(folder / 'command.json', dict(argv=argv, environment=env_extra or {},
                    returncode=status, seconds=time.monotonic() - start))
        if status:
            if status==124 and deadline is not None:
                raise SelectionBudgetExhausted(f"{label} exhausted its build/measurement budget; see {folder}")
            if status == 75:
                raise SystemExit(75)
            raise RuntimeError(f'{label} failed ({status}); see {folder / "stderr.txt"}')
        return folder

    def preflight_gpu(self, record: Path):
        # No CUDA context exists in this process. Any visible compute PID or
        # large invisible allocation belongs to another workload. Reject it
        # before weight loading, or before spending minutes on plan search.
        device = self.env['TILEMEGA_DEVICE_INDEX']
        visible = subprocess.check_output(
            ['nvidia-smi', '-i', device, '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'], text=True)
        rows = [row.split(',') for row in visible.splitlines() if row.strip()]
        used = int(subprocess.check_output(
            ['nvidia-smi', '-i', device, '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'], text=True).splitlines()[0].strip())
        hidden = max(0, used - sum(int(row[-1].strip()) for row in rows))
        accepted = not rows and hidden <= 1024
        atomic_json(record, dict(visible_pids=[int(row[0].strip()) for row in rows],
                                 used_mib=used, hidden_mib=hidden,
                                 accepted=accepted))
        if not accepted:
            raise SystemExit(75)

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
            missing = list(set(missing) | set(missing_serving_fit_sections(before)))
            # The microbenchmarks stamp each section before the suite can
            # validate that the device was idle. A stamped but uncalibrated
            # profile must never be served from cache on a later run.
            # Serving prices BF16 only. Keep an F32 drift visible in the
            # target, but do not repeatedly remeasure an unused profile.
            if before and not before.get('calibration', {}).get('calibrated'):
                self.event('f32_base', False,
                           'F32 base is uncalibrated; serving reads the separate BF16 profile')
            if before and not before.get('calibration_by_dtype', {}).get('bf16', {}).get('calibrated'):
                missing.append('bf16')
            # Serving fits depend on the BF16 profile, not the independent
            # F32 base. An F32 drift therefore remeasures only that section.
            if 'bf16' in missing:
                missing = list(set(missing) | (set(expected) - {'base'}))
            if 'wait' in missing:
                missing = list(set(missing) | {'events', 'hop'})
            self.event('calibration', not missing, 'all section stamps match' if not missing else 'missing or changed sections', sections=sorted(missing))
            if missing:
                base = self.target if self.target.exists() else ROOT / 'configs/targets' / (self.device['arch_tag'] + '.json')
                try:
                    self.command([self.binary, 'calibrate', '--suite', 'serving', '--base', base,
                                  '--out', self.target, '--sections', ','.join(sorted(missing))], 'calibrate', gpu=True)
                except RuntimeError:
                    error=(self.out/'commands/calibrate/stderr.txt').read_text(errors='replace').lower()
                    if 'out of memory' in error or 'memory allocation' in error:
                        from .serving.measure import _gpu_owners
                        owners, visible, used = _gpu_owners()
                        if owners or used - visible > 1024:
                            raise SystemExit(75)
                    raise
                refreshed=json.loads(self.target.read_text())
                fit=refreshed.get('calibration_by_dtype',{}).get('bf16',{}).get('task_body',{})
                prior=before.get('calibration_by_dtype',{}).get('bf16',{}).get('task_body',{})
                if not fit.get('samples') and prior.get('samples'):
                    for field in ('fixed','fixed_physical','latency_scale','loop_body',
                                  'loop_fixed','loop_wait','samples','scalar_fixed_ns',
                                  'source','stage_rate_bytes_per_ns'):
                        if field in prior:fit[field]=prior[field]
                    atomic_json(self.target,refreshed)
                    self.event('task_body_fit',True,
                               'retained prior fixed fit after BF16 bandwidth refresh')
                absent = missing_serving_fit_sections(refreshed)
                if absent:
                    raise RuntimeError('BF16 serving target lacks native fits: ' + ','.join(absent))
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

    def attention_variant(self, base, variant, deadline):
        """Cache a same-geometry structural variant before measuring execution."""
        from .build.identity import verify
        base=Path(base);manifest=json.loads(Path(str(base)+'.plan.json').read_text())
        if matches_attention(manifest,variant):return base
        identity=verify(base)
        digest=key(dict(base=identity['artifact_id'],variant=variant,
                        compiler=self.version['source_sha256'],kind='r14_attention_variant_v1'))
        directory=self.cache/'plans'/digest;directory.mkdir(parents=True,exist_ok=True)
        record=directory/'record.json';library=directory/'plan.so'
        with locked(directory/'.lock'):
            hit=valid_record(record)
            self.event('attention_variant',hit,'pinned measured GEMM geometry',key=digest,variant=variant)
            if not hit:
                if time.monotonic()>=deadline:raise SelectionBudgetExhausted('selection budget exhausted before variant build')
                pinned_geometry(manifest,Path(str(base)+'.classes.tsv'),directory)
                original=json.loads((base.parent/'options.json').read_text())
                options=attention_compile_options(original,variant,directory)
                atomic_json(directory/'options.json',options)
                self.command([self.binary,'compile','--options',directory/'options.json'],
                             'attention-build-'+digest[:16],gpu=True,deadline=deadline)
                actual=json.loads(Path(str(library)+'.plan.json').read_text())
                if not matches_attention(actual,variant):raise RuntimeError('attention variant did not materialize requested switches')
                verify(library)
                record_outputs(record,[library,Path(str(library)+'.plan.json'),
                    Path(str(library)+'.identity.json'),Path(str(library)+'.source.json'),
                    Path(str(library)+'.source.json.patch'),Path(str(library)+'.classes.tsv')],
                    variant=variant,base_artifact_id=identity['artifact_id'])
        return library

    def build(self):
        settings = self.config['solver']; features = self.config['features']; workload = self.config['workload']
        target = self.calibrate()
        if self.version['source_sha256'] != source_fingerprint():
            raise RuntimeError('compiler fingerprint differs from sources; rebuild tilemega')
        self.preflight_gpu(self.out / 'build_guard.json')
        result = {}
        prefill_modes = {}; decode_choices = {}; shared_decodes = {}
        from .moe.deployment import automatic_layout_policy
        layout_policy=automatic_layout_policy(self.model,target,workload)
        if layout_policy['require_shared_layout']:
            atomic_json(self.out/'layout-policy.json',layout_policy)
        phases=('decode','prefill') if layout_policy['require_shared_layout'] else ('prefill','decode')
        for phase in phases:
            export, directory = self.export(phase)
            previous_by_pg = {}
            for batch in sorted(workload['batch']):
                features=batch_features(self.config,batch,phase)
                batch_started=time.monotonic()
                deadline=batch_started+settings['time_budget_s']
                reserve=min(300,settings['time_budget_s']/4)
                pilot_deadline=deadline-reserve
                interval = (0, 0) if phase == 'prefill' else (workload['prompt_len'], workload['prompt_len'] + workload['max_new_tokens'] - 2)
                pg_choices = phase_pg_choices(features, phase)
                built = {}
                build_choices=plan_build_choices(pg_choices,settings,batch,phase)
                for family,pg,family_features in build_choices:
                    choice_features = dict(features, pg=pg, handoff='off',
                                           weight_layout=features['weight_layout'] if pg == 'pages' else 'row')
                    choice_features.update(family_features)
                    shared_manifest=None
                    if layout_policy['require_shared_layout']:
                        choice_features.update(weight_layout='tiled',nonpaged_weight_layout='tiled')
                        if phase=='prefill':
                            shared_manifest=Path(str(shared_decodes[batch])+'.plan.json')
                    target_inputs = dict(target.get('calibration_sections', {}), target_sha256=file_sha(self.target))
                    if shared_manifest:
                        target_inputs['shared_weight_layout_sha256']=file_sha(shared_manifest)
                    if choice_features.get('routing_profile'):
                        target_inputs['routing_profile_sha256']=file_sha(Path(choice_features['routing_profile']))
                    seed_manifest = str(built['l2'])+'.plan.json' if pg=='pages' and 'l2' in built else None
                    prefill_pin=settings['prefill_pins'].get(str(batch)) if phase=='prefill' else None
                    if prefill_pin:
                        target_inputs['prefill_pin']={k:file_sha(Path(v)) for k,v in prefill_pin.items()}
                    if seed_manifest:
                        target_inputs['paged_seed_sha256']=file_sha(Path(seed_manifest))
                    digest = plan_key(export, target_inputs, self.version['source_sha256'], {k:v for k,v in settings.items() if k!='candidate_guard_wait_s'},
                                      choice_features, batch, interval)
                    plan = self.cache / 'plans' / digest;plan.mkdir(parents=True, exist_ok=True)
                    marker = plan / 'record.json'; library = plan / 'plan.so'; manifest = Path(str(library) + '.plan.json')
                    with locked(plan / '.lock'):
                        hit = valid_record(marker)
                        self.event('plan', hit, 'plan key and outputs match' if hit else 'inputs changed or incomplete output', phase=phase, batch=batch, pg=pg, key=digest)
                        if not hit:
                            options = [str(directory / 'bridge.json'), str(library), '--serving', phase,
                            '--batch', str(batch), '--past-range', f'{interval[0]}:{interval[1]}',
                            '--capacity', str(workload['prompt_len'] + workload['max_new_tokens']),
                            '--solver', 'skeleton', '--solve', str(self.target), '--emit', 'serving',
                            '--search-passes', str(settings['passes']), '--top-m', str(settings['top_m']),
                            '--search-jobs', str(settings['jobs']), '--measure-top', str(settings['measure_top']),
                            '--candidate-guard-wait-s', str(settings['candidate_guard_wait_s']),
                            '--candidate-mode', settings['mode'], '--candidate-loop', str(settings['candidate_loop']),
                            '--search-budget-ms', str(first_stage_budget_ms(settings['time_budget_s'],len(build_choices),phase=='decode')),
                            '--serving-pruning', str(int(settings['pruning'])), '--incremental-prepare', '1',
                            '--variant-cache', str(self.cache / 'variants' / self.device_key),
                            '--artifact-cache', str(self.cache / 'artifacts'), '--dump-cg', str(plan / 'selected.mlir'),
                            '--measure-cmd', shlex.join([sys.executable, '-m', 'tilemega.serving.measure_candidate', '--model', str(self.model)])]
                            for name, value in compiler_features(choice_features).items():
                                if phase == 'prefill' and not (shared_manifest and name in ('nonpaged_weight_layout','weight_layout')) and (name.startswith('gemm_impl_') or name in ('attention_noinline','attention_buffers','parallel_argmax','ep_direct','attention_frontier','nonpaged_la','attention_impl','mma_reg_pipe','kphase_mask','lookahead_bytes','v3_poll_ns','l2_slim','page_loop_split','nonpaged_weight_layout','evict_first','evict_last','serve_kv_block','serve_query_rows')):
                                    continue
                                if name == 'handoff' and phase == 'prefill':
                                    value = 'off'
                                options += ['--' + name.replace('_', '-'), str(value)]
                            if prefill_pin:
                                options += pin_prefill(prefill_pin['manifest'],prefill_pin['classes'],plan)
                            if shared_manifest:
                                options += ['--shared-weight-layout',str(shared_manifest)]
                            if seed_manifest:
                                options += ["--paged-seed-from", seed_manifest]
                            previous=previous_by_pg.get(family)
                            if previous:
                                options += ['--serving-warm-start', str(previous)]
                            atomic_json(plan / 'options.json', options)
                            start = time.monotonic()
                            self.command([self.binary, 'compile', '--options', plan / 'options.json'], f'build-{phase}-{family}-B{batch}',deadline=pilot_deadline if phase=='decode' else deadline)
                            seconds = time.monotonic() - start
                            self.command([self.binary, 'audit', 'sass', library, '--out', plan / 'sass.json'], f'sass-{phase}-{family}-B{batch}',deadline=pilot_deadline if phase=='decode' else deadline)
                            self.command([self.binary, 'inspect', 'request-floor', plan / 'selected.mlir', self.target,
                                batch, *interval, plan / 'floor.json', plan / 'floor.tsv'], f'floor-{phase}-{family}-B{batch}',deadline=pilot_deadline if phase=='decode' else deadline)
                            record_outputs(marker, [library, manifest, Path(str(library)+'.identity.json'),
                                Path(str(library)+'.source.json'), Path(str(library)+'.source.json.patch'),
                                plan / 'selected.mlir', plan / 'floor.json', plan / 'floor.tsv'],
                                solve_seconds=seconds, budget_s=settings['time_budget_s'], budget_pass=seconds <= settings['time_budget_s'])
                    previous_by_pg[family] = manifest
                    built[family] = library
                if phase == 'decode':
                    candidates=[]
                    deadline=batch_started+settings['time_budget_s']
                    for family, library in built.items():
                        manifest=json.loads(Path(str(library)+'.plan.json').read_text())
                        pg=manifest['pg']
                        variants=attention_variants(manifest)
                        # Measure the already-built winner first; a budget exhaustion
                        # must not silently turn an unmeasured new body into a winner.
                        variants.sort(key=lambda v:not matches_attention(manifest,v))
                        for variant in variants:
                            for mode, loop in execution_combinations(pg, features['decode_executor'], features['decode_loop']):
                                if pg!='pages' and mode=='L1' and loop and settings['exclude_l1_loop']:
                                    continue
                                candidates.append(dict(pg=pg, family=family, mode=mode, loop=loop,base_library=str(library),
                                    base_variant=matches_attention(manifest,variant),
                                    variant=variant,variant_label=attention_label(variant),samples_ms=[]))
                    candidates=coverage_order(candidates)
                    selection_inputs=dict(libraries={pg:file_sha(library) for pg,library in built.items()},
                                          executor=features['decode_executor'], loop=features['decode_loop'],
                                          prefill_mode=features['prefill_executor'],exclude_l1_loop=settings['exclude_l1_loop'],
                                          objective='uniform_64_1087_coverage_budget_v3',pasts=[64,575,1000],
                                          candidates=candidates)
                    choice_path=self.out / f'decode-choice-B{batch}.json'
                    cached=json.loads(choice_path.read_text()) if choice_path.exists() else {}
                    if cached.get('inputs')==selection_inputs and cached.get('selected') and valid_record(Path(cached['selected']['library']).parent/'record.json'):
                        candidates=cached['candidates'];winner=cached['selected']
                    else:
                        def measure_execution(candidate, round_label):
                            # Always finish three confirmation rounds for measured
                            # survivors. The budget stops admission of further pilots.
                            admit_pilot(candidate,round_label,time.monotonic(),deadline,reserve)
                            candidate['library']=str(self.attention_variant(candidate['base_library'],candidate['variant'],pilot_deadline))
                            stem=f"B{batch}-{candidate['family']}-{candidate['variant_label']}-{candidate['mode']}-{candidate['loop']}-{round_label}"
                            out=self.out / ('decode-choice-'+stem)
                            command=[sys.executable,'-m','tilemega.serving.measure_candidate','--so',candidate['library'],
                                     '--model',str(self.model),'--batch',str(batch),'--past-list','64,575,1000',
                                     '--mode',candidate['mode'],'--loop',str(candidate['loop']),
                                     '--loop-steps','64','--warmup','8','--out',str(out),
                                     '--guard-wait-s',str(settings['candidate_guard_wait_s'])]
                            self.command(command,"choose-"+stem,gpu=True,deadline=pilot_deadline if round_label.startswith("pilot") else deadline)
                            measured=json.loads((out/'measurements.json').read_text())['modes'][candidate['mode']]['by_past']
                            identities=[]
                            for values in measured.values():
                                if bool(values.get('decode_loop_used'))!=bool(candidate['loop']):
                                    raise RuntimeError('candidate did not use requested loop')
                                identities.append(values.get('execution_identity'))
                            if not identities or not identities[0] or any(x!=identities[0] for x in identities):
                                raise ValueError('execution identity differs across past samples')
                            return dict(by_past=measured,execution_identity=identities[0],
                                        spill=identities[0]['spill'],measurement_path=str(out/'measurements.json'))
                        try:
                            winner,candidates=successive_halving(candidates,measure_execution,require_coverage=True)
                        except (RuntimeError,SelectionBudgetExhausted) as error:
                            atomic_json(choice_path,dict(inputs=selection_inputs,candidates=getattr(error,"rows",None),error=str(error),status='failed_or_budget_exhausted',
                                elapsed_s=time.monotonic()-batch_started))
                            raise
                        atomic_json(choice_path,dict(inputs=selection_inputs,candidates=candidates,selected=winner))
                    selected=winner['pg']
                    built[selected]=Path(winner['library'])
                    serving=write_execution(built[selected],winner['mode'],winner['loop'],features['prefill_executor'],
                                            selection=winner,candidates=candidates,stage_one=dict(mode=settings['mode'],loop=settings['candidate_loop']))
                    decode_choices[batch]=(built,winner,candidates)
                    if layout_policy['require_shared_layout']:
                        shared_decodes[batch]=built[selected]
                    result.setdefault(str(batch), {})['decode_pg_choice']=dict(candidates=candidates,selected=winner)
                else:
                    selected, prefill_modes[batch] = self.select_prefill(built, batch)
                result.setdefault(str(batch), {})[phase] = str(built[selected])
                if batch in decode_choices and batch in prefill_modes:
                    decode_built,winner,candidates=decode_choices[batch]
                    result[str(batch)]['serving']=write_execution(decode_built[winner['pg']],
                        winner['mode'],winner['loop'],prefill_modes[batch],selection=winner,candidates=candidates,
                        stage_one=dict(mode=settings['mode'],loop=settings['candidate_loop']))
        atomic_json(self.out / 'plans.json', result)
        return result

    def select_decode(self, built, batch, prefill_mode):
        features=self.config['features']; settings=self.config['solver']
        candidates=[]
        for pg, library in built.items():
            for mode, loop in execution_combinations(pg, features['decode_executor'], features['decode_loop']):
                if pg!='pages' and mode=='L1' and loop and settings['exclude_l1_loop']:
                    continue
                candidates.append(dict(pg=pg, mode=mode, loop=loop, library=str(library), samples_ms=[]))
        selection_inputs=dict(libraries={pg:file_sha(library) for pg,library in built.items()},
                              executor=features['decode_executor'], loop=features['decode_loop'],
                              prefill_mode=prefill_mode,exclude_l1_loop=settings['exclude_l1_loop'])
        choice_path=self.out / f'decode-choice-B{batch}.json'
        cached=json.loads(choice_path.read_text()) if choice_path.exists() else {}
        if cached.get('inputs')==selection_inputs:
            candidates=cached['candidates']
        else:
            for round in range(3):
                ordered=candidates[round:]+candidates[:round]
                for candidate in ordered:
                    if candidate.get('error'):continue
                    out=self.out / f"decode-choice-B{batch}-{candidate['pg']}-{candidate['mode']}-{candidate['loop']}-r{round}"
                    command=[sys.executable,'-m','tilemega.serving.measure_candidate','--so',candidate['library'],
                             '--model',str(self.model),'--batch',str(batch),'--past-mid','575',
                             '--mode',candidate['mode'],'--loop',str(candidate['loop']),
                             '--loop-steps','64','--warmup','8','--out',str(out),
                             '--guard-wait-s',str(settings['candidate_guard_wait_s'])]
                    try:
                        self.command(command,f"choose-B{batch}-{candidate['pg']}-{candidate['mode']}-{candidate['loop']}-r{round}",gpu=True)
                        measured=json.loads((out/'measurements.json').read_text())['modes'][candidate['mode']]
                        if bool(measured.get('decode_loop_used'))!=bool(candidate['loop']):
                            raise RuntimeError('candidate did not use requested loop')
                        candidate['samples_ms'].append(measured['mean_ms'])
                    except RuntimeError as error:
                        candidate['error']=str(error)
            atomic_json(choice_path,dict(inputs=selection_inputs,candidates=candidates))
        winner=select_execution(candidates)
        return winner,candidates

    def select_prefill(self, built, batch):
        features=self.config['features']; settings=self.config['solver']
        candidates=[dict(pg=pg, mode=mode, loop=loop, library=str(library), samples_ms=[])
            for pg,library in built.items()
            for mode,loop in prefill_combinations(pg,features['prefill_executor'])]
        if len(candidates)==1:
            return candidates[0]['pg'], candidates[0]['mode']
        inputs=dict(libraries={pg:file_sha(library) for pg,library in built.items()},
                    executor=features['prefill_executor'])
        path=self.out/f'prefill-choice-B{batch}.json'
        cached=json.loads(path.read_text()) if path.exists() else {}
        if cached.get('inputs')==inputs:
            candidates=cached['candidates']
        else:
            for round in range(3):
                for candidate in candidates[round:]+candidates[:round]:
                    if candidate.get('error'):continue
                    out=self.out/f"prefill-choice-B{batch}-{candidate['pg']}-{candidate['mode']}-r{round}"
                    command=[sys.executable,'-m','tilemega.serving.measure_candidate',
                        '--so',candidate['library'],'--model',str(self.model),'--batch',str(batch),
                        '--mode',candidate['mode'],'--loop','0','--past-mid','0',
                        '--out',str(out),'--guard-wait-s',str(settings['candidate_guard_wait_s'])]
                    try:
                        self.command(command,f"choose-prefill-B{batch}-{candidate['pg']}-{candidate['mode']}-r{round}",gpu=True)
                        measured=json.loads((out/'measurements.json').read_text())['modes'][candidate['mode']]
                        if measured.get('decode_loop_used'):
                            raise RuntimeError('prefill candidate incorrectly used a decode loop')
                        candidate['samples_ms'].append(measured['mean_ms'])
                    except RuntimeError as error:
                        candidate['error']=str(error)
            atomic_json(path,dict(inputs=inputs,candidates=candidates))
        winner=select_execution(candidates)
        atomic_json(self.out/f'prefill-selected-B{batch}.json',winner)
        return winner['pg'],winner['mode']

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
                '--mode', 'auto', '--decode-loop', 'auto', '--prefill-mode', 'auto', '--max-new-tokens', self.config['workload']['max_new_tokens'],
                '--warmup', settings['warmup'], '--repeats', settings['repeats'], '--policy', self.out / 'measurement_policy.json']
            vllm = [settings['vllm_python'], str(ROOT / 'python/tilemega/serving/vllm_baseline.py'), *common,
                    '--out', cell / 'vllm', '--max-tokens', self.config['workload']['max_new_tokens'],
                    '--policy', self.out / 'measurement_policy.json', '--warmup', settings['warmup'], '--repeats', settings['repeats']]
            if settings['vllm']:
                # A vLLM engine and a serving plan cannot share the measured
                # device's memory. Reload each arm for an adjacent pair while
                # keeping initialization and warmup outside the timed call.
                alternation=[]
                for repeat in range(settings['repeats']):
                    arms=[('tilemega',tilemega),('vllm',vllm)]
                    if (index+repeat)%2:arms.reverse()
                    for label, command in arms:
                        out=cell/label/f'round{repeat}'
                        command=list(command)
                        command[command.index('--out')+1]=out
                        command[command.index('--repeats')+1]='1'
                        extra={}
                        if label=='vllm':
                            venv=Path(settings['vllm_python']).expanduser().absolute().parents[1]
                            runtimes=list(venv.glob('lib/python*/site-packages/nvidia/cu*/lib/libcudart.so.*'))
                            if runtimes:
                                paths=list(dict.fromkeys(str(path.parent) for path in runtimes))
                                extra['LD_LIBRARY_PATH']=os.pathsep.join([*paths,os.getenv('LD_LIBRARY_PATH','')])
                        self.command(command,f'bench-{label}-B{batch}-r{repeat}',gpu=True,env_extra=extra)
                        alternation.append(dict(round=repeat,engine=label,
                                                measurements=str(out/'measurements.json')))
                atomic_json(cell/'alternation.json',alternation)
                aggregate_paired_runs(cell,batch,self.config['workload']['max_new_tokens'],
                                      settings['repeats'],self.model)
            else:
                self.command(tilemega, f'bench-tilemega-B{batch}', gpu=True)
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
    if len(sys.argv)>1 and sys.argv[1]=='dnn':
        from .dnn.cli import main as dnn_main
        return dnn_main(sys.argv[2:])
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
