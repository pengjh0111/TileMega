#!/usr/bin/env python3
"""DNN1 baseline queue/run/collect; a whole cell owns one guarded GPU session.

Queue generation and collection are CPU-only. Run uses the existing guard
and identity wrapper. Serial cells hold the NUMA lock; user-authorized
concurrent cells use declared CPU affinity and the guard's trial mode.
External baselines are
frozen independently of final native selection; missing native prefixes are
recorded as incomplete P-CELL, never synthesized from unsafe artifacts.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'CLAUDE.md').is_file() and (p / 'python/tilemega').is_dir())
DEFAULT_PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'
DEFAULT_EXTRA_PATH = '/root/TileMega/runs/r16/env/dnn-export/lib/python3.10/site-packages'
POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
MODELS = ('resnet18', 'resnet34', 'vgg16', 'alexnet', 'bert')
BATCHES = (1, 2, 4, 8, 16)
WORKLOAD_ID = 'dnn-classic-v1'
ACTIVE_GPUS = (0, 1, 2, 3, 4, 5, 6)
CLASSIC_DEVICES = ROOT / 'configs/dnn/classic_h800_devices.json'
# Only these ids may replace the classic constants from a queue spec.
SPEC_WORKLOADS = ('dnn3-v1',)
PCI = {0: '0000:0f:00.0', 1: '0000:10:00.0', 2: '0000:41:00.0',
       3: '0000:44:00.0', 4: '0000:86:00.0', 5: '0000:87:00.0',
       6: '0000:b7:00.0', 7: '0000:b7:00.0'}
ARMS = (
    dict(arm='tensorrt-bf16', system='tensorrt', precision='bf16'),
    dict(arm='tensorrt-fp16', system='tensorrt', precision='fp16'),
    dict(arm='torchcompile-max-autotune', system='torchcompile', precision='bf16',
         compile_mode='max-autotune', freezing=False, fullgraph=True),
    dict(arm='torchcompile-max-autotune-freezing', system='torchcompile', precision='bf16',
         compile_mode='max-autotune', freezing=True, fullgraph=True),
    dict(arm='torchcompile-reduce-overhead', system='torchcompile', precision='bf16',
         compile_mode='reduce-overhead', freezing=False, fullgraph=True),
    dict(arm='tvm-bf16', system='tvm', precision='bf16'),
)
PROTOCOL = dict(warmup=50, iterations=200, timing_processes=3, correctness_processes=5,
    rounds=['forward', 'reverse', 'forward'], extra_correctness_processes=2,
    scope='one complete forward; CUDA events; resident inputs/weights',
    statistic='median of three process medians; median of three process p90s',
    correctness='identical exported BF16 weights promoted to FP32; fixed random inputs; no dataset gate',
    native_scope='optional already-gated opening artifact; final solver/P-SEL artifact is measured later',
    freeze_scope='external baseline freeze; incomplete P-CELL when native prefix unavailable',
    anchor='final native comparison requires DT-ANCHOR; do not silently compare a different card')
FIELDS = ['model', 'batch', 'system', 'arm', 'precision', 'gpu', 'status', 'reason',
    'sm_clock_start_mhz', 'sm_clock_end_mhz', 'sm_clock_source', 'processes',
    'attempt_processes', 'failed_attempts', 'correctness_processes',
    'correctness_passes', 'correctness_metrics_identical', 'correctness_policy',
    'strict_v1_status', 'median_ms', 'p90_ms',
    'process_range_ms', 'process_range_pct', 'correctness', 'versions',
    'engine_or_library_sha256', 'exported_program_sha256', 'source_checkpoint_sha256',
    'inputs_sha256', 'native_prefix_available', 'p_cell_complete', 'cell_session', 'source']
TVM_FIELDS = ['model', 'batch', 'precision', 'gpu', 'status', 'failure_kind',
    'byoc_functions', 'byoc_flop_fraction', 'flop_convention', 'full_graph_flop_fraction',
    'flop_coverage_status', 'scheduled_tir_functions', 'has_wmma', 'has_mma_sync',
    'cuda_wmma', 'ptx_wmma', 'cuda_mma_sync', 'ptx_mma_sync',
    'external_library_tensor_core_instructions', 'links_cublas', 'links_cudnn',
    'cublas_offload_runtime', 'cudnn_offload_runtime', 'trials', 'tune_wall_seconds',
    'library_sha256', 'diagnostics', 'error', 'source']


class HardStop(RuntimeError):
    pass


class PreparationPending(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def gpu_env(gpu, extra=None):
    env = dict(extra or {})
    package_paths = [str(ROOT / 'python'), DEFAULT_EXTRA_PATH]
    package_paths += [value for value in env.get('PYTHONPATH', '').split(os.pathsep)
                      if value and value not in package_paths]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
        TILEMEGA_NUMA_NODE=str(0 if gpu < 4 else 1),
        TILEMEGA_GPU_PID_MODE='namespace_limited',
        PYTHONPATH=os.pathsep.join(package_paths),
        OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def held(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def require_session(cell):
    gpu = cell['gpu']
    if cell.get('workload_id') == WORKLOAD_ID and gpu not in ACTIVE_GPUS:
        raise HardStop('classic workload uses only the currently assigned indices0–6')
    if 'active_gpus' in cell and gpu not in cell['active_gpus']:
        raise HardStop('workload uses only its declared device indices')
    env = os.environ
    if (gpu not in PCI or env.get('TILEMEGA_GPU_LOCK_HELD') != '1' or
            env.get('CUDA_VISIBLE_DEVICES') != str(gpu) or
            env.get('TILEMEGA_DEVICE_INDEX') != str(gpu) or
            env.get('TILEMEGA_DEVICE_GROUP') is not None or
            env.get('TILEMEGA_GPU_LOCK') != f'/tmp/tilemega_gpu{gpu}.lock' or
            env.get('TILEMEGA_TIMING') != '1'):
        raise HardStop('cell execution requires the unchanged single-card timing guard')
    node = 0 if gpu < 4 else 1
    if not held(f'/tmp/tilemega_gpu{gpu}.lock'):
        raise HardStop('the outer session must hold the physical GPU lock')
    if cell.get('parallel'):
        affinity = set(cell.get('cpu_affinity') or [])
        if (env.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL') != '1' or
                env.get('TILEMEGA_NUMA_CONCURRENCY_VERIFIED') == '1' or
                not affinity or not set(os.sched_getaffinity(0)) <= affinity):
            raise HardStop('parallel cells require explicit trial metadata and declared CPU affinity')
    elif not held(f'/tmp/tilemega_numa{node}.lock'):
        raise HardStop('the outer session must hold both physical GPU and NUMA locks')


def telemetry(cell):
    from dnn1_device import gpu
    return gpu(cell['gpu'])


def exit_context_snapshot(cell, timeout_s):
    """Read metadata without creating a CUDA context, with a shared deadline."""
    deadline = time.monotonic() + timeout_s
    def query(flag):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('exit metadata deadline expired')
        return subprocess.check_output(['nvidia-smi', '-i', str(cell['gpu']), flag,
            '--format=csv,noheader,nounits'], text=True, timeout=remaining)
    apps = query('--query-compute-apps=pid,used_memory')
    detail = query('--query-gpu=memory.used,clocks.current.sm')
    owners = {}
    for line in apps.splitlines():
        pid, used = line.split(',')
        owners[int(pid)] = int(used)
    used, sm = detail.strip().split(',')
    return dict(time=time.time(), gpu_index=cell['gpu'], owners=owners, used_mib=float(used),
        sm_clock_mhz=float(sm), power_monitoring=False,
        raw_compute_apps=apps, raw_gpu_detail=detail)


def passive_context_drain(cell, timeout_s=10.0, interval_s=.25, *, observe=None, clock=None, sleep=None):
    """Wait for two empty NVML compute-app samples; the outer guard still decides.

    Child exit can precede CUDA context disappearance. An observed PID is not
    attributed to our subprocess, and no owner is signalled or exempted. This
    exit-only wait never changes a measurement or the guard's acceptance rules.
    """
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    begin = clock()
    record = dict(started=time.time(), timeout_s=timeout_s, interval_s=interval_s,
                  required_empty_samples=2, samples=[], attribution='unknown; observation only')
    empty = 0
    while True:
        try:
            row = (observe(cell) if observe else
                   exit_context_snapshot(cell, max(.001, timeout_s - (clock() - begin))))
            if not isinstance(row.get('owners'), dict):
                raise ValueError('exit metadata omitted the compute-app owners')
        except Exception as error:
            record.update(status='telemetry_unavailable', error=str(error))
            break
        record['samples'].append(dict(elapsed_s=clock() - begin, telemetry=row))
        empty = empty + 1 if not row.get('owners') else 0
        if empty >= record['required_empty_samples']:
            record['status'] = 'empty_observed'
            break
        remaining = timeout_s - (clock() - begin)
        if remaining <= 0:
            record['status'] = 'timeout'
            break
        sleep(min(interval_s, remaining))
    record.update(ended=time.time(), elapsed_s=clock() - begin)
    return record


def strict_correctness(model, metrics):
    if not isinstance(metrics, list) or not metrics:
        return False, 'missing baseline correctness metrics'
    for row in metrics:
        if model == 'nafnet':
            value = row.get('psnr_min_vs_fp32_db', row.get('psnr_vs_fp32_db', float('-inf')))
            if not value >= 40:
                return False, f'PSNR below 40 dB: {value}'
        else:
            value = row.get('cosine_min_vs_fp32', row.get('cosine_vs_fp32', float('-inf')))
            threshold = .999 if model == 'bert' else .9999
            if not math.isfinite(value) or value < threshold:
                return False, f'cosine below {threshold}: {value}'
            if model != 'bert' and row.get('top1_agreement_vs_fp32') != 1.0:
                return False, 'classification top-1 agreement differs from 1.0'
    return True, ''


def validate_tvm_selfcheck(cell):
    path = Path(cell['tvm_selfcheck'])
    if not path.is_file():
        raise PreparationPending(f'awaiting the repaired source build and actual CUDA selfcheck receipt: {path}')
    try:
        value = read(path)
    except json.JSONDecodeError as error:
        raise PreparationPending('CUDA selfcheck receipt is not complete: ' + str(path)) from error
    info = value.get('libinfo', value.get('build_options', {}))
    def on(option):
        if option is True:
            return True
        text = str(option).strip()
        if text.upper() in ('ON', '1', 'TRUE'):
            return True
        return text.startswith('/') and not text.upper().endswith('-NOTFOUND')
    enabled = value.get('enabled') or {}
    options = {
        'CUDA': ('TVM_TIRX_USE_CUDA', 'USE_CUDA', 'cuda'),
        'cuBLAS': ('TVM_USE_CUBLAS', 'USE_CUBLAS', 'cublas'),
        'cuDNN': ('TVM_USE_CUDNN', 'USE_CUDNN', 'cudnn'),
    }
    cuda_exists = value.get('cuda_exist', value.get('cuda_exists', value.get('cuda_device_exists', False)))
    all_enabled = all(any(enabled.get(key) is True for key in keys) and
                      not any(enabled.get(key) is False for key in keys) and
                      any(on(info.get(key)) for key in keys) and
                      all(on(info[key]) for key in keys if key in info)
                      for keys in options.values())
    if cuda_exists is not True or not all_enabled or value.get('error'):
        raise HardStop('DT-TVM branch 3: CUDA device absent or CUDA/cuBLAS/cuDNN not all ON: ' + compact(value))
    if value.get('failure_kind') == 'tvm_cuda_build_or_selfcheck':
        raise HardStop('DT-TVM branch 3: ' + compact(value))
    for library in value.get('libraries') or []:
        if library.get('path') and library.get('sha256'):
            library_path = Path(library['path'])
            if not library_path.is_file() or sha(library_path) != library['sha256']:
                raise PreparationPending('repeat selfcheck after the TVM source-build library changed: ' + str(library_path))
    return dict(path=str(path), sha256=sha(path), receipt=value)


def prefix_status(cell):
    prefix = cell.get('native_prefix')
    if not prefix:
        reason = 'no already-gated native artifact supplied for this cell'
        if cell['batch'] > 1:
            reason = 'opening native B>1 is known incorrect; no verified replacement supplied or attempted'
        if cell['model'] == 'nafnet':
            reason = 'NAFNet sanitizer/numerical repair has not completed; no native prefix attempted'
        return dict(available=False, reason=cell.get('native_prefix_unavailable_reason') or reason)
    required = ('library', 'bridge', 'inputs', 'gate_receipt')
    if any(not Path(prefix.get(key, '')).is_file() for key in required):
        raise HardStop('supplied native prefix lacks an artifact, bridge, inputs or passing gate receipt')
    gate = read(prefix['gate_receipt'])
    if (not gate.get('passed') or gate.get('batch') != cell['batch'] or
            gate.get('model') != cell['model'] or gate.get('library_sha256') != sha(prefix['library'])):
        raise HardStop('native prefix gate does not certify this exact model/batch/library')
    if gate.get('archive_sha256') != cell['exported_program_sha256']:
        raise HardStop('native prefix uses another exported program')
    if (gate.get('source_checkpoint_sha256') != cell['source_checkpoint_sha256'] or
            gate.get('bridge_sha256') != sha(prefix['bridge'])):
        raise HardStop('native prefix gate uses another checkpoint or bridge')
    identity = read(str(prefix['library']) + '.identity.json')
    if identity.get('trace') or any(key.startswith('TILEMEGA_TRACE') and str(value) not in ('0', 'false', 'False')
            for key, value in identity.get('defines', {}).items()):
        raise HardStop('instrumented native artifact cannot be a timing prefix')
    sanitizers = {}
    if cell['model'] == 'nafnet':
        receipts = prefix.get('sanitizer_receipts') or {}
        tools = ('memcheck', 'racecheck', 'synccheck', 'initcheck')
        if any(not Path(receipts.get(tool, '')).is_file() for tool in tools):
            return dict(available=False, reason='NAFNet prefix lacks all four sanitizer receipts; no native prefix attempted')
        for tool in tools:
            path = Path(receipts[tool])
            receipt = read(path)
            if (receipt.get('passed') is not True or receipt.get('batch') != cell['batch'] or
                    receipt.get('library_sha256') != sha(prefix['library']) or
                    receipt.get('tool', tool) != tool):
                raise HardStop('NAFNet sanitizer receipt does not certify this artifact/batch/tool: ' + str(path))
            sanitizers[tool] = dict(path=str(path), sha256=sha(path), receipt=receipt)
    return dict(available=True, artifact=prefix, library_sha256=sha(prefix['library']),
                gate_receipt_sha256=sha(prefix['gate_receipt']),
                sanitizer_evidence=sanitizers,
                scope='opening already-gated artifact only; final coordinates require solver/P-SEL')


def normalize_diag(value):
    """Keep source instruction visibility and library offload/linking separate."""
    def instructions(direct, cuda, ptx):
        if value.get(direct) is True or value.get(direct) is False:
            return value[direct]
        observations = [value.get(cuda), value.get(ptx)]
        if any(item is True for item in observations):
            return True
        if all(item is False for item in observations):
            return False
        return None
    def linked(name):
        libraries = value.get('libraries')
        if not libraries:
            return value.get('links_' + name)
        sections = [row.get('dynamic_section') for row in libraries]
        if any(isinstance(section, str) and 'lib' + name in section and not row.get('readelf_error')
               for row, section in zip(libraries, sections)):
            return True
        if all(isinstance(section, str) and section and not row.get('readelf_error')
               for row, section in zip(libraries, sections)):
            return False
        return None
    count = value.get('byoc_function_count')
    if count is None:
        byoc = value.get('byoc_functions')
        count = len(byoc) if isinstance(byoc, dict) else byoc
    mac_fraction = value.get('byoc_mac_flop_fraction', value.get('byoc_flop_fraction'))
    tuning = value.get('tuning') or {}
    return dict(byoc_functions=count, byoc_flop_fraction=mac_fraction,
        flop_convention='2*MAC; static conv/matmul subtotal; full graph fraction unknown',
        full_graph_flop_fraction=value.get('byoc_total_flop_fraction'),
        flop_coverage_status=value.get('flop_coverage_status', 'unknown for full graph'),
        scheduled_tir_functions=value.get('scheduled_tir_function_count', value.get('scheduled_tir_functions')),
        has_wmma=instructions('has_wmma', 'cuda_wmma', 'ptx_wmma'),
        has_mma_sync=instructions('has_mma_sync', 'cuda_mma_sync', 'ptx_mma_sync'),
        cuda_wmma=value.get('cuda_wmma'), ptx_wmma=value.get('ptx_wmma'),
        cuda_mma_sync=value.get('cuda_mma_sync'), ptx_mma_sync=value.get('ptx_mma_sync'),
        external_library_tensor_core_instructions=value.get('external_library_tensor_core_instructions',
            'unknown; generated sources do not expose external library kernels'),
        links_cublas=linked('cublas'), links_cudnn=linked('cudnn'),
        cublas_offload_runtime=value.get('cublas_offload_runtime'),
        cudnn_offload_runtime=value.get('cudnn_offload_runtime'),
        trials=tuning.get('committed_trials', value.get('trials')),
        tune_wall_seconds=tuning.get('elapsed_s', value.get('tune_wall_seconds')),
        library_sha256=value.get('compiled_library_sha256', value.get('library_sha256')))


def classic_device_identity(gpu):
    if gpu not in ACTIVE_GPUS:
        raise ValueError('new baseline queues use only indices0–6; index7 does not exist')
    devices = read(CLASSIC_DEVICES)
    if (devices.get('schema') != 1 or devices.get('allowed_gpu_indices') != list(ACTIVE_GPUS) or
            set(devices.get('devices', {})) != set(map(str, ACTIVE_GPUS))):
        raise ValueError('current classic device mapping must enumerate exactly indices0–6')
    identity = devices['devices'][str(gpu)]
    if not identity['uuid'].startswith('GPU-') or identity['pci'].lower() != PCI[gpu]:
        raise ValueError('current device UUID/PCI mapping differs from classic topology')
    return identity


def queue_workload(config):
    """Absent fields keep dnn-classic-v1; any other id declares its whole workload."""
    workload_id = config.get('workload_id', WORKLOAD_ID)
    fields = ('models', 'batches', 'arms', 'device_file')
    if workload_id == WORKLOAD_ID:
        if any(name in config for name in fields):
            raise ValueError('the classic workload takes no queue-spec overrides')
        return dict(id=WORKLOAD_ID, models=MODELS, batches=BATCHES, arms=None,
                    device_file=CLASSIC_DEVICES, active_gpus=ACTIVE_GPUS)
    if workload_id not in SPEC_WORKLOADS:
        raise ValueError('new queues require the active classic workload; retain historical queues')
    if any(name not in config for name in fields):
        raise ValueError('a spec workload must declare models, batches, arms and device_file')
    models, batches, arms = map(tuple, (config['models'], config['batches'], config['arms']))
    known = {arm['arm'] for arm in ARMS if arm['system'] != 'tvm'}
    if (not models or len(set(models)) != len(models) or not batches or
            not set(batches) <= set(BATCHES) or not arms or not set(arms) <= known):
        raise ValueError('invalid spec workload models, batches or arms')
    if workload_id == 'dnn3-v1' and 'tensorrt-fp16' in arms:
        raise ValueError('dnn3-v1 collects BF16 baselines only')
    path = Path(config['device_file'])
    path = path if path.is_absolute() else ROOT / path
    devices = read(path)
    if devices.get('workload_id') != workload_id:
        raise ValueError('device file belongs to another workload')
    return dict(id=workload_id, models=models, batches=batches, arms=arms, device_file=path,
                active_gpus=tuple(devices['allowed_gpu_indices']))


def workload_device_identity(gpu, workload):
    devices = read(workload['device_file'])
    allowed = devices.get('allowed_gpu_indices')
    if (devices.get('schema') != 1 or gpu not in allowed or
            set(devices.get('devices', {})) != set(map(str, allowed))):
        raise ValueError('device file must enumerate exactly its allowed indices')
    identity = devices['devices'][str(gpu)]
    if not identity['uuid'].startswith('GPU-') or identity['pci'].lower() != PCI[gpu]:
        raise ValueError('current device UUID/PCI mapping differs from the workload device file')
    return identity


def queue(args):
    config = read(args.spec)
    workload = queue_workload(config)
    classic = workload['id'] == WORKLOAD_ID
    if not classic and not {'exports_root', 'inductor_cache', 'tvm_root'} <= set(config):
        raise ValueError('a spec workload must not fall back to the dnn1 asset roots')
    gpu = config.get('gpu', 0)
    identity = classic_device_identity(gpu) if classic else workload_device_identity(gpu, workload)
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.bench.gpu_guard import require_allowed_gpu
    policy = read(POLICY)
    require_allowed_gpu(policy, str(gpu), timing=True)
    out = args.out.resolve()
    if (out / 'queue-spec.json').exists():
        if not getattr(args, 'refresh_prepared', False):
            raise FileExistsError('retain prior queue evidence; use --refresh-prepared only before execution')
        if (out / 'scheduler').exists() or (out / 'raw').exists():
            raise ValueError('cannot refresh after any scheduler/measurement activity; choose a new run directory')
        revision = out / 'prepared_revisions' / str(time.time_ns())
        revision.mkdir(parents=True)
        for name in ('queue-spec.json', 'queue', 'cells', 'launch.sh', 'preflight.json'):
            source = out / name
            if source.is_dir():
                shutil.copytree(source, revision / name)
            elif source.exists():
                shutil.copy2(source, revision / name)
    out.mkdir(parents=True, exist_ok=True)
    # Resolving the executable symlink would bypass the virtual environment.
    python = str(Path(config.get('python', DEFAULT_PYTHON)).absolute())
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    harness = ROOT / 'python/tilemega/dnn/baselines.py'
    recipe_sources = {str(path): sha(path) if path.is_file() else None for path in (
        Path(__file__).resolve(), harness, harness.with_name('tvm_baseline.py'),
        workload['device_file'], CLASSIC_DEVICES,
        ROOT/'scripts/dnn1_device.py',ROOT/'scripts/dnn1_resource_entry.py',
        ROOT/'scripts/dnn1_supervised_python.py',ROOT/'scripts/dnn1_guard_entry.py')}
    cells = config.get('cells') or [dict(model=m, batch=b) for m in workload['models'] for b in workload['batches']]
    pairs = [(row['model'], row['batch']) for row in cells]
    if len(set(pairs)) != len(pairs) or any(m not in workload['models'] or b not in workload['batches'] for m, b in pairs):
        raise ValueError('cells must be distinct DNN1 model/batch pairs')
    jobs, previous, snapshots = [], [], []
    env = gpu_env(gpu, config.get('env'))
    systems = set(config.get('systems', ('tensorrt', 'torchcompile')))
    if not systems or not systems <= {'tensorrt', 'torchcompile'}:
        raise ValueError('classic primary queues select TRT/TC; TVM uses its fresh10000 entry')
    arms = [arm for arm in ARMS if arm['system'] in systems and
            (workload['arms'] is None or arm['arm'] in workload['arms'])]
    parallel = config.get('parallel', False)
    affinity = config.get('cpu_affinity') or []
    if parallel:
        if not affinity or any(type(cpu) is not int or cpu < 0 for cpu in affinity):
            raise ValueError('parallel cells require an explicit nonempty CPU affinity')
        env['TILEMEGA_NUMA_CONCURRENCY_TRIAL'] = '1'
        env.pop('TILEMEGA_NUMA_CONCURRENCY_VERIFIED', None)
    uuid, pci = identity['uuid'], identity['pci']
    node = 0 if gpu < 4 else 1
    inductor_root = Path(config.get('inductor_cache', ROOT / 'runs/dnn1/inductor'))
    for model, batch in pairs:
        key = f'{model}_b{batch}'
        cell_root = out / 'cells' / key
        export = Path(config.get('exports_root', ROOT / 'runs/dnn1/exports')) / model
        export = export.resolve()
        manifest = read(export / 'manifest.json')
        ep_sha = sha(export / 'exported_program.pt2')
        checkpoint_sha = sha(export / 'checkpoint/model.safetensors')
        if manifest['model'] != model or manifest['artifacts']['exported_program.pt2'] != ep_sha:
            raise ValueError('export identity differs: ' + key)
        if manifest.get('checkpoint', {}).get('sha256') != checkpoint_sha:
            raise ValueError('export checkpoint identity differs: ' + key)
        if not manifest.get('timing_eligible', manifest.get('accuracy_eligible', False)):
            raise ValueError('export is not timing eligible: ' + key)
        cell = dict(schema=1, model=model, batch=batch, gpu=gpu, export=str(export),
            workload_id=workload['id'],
            cell_root=str(cell_root), python=python, baselines=str(harness),
            baseline_harness_sha256=sha(harness), collector=str(Path(__file__).resolve()),
            recipe_sources=recipe_sources,
            collector_sha256=sha(__file__), exported_program_sha256=ep_sha,
            source_checkpoint_sha256=checkpoint_sha, seed=20261007,
            correctness_policy=config.get('correctness_policy', 'identical-metrics-v1'),
            tvm_selfcheck=str(Path(config.get('tvm_selfcheck', ROOT / 'runs/dnn1/tvm/selfcheck.json')).resolve()),
            tvm_root=str(config.get('tvm_root', ROOT / 'runs/dnn1/tvm')), engine_cache=str(out / 'engine-cache'),
            onnx_preparation=str(Path(config.get('onnx_preparation',
                Path(config.get('exports_root', ROOT / 'runs/dnn1/exports')) / 'preparation_v2.json')).resolve()),
            inductor_cache=str(inductor_root), env=env,
            libraries=config.get('libraries', {}), logical_timeout_s=config.get('logical_timeout_s', 3600),
            native_prefix=config.get('native_prefixes', {}).get(key),
            native_prefix_unavailable_reason=config.get('native_prefix_unavailable_reasons', {}).get(key),
            parallel=parallel, cpu_affinity=affinity,
            scheduling_amendment=config.get('scheduling_amendment'),
            protocol=PROTOCOL, arms=arms, order=[
                [arm['arm'] for arm in (reversed(arms) if r == 1 else arms)] for r in range(3)])
        if not classic:
            cell.update(active_gpus=list(workload['active_gpus']), device_file=str(workload['device_file']))
        if parallel:
            cell['inductor_cache'] = str(inductor_root / f'gpu{gpu}')
        cell['prefix'] = prefix_status(cell)
        cell['p_cell_complete'] = cell['prefix']['available']
        cell_path = cell_root / 'spec.json'
        write(cell_path, cell)
        snapshots.append(str(cell_path))
        export_name, name = 'dnn1-export-' + key, 'dnn1-cell-' + key
        export_command = [python, str(Path(__file__).resolve()), 'run', '--kind', 'export',
                          '--cell', str(cell_path), '--out', '{attempt_dir}/job']
        jobs.append(dict(name=export_name, gpu=False, timing=False, priority=10,
            timeout_s=7200, cwd=str(ROOT), after=previous, env=dict(env, CUDA_VISIBLE_DEVICES=''),
            command=['flock', f'/tmp/tilemega_gpu{gpu}.lock', *export_command]))
        binding = '--physcpubind=' + ','.join(map(str, affinity)) if parallel else f'--cpunodebind={node}'
        command = ['numactl', binding, python,
            str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(POLICY),
            '--out', '{attempt_dir}/job', '--expected-uuid', uuid, '--expected-pci', pci,
            '--numa-node', str(node), '--allow-no-measurements', '--', python,
            str(Path(__file__).resolve()), 'run', '--cell', str(cell_path), '--out', '{attempt_dir}/job']
        jobs.append(dict(name=name, gpu=True, timing=True, priority=10, timeout_s=config.get('cell_timeout_s', 86400),
            needs_free_mib=config.get('needs_free_mib', 8192), cwd=str(ROOT), after=[export_name],
            env=env, command=command))
        previous = [name]
    write(out / 'queue' / 'queue_dnn1_baselines.json', jobs)
    write(out / 'queue-spec.json', dict(schema=1, config=config, cells=snapshots,
        workload_id=workload['id'], workload=dict(models=list(workload['models']), batches=list(workload['batches'])),
        gpu=gpu, protocol=PROTOCOL, source_spec=str(args.spec.resolve()), source_spec_sha256=sha(args.spec)))
    launcher = out / 'launch.sh'
    start = ['env', '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD',
        '-u', 'TILEMEGA_RUN_NONCE', '-u', 'TILEMEGA_NUMA_CONCURRENCY_VERIFIED',
        '-u', 'TILEMEGA_NUMA_CONCURRENCY_TRIAL', *[f'{k}={v}' for k, v in env.items()],
        python, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(out / 'queue'),
        '--out', str(out / 'scheduler'), '--policy', str(POLICY), '--deadline-hours', '96']
    scheduler_lock = f'/tmp/tilemega_dnn1_baseline_scheduler_gpu{gpu}.lock' if parallel else '/tmp/tilemega_dnn1_baseline_scheduler.lock'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n' +
        shlex.join([python, str(Path(__file__).resolve()), 'preflight', '--run-root', str(out)]) + '\n' +
        shlex.join(['flock', scheduler_lock, *start]) + '\n')
    launcher.chmod(0o755)
    print(compact(dict(queue=str(out / 'queue'), cells=len(cells), gpu=gpu,
                       launcher=str(launcher), executed_gpu=False)))


def preflight(args):
    spec = read(args.run_root / 'queue-spec.json')
    checked = {}
    sources = set()
    try:
        for cell_path in spec['cells']:
            cell = read(cell_path)
            for path, expected in cell.get('recipe_sources', {}).items():
                if path in sources:
                    continue
                sources.add(path)
                if not Path(path).is_file():
                    raise PreparationPending('awaiting the final baseline source: ' + path)
                if expected is None or sha(path) != expected:
                    raise PreparationPending('baseline source changed; refresh the unstarted queue: ' + path)
            if any(a['system']=='tvm' for a in cell['arms']) and cell['tvm_selfcheck'] not in checked:
                checked[cell['tvm_selfcheck']] = validate_tvm_selfcheck(cell)
        record = dict(schema=1, status='ready', checked_selfchecks=checked, source_files=sorted(sources),
                      gpu=spec['gpu'], cells=len(spec['cells']), executed_gpu=False)
    except PreparationPending as error:
        record = dict(schema=1, status='preparation_pending', error=str(error),
                      gpu=spec['gpu'], executed_gpu=False)
        write(args.run_root / 'preflight.json', record)
        raise
    write(args.run_root / 'preflight.json', record)
    print(compact(record))


def execute(command, folder, env, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'command.json', dict(command=command, cwd=str(ROOT)))
    started = time.time()
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            sys.path.insert(0, str(ROOT / 'python'))
            from tilemega.bench.gpu_guard import stop
            stop(process)
    return dict(returncode=code, pid=process.pid, started=started, ended=time.time(),
                stdout=str(folder / 'stdout.log'), stderr=str(folder / 'stderr.log'))


def arm_command(cell, arm, role, index, folder):
    command = [cell['python'], cell['baselines'], '--export', cell['export'],
        '--system', arm['system'], '--precision', arm['precision'], '--batch', str(cell['batch']),
        '--seed', str(cell['seed']), '--warmup', '50', '--iterations', '200',
        '--round-index', str(index), '--out', str(folder)]
    if role == 'correctness':
        command += ['--correctness-only']
    if arm['system'] == 'tensorrt':
        command += ['--engine-cache', cell['engine_cache']]
    elif arm['system'] == 'torchcompile':
        command += ['--compile-mode', arm['compile_mode'], '--fullgraph', str(arm['fullgraph']).lower()]
        if arm['freezing']:
            command += ['--freezing']
    else:
        command += ['--tvm-work-dir', str(Path(cell['tvm_root']) / 'tune' /
            f"{cell['model']}_b{cell['batch']}" / arm['precision']),
            '--tvm-max-trials', '2000', '--tvm-timeout-s', '1800']
    return command


def receipt_problem(cell, arm, role, result):
    if (result.get('model') != cell['model'] or result.get('batch') != cell['batch'] or
            result.get('gpu_index') != cell['gpu'] or result.get('system') != arm['system'] or
            result.get('precision') != arm['precision']):
        return 'measurement model/batch/system/precision/GPU identity differs'
    if (result.get('exported_program_sha256') != cell['exported_program_sha256'] or
            result.get('source_checkpoint_sha256') != cell['source_checkpoint_sha256']):
        return 'measurement export/checkpoint hashes differ'
    if role == 'measurement':
        timing = result.get('measurement') or {}
        if timing.get('warmup') != 50 or timing.get('iterations') != 200:
            return 'timing does not use 50 warmups and 200 CUDA events'
        if not all(isinstance(result.get(key), (int, float)) and math.isfinite(result[key]) and result[key] > 0
                   for key in ('median_ms', 'p90_ms')):
            return 'invalid process median/p90'
    elif result.get('process_role') != 'correctness' or result.get('median_ms') is not None:
        return 'correctness-only process was measured or lacks its role marker'
    if arm['system'] == 'torchcompile':
        cfg = result.get('configuration', {})
        if cfg.get('fullgraph') != arm['fullgraph'] or cfg.get('freezing') != arm['freezing']:
            return 'actual fullgraph/freezing configuration differs from the named arm'
    if arm['system'] == 'tensorrt':
        identity = result.get('configuration', {}).get('engine_cache', {}).get('identity', {})
        if identity.get('static_batch') != cell['batch']:
            return 'TensorRT cache identity lacks the correct static batch'
    if not result.get('configuration', {}).get('inputs_sha256'):
        return 'measurement lacks common CPU input identity'
    return ''


def measurement_clocks(result, gpu):
    path = Path(result.get('guard_log', ''))
    if not path.is_file():
        raise ValueError('receipt lacks guard_log measurement-boundary samples')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    samples = {}
    for name, label in [('clock_start', 'before-measurement'), ('clock_end', 'after-measurement')]:
        matches = [row for row in rows if row.get('label') == label and row.get('gpu_index') == gpu]
        if len(matches) != 1 or matches[0].get('accepted') is not True:
            raise ValueError('missing, duplicated or rejected measurement-boundary sample: ' + label)
        try:
            clock = float(matches[0].get('telemetry', {}).get('clocks.sm', 'nan'))
        except (TypeError, ValueError) as error:
            raise ValueError('invalid measurement-boundary SM clock: ' + label) from error
        if not math.isfinite(clock) or clock <= 0:
            raise ValueError('invalid measurement-boundary SM clock: ' + label)
        samples[name] = dict(sm_clock_mhz=clock, sample=matches[0],
                             source=str(path), source_sha256=sha(path))
    if samples['clock_end']['sample']['time'] < samples['clock_start']['sample']['time']:
        raise ValueError('measurement-boundary samples are reversed')
    return samples


def logical(cell, arm, role, index, out):
    attempts = []
    for attempt in range(2):
        folder = out / 'measurements' / f'{role}-{index}' / arm['arm'] / f'attempt{attempt + 1}'
        env = dict(os.environ, **cell['env'], TORCHINDUCTOR_CACHE_DIR=str(
            Path(cell['inductor_cache']) / f"{cell['model']}_b{cell['batch']}" / arm['arm']))
        before = telemetry(cell)
        record = execute(arm_command(cell, arm, role, index, folder), folder, env, cell['logical_timeout_s'])
        after = telemetry(cell)
        record.update(attempt=attempt + 1, folder=str(folder), process_before=before, process_after=after)
        failure_path = folder / 'failure.json'
        failure = read(failure_path) if failure_path.is_file() else {}
        measure_path = folder / 'measurements.json'
        value = read(measure_path) if measure_path.is_file() else None
        reason = receipt_problem(cell, arm, role, value) if record['returncode'] == 0 and value else ''
        if record['returncode'] == 0 and value:
            try:
                record.update(measurement_clocks(value, cell['gpu']))
            except (ValueError, KeyError) as error:
                reason = str(error)
        if value and value.get('pid') != record['pid']:
            reason = 'measurement PID differs from the fresh subprocess'
        good, correctness_reason = strict_correctness(cell['model'], value.get('correctness')) if value else (False, '')
        if record['returncode'] == 0 and value and not reason and good:
            record.update(status='pass', measurement=str(measure_path), correctness=value['correctness'])
        else:
            if reason:
                failure = dict(failure_kind='metadata', error=reason)
            elif record['returncode'] == 0 and value and not good:
                failure = dict(failure_kind='correctness', error=correctness_reason,
                               raw_correctness=value.get('correctness'))
            elif not failure:
                failure = dict(failure_kind='unclassified', error='subprocess failed or no receipt; retain stdout/stderr verbatim')
            record.update(status='fail', failure=failure,
                          measurement=str(measure_path) if measure_path.is_file() else None)
        write(folder / 'logical-attempt.json', record)
        attempts.append(record)
        if record['returncode'] == 75 or any(row.get('throttle_mask', 0) & (0x8 | 0x20 | 0x40 | 0x80)
                                            for row in (before, after)):
            raise HardStop('whole cell invalidated by timing guard/throttle; discard its measurements: ' + str(folder))
        if (record['status'] == 'pass' or arm['system'] == 'tvm' or
                (arm['system'] == 'torchcompile' and arm.get('fullgraph') and
                 failure.get('failure_kind') == 'compile')):
            break
    return dict(arm=arm, role=role, index=index, attempts=attempts,
                status=attempts[-1]['status'], final=attempts[-1])


def fallback_arm(arm, result):
    kind = result['final'].get('failure', {}).get('failure_kind')
    if arm['system'] == 'torchcompile' and arm.get('fullgraph') and kind == 'compile':
        return dict(arm, arm=arm['arm'] + '-fullgraph-false', fullgraph=False)
    if arm['system'] == 'tvm' and arm['precision'] == 'bf16' and kind in ('compile', 'correctness'):
        return dict(arm, arm='tvm-fp16', precision='fp16')
    return None


def native_visit(cell, index, out):
    prefix = cell['prefix']['artifact']
    spec = dict(name='dnn1-baseline-prefix', kind='final', model=cell['model'], batch=cell['batch'], gpu=cell['gpu'],
        arms={'prefix': dict(root=str(ROOT), library=prefix['library'], export=cell['export'],
            bridge=prefix['bridge'], inputs=prefix['inputs'], library_sha256=sha(prefix['library']),
            inputs_sha256=sha(prefix['inputs']), root_commit=subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())})
    path = out / 'native-prefix-spec.json'
    write(path, spec)
    folder = out / 'measurements' / f'native-prefix-{index}'
    command = [cell['python'], str(ROOT / 'scripts/sm90r3_dnn.py'), 'measure', '--spec', str(path),
               '--arm', 'prefix', '--fresh', str(index), '--out', str(folder)]
    record = execute(command, folder, dict(os.environ, **cell['env']), 1800)
    if record['returncode']:
        raise HardStop('already-gated native prefix failed; no further GPU jobs in this line: ' + compact(record))
    return record


def cached_onnx(cell):
    directory = Path(cell['export'])
    metadata = directory / f"onnx_b{cell['batch']}.json"
    if not metadata.is_file():
        return None
    value = read(metadata)
    if (value.get('batch') != cell['batch'] or
            value.get('exported_program_sha256') != cell['exported_program_sha256'] or
            value.get('input_seed') != cell['seed']):
        raise ValueError('prepared ONNX static batch, exported program hash or input seed differs')
    for path_key, digest_key in [('path', 'sha256'), ('native_inputs_file', 'native_inputs_sha256')]:
        path = Path(value[path_key])
        if not path.is_file() or sha(path) != value[digest_key]:
            raise ValueError('prepared ONNX/input file hash differs: ' + path_key)
    return dict(metadata=str(metadata), metadata_sha256=sha(metadata), receipt=value)


def run_export(cell, out):
    attempts = []
    command = [cell['python'], cell['baselines'], '--system', 'export-onnx',
               '--export', cell['export'], '--batch', str(cell['batch']), '--seed', str(cell['seed'])]
    directory = Path(cell['export'])
    with (directory / f".onnx_b{cell['batch']}.lock").open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = dict(status='fail', attempts=attempts, model=cell['model'], batch=cell['batch'],
                      cpu_only=True, onnx_exports_successful=0, reused=False)
        try:
            cache = cached_onnx(cell)
            preparation = Path(cell.get('onnx_preparation', directory.parent / 'preparation_v2.json'))
            prior = [row for row in read(preparation) if row.get('model') == cell['model']
                     and row.get('batch') == cell['batch']] if preparation.is_file() else []
            if cache:
                record.update(status='pass', reused=True, cache=cache)
            elif prior and prior[-1]['status'] == 'failed':
                record.update(preparation=prior[-1], error='prior ONNX preparation failed; no additional retries')
            elif preparation.name == 'preparation_v2.json' and any(directory.glob(f"onnx_b{cell['batch']}_v2_attempt*.log")):
                record['error'] = 'preparation attempt logs exist without a valid receipt; retain evidence and do not duplicate export'
            else:
                for index in range(2):
                    env = dict(os.environ, **cell['env'])
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    value = execute(command, out / f'attempt{index + 1}', env, 7200)
                    attempts.append(value)
                    if value['returncode'] == 0:
                        cache = cached_onnx(cell)
                        if cache is None:
                            raise ValueError('successful ONNX exporter left no identified receipt')
                        record.update(status='pass', cache=cache, onnx_exports_successful=1)
                        break
        except Exception as error:
            record.update(error=str(error), traceback=traceback.format_exc())
    write(out / 'export-result.json', record)
    write(Path(cell['cell_root']) / 'onnx-result.json', record)
    print(compact(dict(export_status=record['status'], model=cell['model'], batch=cell['batch'])))


def run(args):
    cell = read(args.cell)
    if sha(__file__) != cell['collector_sha256'] or sha(cell['baselines']) != cell['baseline_harness_sha256']:
        raise HardStop('collector or frozen baseline recipe changed after queue creation')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.kind == 'export':
        run_export(cell, args.out)
        return
    try:
        selfcheck = (validate_tvm_selfcheck(cell) if any(a['system']=='tvm' for a in cell['arms']) else
                     dict(status='deferred', reason='user prioritizes TRT/torch.compile; TVM evidence collected separately'))
    except PreparationPending as error:
        write(args.out / 'cell-result.json', dict(schema=1, status='preparation_pending',
            model=cell['model'], batch=cell['batch'], gpu=cell['gpu'], error=str(error),
            logical_jobs=[], order=[], prefix=cell['prefix'], p_cell_complete=False))
        raise
    require_session(cell)
    cell_lock = (f"/tmp/tilemega_dnn1_baseline_cell_gpu{cell['gpu']}.lock" if cell.get('parallel') else
                 '/tmp/tilemega_dnn1_baseline_cell.lock')
    with Path(cell_lock).open('a') as serial:
        fcntl.flock(serial, fcntl.LOCK_EX)
        record = dict(schema=1, cell=str(args.cell.resolve()), cell_sha256=sha(args.cell),
            model=cell['model'], batch=cell['batch'], gpu=cell['gpu'], protocol=cell['protocol'],
            prefix=cell['prefix'], p_cell_complete=cell['p_cell_complete'], started=time.time(),
            order=[], logical_jobs=[], native_visits=[], status='running', runtime_libraries={},
            parallel=cell.get('parallel', False), cpu_affinity=sorted(os.sched_getaffinity(0)),
            numa_concurrency_verified=False if cell.get('parallel') else None,
            scheduling_amendment=cell.get('scheduling_amendment'))
        destination = args.out / 'cell-result.json'
        def save():
            write(destination, record)
        try:
            record['tvm_selfcheck'] = selfcheck
            torch_spec = importlib.util.find_spec('torch')
            if torch_spec and torch_spec.origin:
                library = Path(torch_spec.origin).parent / 'lib/libtorch_cuda.so'
                if library.is_file():
                    record['runtime_libraries']['torchcompile'] = {str(library): sha(library)}
            for system, paths in cell.get('libraries', {}).items():
                if isinstance(paths, str):
                    paths = [paths]
                record['runtime_libraries'][system] = {str(Path(p).resolve()): sha(p) for p in paths}
            onnx = read(Path(cell['cell_root']) / 'onnx-result.json')
            dynamic = {}
            def visit(arm, role, index, phase):
                if any(row['arm']['arm'] == arm['arm'] and row['role'] == role and row['index'] == index
                       for row in record['logical_jobs']):
                    return
                if arm['system'] == 'tensorrt' and onnx['status'] != 'pass':
                    record['logical_jobs'].append(dict(arm=arm, role=role, index=index,
                        status='N/A', final=dict(failure=dict(failure_kind='onnx_export', error=compact(onnx))), attempts=[]))
                    save()
                    return
                record['order'].append(dict(position=len(record['order']), arm=arm['arm'], role=role,
                                            index=index, phase=phase))
                result = logical(cell, arm, role, index, args.out)
                record['logical_jobs'].append(result)
                save()
                if result['status'] == 'pass':
                    inputs = read(result['final']['measurement'])['configuration']['inputs_sha256']
                    previous = record.get('inputs_sha256')
                    if previous and previous != inputs:
                        raise HardStop('systems in one cell used different input tensors')
                    record['inputs_sha256'] = inputs
                    return
                kind = result['final'].get('failure', {}).get('failure_kind')
                if kind == 'tvm_cuda_build_or_selfcheck':
                    raise HardStop('DT-TVM branch 3: ' + compact(result['final']['failure']))
                alternate = fallback_arm(arm, result)
                if alternate:
                    dynamic[alternate['arm']] = alternate
                    visit(alternate, role, index, 'explicit reference after ' + arm['arm'])
                elif arm['system'] == 'tvm' and kind not in ('compile', 'correctness'):
                    raise HardStop('TVM failure outside DT-TVM fp16 branch requires diagnosis: ' + compact(result['final']))
            for index in range(3):
                if cell['prefix']['available'] and index != 1:
                    record['order'].append(dict(arm='tilemega-prefix', role='measurement', index=index))
                    record['native_visits'].append(native_visit(cell, index, args.out))
                arms = reversed(cell['arms']) if index == 1 else cell['arms']
                for arm in arms:
                    visit(arm, 'measurement', index, 'primary round')
                if cell['prefix']['available'] and index == 1:
                    record['order'].append(dict(arm='tilemega-prefix', role='measurement', index=index))
                    record['native_visits'].append(native_visit(cell, index, args.out))
            completed_arms = set()
            while True:
                pending = [arm for arm in [*cell['arms'], *dynamic.values()]
                           if arm['arm'] not in completed_arms]
                if not pending:
                    break
                arm = pending[0]
                completed_arms.add(arm['arm'])
                if arm['arm'] in dynamic:
                    for index in range(3):
                        visit(arm, 'measurement', index, 'complete explicit reference rounds')
                measured = [row for row in record['logical_jobs'] if row['arm']['arm'] == arm['arm']
                            and row['role'] == 'measurement']
                if len(measured) != 3 or any(row['status'] != 'pass' for row in measured):
                    continue
                for index in range(2):
                    visit(arm, 'correctness', index, 'additional P-CORR process')
            record.update(status='complete', ended=time.time())
        except Exception as error:
            record.update(status='hard_stop', ended=time.time(), error=str(error), traceback=traceback.format_exc())
            save()
            raise
        finally:
            record['exit_context_drain'] = passive_context_drain(cell)
            save()
        save()
        print(compact(dict(status=record['status'], model=cell['model'], batch=cell['batch'],
            jobs=len(record['logical_jobs']), native_prefix=cell['prefix'], p_cell_complete=cell['p_cell_complete'])))


def artifact_hashes(measurement, session, system):
    cfg = measurement.get('configuration', {})
    diag = cfg.get('tvm_diag', {})
    if system == 'torchcompile' and cfg.get('library_sha256'):
        return {key: cfg[key] for key in ('library_sha256', 'compiled_wrapper_digest') if cfg.get(key)}
    for key in ('engine_sha256', 'compiled_library_sha256', 'library_sha256'):
        if cfg.get(key):
            return {key: cfg[key]}
        if diag.get(key):
            return {key: diag[key]}
    return session.get('runtime_libraries', {}).get(system, {})


def aggregate_arm(cell, session, name, jobs, accepted):
    arm = jobs[0]['arm']
    passing = [row for row in jobs if row['status'] == 'pass'] if accepted else []
    measured = [row for row in passing if row['role'] == 'measurement']
    values = [read(row['final']['measurement']) for row in passing]
    timing = [read(row['final']['measurement']) for row in measured]
    correctness = [value['correctness'] for value in values]
    identical = len(correctness) == 5 and len({compact(value) for value in correctness}) == 1
    policy = cell.get('correctness_policy', 'identical-metrics-v1')
    if policy not in ('identical-metrics-v1', 'five-independent-gates-v2'):
        raise ValueError('unknown baseline correctness policy: ' + policy)
    # Independent compiler/autotuner processes may choose different legal
    # reductions. V2 retains every numerical gate, recording reproducibility
    # separately; it is only enabled by an explicitly amended cell spec.
    gates = all(strict_correctness(cell['model'], metrics)[0] for metrics in correctness)
    complete = (len(measured) == 3 and len(passing) == 5 and gates
                and (identical or policy == 'five-independent-gates-v2'))
    failures = [dict(failure_kind=row.get('final', {}).get('failure', {}).get('failure_kind'),
                     error=row.get('final', {}).get('failure', {}).get('error'),
                     source=row.get('final', {}).get('folder'))
                for row in jobs if row['status'] != 'pass']
    reason = ''
    if not accepted:
        reason = 'outer physical-card guard rejected or session not completed; no timings admitted'
        if session.get('error'):
            reason += ': ' + session['error']
    elif failures:
        reason = compact(list({compact(value): value for value in failures}.values()))
    elif not complete:
        reason = ('requires exactly 3 timing processes and 5 identical passing correctness metrics'
                  if policy == 'identical-metrics-v1' else
                  'requires exactly 3 timing processes and 5 independently passing numerical gates')
    versions = [value.get('versions', {}) for value in values]
    hashes = [artifact_hashes(value, session, arm['system']) for value in timing]
    if complete and (not all(hashes) or len({compact(value) for value in versions}) != 1):
        complete = False
        reason = 'missing engine/library identity or process version mismatch'
    medians = [value['median_ms'] for value in timing]
    row = dict(model=cell['model'], batch=cell['batch'], system=arm['system'], arm=name,
        precision=arm['precision'], gpu=cell['gpu'], status='pass' if complete else 'N/A', reason=reason,
        sm_clock_start_mhz=compact([r['final']['clock_start']['sm_clock_mhz'] for r in measured]),
        sm_clock_end_mhz=compact([r['final']['clock_end']['sm_clock_mhz'] for r in measured]),
        sm_clock_source='receipt.guard_log before-measurement/after-measurement' if measured else '',
        processes=len(measured), attempt_processes=sum(len(row.get('attempts', [])) for row in jobs),
        failed_attempts=sum(attempt.get('status') != 'pass' for row in jobs for attempt in row.get('attempts', [])),
        correctness_processes=len(passing), correctness_passes=f'{len(passing)}/5',
        correctness_policy=policy, strict_v1_status='pass' if complete and identical else 'N/A',
        correctness_metrics_identical=identical, median_ms=statistics.median(medians) if complete else '',
        p90_ms=statistics.median(v['p90_ms'] for v in timing) if complete else '',
        process_range_ms=max(medians) - min(medians) if complete else '',
        process_range_pct=(max(medians) / min(medians) - 1) * 100 if complete else '',
        correctness=compact(correctness[0] if identical else correctness), versions=compact(versions[0] if versions else {}),
        engine_or_library_sha256=compact(list({compact(value): value for value in hashes}.values())),
        exported_program_sha256=cell['exported_program_sha256'],
        source_checkpoint_sha256=cell['source_checkpoint_sha256'], inputs_sha256=session.get('inputs_sha256', ''),
        native_prefix_available=cell['prefix']['available'], p_cell_complete=session.get('p_cell_complete', False),
        cell_session=session.get('source', ''), source=compact([r['final'].get('measurement') for r in passing]))
    return row


def collect(args):
    spec = read(args.run_root / 'queue-spec.json')
    state = read(args.run_root / 'scheduler/state.json')
    baseline_rows, tvm_rows, audit = [], [], []
    for cell_path in spec['cells']:
        cell = read(cell_path)
        key = f"{cell['model']}_b{cell['batch']}"
        job = state.get('dnn1-cell-' + key, {})
        folder = Path(job['out']) if job.get('out') else None
        guard_path = folder / 'guard_result.json' if folder else None
        guard = read(guard_path) if guard_path and guard_path.is_file() else {}
        session_path = folder / 'job/cell-result.json' if folder else None
        session = read(session_path) if session_path and session_path.is_file() else dict(logical_jobs=[])
        session['source'] = str(session_path) if session_path else ''
        accepted = job.get('status') == 'done' and guard.get('code') == 0 and session.get('status') == 'complete'
        grouped = {}
        for logical_job in session['logical_jobs']:
            grouped.setdefault(logical_job['arm']['arm'], []).append(logical_job)
        for arm in cell['arms']:
            grouped.setdefault(arm['arm'], [dict(arm=arm, role='measurement', index=0,
                status='N/A', attempts=[], final=dict(failure=dict(failure_kind='not_run',
                    error=session.get('error', 'cell did not run'))))])
        for arm, rows in grouped.items():
            baseline_row = aggregate_arm(cell, session, arm, rows, accepted)
            baseline_rows.append(baseline_row)
            if rows[0]['arm']['system'] != 'tvm':
                continue
            diagnostics, errors = [], []
            for row in rows:
                if not row.get('attempts') and row.get('final', {}).get('failure'):
                    errors.append(row['final']['failure'])
                for attempt in row.get('attempts', []):
                    if attempt.get('failure'):
                        errors.append(attempt['failure'])
                        failed_diag = attempt['failure'].get('configuration', {}).get('tvm_diag')
                        if failed_diag is not None:
                            diagnostics.append(failed_diag)
                    if attempt.get('measurement'):
                        value = read(attempt['measurement'])
                        diag = value.get('configuration', {}).get('tvm_diag')
                        if diag is not None:
                            diagnostics.append(diag)
                    alternate = Path(attempt['folder']) / 'tvm_diag.json'
                    if alternate.is_file():
                        diagnostics.append(read(alternate))
            normalized = [normalize_diag(value) for value in diagnostics]
            diag = normalized[0] if normalized else normalize_diag({})
            error_summary = [dict(failure_kind=e.get('failure_kind'), phase=e.get('phase'),
                                  error_type=e.get('error_type'), error=e.get('error')) for e in errors]
            error_summary = list({compact(value): value for value in error_summary}.values())
            diagnostic_summary = normalized
            diagnostic_summary = list({compact(value): value for value in diagnostic_summary}.values())
            tvm_rows.append(dict(model=cell['model'], batch=cell['batch'],
                precision=rows[0]['arm']['precision'], gpu=cell['gpu'],
                status=baseline_row['status'],
                failure_kind=compact([e.get('failure_kind') for e in errors]),
                **diag,
                diagnostics=compact(diagnostic_summary), error=compact(error_summary), source=str(session_path)))
        audit.append(dict(model=cell['model'], batch=cell['batch'], gpu=cell['gpu'], accepted=accepted,
            guard=guard, status=job.get('status'), session=str(session_path),
            order=session.get('order', []), prefix=cell['prefix'], p_cell_complete=cell['p_cell_complete']))
    args.out.mkdir(parents=True, exist_ok=True)
    classic = spec.get('workload_id') == WORKLOAD_ID
    stem = 'DNN_classic' if classic else spec.get('config', {}).get('output_stem', 'DNN1')
    baseline_fields = FIELDS
    if spec.get('workload_id') in SPEC_WORKLOADS:
        baseline_fields = FIELDS + ['workload_id']
        baseline_rows = [dict(row, workload_id=spec['workload_id']) for row in baseline_rows]
    for filename, fields, rows in [(stem + '_baselines.csv', baseline_fields, baseline_rows),
                                    (stem + '_tvm_diag.csv', TVM_FIELDS, tvm_rows)]:
        with (args.out / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    write(args.out / 'collection-audit.json', dict(schema=1, evidence='verified', protocol=PROTOCOL,
        workload_id=spec.get('workload_id', 'dnn1-historical'),
        collector_sha256=sha(__file__), cells=audit, rows=len(baseline_rows),
        passing_rows=sum(row['status'] == 'pass' for row in baseline_rows)))
    print(compact(dict(out=str(args.out), rows=len(baseline_rows),
        passing=sum(row['status'] == 'pass' for row in baseline_rows), executed_gpu=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--spec', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--refresh-prepared', action='store_true',
                   help='archive and refresh a queue that has never started')
    r = sub.add_parser('run')
    r.add_argument('--cell', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--kind', choices=('cell', 'export'), default='cell')
    c = sub.add_parser('collect')
    c.add_argument('--run-root', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        {'queue': queue, 'run': run, 'collect': collect, 'preflight': preflight}[args.action](args)
    except PreparationPending as error:
        print(compact(dict(status='preparation_pending', error=str(error), executed_gpu=False)), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
