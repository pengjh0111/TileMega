#!/usr/bin/env python3
"""Prepare and run the strict TP1 A100 baseline queue."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from tilemega.bench.gpu_guard import numa_node_of, single_card_mask, smi, verify_device
from tilemega.bench.scheduler import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/a100'
POLICY = ROOT / 'configs/bench/a100_guard_policy.json'
BATCHES = (1, 2, 4, 8, 16)
MODELS = {
    'qwen3_0_6b': ('sm90d1', 'qwen'),
    'qwen3_1_7b': ('sm90d1', 'qwen'),
    'llama3_2_1b': ('sm90d1', 'llama3'),
    'llama3_2_3b': ('sm90d1', 'llama3'),
    'llama3_1_8b': ('sm90d1', 'llama3'),
    'qwen3_30b_a3b': ('sm90d1', 'qwen'),
    'mistral_7b_v0_3': ('sm90r4', 'mistral'),
    'olmoe_1b_7b': ('sm90r4', 'olmo'),
}
MOE = {'qwen3_30b_a3b', 'olmoe_1b_7b'}
ENGINES = {'vllm': 'vllm', 'sglang': 'sglang',
           'trtllm': 'tensorrt-llm', 'torchcompile': 'tilemega'}
SERVING = ROOT / 'python/tilemega/serving'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suite_python(suite, envs=RUN/'envs'):
    path = envs / suite / 'bin/python'
    if not os.access(path, os.X_OK):
        raise ValueError(f'missing isolated {suite} environment: {path}')
    return path


def inputs(model):
    origin, family = MODELS[model]
    return (ROOT / 'configs/serving' / origin / 'inputs' / model,
            ROOT / 'configs/prompts' / f'{family}_isl64.json')


def device_environment(policy, model='A100'):
    index, = policy['allowed_gpu_indices']
    verify_device(policy, {'gpu_index': index})
    fields = 'name,pci.bus_id,memory.total,mig.mode.current,compute_cap,driver_version'
    raw = smi('-i', str(index), f'--query-gpu={fields}', '--format=csv,noheader,nounits')
    name, bus, memory, mig, capability, driver = [v.strip() for v in raw.split(',')]
    if model not in name or capability != '8.0' or mig != 'Disabled':
        raise ValueError(f'expected the mapped {model} with MIG disabled: {raw}')
    domain, address = bus.lower().split(':', 1)
    pci = f'{domain[-4:]}:{address}'
    node = numa_node_of(index)
    local = Path('/sys/bus/pci/devices') / pci / 'local_cpulist'
    cpu_ids = set()
    for span in local.read_text().strip().split(','):
        lo, _, hi = span.partition('-')
        cpu_ids.update(range(int(lo), int(hi or lo) + 1))
    allowed = sorted(cpu_ids & os.sched_getaffinity(0))[:4]
    if not allowed:
        raise ValueError('no permitted CPU cores local to the GPU')
    return dict(name=name, pci=pci, memory_mib=int(memory), mig=mig, capability=capability,
                driver=driver, index=index, uuid=policy['gpu_uuid'], numa_node=node,
                cpus=','.join(map(str, allowed)))


def inspect_inputs(inventory):
    manifest = {}
    for model in MODELS:
        config, prompt = inputs(model)
        desc = json.loads((inventory/f'{model}.json').read_text())
        if desc['config_sha256'] != sha(config/'config.json'):
            raise ValueError(f'stale checkpoint inventory: {model}')
        settings = json.loads((config/'config.json').read_text())
        ids = json.loads(prompt.read_text())
        if len(ids) < max(BATCHES) or any(len(row) != 64 or any(
                type(token) is not int or not 0 <= token < settings['vocab_size'] for token in row)
                for row in ids):
            raise ValueError(f'invalid frozen prompt for {model}')
        manifest[model] = dict(config_sha256=desc['config_sha256'], prompt_sha256=sha(prompt),
                               checkpoint_bytes=desc['total_bytes'])
    return manifest


def mpk_serving_path(model):
    name = 'a800_mpk_llama8.py' if model == 'llama3_1_8b' else 'a800_mpk_serving.py'
    return ROOT/'scripts'/name


def prepare(policy_path=POLICY, run=RUN, inventory=RUN/'inventory', envs=RUN/'envs',
            gpu_model='A100', timeout_s=14400, heartbeat_timeout_s=None,
            seed_suite='vllm', expected_checkpoint=None, engines=None, mpk_include_root=None):
    policy = json.loads(policy_path.read_text())
    dev = device_environment(policy, gpu_model)
    env = dict(TILEMEGA_DEVICE_INDEX=str(dev['index']),
               CUDA_VISIBLE_DEVICES=single_card_mask(policy, dev['index']), TILEMEGA_NUMA_NODE=dev['numa_node'],
               TILEMEGA_GPU_PID_MODE='strict', TORCH_CUDA_ARCH_LIST=dev['capability'],
               CUDA_HOME='/usr/local/cuda-12.8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(ROOT/'python'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2',
               TORCHINDUCTOR_COMPILE_THREADS='2', TOKENIZERS_PARALLELISM='false')
    if 'cuda_visible_devices' not in policy:
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device_job = f'{gpu_model.lower()}_device_identity'
    identity = [
        'taskset', '--cpu-list', dev['cpus'], suite_python('tilemega', envs), '-u',
        ROOT/'scripts/run_r16_with_device_identity.py', '--out', '{attempt_dir}/job',
        '--guard-policy', policy_path, '--expected-uuid', dev['uuid'], '--expected-pci', dev['pci'],
        '--numa-node', dev['numa_node'],
    ]
    checkpoint_hash_args = ['--expected-sha256', expected_checkpoint[1]] if expected_checkpoint else []
    jobs = [dict(name=device_job, gpu=True, timing=False, priority=0,
                 needs_free_mib=1024, timeout_s=120, cwd=str(ROOT), env=env,
                 command=[*identity, '--allow-no-measurements', '--', '/usr/bin/true'])]
    manifest = inspect_inputs(inventory)
    for model, description in manifest.items():
        if expected_checkpoint and model != expected_checkpoint[0]:
            continue
        config, prompt = inputs(model)
        settings = json.loads((config/'config.json').read_text())
        checkpoint_bytes = description['checkpoint_bytes']
        for engine, suite in (ENGINES if engines is None else engines).items():
            if seed_suite != 'vllm' and suite != seed_suite:
                continue
            harness = f'{engine}_baseline.py'
            if engine == 'torchcompile' and model in MOE:
                harness = 'torchcompile_moe_baseline.py'
            harness_path = mpk_serving_path(model) if engine == 'mpk' else SERVING/harness
            cache = run/'cache'/suite
            tmp = run/'tmp'/suite
            for folder in (cache, tmp):
                folder.mkdir(parents=True, exist_ok=True)
            job_env = dict(env, VIRTUAL_ENV=str(suite_python(suite, envs).parent.parent),
                           PATH=f'{suite_python(suite, envs).parent}:{env["CUDA_HOME"]}/bin:{os.defpath}',
                           HF_HOME=str(cache/'hf'), XDG_CACHE_HOME=str(cache),
                           TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                           TRITON_CACHE_DIR=str(cache/'triton'), CUDA_CACHE_PATH=str(cache/'cuda'),
                           TMPDIR=str(tmp))
            if engine == 'vllm':
                # ZMQ appends a UUID, so a cohort's long TMPDIR may exceed sun_path.
                job_env.update(VLLM_USE_V1='1', VLLM_CACHE_ROOT=str(cache/'vllm'),
                               VLLM_RPC_BASE_PATH='/tmp')
            if engine == 'mpk':
                job_env['MIRAGE_HOME'] = str(ROOT.parent/'software/mpk-63af900')
                if mpk_include_root is not None:
                    job_env['MIRAGE_ROOT'] = str(mpk_include_root)
            if engine == 'trtllm':
                job_env['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
            for batch in BATCHES:
                heads = settings.get('num_key_value_heads', settings['num_attention_heads'])
                head_dim = settings.get('head_dim') or settings['hidden_size']//settings['num_attention_heads']
                kv_bytes = 2 * 2 * settings['num_hidden_layers'] * heads * head_dim * batch * 1088
                needs = math.ceil((checkpoint_bytes + kv_bytes) / 2**20) + 4096
                if engine in ('vllm', 'sglang'):
                    needs = max(needs, math.ceil(0.85 * dev['memory_mib']))
                # Near capacity, let the engine report OOM under the full protocol.
                needs = min(needs, dev['memory_mib'] - policy['max_hidden_mib'])
                for round_index in range(3):
                    command = [
                        *identity, '--',
                        suite_python(seed_suite, envs), '-u', SERVING/'random_checkpoint.py',
                        '--config', config, '--out', '{attempt_dir}/checkpoint', '--seed', '0',
                        '--max-gib', math.ceil(checkpoint_bytes/2**30), *checkpoint_hash_args, '--',
                        suite_python(suite, envs), '-u', harness_path,
                        '--model', '{attempt_dir}/checkpoint', '--prompt-ids', prompt,
                        '--batch', batch, '--max-tokens', 1024, '--warmup', 1, '--repeats', 3,
                        '--round-index', round_index, '--tp', 1, '--load-format', 'auto',
                        '--out', '{attempt_dir}/job/measurements',
                    ]
                    jobs.append(dict(name=f'{model}_{engine}_b{batch:02d}_r{round_index}',
                                     model=model, engine=engine, batch=batch, round=round_index,
                                     gpu=True, timing=True, after=[device_job],
                                     priority=round_index + 10, timeout_s=timeout_s, needs_free_mib=needs,
                                     cwd=str(ROOT), env=job_env, command=command,
                                     harness_sha256=sha(harness_path)))
                    if heartbeat_timeout_s is not None:
                        jobs[-1]['heartbeat_timeout_s'] = heartbeat_timeout_s
    for item in jobs:
        item['command'] = list(map(str, item['command']))
    (run/'queue').mkdir(parents=True, exist_ok=True)
    write_json(run/'queue/queue_baselines.json', jobs)
    write_json(run/'baseline_inputs.json', dict(device=dev, models=manifest,
               source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               queue_sha256=sha(run/'queue/queue_baselines.json'), guard_policy_sha256=sha(policy_path)))
    print(f'Prepared {len(jobs) - 1} baseline jobs and one guarded device probe')
    return jobs


def run(args):
    from tilemega.bench.scheduler import main as schedule
    policy = json.loads(POLICY.read_text())
    dev = device_environment(policy)
    os.environ.update(TILEMEGA_DEVICE_INDEX=str(dev['index']), TILEMEGA_GPU_LOCK=policy['gpu_lock_path'])
    os.environ.pop('TILEMEGA_DEVICE_GROUP', None)
    options = ['--queue-dir', str(run/'queue'), '--policy', str(POLICY),
               '--out', str(RUN/'baseline'), '--deadline-hours', 'inf']
    if args.resume_timing:
        options.append('--resume-timing')
    schedule(options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest='action', required=True)
    action.add_parser('prepare')
    action.add_parser('run').add_argument('--resume-timing', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        run(args)
