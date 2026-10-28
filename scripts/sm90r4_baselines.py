#!/usr/bin/env python3
"""SM90R4 new baselines (prompt §2.6): R16 scripts on the shared random checkpoint.

One queue job is one fresh process over B1-16 (warmup 1, N1 and N1024 x 3);
rounds 0-2 are the three fresh processes. Every process rebuilds the
name-seeded memfd checkpoint, so all engines and TileMega read identical bytes
(the checkpoint_sha256 each process records is the identity check).
"""
import argparse
import json
from pathlib import Path

from sm90d1_common import ROOT, PYTHON, read, write

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
PCI = {0: '0f', 1: '10', 2: '41', 3: '44'}
ENVS = {'vllm': '/workspace/llm-stack/envs/vllm/bin/python',
        'sglang': '/workspace/llm-stack/envs/sglang/bin/python',
        'trtllm': '/workspace/llm-stack/envs/trtllm/bin/python',
        'torchcompile': '/workspace/llm-stack/envs/vllm/bin/python'}
MODELS = {'mistral_7b_v0_3': 'configs/prompts/mistral_isl64.json',
          'olmoe_1b_7b': 'configs/prompts/olmo_isl64.json'}


def job(model, engine, round_index, gpu, concurrency):
    uuid = read(POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
    inputs = ROOT / 'configs/serving/sm90r4/inputs' / model
    baseline = [ENVS[engine], '-u', str(ROOT / f'python/tilemega/serving/{engine}_baseline.py'),
                '--model', '{attempt_dir}/checkpoint', '--prompt-ids', str(ROOT / MODELS[model]),
                '--batch', 'all', '--max-tokens', '1024', '--warmup', '1', '--repeats', '3',
                '--round-index', str(round_index), '--load-format', 'auto',
                '--policy', str(ROOT / 'configs/bench/measurement_policy.json'),
                '--out', '{attempt_dir}/job/measurements']
    checkpoint = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'), '--config', str(inputs),
                  '--out', '{attempt_dir}/checkpoint', '--seed', '0', '--max-gib', '16', '--', *baseline]
    command = ['numactl', '--cpunodebind=0', PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
               '--guard-policy', str(POLICY), '--out', '{attempt_dir}/job', '--expected-uuid', uuid,
               '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', '0', '--', *checkpoint]
    env = dict(TILEMEGA_DEVICE_INDEX=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_GPU_PID_MODE='namespace_limited',
               CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='9.0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', VLLM_USE_V1='1')
    if engine == 'trtllm':
        env['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
    if concurrency == 'trial':
        env.update(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', TILEMEGA_NUMA_CONCURRENCY_VERIFIED='0')
    return dict(name=f'r4base_{model}_{engine}_r{round_index}_g{gpu}', gpu=True, timing=True,
                timeout_s=14400, needs_free_mib=74000, cwd=str(ROOT), env=env, command=command,
                priority=8.0, after=[],
                protocol=dict(isl=64, osl=1024, warmup=1, repeats=3, fresh_process_round=round_index,
                              precision='BF16 weights/activations/KV', random_weights=True,
                              checkpoint='tilemega-r16-name-seeded-uniform-v1 seed 0, rebuilt per process',
                              concurrency=concurrency))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--engines', nargs='+', default=list(ENVS))
    p.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--concurrency', choices=('serial', 'trial'), default='trial')
    p.add_argument('--queue-root', type=Path, required=True)
    a = p.parse_args()
    queues = {gpu: [] for gpu in a.gpus}
    cells = [(m, e, r) for r in a.rounds for m in a.models for e in a.engines]
    for index, (model, engine, round_index) in enumerate(cells):
        gpu = a.gpus[index % len(a.gpus)]
        queues[gpu].append(job(model, engine, round_index, gpu, a.concurrency))
    for gpu, jobs in queues.items():
        folder = a.queue_root / f'gpu{gpu}' / 'queue'
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / 'queue_baselines.json', jobs)
    print(json.dumps({gpu: len(jobs) for gpu, jobs in queues.items()}))


if __name__ == '__main__':
    main()
