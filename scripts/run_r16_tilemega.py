#!/usr/bin/env python3
"""Prepare independent random-weight model execution and seed timing queues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_r16_baselines import MODELS


def prepare(root, compiler, source, python, gpu, node, models, batches, label):
    root, compiler, source, python = map(lambda p: Path(p).resolve(),
                                       (root, compiler, source, python))
    if not compiler.is_file() or not python.is_file():
        raise FileNotFoundError('compiler and environment interpreter must exist')
    target = source/'configs/targets/sm_90_h800.json'
    if not target.is_file():
        raise FileNotFoundError(target)
    env = {'TILEMEGA_DEVICE_INDEX': str(gpu), 'CUDA_VISIBLE_DEVICES': str(gpu),
           'TILEMEGA_NUMA_NODE': str(node), 'TILEMEGA_GPU_PID_MODE': 'namespace_limited',
           'PYTHONPATH': str(root/'python'), 'OMP_NUM_THREADS': '8',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = []
    def job(name, command, priority, after=(), timing=False, timeout=1800):
        return dict(name=name, command=['numactl', f'--cpunodebind={node}', *command],
            gpu=True, timing=timing, priority=priority, after=list(after), timeout_s=timeout,
            needs_free_mib=74000, cwd=str(root), env=dict(env))
    for model_index, model in enumerate(models):
        config = root/'runs/r16/models'/model
        prompts = root/'configs/prompts'/('qwen_isl64.json' if model.startswith('qwen')
                                        else 'llama3_isl64.json')
        if not (config/'config.json').is_file():
            raise FileNotFoundError(config)
        for batch in batches:
            pair = root/f'runs/r16/plans/{label}/{model}/B{batch}'
            pair.mkdir(parents=True, exist_ok=True)
            builds = []
            for phase in ('prefill', 'decode'):
                bridge = root/f'runs/r16/exports/{model}/{phase}/export.json'
                if not bridge.is_file():
                    raise FileNotFoundError(bridge)
                name = f'tmseed_{label}_{model}_B{batch}_{phase}_build_gpu{gpu}'
                command = [str(compiler), 'compile', str(bridge), str(pair/f'{phase}.so'),
                    '--serving', phase, '--emit', 'serving', '--batch', str(batch),
                    '--runtime-target', str(target), '--pg', 'off',
                    '--nonpaged-weight-layout', 'tiled', '--deferred-norm', '0',
                    '--resource-probes', '0', '--capacity', '1088', '--prompt-len', '64']
                result.append(job(name, command, 50+model_index*10))
                builds.append(name)
            common = [str(python), '-m', 'tilemega.serving.random_checkpoint',
                '--config', str(config), '--out', '{attempt_dir}/checkpoint',
                '--seed', '0', '--max-gib', '70', '--', str(python),
                '-m', 'tilemega.serving.r16_run', '--prefill', str(pair/'prefill.so'),
                '--decode', str(pair/'decode.so'), '--prompt-ids', str(prompts),
                '--batch', str(batch), '--system-label', 'TM-seed',
                '--policy', str(root/'configs/bench/measurement_policy.json'),
                '--out', '{attempt_dir}/measurements']
            smoke = f'tmseed_{label}_{model}_B{batch}_smoke_gpu{gpu}'
            result.append(job(smoke, common+['--max-tokens', '16', '--warmup', '1',
                '--repeats', '1', '--check-executors', '--hf-check'],
                51+model_index*10, builds, timeout=3600))
            for round_index in range(3):
                name = f'm1_tilemega_seed_{model}_B{batch}_r{round_index}_gpu{gpu}'
                command = common+['--max-tokens', '1024', '--warmup', '1', '--repeats', '3',
                                  '--round-index', str(round_index)]
                if round_index == 0:
                    command += ['--check-executors', '--hf-check']
                result.append(job(name, command, 52+round_index*100+model_index*10,
                                  [smoke], timing=True, timeout=7200))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path('/workspace/llm-stack/envs/vllm/bin/python'))
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--numa', type=int, default=1)
    parser.add_argument('--models', choices=MODELS, nargs='+', default=MODELS)
    parser.add_argument('--batches', type=int, choices=(1, 2, 4, 8, 16), nargs='+', default=[1, 16])
    parser.add_argument('--label', default='seed')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    jobs = prepare(args.root, args.compiler, args.source, args.python, args.gpu, args.numa,
                   args.models, args.batches, args.label)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'queue_tilemega_seed.json').write_text(json.dumps(jobs, indent=2)+'\n')
    (args.out/'plan.json').write_text(json.dumps(dict(evidence='queue_generation_only',
        jobs=len(jobs), models=list(args.models), batches=args.batches,
        implementation='unsolved seed; not RF-2 final default',
        checkpoints='shared deterministic memfd weights; no pretrained accuracy claim',
        ownership='provisional_container_pid_namespace', numa_binding='CPU only; memory binding unavailable',
        enqueued=False), indent=2)+'\n')
    print(json.dumps({'jobs': len(jobs), 'enqueued': False, 'out': str(args.out.resolve())}))


if __name__ == '__main__':
    main()
