#!/usr/bin/env python3
"""Reproduce one baseline cell or one frozen native candidate under the guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_r16_baselines import MODELS, SYSTEMS, step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--system', choices=SYSTEMS)
    choice.add_argument('--native-candidate', type=Path)
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), default=1)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--numa', type=int, required=True)
    p.add_argument('--pid-mode', choices=('strict', 'namespace_limited'), default='strict')
    p.add_argument('--cpu-bind-only', action='store_true')
    p.add_argument('--env-root', type=Path, default=Path('/workspace/llm-stack/envs'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', action='store_true', help='run the generated guarded queue')
    a = p.parse_args()
    if a.system and not a.model:
        p.error('--system requires --model')
    if a.gpu < 0 or a.numa < 0:
        p.error('GPU and NUMA indices must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    out = a.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    queue = out/'queue'
    queue.mkdir()
    jobs = []
    if a.system:
        for rnd in range(3):
            j = step(root, a.env_root, a.model, a.system, a.gpu, a.numa, rnd,
                     phase='m1_reproduce', batch=str(a.batch), pid_mode=a.pid_mode,
                     cpu_bind_only=a.cpu_bind_only)
            if a.system == 'trtllm':
                j['env']['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
                j['protocol']['executor_deviation'] = 'official TP1 single-process mode'
            j['priority'] = rnd
            jobs.append(j)
    else:
        original = a.native_candidate.absolute()
        candidate = json.loads(original.read_text())
        driver = Path(candidate['script'])
        if hashlib.sha256(driver.read_bytes()).hexdigest() != candidate['script_sha256']:
            raise ValueError('frozen native driver SHA differs')
        candidate.update(gpu=a.gpu, cpu_bind_only=a.cpu_bind_only,
                         file=str(out/'candidate.json'))
        candidate['env'].update(CUDA_VISIBLE_DEVICES=str(a.gpu),
            TILEMEGA_DEVICE_INDEX=str(a.gpu), TILEMEGA_NUMA_NODE=str(a.numa),
            TILEMEGA_GPU_PID_MODE=a.pid_mode,
            TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{a.gpu}.lock')
        Path(candidate['file']).write_text(json.dumps(candidate, indent=2)+'\n')
        # Native execution independently verifies both complete artifact identities,
        # actual HGMMA and runtime executor usage before accepting a measurement.
        for index, (stage, rnd) in enumerate([('smoke', 0), ('final', 0), ('final', 1), ('final', 2)]):
            binding = ['numactl', '--cpunodebind='+str(a.numa)]
            if not a.cpu_bind_only:
                binding.append('--membind='+str(a.numa))
            jobs.append(dict(name='native_reproduce_'+stage+'_'+str(rnd),
                command=binding+[candidate['python'], str(driver), 'execute',
                    '--candidate', candidate['file'], '--stage', stage,
                    '--round-index', str(rnd), '--out', '{attempt_dir}/measurements'],
                cwd=str(root), env=candidate['env'], gpu=True, timing=stage != 'smoke',
                timeout_s=3600, needs_free_mib=candidate['needs_free_mib'], priority=index,
                after=[] if stage == 'smoke' else ['native_reproduce_smoke_0']))
    (queue/'queue_cell.json').write_text(json.dumps(jobs, indent=2)+'\n')
    command = [sys.executable, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(queue),
               '--out', str(out/'scheduler'), '--policy', str(root/'configs/bench/guard_policy.json')]
    env = dict(os.environ, PYTHONPATH=str(root/'python'), CUDA_VISIBLE_DEVICES=str(a.gpu),
        TILEMEGA_DEVICE_INDEX=str(a.gpu), TILEMEGA_NUMA_NODE=str(a.numa),
        TILEMEGA_GPU_PID_MODE=a.pid_mode, TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{a.gpu}.lock')
    receipt = dict(evidence='reproduction queue prepared', command=command,
        env={k:env[k] for k in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'TILEMEGA_DEVICE_INDEX',
             'TILEMEGA_NUMA_NODE', 'TILEMEGA_GPU_PID_MODE', 'TILEMEGA_GPU_LOCK')},
        protocol=dict(isl=64, osl=1024, warmup=1, repeats=3, fresh_processes=3),
        ownership='strict' if a.pid_mode == 'strict' else 'provisional_container_pid_namespace',
        checkpoint='same frozen random checkpoint for native; backend dummy weights for external baseline',
        queue_sha256=hashlib.sha256((queue/'queue_cell.json').read_bytes()).hexdigest())
    (out/'reproduction.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    if a.run:
        raise SystemExit(subprocess.run(command, cwd=root, env=env).returncode)


if __name__ == '__main__':
    main()
