"""Run independently guarded cells; resume only results accepted by the guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL_PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'
LAUNCHER = '/root/python/run_repo_script.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/moe_profile/benchmark')
    parser.add_argument('--framework', choices=['all', 'vllm', 'sglang', 'trtllm'], default='all')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--gap-seconds', type=float, default=20)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'configs/moe_profile/workload.json').read_text())
    gpu_indices = [int(value) for value in args.gpus.split(',')]
    if any(str(index) not in json.loads((ROOT / 'configs/moe_profile/devices.json').read_text())['devices']
           for index in gpu_indices):
        raise ValueError('GPU ordinal outside the current physical device manifest')
    frameworks = list(plan['frameworks']) if args.framework == 'all' else [args.framework]
    completed = 0
    for length, output in plan['lengths']:
        for batch in plan['batches']:
            for framework in frameworks:
                name = f'{framework}_tp1_b{batch}_i{length}_o{output}'
                folder = args.out / name
                accepted = []
                if folder.exists():
                    for attempt in sorted(folder.glob('attempt*')):
                        guard = attempt / 'guard/guard_result.json'
                        result = attempt / 'worker/result.json'
                        if guard.exists() and result.exists():
                            if json.loads(guard.read_text())['code'] == 0 and json.loads(result.read_text())['status'] == 'ok':
                                accepted.append(attempt)
                if accepted:
                    print(json.dumps(dict(cell=name, status='already_accepted', path=str(accepted[-1]))), flush=True)
                    continue
                if args.limit is not None and completed >= args.limit:
                    return
                folder.mkdir(exist_ok=True)
                base_attempt = len(list(folder.glob('attempt*')))
                # Foreign ownership only defers our own job. No other process is signalled.
                for retry in range(len(gpu_indices) * 2):
                    gpu = gpu_indices[retry % len(gpu_indices)]
                    attempt = folder / f'attempt{base_attempt+retry:02d}'
                    attempt.mkdir()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               TILEMEGA_DEVICE_INDEX=str(gpu), TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
                               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_NODE='0',
                               TILEMEGA_TIMING='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                    command = ['taskset', '-c', '16-31', CONTROL_PYTHON, LAUNCHER, '--entry',
                               str(ROOT / 'scripts/moe_profile_guard.py'), '--policy',
                               str(ROOT / 'configs/moe_profile/guard_policy.json'), '--out',
                               str(attempt / 'guard'), '--needs-free-mib', '75000', '--timeout-s', '1800', '--',
                               plan['frameworks'][framework]['python'], '-u', LAUNCHER, '--entry',
                               str(ROOT / 'scripts/moe_profile_benchmark.py'), '--framework', framework,
                               '--model', str(args.model), '--inputs', str(args.inputs),
                               '--batch', str(batch), '--input-tokens', str(length), '--output-tokens',
                               str(output), '--out', str(attempt / 'worker')]
                    (attempt / 'command.json').write_text(json.dumps(dict(command=command,
                        environment={key: value for key, value in env.items() if key.startswith((
                            'CUDA_', 'TILEMEGA_', 'OMP_', 'TOKENIZERS_', 'HF_HUB_', 'TRANSFORMERS_'))}), indent=2) + '\n')
                    with (attempt / 'worker.log').open('w') as log:
                        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                              stderr=subprocess.STDOUT).returncode
                    print(json.dumps(dict(cell=name, gpu=gpu, attempt=str(attempt), exit_code=code)), flush=True)
                    if code != 75:
                        break
                    time.sleep(args.gap_seconds)
                completed += 1
                time.sleep(args.gap_seconds)
    if args.framework == 'all' and args.limit is None:
        (ROOT / 'runs/moe_profile/main_matrix_done.json').write_text(json.dumps(dict(
            time=time.time(), scope='All configured ordinary cells attempted;inspect per-cell guard/result for failures.'),
            indent=2) + '\n')


if __name__ == '__main__':
    main()
