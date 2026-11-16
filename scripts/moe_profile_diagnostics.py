"""Short, independently guarded diagnostics after the ordinary matrix completes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from moe_profile_matrix import CONTROL_PYTHON, LAUNCHER, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['timeline', 'routing', 'attribution', 'control'], required=True)
    parser.add_argument('--framework', choices=['all', 'vllm', 'sglang', 'trtllm'], default='all')
    parser.add_argument('--gpus', default='1,0')
    parser.add_argument('--wait-main', action='store_true')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--untimed', action='store_true', help='Source/route evidence only;no timing attribution from this run')
    parser.add_argument('--cpus')
    parser.add_argument('--phase', choices=['prefill', 'decode'])
    parser.add_argument('--graph-trace', choices=['node', 'graph'], default='node')
    parser.add_argument('--refresh', action='store_true', help='Retain old attempts and replace incomplete diagnostic evidence')
    args = parser.parse_args()
    if args.untimed and args.kind not in ('routing', 'attribution'):
        raise ValueError('ordinary/control/native timeline measurements require the timing guard')
    plan = json.loads((ROOT / 'configs/moe_profile/workload.json').read_text())
    output = ROOT / 'runs/moe_profile' / (args.kind + ('_graph' if args.graph_trace == 'graph' else ''))
    output.mkdir(exist_ok=True)
    frameworks = list(plan['frameworks']) if args.framework == 'all' else [args.framework]
    points = plan['profiles']['points']
    if args.batch:
        points = [point for point in points if point[0] == args.batch]
    if not points:
        raise ValueError('no representative point for requested batch')
    if args.wait_main:
        expected = [ROOT / 'runs/moe_profile/benchmark' / f'{framework}_tp1_b{batch}_i{length}_o{count}'
                    for framework in plan['frameworks'] for length, count in plan['lengths']
                    for batch in plan['batches']]
        deadline = time.monotonic() + 7200
        while (not (ROOT / 'runs/moe_profile/main_matrix_done.json').exists()
               or not all(list(folder.glob('attempt*/guard/guard_result.json')) for folder in expected)):
            if time.monotonic() > deadline:
                raise TimeoutError('ordinary matrix did not complete before diagnostics')
            time.sleep(10)
    for framework in frameworks:
        for batch, length, count in points:
            baseline = ROOT / 'runs/moe_profile/benchmark' / f'{framework}_tp1_b{batch}_i{length}_o{count}'
            accepted = [path for path in baseline.glob('attempt*') if (path / 'guard/guard_result.json').exists()
                        and json.loads((path / 'guard/guard_result.json').read_text())['code'] == 0]
            if not accepted:
                (output / f'{framework}_b{batch}_i{length}_unavailable.json').write_text(json.dumps(dict(
                    reason='No accepted ordinary baseline for this representative configuration',
                    baseline=str(baseline)), indent=2) + '\n')
                continue
            phases = ['prefill', 'decode'] if args.kind == 'timeline' else ['decode']
            if args.phase:
                phases = [args.phase]
            for phase in phases:
                folder = output / f'{framework}_b{batch}_i{length}_{phase}'
                folder.mkdir(exist_ok=True)
                if not args.refresh and any(json.loads(path.read_text()).get('code') == 0
                       and not (path.parents[1] / 'superseded_for_missing_kv_metadata.json').exists()
                       and ((path.parents[1] / 'export/gpu_activity.json').exists()
                            if args.kind in ('timeline', 'attribution') else True)
                       for path in folder.glob('attempt*/guard/guard_result.json')):
                    continue
                base_attempt = len(list(folder.glob('attempt*')))
                for retry, gpu in enumerate(args.gpus.split(',')):
                    attempt = folder / f'attempt{base_attempt+retry:02d}'
                    attempt.mkdir()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TILEMEGA_DEVICE_INDEX=gpu,
                               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
                               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_NODE='0',
                               TILEMEGA_TIMING='0' if args.untimed else '1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                    if args.kind in ('attribution', 'routing') and framework == 'sglang':
                        env['PYTHONPATH'] = str(ROOT / 'scripts/moe_profile_site') + ':' + str(ROOT / 'scripts') + ':' + env.get('PYTHONPATH', '')
                        env['MOE_PROFILE_NVTX_OUT' if args.kind == 'attribution' else 'MOE_PROFILE_ROUTE_META_OUT'] = str(attempt / 'worker/nvtx_sources')
                    mode = 'routing' if args.kind == 'routing' else 'benchmark' if args.kind == 'control' else 'profile'
                    worker = [plan['frameworks'][framework]['python'], '-u', LAUNCHER, '--entry',
                              str(ROOT / 'scripts/moe_profile_benchmark.py'), '--framework', framework,
                              '--model', '/dev/shm/tilemega-moe-profile-qwen3-ad44e777', '--inputs',
                              str(ROOT / 'configs/moe_profile/inputs.json.gz'), '--batch', str(batch),
                              '--input-tokens', str(length), '--output-tokens', str(count),
                              '--mode', mode, '--phase', phase, '--rounds', '3' if mode == 'benchmark' else '1',
                              '--out', str(attempt / 'worker')]
                    if args.kind == 'attribution':
                        worker.append('--attribution')
                    if args.kind == 'control':
                        worker.append('--no-cuda-graph')
                    if mode == 'profile':
                        worker = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                                  '--cuda-graph-trace=' + args.graph_trace, '--trace-fork-before-exec=true',
                                  '--flush-on-cudaprofilerstop=true',
                                  '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                                  '--force-overwrite=true', '-o', str(attempt / 'trace'), *worker]
                    command = ['taskset', '-c', args.cpus or ('32-47' if args.untimed else '16-31'), CONTROL_PYTHON, LAUNCHER, '--entry',
                               str(ROOT / 'scripts/moe_profile_guard.py'), '--policy',
                               str(ROOT / 'configs/moe_profile/guard_policy.json'), '--out',
                               str(attempt / 'guard'), '--needs-free-mib', '75000', '--timeout-s', '900', '--', *worker]
                    (attempt / 'command.json').write_text(json.dumps(dict(command=command,
                        environment={key: value for key, value in env.items() if key.startswith((
                            'CUDA_', 'TILEMEGA_', 'OMP_', 'MOE_PROFILE_', 'PYTHONPATH'))}), indent=2) + '\n')
                    with (attempt / 'worker.log').open('w') as log:
                        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                              stderr=subprocess.STDOUT).returncode
                    print(json.dumps(dict(kind=args.kind, framework=framework, batch=batch,
                                          length=length, phase=phase, exit_code=code, attempt=str(attempt))), flush=True)
                    if code == 0 and mode == 'profile':
                        with (attempt / 'export.log').open('w') as log:
                            subprocess.run(['taskset', '-c', '136-143', 'nsys', 'export', '--type', 'sqlite',
                                            '--force-overwrite=true', '-o', str(attempt / 'trace.sqlite'),
                                            str(attempt / 'trace.nsys-rep')], stdout=log, stderr=subprocess.STDOUT, check=True)
                        parse = [CONTROL_PYTHON, LAUNCHER, '--entry', str(ROOT / 'scripts/moe_profile_nsys.py'),
                                 '--sqlite', str(attempt / 'trace.sqlite'), '--out', str(attempt / 'export')]
                        if args.kind == 'attribution':
                            parse.extend(['--nvtx-proof', str(attempt / 'worker/engine.json')])
                        subprocess.run(['taskset', '-c', '136-143', *parse], check=True)
                    if code != 75:
                        break
                    time.sleep(5)
                time.sleep(5)


if __name__ == '__main__':
    main()
