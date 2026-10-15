#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare isolated full-model DNN candidate builds and guarded numerical/timing jobs."""
from __future__ import annotations
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODELS = ('mbv1', 'mbv2', 'bert', 'resnet18', 'nafnet')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def definitions(model):
    if model == 'mbv1':
        return [('h7c-mma16', ['--dwpw-fuse', 'auto', '--dwpw-impl', 'mma16'], {'dwpw_impl':'mma16'}),
                ('h7c-wgmma', ['--dwpw-fuse', 'auto', '--dwpw-impl', 'wgmma'],
                 {'dwpw_impl': 'wgmma', 'requires': ['HGMMA']})]
    if model == 'mbv2':
        return [('h7b-cpasync', ['--dwpw-fuse', '0', '--depthwise-impl', 'cpasync'], {'depthwise_impl':'cpasync'}),
                ('h7b-tma', ['--dwpw-fuse', '0', '--depthwise-impl', 'tma'],
                 {'depthwise_impl': 'tma', 'requires': ['UTMALDG']})]
    if model == 'bert':
        return [('h8-mma16', ['--encoder-attention-impl', 'mma16'], {'encoder_attention_impl':'mma16'}),
                ('h8-wgmma', ['--encoder-attention-impl', 'wgmma'],
                 {'encoder_attention_impl': 'wgmma', 'requires': ['HGMMA', 'UTMALDG']})]
    if model in ('resnet18', 'nafnet'):
        return [('h7a-tma-im2col', ['--dwpw-fuse', 'auto', '--forward-gemm-impl', 'wgmma', '--conv-impl', 'tma_im2col'],
                 {'conv_impl':'tma_im2col', 'forward_gemm_impl':'wgmma', 'requires':['HGMMA','UTMALDG']})]
    raise ValueError('unknown candidate model')


def verified_config(path):
    config = read(path)
    if sha(config['cli']) != config['cli_sha256']:
        raise ValueError('completed immutable CLI changed')
    if sha(config['target']) != config['target_sha256']:
        raise ValueError('frozen target changed')
    if config.get('single_point_case_file') and sha(config['single_point_case_file']) != config['single_point_case_sha256']:
        raise ValueError('predeclared evaluation case changed')
    for filename, expected in config['export_files'].items():
        if sha(Path(config['export']) / filename) != expected:
            raise ValueError('shared exported checkpoint/input changed: ' + filename)
    return config


def child(command, cwd, env, log):
    start = time.time()
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        code = process.wait()
    return dict(returncode=code, child_pid=process.pid, supervisor_pid=os.getpid(),
                elapsed_seconds=time.time() - start, command=command, log=str(log))


def cpu_env(config):
    env = dict(os.environ, PYTHONPATH=str(Path(config['source']) / 'python'),
               CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('TILEMEGA_DEVICE_INDEX', None)
    return env


def build(path):
    config = verified_config(path)
    folder = Path(config['directory'])
    env = cpu_env(config)
    cli = config['cli']
    flags = config['flags']
    cg = folder / 'selected-seed.mlir'
    commands = [[cli, 'compile', config['bridge'], str(folder / 'seed.cu')] + flags +
                ['--dump-cg', str(cg)],
                [cli, 'compile', config['bridge'] if config.get('direct_single_point') else str(cg), str(folder / 'plan.so')] + flags]
    result = dict(candidate_id=config['candidate_id'], gpu_executed=False,
                  source_head=config['source_head'], cli_sha256=config['cli_sha256'], steps=[])
    try:
        for index, command in enumerate(commands):
            row = child(command, config['source'], env, folder / f'cpu-build-{index}.log')
            result['steps'].append(row)
            if row['returncode']:
                raise ValueError('official CPU candidate build failed')
        manifest = read(folder / 'plan.so.plan.json')
        identity = read(folder / 'plan.so.identity.json')
        if identity['binary_sha256'] != sha(folder / 'plan.so'):
            raise ValueError('binary does not match official identity')
        if identity.get('source', {}).get('head') != config['source_head']:
            raise ValueError('official identity belongs to another source snapshot')
        for field, expected in config['expected'].items():
            if field != 'requires' and manifest.get(field) != expected:
                raise ValueError(f'requested {field} did not bind: {manifest.get(field)}')
        if identity['implementations']['execution_arch_id'] != 900:
            raise ValueError('candidate is not bound to native Hopper')
        build_log = (folder / 'cpu-build-1.log').read_text()
        # nvcc stderr is retained by the official compiler as a separate receipt.
        for log in folder.glob('plan.so*stderr*'):
            build_log += log.read_text(errors='replace')
        if '(C7510)' in build_log or 'wgmma.mma_async instructions are serialized' in build_log:
            raise ValueError('full model has serialized WGMMA; numerical/timing jobs refused')
        counts = dict(HGMMA=0, UTMALDG=0, UBLKCP=0)
        with gzip.open(folder / 'sass.txt.gz', 'wt', compresslevel=1) as stream:
            proc = subprocess.Popen(['/usr/local/cuda/bin/cuobjdump', '--dump-sass',
                                     str(folder / 'plan.so')], stdout=subprocess.PIPE,
                                    text=True, env=env)
            for line in proc.stdout:
                stream.write(line)
                for instruction in counts:
                    counts[instruction] += int(instruction in line)
            if proc.wait():
                raise ValueError('SASS disassembly failed')
        if any(not counts[required] for required in config['expected'].get('requires', [])):
            raise ValueError('requested instructions absent from full model SASS')
        result.update(passed=True, artifact_id=identity['artifact_id'], sass_counts=counts,
                      library_sha256=sha(folder / 'plan.so'), gemms=manifest['gemms'],
                      identity_path=str(folder / 'plan.so.identity.json'))
    except Exception as error:
        result.update(passed=False, error=str(error))
        write(folder / 'cpu-build-result.json', result)
        raise
    write(folder / 'cpu-build-result.json', result)
    print(json.dumps(result), flush=True)


def validate_built(path):
    config = verified_config(path)
    folder = Path(config['directory'])
    result = read(folder / 'cpu-build-result.json')
    identity = read(folder / 'plan.so.identity.json')
    if not result.get('passed') or result.get('library_sha256') != sha(folder / 'plan.so'):
        raise ValueError('completed CPU build is absent or changed')
    if result['artifact_id'] != identity['artifact_id'] or identity['source']['head'] != config['source_head']:
        raise ValueError('completed CPU artifact identity differs')


def require_lease(config):
    device = str(config['gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != device or os.environ.get('TILEMEGA_DEVICE_INDEX') != device:
        raise ValueError('explicit matching physical GPU variables required')
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('external GPU guard and lock required')


def check(path, fresh, attempt):
    config = verified_config(path)
    require_lease(config)
    if not read(Path(config['directory']) / 'cpu-build-result.json').get('passed'):
        raise ValueError('candidate full-model build is not accepted')
    attempt = Path(attempt).absolute()
    attempt.mkdir(parents=True, exist_ok=True)
    gate = attempt / 'full-model-gate.json'
    command = [config['python'], '-m', 'tilemega.dnn.check_generated', '--library',
               str(Path(config['directory']) / 'plan.so'), '--bridge', config['bridge'],
               '--export', config['export'], '--batch', '1', '--input-tensors',
               str(Path(config['export']) / 'inputs_b1.safetensors'), '--out', str(gate)]
    status = child(command, (config.get('run_root') or str(Path(config['source']).parents[3])), dict(os.environ), attempt / 'check.log')
    status.update(fresh_index=fresh, candidate_id=config['candidate_id'], gate=str(gate),
                  candidate_receipt_sha256=sha(path))
    write(attempt / 'check-process.json', status)
    if status['returncode'] or not gate.is_file() or not read(gate).get('passed'):
        raise ValueError('full-model numerical gate failed')
    write(Path(config['directory']) / f'fresh-{fresh}.json', status)


def aggregate(path):
    config = verified_config(path)
    directory = Path(config['directory'])
    rows = [read(directory / f'fresh-{index}.json') for index in range(5)]
    if len({row['child_pid'] for row in rows}) != 5:
        raise ValueError('correctness requires five distinct fresh child processes')
    gates = [read(row['gate']) for row in rows]
    identity = read(directory / 'plan.so.identity.json')
    for gate in gates:
        if not gate.get('passed') or gate['artifact_id'] != identity['artifact_id']:
            raise ValueError('numerical receipts differ from the candidate binary')
        if gate['input_tensors_sha256'] != config['export_files']['inputs_b1.safetensors']:
            raise ValueError('numerical receipt used different full-model inputs')
        if gate['archive_sha256'] != config['export_files']['exported_program.pt2']:
            raise ValueError('numerical receipt used another checkpoint')
        modes = {row['mode'] for row in gate['cases']}
        if not {1, 2} <= modes:
            raise ValueError('full-model same-binary L1/L2 gate is missing')
    result = dict(gates[0], fresh_processes=5, fresh_passes=5,
                  fresh_child_pids=[row['child_pid'] for row in rows],
                  fresh_receipts=[row['gate'] for row in rows],
                  candidate_id=config['candidate_id'], candidate_receipt_sha256=sha(path))
    write(directory / 'gate-5fresh.json', result)


def measure(path, round_index, attempt):
    config = verified_config(path)
    require_lease(config)
    attempt = Path(attempt).absolute()
    attempt.mkdir(parents=True, exist_ok=True)
    command = [config['python'], '-m', 'tilemega.dnn.benchmark', '--library',
               str(Path(config['directory']) / 'plan.so'), '--bridge', config['bridge'],
               '--export', config['export'], '--inputs',
               str(Path(config['export']) / 'inputs_b1.safetensors'), '--gate',
               str(Path(config['directory']) / 'gate-5fresh.json'), '--executor', 'L2',
               '--round-index', str(round_index), '--warmup', '50', '--iterations', '200',
               '--out', str(attempt / 'measurements')]
    status = child(command, (config.get('run_root') or str(Path(config['source']).parents[3])), dict(os.environ), attempt / 'measure.log')
    write(attempt / 'measure-process.json', status)
    if status['returncode']:
        raise ValueError('full-model end-to-end measurement child failed')
    target = attempt / 'measurements/measurements.json'
    value = read(target)
    # Annotate successful raw measurements; never alter elapsed times or precision.
    metadata = dict(candidate_id=config['candidate_id'], candidate_receipt_sha256=sha(path))
    value.update(metadata)
    value['configuration'] = {**value.get('configuration', {}), **metadata}
    write(target, value)


def prepare(args):
    root = args.root.absolute()
    receipt = read(args.receipt)
    if not receipt.get('build_completed') or not receipt.get('snapshot_source_unchanged'):
        raise ValueError('requires completed immutable CLI build receipt')
    if sha(receipt['cli']) != receipt['cli_sha256']:
        raise ValueError('CLI differs from snapshot receipt')
    out = args.out.absolute()
    if out.exists():
        raise FileExistsError('new output required; previous candidate receipts retained')
    out.mkdir(parents=True)
    driver = out / 'driver.py'
    shutil.copy2(__file__, driver)
    target = out / 'target.json'
    shutil.copy2(args.target, target)
    python = str(args.python.absolute())
    jobs, configurations, rejected = [], [], []
    node = 0 if args.gpu < 4 else 1
    for model in args.models:
        export = root / 'runs/r16/dnn/export' / model
        needed = ('bridge_b1.json', 'manifest.json', 'inputs_b1.safetensors', 'exported_program.pt2')
        if any(not (export / name).is_file() for name in needed):
            rejected.append(dict(model=model, reason='full-model export/checkpoint/input missing'))
            continue
        for arm, extra, expected in definitions(model):
            directory = out / model / arm
            directory.mkdir(parents=True)
            bridge = directory / 'export.json'
            shutil.copy2(export / 'bridge_b1.json', bridge)
            flags = ['--emit', 'serving', '--serving', 'forward', '--frontend', 'dnn',
                     '--batch', '1', '--runtime-target', str(target), '--pg', 'l2',
                     '--resource-probes', '0', '--seed-arch', 'native',
                     '--nonpaged-weight-layout', 'tiled', '--deferred-ln', 'auto',
                     '--reuse', 'l2', '--global-la', 'auto'] + extra
            direct_single_point = model in ('resnet18', 'nafnet')
            if direct_single_point:
                if args.class_cases_helper is None:
                    rejected.append(dict(model=model, reason='official semantic-class cases helper required for bounded H7a solve'))
                    continue
                cases = directory / 'single-point-cases.json'
                helper = [str(args.class_cases_helper.absolute()), str(bridge), str(target), str(cases)]
                with (directory / 'classes.log').open('w') as stream:
                    result = subprocess.run(helper, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    rejected.append(dict(model=model, reason='official class-case preparation failed'))
                    continue
                position = flags.index('--seed-arch')
                del flags[position:position+2]
                flags += ['--solve', str(target), '--selection', 'predicted', '--evaluate-configs', str(cases), '--search-passes', '1']
            config = dict(candidate_id=f'{receipt["source_dir"].split("/")[-1]}:{model}:{arm}:B1',
                          source=receipt['source_dir'], source_head=receipt['snapshot_head'], run_root=str(root),
                          cli=receipt['cli'], cli_sha256=receipt['cli_sha256'],
                          directory=str(directory), export=str(export), bridge=str(bridge),
                          python=python, gpu=args.gpu, target=str(target), target_sha256=sha(target),
                          flags=flags, expected=expected, direct_single_point=direct_single_point,
                          single_point_case_file=str(cases) if direct_single_point else None,
                          single_point_case_sha256=sha(cases) if direct_single_point else None,
                          class_helper_sha256=sha(args.class_cases_helper) if direct_single_point else None, export_files={n: sha(export / n) for n in needed},
                          checkpoint_scope='fixed random exported checkpoint and fixed full-model inputs; not pretrained dataset G-DNN',
                          search_scope='explicit native seed; same structure within each A/B pair; not RF2 final defaults')
            path = directory / 'candidate.json'
            write(path, config)
            configurations.append(str(path))
            stem = 'dnn-candidate-' + out.name + '-' + model + '-' + arm
            prefix = [python, str(driver)]
            env = {'PYTHONPATH': str(Path(receipt['source_dir']) / 'python'),
                   'CUDA_VISIBLE_DEVICES': str(args.gpu), 'TILEMEGA_DEVICE_INDEX': str(args.gpu),
                   'TILEMEGA_NUMA_NODE': str(node), 'TILEMEGA_GPU_PID_MODE': args.pid_mode,
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '8'}
            common = dict(cwd=str(root), env=env, timeout_s=900,
                          timing=False, gpu=True, needs_free_mib=4096)
            jobs.append(dict(common, name=stem+'-build', gpu=False, timeout_s=3600,
                             priority=6, command=prefix+['build', '--candidate', str(path)]))
            check_names = []
            for fresh in range(5):
                name = stem+f'-check-f{fresh}'
                check_names.append(name)
                jobs.append(dict(common, name=name, priority=7, after=[stem+'-build'],
                                 command=['numactl', f'--cpunodebind={node}']+prefix+
                                 ['check', '--candidate', str(path), '--fresh', str(fresh),
                                  '--attempt', '{attempt_dir}']))
            jobs.append(dict(common, name=stem+'-gate', gpu=False, priority=7,
                             after=check_names, command=prefix+['aggregate', '--candidate', str(path)]))
            for round_index in range(3):
                jobs.append(dict(common, name=stem+f'-measure-r{round_index}', timing=True,
                                 timeout_s=1800, priority=8+round_index*.01, after=[stem+'-gate'],
                                 command=['numactl', f'--cpunodebind={node}']+prefix+
                                 ['measure', '--candidate', str(path), '--round-index', str(round_index),
                                  '--attempt', '{attempt_dir}']))
    write(out / 'queue-preview.json', jobs)
    write(out / 'preparation.json', dict(source_receipt=str(args.receipt.absolute()),
          driver_sha256=sha(driver), gpu_execution_performed=False, configurations=configurations,
          rejected=rejected, jobs=len(jobs), h7a='one explicit official evaluation case, not coordinate descent or RF2 final selection'))
    print(json.dumps({'queue': str(out / 'queue-preview.json'), 'jobs': len(jobs), 'rejected': rejected}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--root', type=Path, default=Path(__file__).absolute().parents[1])
    command.add_argument('--receipt', type=Path, required=True)
    command.add_argument('--out', type=Path, required=True)
    command.add_argument('--target', type=Path, required=True)
    command.add_argument('--python', type=Path, required=True)
    command.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS[:3]))
    command.add_argument('--class-cases-helper', type=Path)
    command.add_argument('--gpu', type=int, default=7)
    command.add_argument('--pid-mode', choices=('strict', 'namespace_limited'), default='strict')
    for name in ('build', 'validate-built', 'check', 'aggregate', 'measure'):
        command = sub.add_parser(name)
        command.add_argument('--candidate', type=Path, required=True)
        if name in ('check', 'measure'):
            command.add_argument('--attempt', type=Path, required=True)
        if name == 'check':
            command.add_argument('--fresh', type=int, required=True)
        if name == 'measure':
            command.add_argument('--round-index', type=int, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args)
    elif args.operation == 'build':
        build(args.candidate)
    elif args.operation == 'validate-built':
        validate_built(args.candidate)
    elif args.operation == 'check':
        check(args.candidate, args.fresh, args.attempt)
    elif args.operation == 'aggregate':
        aggregate(args.candidate)
    else:
        measure(args.candidate, args.round_index, args.attempt)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f'DNN candidate rejected: {error}', file=sys.stderr, flush=True)
        raise SystemExit(3)
