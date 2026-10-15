#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build explicit full-model forward seeds, then certify and measure each arm."""
import argparse
import ctypes
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import check_r16_forward_seed as contract
import run_r16_h7a_recertify as probe

recovery = probe.recovery
base = recovery.base
SHAPE = dict(tile_m=64, tile_n=128, tile_k=64, stages=2, split_k=1)
KINDS = ('tilemega_l1_kernel', 'tilemega_l2_kernel',
         'tilemega_l1_loop_kernel', 'tilemega_stage_kernel')


def completed_receipt(path):
    receipt = base.read(path)
    if receipt.get('status') == 'completed' and receipt.get('compiler'):
        source = Path(receipt['source_dir'])
        files = receipt.get('source_files', {})
        if not files:
            raise ValueError('completed receipt has no captured source checksums')
        if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                   text=True).strip() != receipt['snapshot_head']:
            raise ValueError('completed receipt source HEAD changed')
        for name, expected in files.items():
            if base.sha(source / name) != expected:
                raise ValueError('captured source changed: ' + name)
        receipt = dict(receipt, cli=receipt['compiler'],
            cli_sha256=receipt['compiler_sha256'], build_completed=True,
            snapshot_source_unchanged=True, source_files_reverified=len(files),
            receipt_schema='completed/compiler; independently reverified source HEAD/files')
    if (not receipt.get('build_completed') or not receipt.get('snapshot_source_unchanged') or
        base.sha(receipt['cli']) != receipt['cli_sha256']):
        raise ValueError('completed immutable CLI receipt is required')
    return receipt


def verified(path):
    c = base.verified_config(path)
    for name, expected in c['dependencies'].items():
        if base.sha(name) != expected:
            raise ValueError('frozen candidate dependency changed: ' + name)
    if base.sha(c['full_model_reference']) != c['full_model_reference_sha256']:
        raise ValueError('independent complete-model reference changed')
    if base.sha(c['bridge']) != c['export_files']['bridge_b1.json']:
        raise ValueError('candidate bridge differs from the complete exported model')
    return c


def resource_gate(actual, target, dynamic_shared, requested=1):
    if (actual['num_sms'] != target['resources']['num_sms'] or
        (actual['major'], actual['minor']) != (target['sm_major'], target['sm_minor'])):
        raise ValueError('actual device differs from the bound target')
    rows = actual['kernels']
    if len(rows) != len(KINDS) or any(sum(k in row['name'] for row in rows) != 1 for k in KINDS):
        raise ValueError('full-model executor kernel set is incomplete')
    for row in rows:
        if (row['resident'] < requested or row['registers'] <= 0 or
            row['threads'] != 128 or row['max_threads'] < 128 or
            row['dynamic_shared'] != dynamic_shared):
            raise ValueError('actual final kernel cannot support declared placement')
    return min(row['resident'] for row in rows)


def certify(path, attempt):
    c = verified(path)
    base.require_lease(c)
    d = Path(c['directory'])
    out = Path(attempt).absolute()
    out.mkdir(parents=True, exist_ok=True)
    pending = base.read(d / 'cpu-compile-result.json')
    if pending['exit_code'] or pending['binary_sha256'] != base.sha(d / 'plan.so'):
        raise ValueError('immutable successful CPU output is missing')
    command = [c['host_probe'], str(d / 'plan.so'), '128',
               str(pending['dynamic_shared']), *pending['kernel_symbols']]
    p = subprocess.run(command, text=True, capture_output=True)
    (out / 'resources.stdout').write_text(p.stdout)
    (out / 'resources.stderr').write_text(p.stderr)
    if p.returncode:
        raise ValueError('public CUDA resource query failed on the actual binary')
    actual = json.loads(p.stdout)
    resident = resource_gate(actual, base.read(c['target']), pending['dynamic_shared'])
    base.write(d / 'guarded-build-result.json', dict(exit_code=0, time=time.time(),
        gpu=c['gpu'], pid=os.getpid(), binary_sha256=pending['binary_sha256'],
        actual_final_artifact_resources=actual, actual_min_residency=resident,
        command=command, no_compilation_performed=True,
        resource_query='actual dlopen hoststubs: public CUDA attributes, opt-in and occupancy',
        cpu_compilation_was_gpu_guarded=False, native_seed_not_solver_optimal=True))


def accept(path):
    c = verified(path)
    d = Path(c['directory'])
    state = base.read(Path(c['run_root']) / f"runs/r16/scheduler/gpu{c['gpu']}/state.json")
    event = state[c['build_job']]
    out = Path(event['out'])
    recovery.completed_guard(out if out.is_absolute() else Path(c['run_root']) / out, event)
    identity = recovery.native_identity(d / 'plan.so')
    m = identity['plan']
    resource = base.read(d / 'guarded-build-result.json')
    if resource['binary_sha256'] != identity['binary_sha256']:
        raise ValueError('resource receipt belongs to another binary')
    if (identity['source']['head'] != c['source_head'] or
        identity['source']['compiler_sha256'] != c['cli_sha256'] or
        int(identity['defines'].get('TILEMEGA_ARCH_ID', 0)) != 900):
        raise ValueError('artifact source/compiler/architecture identity differs')
    if any(m.get(k) != v for k, v in c['expected'].items()):
        raise ValueError('requested forward coordinate did not bind')
    contract.check_matrix(m, (d / 'plan.so.cu').read_text(),
                          tuple(c['full_model_counts']), SHAPE)
    log = (d / 'plan.so.ptxas.log').read_text(errors='replace')
    if 'C7510' in log or re.search(r'wgmma[^\n]*serializ', log, re.I):
        raise ValueError('full model contains serialized WGMMA')
    p = subprocess.Popen(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(d / 'plan.so')],
                         stdout=subprocess.PIPE, text=True, env=base.cpu_env(c))
    def lines():
        with gzip.open(d / 'sass.txt.gz', 'wt', compresslevel=1) as saved:
            for line in p.stdout:
                saved.write(line)
                yield line
    audit = recovery.audit_sass(lines())
    base.write(d / 'sass-audit.json', audit)
    if p.wait():
        raise ValueError('actual binary SASS audit failed')
    n = sum(g['impl'] == 'wgmma' for g in m['gemms'])
    if c['arm'] == 'tma-im2col':
        if (not n or any(not audit['counts'].get(k) for k in ('HGMMA', 'UTMALDG.4D')) or
            audit['hot_local_instructions']):
            raise ValueError('actual WGMMA/im2col instructions or compute-loop spill gate failed')
    elif n or audit['counts'].get('HGMMA'):
        raise ValueError('MMA control actually contains WGMMA')
    base.write(d / 'cpu-build-result.json', dict(passed=True, candidate_id=c['candidate_id'],
        artifact_id=identity['artifact_id'], library_sha256=identity['binary_sha256'],
        source_head=c['source_head'], cli_sha256=c['cli_sha256'], gemms=m['gemms'],
        actual_wgmma_gemms=n, sass_counts=audit['counts'],
        hot_local_gate=audit['hot_local_instructions'], resource_recertification=resource,
        whole_kernel_spill_claim='not asserted; complete ptxas and loop audit retained',
        native_seed_not_solver_optimal=True))


def build(path):
    c = verified(path)
    d = Path(c['directory'])
    env = base.cpu_env(c)
    env['TMPDIR'] = c['compiler_tmpdir']
    if not Path(env['TMPDIR']).is_dir():
        raise ValueError('declared executable RAM compiler scratch is absent')
    command = [c['cli'], 'compile', c['bridge'], str(d / 'plan.so'), *c['flags']]
    start = time.time()
    row = base.child(command, c['run_root'], env, d / 'build.log')
    result = dict(row, exit_code=row['returncode'], started=start, ended=time.time(),
                  gpu_execution=False, cpu_compilation_was_gpu_guarded=False,
                  source_head=c['source_head'], cli_sha256=c['cli_sha256'])
    base.write(d / 'cpu-compile-result.json', result)
    if row['returncode']:
        raise ValueError('official complete-model native-seed compilation failed')
    identity = recovery.native_identity(d / 'plan.so')
    contract.check_matrix(identity['plan'], (d / 'plan.so.cu').read_text(),
                          tuple(c['full_model_counts']), SHAPE)
    lib = ctypes.CDLL(str(d / 'plan.so'))
    lib.tm_plan_shared_bytes.restype = ctypes.c_size_t
    # This exported constant accessor does not call the CUDA device probe.
    result['dynamic_shared'] = int(lib.tm_plan_shared_bytes())
    symbols = subprocess.check_output(['nm', '-D', str(d / 'plan.so')], text=True)
    selected = []
    for kind in KINDS:
        matches = [s.split()[-1] for s in symbols.splitlines()
                   if kind in s and '__device_stub__' not in s]
        if len(matches) != 1:
            raise ValueError('actual executor hoststub is not unique: ' + kind)
        selected.append(matches[0])
    result.update(binary_sha256=identity['binary_sha256'], kernel_symbols=selected)
    base.write(d / 'cpu-compile-result.json', result)
    jobs = base.read(Path(c['candidate_path']).parent / 'queue-preview.json')
    queue = Path(c['publication_queue'])
    if queue.exists():
        raise FileExistsError('new run queue already published')
    base.write(queue, jobs)
    base.write(d / 'publication.json', dict(time=time.time(), queue=str(queue), jobs=len(jobs),
        gpu_resource_certified=False, numerical_gate_passed=False))
    print(json.dumps(dict(cpu_compile_succeeded=True, queue=str(queue), jobs=len(jobs))), flush=True)


def prepare(args):
    root = args.root.absolute()
    receipt = completed_receipt(args.receipt)
    out = args.out.absolute()
    if out.exists():
        raise FileExistsError('new run-id required')
    out.mkdir(parents=True)
    base.write(out / 'normalized-cli-receipt.json', dict(receipt,
        original_receipt=str(args.receipt.absolute()), original_receipt_sha256=base.sha(args.receipt)))
    dependencies = {}
    for module in (sys.modules[__name__], contract, probe, recovery, base):
        source = Path(module.__file__)
        dest = out / source.name
        shutil.copy2(source, dest)
        dependencies[str(dest)] = base.sha(dest)
    validation = Path(base.__file__).with_name('r16_measurement_validation.py')
    shutil.copy2(validation, out / validation.name)
    dependencies[str(out / validation.name)] = base.sha(out / validation.name)
    target = out / 'target.json'
    shutil.copy2(args.target, target)
    source = out / 'actual-artifact-probe.cpp'
    source.write_text(probe.HOST_PROBE)
    host_probe = out / 'actual-artifact-probe'
    subprocess.run(['g++', '-std=c++17', '-O2', str(source), '-I/usr/local/cuda/include',
        '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-ldl',
        '-o', str(host_probe)], check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    dependencies[str(host_probe)] = base.sha(host_probe)
    reference = args.full_model_reference.absolute()
    counts = contract.model_counts(reference.read_text())
    export = args.export.absolute()
    d = out / args.model
    d.mkdir()
    bridge = d / 'export.json'
    shutil.copy2(export / 'bridge_b1.json', bridge)
    # The parent directory names the actual model in the official manifest.
    for arm in args.arms:
        folder = d / arm
        folder.mkdir()
        expected = dict(model=args.model, conv_impl='cpasync' if arm == 'mma' else 'tma_im2col',
                        forward_gemm_impl='mma16' if arm == 'mma' else 'wgmma',
                        forward_seed_tile=SHAPE, forward_seed_explicit=True)
        flags = ['--emit', 'serving', '--serving', 'forward', '--frontend', 'dnn', '--batch', '1',
                 '--runtime-target', str(target), '--pg', 'l2', '--resource-probes', '0',
                 '--seed-arch', 'native', '--forward-seed-tile', 'M64N128K64S2',
                 '--nonpaged-weight-layout', 'tiled', '--deferred-ln', 'auto', '--reuse', 'l2',
                 '--global-la', 'auto', '--dwpw-fuse', 'auto', '--forward-gemm-impl',
                 expected['forward_gemm_impl'], '--conv-impl', expected['conv_impl']]
        config_path = folder / 'candidate.json'
        stem = 'h7a-' + out.name + '-' + args.model + '-' + arm
        c = dict(candidate_id=stem + ':B1', arm=arm, source=receipt['source_dir'],
            source_head=receipt['snapshot_head'], cli=receipt['cli'], cli_sha256=receipt['cli_sha256'],
            run_root=str(root), python=str(args.python.absolute()), gpu=args.gpu,
            directory=str(folder), bridge=str(bridge), export=str(export), flags=flags,
            expected=expected, target=str(target), target_sha256=base.sha(target),
            export_files={n:base.sha(export / n) for n in ('bridge_b1.json', 'manifest.json',
                'inputs_b1.safetensors', 'exported_program.pt2')}, dependencies=dependencies,
            full_model_reference=str(reference), full_model_reference_sha256=base.sha(reference),
            full_model_counts=counts, host_probe=str(host_probe), compiler_tmpdir=str(args.tmpdir.absolute()),
            candidate_path=str(config_path), build_job=stem + '-certify',
            publication_queue=str(root / f'runs/r16/queue/gpu{args.gpu}' / (stem + '.json')),
            selection_scope='explicit public native seed; not a solver-optimal or final-default claim')
        base.write(config_path, c)
        node = 0 if args.gpu < 4 else 1
        common = dict(cwd=str(root), env=dict(PYTHONPATH=c['source'] + '/python',
            CUDA_VISIBLE_DEVICES=str(args.gpu), TILEMEGA_DEVICE_INDEX=str(args.gpu),
            TILEMEGA_NUMA_NODE=str(node), TILEMEGA_GPU_PID_MODE='namespace_limited',
            OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'),
            priority=3, needs_free_mib=4096)
        prefix = [c['python'], str(out / Path(__file__).name)]
        jobs = [dict(common, name=c['build_job'], gpu=True, timing=False, timeout_s=300,
            command=['numactl', f'--cpunodebind={node}', *prefix, 'certify', '--candidate',
                     str(config_path), '--attempt', '{attempt_dir}']),
            dict(common, name=stem+'-accept', gpu=False, timing=False, timeout_s=300,
                 after=[c['build_job']], command=[*prefix, 'accept', '--candidate', str(config_path)])]
        checks = []
        for fresh in range(5):
            name = stem + f'-check-f{fresh}'
            checks.append(name)
            jobs.append(dict(common, name=name, gpu=True, timing=False, timeout_s=1200,
                after=[stem+'-accept'], command=['numactl', f'--cpunodebind={node}', *prefix,
                'check', '--candidate', str(config_path), '--fresh', str(fresh), '--attempt', '{attempt_dir}']))
        jobs.append(dict(common, name=stem+'-gate', gpu=False, timing=False, timeout_s=120,
            after=checks, command=[*prefix, 'aggregate', '--candidate', str(config_path)]))
        for rnd in range(3):
            jobs.append(dict(common, name=stem+f'-measure-r{rnd}', gpu=True, timing=True,
                timeout_s=1800, after=[stem+'-gate'], command=['numactl', f'--cpunodebind={node}',
                *prefix, 'measure', '--candidate', str(config_path), '--round-index', str(rnd),
                '--attempt', '{attempt_dir}']))
        base.write(folder / 'queue-preview.json', jobs)
    print(json.dumps(dict(out=str(out), full_model_counts=counts, gpu_jobs_published=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    a = sub.add_parser('prepare')
    for name in ('receipt', 'out', 'target', 'export', 'full-model-reference'):
        a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--model', choices=('resnet18', 'nafnet'), required=True)
    a.add_argument('--arms', nargs='+', choices=('mma', 'tma-im2col'), default=['mma', 'tma-im2col'])
    a.add_argument('--root', type=Path, default=Path('/root/TileMega'))
    a.add_argument('--python', type=Path, default=Path('/root/TileMega/runs/r16/env/dnn-export/bin/python'))
    a.add_argument('--tmpdir', type=Path, default=Path('/root/TileMega/runs/r16/tmp/compiler'))
    a.add_argument('--gpu', type=int, choices=range(8), default=0)
    for op in ('build', 'certify', 'accept', 'check', 'aggregate', 'measure'):
        a = sub.add_parser(op)
        a.add_argument('--candidate', type=Path, required=True)
        if op in ('certify', 'check', 'measure'):
            a.add_argument('--attempt', type=Path, required=True)
        if op == 'check':
            a.add_argument('--fresh', type=int, required=True)
        if op == 'measure':
            a.add_argument('--round-index', type=int, required=True)
    args = p.parse_args()
    if args.operation == 'prepare':
        prepare(args)
    elif args.operation == 'build':
        build(args.candidate)
    elif args.operation == 'certify':
        certify(args.candidate, args.attempt)
    elif args.operation == 'accept':
        accept(args.candidate)
    else:
        verified(args.candidate)
        if args.operation == 'check':
            base.check(args.candidate, args.fresh, args.attempt)
        elif args.operation == 'aggregate':
            base.aggregate(args.candidate)
        else:
            base.measure(args.candidate, args.round_index, args.attempt)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print('Forward seed candidate rejected: '+str(error), file=sys.stderr)
        raise SystemExit(3)
