#!/usr/bin/env python3
"""Prepare guarded dense25 closeout manifests/queues from existing artifacts.

CPU only: this command neither builds artifacts nor starts a scheduler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b')
DEFAULT_GPUS = dict(zip(MODELS, range(5)))
# Same machine mapping as queue_sm90r2.py; the execution wrapper checks it again.
PCI = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 7: 'b7'}
FROZEN = 'configs/serving/sm90r2/closeout/manifest.json'
POLICY = 'configs/bench/sm90r2_guard_policy.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def git_bytes(root, revision, path):
    return subprocess.check_output(['git', '-C', str(root), 'show', f'{revision}:{path}'])


def tracked(root, record, revision='HEAD'):
    path = root / record['path']
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError('invalid tracked input path: ' + str(path))
    actual = sha(path)
    if actual != record['sha256'] or hashlib.sha256(git_bytes(root, revision, record['path'])).hexdigest() != actual:
        raise ValueError('tracked input differs: ' + str(path))
    return path


def prefill_signature(plan):
    fields = ('tile_m', 'tile_n', 'tile_k', 'stages', 'split_k', 'impl')
    return dict(pg=plan['pg'], residency_requested=str(plan['requested_residency']),
                attention_impl=plan['attention_impl'], attention_kv_block=plan['attention_kv_block'],
                gemm_table_sha256=hashlib.sha256(json.dumps(plan['gemms'], sort_keys=True).encode()).hexdigest(),
                geometries=[list(v) for v in sorted({tuple(g[k] for k in fields) for g in plan['gemms']})])


def canonical_coordinates(coords):
    result = json.loads(json.dumps(coords))
    # Sm90DenseCoordinates::FromJson supplies these optional schema-1 defaults.
    for gemm in result['gemms'].values():
        for key, value in dict(register_stages=1, workspace_offset=0, preissue_late=False).items():
            gemm.setdefault(key, value)
    for key, value in dict(workspace_offset=0, preissue_late=False).items():
        result['attention'].setdefault(key, value)
    return result


def environment(root, gpu):
    node = 0 if gpu < 4 else 1
    return dict(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE=str(node),
                TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
                TILEMEGA_NUMA_CONCURRENCY_VERIFIED='0', PYTHONPATH=str(root / 'python'),
                OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def prepare(args):
    root, output = args.repo_root.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError('retain existing preparation; choose a new --out')
    gpu_for = dict(DEFAULT_GPUS)
    gpu_for.update(json.loads(args.model_gpus))
    if set(gpu_for) != set(MODELS) or any(type(g) is not int or g not in PCI for g in gpu_for.values()):
        raise ValueError('model GPUs must be timing cards 0/1/2/3/4/7; GPU6 forbidden, GPU5 functional-only')
    if args.tag and (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag) or args.tag in ('.', '..')):
        raise ValueError('tag must be one safe directory component')
    build_root = args.build_root.resolve() if args.build_root else root / 'runs/sm90r2/build' / args.tag
    if output == build_root or output in build_root.parents or build_root in output.parents:
        raise ValueError('--out and artifact tree must be disjoint')
    frozen_path = root / FROZEN
    frozen = read(tracked(root, dict(path=FROZEN, sha256=sha(frozen_path))))
    expected = {(m, b) for m in MODELS for b in (1, 2, 4, 8, 16)}
    cells = frozen['cells']
    if (len(cells) != 25 or {(c['model'], c['batch']) for c in cells} != expected
            or len({c['cell'] for c in cells}) != 25
            or frozen['kind'] != 'sm90r2_closeout_dense_fixed_coordinates'
            or frozen['solver_validated'] is not False or frozen['e3_validated'] is not False):
        raise ValueError('expected the tracked fixed, unvalidated dense25 manifest')
    for key in ('target', 'inputs_manifest'):
        tracked(root, frozen[key])
    policy = read(root / POLICY)
    allowed = set(policy['allowed_timing_gpu_indices']) & set(policy['allowed_gpu_indices'])
    if not set(gpu_for.values()) <= allowed:
        raise ValueError('requested GPU is excluded by guard policy')
    uuid_for = policy['idle_power_mapping_provenance']['uuid_by_gpu']
    # Import only after selecting the repository; staged copies work with --repo-root.
    sys.path[:0] = [str(root / 'scripts'), str(root / 'python')]
    from tilemega.build.identity import verify
    from run_sm90r2_closeout import validate
    prepared, missing, invalid = [], [], []
    cohort = None
    for cell in cells:
        try:
            config = read(tracked(root, cell['build_config']))
            coords = read(tracked(root, cell['coordinates']))
            if (config['model'], config['batch']) != (cell['model'], cell['batch']):
                raise ValueError('configuration model/batch differs from manifest')
            folder = build_root / f"{cell['model']}-B{cell['batch']}"
            absent = [str(folder / (phase + '.so' + suffix)) for phase in ('prefill', 'decode')
                      for suffix in ('', '.identity.json', '.cu', '.plan.json', '.ptxas.log', '.build_command.txt')
                      if not (folder / (phase + '.so' + suffix)).is_file()]
            if absent:
                missing.append(dict(cell=cell['cell'], absent=absent))
                continue
            libraries, identities = {}, {}
            for phase in ('prefill', 'decode'):
                binary = folder / (phase + '.so')
                identity = verify(binary)
                source = identity['source']
                binding = (source['head'], source['source_digest'], source['compiler_sha256'])
                if source['worktree_diff_sha256'] != hashlib.sha256(b'').hexdigest():
                    raise ValueError('build includes an uncommitted source patch')
                if args.expected_head and source['head'] != args.expected_head:
                    raise ValueError('artifact differs from --expected-head')
                if cohort is not None and binding != cohort:
                    raise ValueError('artifacts do not share one HEAD/source/compiler')
                if identity['target_sha256'] != frozen['target']['sha256']:
                    raise ValueError('artifact runtime target differs from frozen target')
                if phase == 'decode' and (identity['plan'].get('native_dense') != canonical_coordinates(coords)
                                         or identity['plan'].get('native_origin') != 'fixed'):
                    raise ValueError('native coordinates/origin differ from fixed selection')
                if phase == 'prefill' and prefill_signature(identity['plan']) != config['coordinates']['prefill']:
                    raise ValueError('prefill geometry differs from frozen configuration')
                cohort = binding
                identities[phase] = identity
                libraries[phase] = dict(path=str(binary), sha256=identity['binary_sha256'],
                    artifact_id=identity['artifact_id'], identity_sha256=sha(str(binary) + '.identity.json'))
            # These files must also have existed unchanged at the actual build HEAD.
            for record in (dict(path=FROZEN, sha256=sha(frozen_path)), cell['build_config'], cell['coordinates']):
                tracked(root, record, cohort[0])
            manifest = {k: config[k] for k in ('model', 'batch', 'model_dir', 'model_config_sha256',
                        'prompt_ids', 'prompt_ids_sha256', 'execution')}
            manifest.update(schema=1, kind='sm90r2_closeout_llm', build_head=cohort[0],
                protocol=frozen['protocol'], e3_validated=False, solver_validated=False,
                selection='frozen native control; solver E3 not met; concurrent timing consistency unvalidated',
                configuration=cell['build_config'], coordinates=cell['coordinates'], libraries=libraries)
            prepared.append((cell, manifest))
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
            invalid.append(dict(cell=cell['cell'], error=str(error)))
    report = dict(schema=1, kind='sm90r2_closeout_preparation', configuration_manifest_sha256=sha(frozen_path),
        build_root=str(build_root), build_head=cohort[0] if cohort else None,
        source_digest=cohort[1] if cohort else None, compiler_sha256=cohort[2] if cohort else None,
        cells_expected=25, cells_ready=len(prepared), missing=missing, invalid=invalid,
        allow_partial=args.allow_partial, model_gpus=gpu_for, e3_validated=False, solver_validated=False,
        numa_concurrency_trial=True, numa_concurrency_verified=False, gpu_started=False, cuda_builds_started=False,
        guard_policy=dict(path=POLICY, sha256=sha(root / POLICY)), helper_sha256=sha(__file__))
    output.mkdir(parents=True)
    if invalid or (missing and not args.allow_partial):
        report['queues_generated'] = False
        save(output / 'report.json', report)
        print(json.dumps(report))
        return 2
    queues = {g: [] for g in sorted(set(gpu_for.values()))}
    for cell, manifest in prepared:
        path = output / 'manifests' / f"{cell['cell']}.json"
        save(path, manifest)
        # Existing runner checks prompts, identity, static-loop mode and host weight ABI; no plan/create.
        try:
            validate(path)
        except (ValueError, KeyError, OSError) as error:
            report.update(queues_generated=False,
                          invalid=[dict(cell=cell['cell'], error='runner CPU validation: ' + str(error))])
            save(output / 'report.json', report)
            print(json.dumps(report))
            return 2
        gpu = gpu_for[cell['model']]
        node = 0 if gpu < 4 else 1
        for fresh in range(5 if cell['model'] == 'llama3_1_8b' else 3):
            name = f'closeout-{cell["cell"]}-f{fresh}'
            command = ['numactl', f'--cpunodebind={node}', args.python,
                str(root / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(root / POLICY),
                '--out', '{attempt_dir}/job', '--expected-uuid', uuid_for[str(gpu)],
                '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', str(node), '--',
                args.python, str(root / 'scripts/run_sm90r2_closeout.py'), '--manifest', str(path),
                '--fresh', str(fresh), '--out', '{attempt_dir}/job']
            queues[gpu].append(dict(name=name, gpu=True, timing=True, priority=10, timeout_s=3600,
                needs_free_mib=32768, cwd=str(root),
                after_any=[queues[gpu][-1]['name']] if queues[gpu] else [],
                env=environment(root, gpu), command=command))
    launch = []
    for gpu, jobs in queues.items():
        if not jobs:
            continue
        directory = output / f'gpu{gpu}'
        save(directory / 'queues/queue_closeout.json', jobs)
        command = [args.python, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(directory / 'queues'),
                   '--out', str(directory / 'scheduler'), '--policy', str(root / POLICY), '--deadline-hours', '24']
        launch.append(dict(gpu=gpu, env=environment(root, gpu), cwd=str(root), command=command))
    save(output / 'scheduler-commands.json', launch)
    report.update(queues_generated=True, jobs=sum(len(v) for v in queues.values()),
                  prepared_cells=[c['cell'] for c, _ in prepared],
                  runtime_manifests_cpu_validated=len(prepared))
    save(output / 'report.json', report)
    print(json.dumps(report))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tag', help='existing runs/sm90r2/build/TAG artifact tree')
    source.add_argument('--build-root', type=Path, help='tree containing MODEL-BATCH directories, e.g. MODEL-B16')
    parser.add_argument('--out', type=Path, required=True, help='new, separate preparation directory')
    parser.add_argument('--expected-head', help='optional full artifact HEAD; otherwise infer and enforce one cohort')
    parser.add_argument('--allow-partial', action='store_true', help='prepare complete cells; report absent binaries')
    parser.add_argument('--model-gpus', default='{}', help='JSON overrides for default five-model GPU0/1/2/3/4 map')
    parser.add_argument('--python', default='/workspace/llm-stack/envs/vllm/bin/python', help='existing runtime Python')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        return prepare(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
