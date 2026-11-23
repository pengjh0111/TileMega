#!/usr/bin/env python3
"""Prepare immutable fixed-coordinate R10 trials and verify their common inputs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import BATCHES, ROOT, sha
from a800_native import artifact
from tilemega.bench.scheduler import write_json

MODELS = ('qwen3_30b_a3b', 'olmoe_1b_7b', 'llama3_1_8b', 'mistral_7b_v0_3')
SPLIT_FAMILIES = ('qkv', 'o', 'down', 'router', 'expert_down')


def read(path):
    return json.loads(Path(path).read_text())


def suffix(path, ending):
    return Path(str(path) + ending)


def archive(record):
    source = Path(record['timing_source'])
    if sha(source) != record['timing_sha256']:
        raise ValueError('archived R9 timing receipt changed')
    receipt = read(source)
    raw = read(Path(receipt['evidence']) / f"B{record['batch']}/measurements.json")
    decode, identity, manifest = artifact(Path(raw['configuration']['decode_binary']))
    prefill = ('prompt' if record['prefill_sha256'] == 'prompt'
               else raw['configuration']['prefill_binary'])
    targets = dict(decode_sha256=identity['binary_sha256'],
                   prefill_sha256=prefill if prefill == 'prompt' else sha(prefill),
                   target_sha256=sha(manifest['runtime_target']))
    if any(record[key] != digest or receipt[key] != digest for key, digest in targets.items()):
        raise ValueError('archived artifacts differ from the published R9 result')
    families = {}
    for row in manifest['gemms']:
        family, shape = row['family'], (row['n'], row['k'])
        if families.setdefault(family, shape) != shape:
            raise ValueError(f'inconsistent ModelPlan shapes for {family}')
    return prefill, manifest, families, raw['checkpoint']['checkpoint_sha256']


def schema10(coordinates):
    coordinates.update(schema=10, tp=coordinates.get('tp', 1),
                       argmax=coordinates.get('argmax', 'la'))
    return coordinates


def split_arms(anchor, families, sms):
    grid = sms * anchor['residency']
    # Keep the exact same QKV-to-attention handoff in the paired control.
    control = deepcopy(anchor)
    control['group_wait'] = False
    yield 'control', control
    for slices in (2, 4):
        trial = schema10(deepcopy(control))
        for family in SPLIT_FAMILIES:
            if family not in families:
                continue
            n, k = families[family]
            g = trial['gemms'][family]
            tn = g['tile_n']
            tk = g['slot_bytes'] // (2 * tn)
            tiles = (n + tn - 1) // tn
            if tiles >= grid or k % (tk * slices):
                continue
            if family in ('o', 'down') and tiles * slices > 3 * grid:
                continue
            g['split_k'] = slices
        if any(g.get('split_k') == slices for g in trial['gemms'].values()):
            yield f'split-{slices}', trial


def mainloop_arms(anchor):
    control = deepcopy(anchor)
    control.pop('producer_warps', None)
    yield 'control', control
    yield 'multistage', dict(schema10(deepcopy(control)), multistage=True)


def router_arms(anchor, families):
    if 'router' not in families:
        return
    control = schema10(deepcopy(anchor))
    if control['weight_stream']:
        raise ValueError('padded routing requires nonstreamed, retired payload')
    control['payload'] = 'reuse'
    yield 'control', control
    n, k = families['router']
    original = control['gemms']['router']
    warps_n = original['warps_n']
    for width in (24 * warps_n, 40 * warps_n):
        if width > n or not n % width:
            continue
        trial = deepcopy(control)
        tn = original['tile_n']
        tk = original['slot_bytes'] // (2 * tn)
        # Keep the K depth and use the same MMA instruction and output layout.
        if k % tk:
            raise ValueError('router coordinate uses an incomplete K tile')
        trial['gemms']['router'].update(tile_n=width, slot_bytes=2 * width * tk)
        yield f'router-tn{width}', trial


def prepare_options(args, source, target, folder, prefill, anchor):
    cache = dict(weight_evict_first=anchor.get('weight_evict_first', False),
                 persist_l2_eighths=anchor.get('persist_l2_eighths', 0))
    binding = folder / 'cache-policy.json'
    if not binding.exists():
        write_json(binding, {'cache_policy': cache})
    elif read(binding) != {'cache_policy': cache}:
        raise ValueError('the option control changed its cache policy')
    seeded = None
    if anchor['model'] == 'moe':
        page = schema10(deepcopy(anchor))
        page['expert_page_copy'] = 'segments'
        page.pop('expert_aligned_gate_pages', None)
        seeded = folder / 'page-segments.anchors.json'
        contract = {'cache_policy': cache, 'required_anchors': [page]}
        if not seeded.exists():
            write_json(seeded, contract)
        elif read(seeded) != contract:
            raise ValueError('the page-segments anchor changed')
    linked, rejected = {}, {}
    scenarios = [('enum', binding)]
    if seeded:
        scenarios.append(('seed', seeded))
    for label, seed in scenarios:
        output = folder / f'{label}.so'
        evidence = Path(str(output) + '.search') / 'prepared.json'
        cmd = [sys.executable, str(ROOT/'scripts/a800_solve.py'), 'prepare',
               '--model', args.model, '--batch', str(args.batch), '--export', str(source),
               '--target', target, '--prefill', prefill, '--cli', str(args.cli.resolve()),
               '--compile-jobs', '1', '--top', '2', '--search-budget-ms', '600000',
               '--timeout-s', '2600', '--out', str(output)]
        cmd.extend(('--seed-coordinates', str(seed)))
        if not evidence.is_file():
            with (folder / f'{label}.search.log').open('w') as log:
                checked = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=2660)
            if checked.returncode:
                rejected[label] = {'exit_code': checked.returncode, 'receipt': str(evidence)}
                continue
        abi = read(evidence.parent/'weight-abi.json')
        expected = {row['binary_sha256'] for row in abi['candidates']}
        document = read(evidence)
        receipt = document['receipt']
        if document['scope']['compiler_sha256'] != sha(args.cli) or not expected:
            raise ValueError('the solver shortlist predates this compiler')
        for candidate in receipt['candidates']:
            if candidate['status'] != 'compiled':
                continue
            lib = candidate['binary']
            _, identity, _ = artifact(Path(lib))
            rank = candidate['rank']
            if (identity['binary_sha256'] != candidate['binary_sha256'] or
                    identity['binary_sha256'] not in expected):
                raise ValueError('the shortlist does not match the validated weight packing')
            linked[f'{label}-{rank}'] = lib
    if seeded and 'seed' in rejected:
        # A rejected solver price cannot eliminate the experimental body. Link
        # the same coordinates directly and keep the solver error as evidence.
        trial = read(seeded)['required_anchors'][0]
        try:
            linked['seed-fixed'] = str(build(args, source, target, folder, 'seed-fixed', trial))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            rejected['seed-fixed'] = str(error)
    # A required anchor and unseeded search can select the same coordinates.
    distinct = {json.dumps(anchor, sort_keys=True): 'control'}
    for label, lib in linked.items():
        _, _, manifest = artifact(Path(lib))
        fingerprint = json.dumps(manifest['native_dense'], sort_keys=True)
        equal = distinct.setdefault(fingerprint, label)
        if equal != label:
            rejected[label] = {'same_coordinates_as': equal}
    return {label: linked[label] for label in distinct.values() if label != 'control'}, rejected


def build(args, source, target, folder, label, coordinates):
    lib = folder / f'{label}.so'
    config = folder / f'{label}.coordinates.json'
    if config.exists():
        if read(config) != coordinates:
            raise ValueError(f'coordinate changed for {label}')
    else:
        write_json(config, coordinates)
    if suffix(lib, '.identity.json').is_file():
        artifact(lib)
        return lib
    command = [str(args.cli.resolve()), 'compile', str(source), str(lib),
               '--tmexec-config', str(config), '--serving', 'decode', '--emit', 'serving',
               '--runtime-target', target, '--batch', str(args.batch), '--capacity', '1088',
               '--prompt-len', '64', '--past-range', '64:1086', '--nvcc-split-compile', '1']
    with (folder / f'{label}.log').open('w') as log:
        subprocess.run(['timeout', '--kill-after=15s', '900', *command], cwd=ROOT,
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    artifact(lib)
    return lib


def prepare_probes(args, folder, libraries, target, prefill):
    proofs, missing = {}, {}
    if args.phase not in ('splitk', 'mainloop'):
        return proofs, missing
    for label, library in libraries.items():
        lib, identity, manifest = artifact(Path(library))
        coords = manifest['native_dense']
        family_names = [family for family in ('qkv', 'o', 'down') if family in coords['gemms']]
        options = ['--role-clock'] if args.phase == 'mainloop' else ['--width-scan']
        splits = {coords['gemms'][family].get('split_k', 1) for family in family_names}
        for slices in sorted(splits):
            families = [f for f in family_names if coords['gemms'][f].get('split_k', 1) == slices]
            if args.phase == 'splitk' and label != 'control' and slices == 1:
                continue
            stem = f'{label}-service-k{slices}'
            out = folder / stem
            anchor = folder / f'{stem}.anchor.json'
            write_json(anchor, dict(model=args.model, batch=args.batch, decode=str(lib),
                                    decode_sha256=identity['binary_sha256'], coordinates=coords,
                                    prefill=prefill, artifact_target=target))
            command = [sys.executable, str(ROOT/'scripts/prepare_a800_dense_service.py'),
                       '--anchor', str(anchor), '--out', str(out), '--families', *families,
                       '--split-k', str(slices), '--reserve-program-shared', *options]
            if coords.get('multistage'):
                command.append('--multistage')
            if coords.get('epilogue_rows') == 'live':
                command.append('--live')
            if coords.get('copy_issue_group'):
                command.extend(('--copy', str(coords['copy_issue_group'])))
            if coords.get('residual_copy_issue') and any(f in ('o', 'down') for f in families):
                command.append('--residual-copy')
            artifact_id = out/'probe.identity.json'
            log_path = folder / f'{stem}.log'
            if not artifact_id.exists():
                with log_path.open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=310)
                if result.returncode:
                    missing[stem] = {'exit_code': result.returncode, 'log': str(log_path)}
                    continue
            expected = read(artifact_id)
            if (expected['anchor_sha256'] != sha(anchor) or expected['split_k'] != slices
                    or expected['multistage'] != bool(coords.get('multistage'))):
                raise ValueError('linked stage probe differs from its production geometry')
            proofs[stem] = {'binary': str(out/'probe'), 'identity_sha256': sha(artifact_id),
                            'target': target, 'target_sha256': sha(target), 'arm': label,
                            'families': families, 'split_k': slices}
    return proofs, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('options', 'splitk', 'mainloop', 'router'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probes', action='store_true',
                        help='link split-K stage bandwidth and multistage role probes')
    parser.add_argument('--e1-selection', type=Path,
                        help='accepted cache policy coordinates keyed by model/Bn')
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('prepare these artifacts on the login node with CUDA devices hidden')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        parser.error('prepare from a clean pinned worktree')
    source = args.export.resolve(strict=True)
    key = f'{args.model}/B{args.batch}'
    record = read(args.catalog)[key]
    prefill, manifest, families, checkpoint = archive(record)
    target = manifest['runtime_target']
    anchor = deepcopy(manifest['native_dense'])
    if args.e1_selection:
        cache = read(args.e1_selection)[key]
        if not set(cache) <= {'weight_evict_first', 'persist_l2_eighths'}:
            parser.error('the E1 receipt can change only the two cache policy coordinates')
        if cache:
            schema10(anchor).update(cache)
    if args.phase == 'splitk':
        arms = split_arms(anchor, families, read(target)['resources']['num_sms'])
    elif args.phase == 'mainloop':
        arms = mainloop_arms(anchor)
    elif args.phase == 'router':
        arms = router_arms(anchor, families)
    else:
        arms = (('control', anchor),)
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    plan = folder / 'plan.json'
    if plan.exists():
        parser.error('retain the existing immutable trial plan')
    linked, rejected = {}, {}
    for label, coords in arms:
        try:
            linked[label] = str(build(args, source, target, folder, label, coords))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            if label == 'control':
                raise
            rejected[label] = str(error)
    if args.phase == 'options':
        candidates, excluded = prepare_options(args, source, target, folder, prefill, anchor)
        linked.update(candidates)
        rejected.update(excluded)
    if 'control' not in linked or len(linked) < 2:
        write_json(folder / 'prepare-rejected.json', dict(
            phase=args.phase, model=args.model, batch=args.batch, rejected=rejected,
            reason='no legal trial artifact linked'))
        return 1
    probes, missing = prepare_probes(args, folder, linked, target, prefill) if args.probes else ({}, {})
    control = linked.pop('control')
    label = next(iter(linked))
    trial = linked.pop(label)
    timeout = 1080 if args.model == 'qwen3_30b_a3b' else 570
    command = ['--model', args.model, '--batch', str(args.batch), '--target', target,
               '--prefill', prefill, '--control', control, '--trial', trial,
               '--expected-checkpoint-sha256', checkpoint, '--process-timeout-s', str(timeout)]
    if args.phase == 'router':
        command.extend(('--routing-export', str(source)))
    for name, path in linked.items():
        command.extend(('--additional-trial', f'{name}={path}'))
    received = folder / 'pair.plan.json'
    with received.with_suffix('.tmp').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts/a800_pair_native.py'),
                        'plan', *command, '--budget-s', '9500'],
                       check=True, stdout=log)
    received.with_suffix('.tmp').replace(received)
    write_json(plan, dict(phase=args.phase, model=args.model, batch=args.batch,
                         command=command, labels={'trial': label}, rejected=rejected,
                         probes=probes, probe_rejections=missing,
                         plan_sha256=sha(received)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
