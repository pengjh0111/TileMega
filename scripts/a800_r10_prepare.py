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
    for width in (24, 40):
        if width > n or not n % width:
            continue
        trial = deepcopy(control)
        tn = original['tile_n']
        tk = original['slot_bytes'] // (2 * tn)
        # Keep the K depth and use the same MMA instruction and output layout.
        if k % tk:
            raise ValueError('router coordinate uses an incomplete K tile')
        warps_n = max(w for w in (1, 2, 4) if width % (8 * w) == 0)
        trial['gemms']['router'].update(tile_n=width, slot_bytes=2 * width * tk,
                                      warps_n=warps_n, warps_k=4 // warps_n)
        yield f'router-tn{width}', trial


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('splitk', 'mainloop', 'router'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
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
    anchor = manifest['native_dense']
    if args.phase == 'splitk':
        arms = split_arms(anchor, families, read(target)['resources']['num_sms'])
    elif args.phase == 'mainloop':
        arms = mainloop_arms(anchor)
    else:
        arms = router_arms(anchor, families)
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
    if 'control' not in linked or len(linked) < 2:
        write_json(folder / 'prepare-rejected.json', dict(
            phase=args.phase, model=args.model, batch=args.batch, rejected=rejected,
            reason='no legal trial artifact linked'))
        return 1
    control = linked.pop('control')
    label = next(iter(linked))
    trial = linked.pop(label)
    timeout = 1080 if args.model == 'qwen3_30b_a3b' else 570
    command = ['--model', args.model, '--batch', str(args.batch), '--target', target,
               '--prefill', prefill, '--control', control, '--trial', trial,
               '--expected-checkpoint-sha256', checkpoint, '--process-timeout-s', str(timeout)]
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
                         plan_sha256=sha(received)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
