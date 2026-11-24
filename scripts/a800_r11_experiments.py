#!/usr/bin/env python3
"""Link fixed-coordinate boundary, router, and mainloop experiments on the login node."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess

from a100_baselines import inputs, sha
from a800_native import artifact
from a800_r10_prepare import build, schema10
from a800_r11_gate import read
from a800_r11_prepare import router_changed
from a800_r11_release import clean_head
from a800_shortlist_abi import freeze_host_preparation
from tilemega.bench.scheduler import write_json


def remove_cooperative(coords):
    coords.pop('multistage', None)
    coords.pop('multistage_single_join', None)
    explicit = {'weight_evict_first', 'persist_l2_eighths', 'epilogue_rows', 'payload', 'producer_warps',
                'copy_issue_group', 'expert_page_copy', 'expert_gu_routes', 'expert_slot_padding',
                'expert_aligned_gate_pages', 'task_group_wait', 'l2_weight_preissue'}
    if (coords['schema'] != 10 or explicit & coords.keys() or coords['attention'].get('compact_workspace')
            or any(tile.get('split_k', 1)>1 for tile in coords['gemms'].values())):
        return
    if 'route' in coords:
        coords['schema'] = 9
    elif coords['argmax'] != 'la' or coords['attention']['merge'] == 'stage':
        coords['schema'] = 8
    else:
        coords['schema'] = 6
        coords.pop('tp', None)
        coords.pop('argmax', None)


def boundary(base, config):
    trial = schema10(deepcopy(base))
    trial['task_group_wait'] = True
    yield 'whole-output-groups', trial
    if not base['group_wait'] and config.get('model_type') != 'olmoe':
        projected = deepcopy(trial)
        heads, kv = config['num_attention_heads'], config['num_key_value_heads']
        dim = config.get('head_dim', config['hidden_size']//heads)
        columns = (heads//kv+2)*dim
        qkv = projected['gemms']['qkv']
        tn, tk = qkv['tile_n'], qkv['slot_bytes']//(2*qkv['tile_n'])
        widths = [width for width in range(8*qkv['warps_n'], columns+1, 8*qkv['warps_n']) if columns%width == 0]
        if widths and qkv.get('split_k', 1) == 1:
            width = min(widths, key=lambda width: abs(width-tn))
            qkv.update(tile_n=width, slot_bytes=2*width*tk)
            projected['group_wait'] = True
            yield 'whole-output-and-kv-groups', projected


def qwen(base, manifest):
    if base.get('route', 'local') == 'local':
        route = deepcopy(base)
        route.update(schema=max(base['schema'], 9), route='la', combine='la',
                     argmax=base.get('argmax', 'la'), tp=base.get('tp', 1))
        yield 'router-last-arriver', route
    if not base.get('multistage') and not base.get('producer_warps') and not base['weight_stream']:
        yield 'l2-weight-preissue', dict(schema10(deepcopy(base)), l2_weight_preissue=True)
    option = base['gemms']['o']
    width, depth = option['tile_n'], option['slot_bytes']//(2*option['tile_n'])
    current_slots = option['slots']
    shape = next(record for record in manifest['gemms'] if record['family'] == 'o')
    for slots in (2, 4, 8):
        if slots != current_slots:
            trial = deepcopy(base)
            trial['gemms']['o']['slots'] = slots
            yield f'o-slots-{slots}', trial
    for registers in (2, 4):
        if registers != option['register_stages'] and 16*option['warps_k']*registers <= depth:
            trial = deepcopy(base)
            trial['gemms']['o']['register_stages'] = registers
            yield f'o-register-{registers}', trial
    for new_depth in (depth//2, depth*2):
        if (new_depth >= 64 and new_depth >= 16*option['warps_k']*option['register_stages']
                and shape['k'] % (new_depth*option.get('split_k', 1)) == 0):
            trial = deepcopy(base)
            trial['gemms']['o']['slot_bytes'] = 2*width*new_depth
            yield f'o-tk-{new_depth}', trial
    if option.get('split_k', 1) == 1 and shape['k']%(2*depth) == 0:
        trial = schema10(deepcopy(base))
        trial['gemms']['o']['split_k'] = 2
        yield 'o-split-k2', trial


def mainloop(base):
    specialized = deepcopy(base)
    remove_cooperative(specialized)
    yield 'specialized', specialized
    cooperative = schema10(deepcopy(base))
    cooperative['multistage'] = True
    cooperative.pop('producer_warps', None)
    cooperative.pop('l2_weight_preissue', None)
    cooperative.pop('multistage_single_join', None)
    yield 'cooperative', cooperative
    yield 'cooperative-single-join', dict(cooperative, multistage_single_join=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('boundary', 'qwen', 'mainloop'))
    for name in ('control', 'target', 'cli', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('prepare experiments on the login node with CUDA devices hidden')
    source_commit = clean_head()
    control = read(args.control)
    args.model, args.batch = control['model'], control['batch']
    target = args.target.resolve(strict=True)
    exported = Path(control['export'])
    _, identity, manifest = artifact(Path(control['decode']))
    base = manifest['native_dense']
    if (control['decode_sha256'] != identity['binary_sha256'] or control['coordinates'] != base
            or control['export_sha256'] != sha(exported)):
        raise ValueError('the experiment changed a published control or export')
    if args.experiment == 'qwen' and args.model != 'qwen3_30b_a3b':
        parser.error('use the Qwen3-30B-A3B routing and output-stage sweep')
    config = read(inputs(args.model)[0]/'config.json')
    candidates = {'boundary': lambda: boundary(base, config), 'qwen': lambda: qwen(base, manifest),
                  'mainloop': lambda: mainloop(base)}[args.experiment]()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    packing_prefix = () if control['prefill'] == 'prompt' else (Path(control['prefill']),)
    arms, failures = [], {}
    for name, coordinates in candidates:
        if coordinates == base:
            continue
        try:
            library = build(args, exported, str(target), out, name, coordinates)
            _, candidate, linked = artifact(library)
            if candidate['target_sha256'] != sha(target):
                raise ValueError('the linked arm uses a different target')
            packing = library.with_suffix('.host-preparation.json')
            freeze_host_preparation(config, args.batch, (*packing_prefix, library), packing)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            failures[name] = str(error)
            continue
        arms.append(dict(label=name, rank=len(arms)+1, coordinates=linked['native_dense'],
            binary=str(library), binary_sha256=candidate['binary_sha256'],
            host_preparation=str(packing), host_preparation_sha256=sha(packing),
            route_check=router_changed(base, linked['native_dense'])))
    prepared = out/'experiment.json'
    write_json(prepared, dict(source=source_commit, phase=args.experiment, failures=failures,
                 compiler_sha256=sha(args.cli), target_sha256=sha(target)))
    if not arms:
        raise ValueError(f'no linked {args.experiment} trials: {prepared}')
    write_json(out/'finalists.json', dict(model=args.model, batch=args.batch, control=str(args.control.resolve()),
        control_sha256=sha(args.control), prepared=str(prepared), prepared_sha256=sha(prepared),
        target=str(target), target_sha256=sha(target), candidates=arms, routing_export=str(exported)))


if __name__ == '__main__':
    main()
