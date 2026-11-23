#!/usr/bin/env python3
"""Bind final normal-search seeds to the R10 paired screening artifacts."""
import argparse
from copy import deepcopy
from pathlib import Path

from a100_baselines import BATCHES, sha
from a800_native import artifact
from a800_r10_prepare import MODELS, paired_libraries, read, schema10
from tilemega.bench.scheduler import write_json


def measured_arm(campaign, phase, model, batch, label, evidence):
    receipt = campaign/phase/model/f'B{batch}'/'plan.json'
    plan = read(receipt)
    if (plan['phase'], plan['model'], plan['batch']) != (phase, model, batch):
        raise ValueError(f'screening plan does not belong to {model}/B{batch} {phase}')
    library, identity, manifest = artifact(Path(paired_libraries(plan)[label]))
    if identity['binary_sha256'] != evidence['decode_sha256']:
        raise ValueError(f'{phase} {label} changed after paired screening')
    proof = dict(evidence, library=str(library), screen_plan_sha256=sha(receipt))
    return manifest['native_dense'], proof


def combine(base, split, mainloop, router, shapes):
    coords = schema10(deepcopy(base))
    if router:
        coords['gemms']['router'] = deepcopy(router['gemms']['router'])
        coords['payload'] = router['payload']
    if split:
        for family, gemm in split['gemms'].items():
            if gemm.get('split_k', 1) == 1:
                continue
            current = coords['gemms'][family]
            # Preserve the measured K decomposition when the R2 output tile
            # differs. A conflicting geometry still enters by its own arm.
            if current['tile_n'] != gemm['tile_n'] or current['slot_bytes'] != gemm['slot_bytes']:
                continue
            if shapes[family][1] % (current['slot_bytes']//(2*current['tile_n'])*gemm['split_k']):
                raise ValueError('measured split K does not divide the frozen ModelPlan')
            current['split_k'] = gemm['split_k']
            if family == 'qkv':
                coords['group_wait'] = False
    if mainloop:
        coords['multistage'] = True
        coords.pop('producer_warps', None)
    return coords


def receipt(catalog, samples, campaign, model, batch):
    cell = f'{model}/B{batch}'
    old = catalog['cells'][cell]
    cache = dict(weight_evict_first=False, persist_l2_eighths=0)
    selected_cache = samples['e1'][cell]
    if selected_cache.keys() - cache.keys():
        raise ValueError('invalid paired E1 policy')
    cache.update(selected_cache)
    selected, proofs, protected = {}, {}, None
    for phase in ('options', 'splitk', 'mainloop', 'router'):
        row = samples[phase].get(cell, {})
        label = row.get('winner')
        if label not in (None, 'control'):
            evidence = row['arms'][label]
            if not evidence['beats_control']:
                raise ValueError(f'{phase} {label} did not beat its paired control')
            selected[phase], proofs[phase] = measured_arm(campaign, phase, model, batch, label, evidence)
        if phase == 'options' and row.get('cost_model_blind'):
            enum = min(v['tpot_seconds'] for k, v in row['arms'].items() if k.startswith('enum-'))
            seeds = [(k, v) for k, v in row['arms'].items()
                     if k.startswith('seed-') and v['tpot_seconds'] < .995*enum]
            for label, evidence in sorted(seeds, key=lambda item: item[1]['tpot_seconds']):
                option, proof = measured_arm(campaign, phase, model, batch, label, evidence)
                if option.get('expert_page_copy') == 'segments':
                    protected, proofs['protected_page'] = option, proof
                    break

    def seeded(coords):
        result = schema10(deepcopy(coords))
        for field, value in cache.items():
            if value:
                result[field] = value
            else:
                result.pop(field, None)
        return result

    base = selected.get('options', old['coords'])
    combined = combine(base, selected.get('splitk'), selected.get('mainloop'),
                       selected.get('router'), old['shape'])
    candidates = [seeded(combined)]
    if protected:
        pinned = seeded(protected)
        if pinned not in candidates:
            candidates.append(pinned)
    # A failed joint projection must not remove a separately observed arm.
    alternatives = [seeded(item) for item in (*selected.values(), old['coords'])]
    return dict(cache_policy=cache, required_anchors=candidates,
                measure_unpriced_anchors=True), alternatives, proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'results', 'campaign', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('retain the previous seed and screening bindings')
    files = {'e1': args.results/'e1.selection.json'}
    for phase in ('options', 'splitk', 'mainloop', 'router'):
        files[phase] = args.results/phase/f'{phase}.screening.json'
    samples = {phase: read(path) for phase, path in files.items()}
    catalog = read(args.inventory)
    wins = sum(sample.get('winner') == 'multistage' for sample in samples['mainloop'].values())
    measured = sum('multistage' in sample.get('arms', {}) for sample in samples['mainloop'].values())
    bindings = {}
    for model in MODELS:
        for batch in BATCHES:
            seed, alternatives, proofs = receipt(catalog, samples, args.campaign, model, batch)
            cell = f'{model}/B{batch}'
            folder = args.out/cell
            folder.mkdir(parents=True, exist_ok=False)
            path = folder/'seed-coordinates.json'
            write_json(path, seed)
            write_json(folder/'alternative-anchors.json', alternatives)
            bindings[cell] = dict(seed=str(path), seed_sha256=sha(path), paired_evidence=proofs)
    write_json(args.out/'selection.json', dict(cells=bindings, inventory_sha256=sha(args.inventory),
               screened={phase: sha(path) for phase, path in files.items()},
               multistage={'wins': wins, 'measured': measured,
                           'clears_default_screen': measured == 20 and wins >= 12}))


if __name__ == '__main__':
    main()
