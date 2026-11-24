"""Turn Phase 1 measured stage rows into the exact-key native cost input."""

import argparse
import csv
from pathlib import Path

from calibration_queue import CALIB, CASE
from guarded import ROOT, read, sha, write


VARIANTS = ('overlap', 'head-startup', 'tail-publication')
GEMM_KEYS = ('n', 'k', 'residency', 'tile_n', 'tile_k', 'slots', 'warps_n', 'warps_k', 'register_stages')
ATTENTION_KEYS = ('hkv', 'head_dim', 'queries', 'norm', 'residency', 'chunks', 'block_keys', 'slots', 'new_row_work')


def table(variant):
    target = ROOT / 'configs/targets/sm_90_h800_sq2.json'
    calibration = read(CALIB / 'collection.json')
    assert calibration['target_sha256'] == sha(target)
    assert (CASE / 'progress.md').is_file()
    binding = read(CALIB / 'production_source_binding.json')['production_headers']
    for header, expected in binding.items():
        assert sha(ROOT / header) == expected
    paths = {kind: CASE / f'calib_{kind}.csv' for kind in ('gemm', 'attention')}
    for path in paths.values():
        assert calibration['tables'][path.name]['sha256'] == sha(path)
    with paths['gemm'].open() as handle:
        gemm = []
        for row in csv.DictReader(handle):
            assert row['mma'] == 'mma_sync' and row['multicast'] == 'False'
            entry = {k: int(row[k]) for k in GEMM_KEYS}
            entry.update({k: row[k] for k in ('geometry_id', 'family', 'model')})
            entry.update({k: float(row[k]) for k in ('stage_body_ns', 'head_body_ns', 'epilogue_ns')})
            if row['steady_ns']:
                entry['steady_ns'] = float(row['steady_ns'])
            gemm.append(entry)
    with paths['attention'].open() as file:
        measured = [row for row in csv.DictReader(file) if row['past'] == '575']
    independent = {
        tuple(row[k] for k in ATTENTION_KEYS): float(row['stage_body_ns'])
        for row in measured if row['merge'] == 'stage'
    }
    attention = []
    for row in measured:
        full = float(row['combined_body_ns'])
        if row['merge'] == 'stage' or int(row['chunks']) == 1:
            compute = float(row['stage_body_ns'])
        else:
            compute = independent[tuple(row[k] for k in ATTENTION_KEYS)]
        reducer = full - compute
        if reducer < 0:
            raise ValueError(f'measured merge delta is negative for {row["geometry_id"]}')
        attention.append({
            **{k: int(row[k]) for k in ATTENTION_KEYS},
            **{k: row[k] for k in ('geometry_id', 'model', 'merge')},
            'compute_ns': compute, 'reducer_ns': reducer, 'combined_body_ns': full,
        })
    return {
        'schema': 1, 'variant': variant, 'target_sha256': sha(target),
        'sources': {p.name: sha(p) for p in paths.values()},
        'gemms': gemm, 'attention': attention,
        'normalization': 'Measured per-task head/steady service envelope, matched to isolated stage makespan with raw DRAM bytes retained.',
        'attention_context': 575,
        'attention_reducer': 'Explicit stage uses its measured body time. LA and DSMEM use their measured full stage minus corresponding StageMerge compute.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = table(args.variant)
    if args.destination.exists():
        assert read(args.destination) == result
    else:
        write(args.destination, result)


if __name__ == '__main__':
    main()
