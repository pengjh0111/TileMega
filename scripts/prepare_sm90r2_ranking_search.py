#!/usr/bin/env python3
"""Freeze diagnostic search cohorts before measuring their candidate ordering."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess

from sm90d1_common import ROOT, read, sha, write


def schedule_key(coordinates, prediction):
    structural = json.loads(json.dumps(coordinates))
    for body in [*structural['gemms'].values(), structural['attention']]:
        for key in ('workspace_offset', 'register_stages', 'preissue_late', 'prefetch_l2_bytes'):
            body.pop(key, None)
    # Preserve a timing, lookahead, placement or L2 change when its features differ.
    # Otherwise the diagnostic cohort tests another division of work instead
    # of spending its eight programs on compile-time translations and aliases.
    return json.dumps([structural, prediction], sort_keys=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--config', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline-tag', default='B15Z')
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new evidence directory under runs/sm90r2')
    a.out.mkdir(parents=True)
    target = a.out / 'target.json'
    shutil.copy2(a.target.resolve(), target)
    compiler = a.compiler.resolve()
    compiler_sha = sha(compiler)
    inputs = read(ROOT / 'configs/serving/sm90d1/inputs/manifest.json')
    policy = dict(selection='baseline, first three distinct predicted schedules, exact/preissue seeds for each residency, then ranked fill',
                  equivalence='equal-feature timing, offset, lookahead and clipped-L2 variants share one diagnostic schedule',
                  candidates=8, final_selection=False, compiler_sha256=compiler_sha,
                  target_sha256=sha(target), search_passes=64, search_budget_ms=600000,
                  max_parallel_searches=2)
    write(a.out / 'policy.json', policy)

    def prepare(path):
        config = read(path)
        cell = f"{config['model']}_b{config['batch']}"
        directory = a.out / cell
        directory.mkdir()
        packed = ROOT / f"configs/serving/sm90d1/inputs/{config['model']}/decode.export.json.xz"
        expected = inputs[str(packed.relative_to(ROOT))]
        if sha(packed) != expected['compressed_sha256']:
            raise ValueError('repository export changed')
        graph = directory / 'decode.export.json'
        graph.write_bytes(lzma.decompress(packed.read_bytes()))
        if sha(graph) != expected['raw_sha256']:
            raise ValueError('export decompression changed')
        output = directory / 'predicted.cu'
        command = [str(compiler), 'compile', str(graph), str(output), '--solve', str(target),
                   '--serving', 'decode', '--emit', 'serving', '--batch', str(config['batch']),
                   '--seed-arch', 'native', '--selection', 'predicted', '--search-passes', '64',
                   '--search-budget-ms', '600000']
        with (directory / 'search.log').open('w') as log:
            status = subprocess.call(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                     stdout=log, stderr=subprocess.STDOUT)
        write(directory / 'command.json', dict(command=command, exit_code=status))
        if status:
            raise RuntimeError(f'{cell}: diagnostic search failed; retained search.log')
        search_path = Path(str(output) + '.search') / 'receipt.json'
        search = read(search_path)
        baseline = ROOT / f"runs/sm90r2/build/{a.baseline_tag}/{config['model']}-B{config['batch']}/decode.so"
        identity = read(str(baseline) + '.identity.json')
        calibration = read(target)['sm90_dense_decode']
        if sha(baseline) != identity['binary_sha256'] or identity['trace']:
            raise ValueError('baseline must be an unchanged uninstrumented artifact')
        if any(identity['source']['source_files'].get(k) != v
               for k, v in calibration['native_headers'].items()):
            raise ValueError('baseline and search use different native implementations')
        selected, seen, seen_coordinates = [], set(), set()

        def add(coordinates, origin, rank=None, prediction=None, binary=None):
            key = schedule_key(coordinates, prediction)
            coordinate_key = json.dumps(coordinates, sort_keys=True)
            if key in seen or coordinate_key in seen_coordinates or len(selected) == 8:
                return
            seen.add(key)
            seen_coordinates.add(coordinate_key)
            coordinate_path = directory / f'candidate-{len(selected)}.tmexec.json'
            write(coordinate_path, coordinates)
            selected.append(dict(index=len(selected), origin=origin, rank=rank,
                coordinates=str(coordinate_path), coordinates_sha256=sha(coordinate_path),
                prediction=prediction, reused_binary=str(binary) if binary else None,
                binary_sha256=sha(binary) if binary else None))

        ranked = search['ranked']
        baseline_coordinates = read(str(baseline) + '.plan.json')['native_dense']
        baseline_prediction = next((r['prediction'] for r in ranked
                                    if r['coordinates'] == baseline_coordinates), None)
        add(baseline_coordinates, 'development_control', prediction=baseline_prediction, binary=baseline)
        for row in ranked:
            if len(selected) == 4:
                break
            add(row['coordinates'], 'predicted_top_three', row['rank'], row['prediction'])
        for origin in ('resident1_exact', 'resident1_preissue', 'resident2_exact', 'resident2_preissue'):
            for row in ranked:
                if row['origin'] == origin:
                    add(row['coordinates'], origin, row['rank'], row['prediction'])
        for row in ranked:
            add(row['coordinates'], 'ranked_fill', row['rank'], row['prediction'])
        if len(selected) != 8:
            raise ValueError('search did not provide eight distinct coordinates')
        if sha(compiler) != compiler_sha or sha(target) != policy['target_sha256']:
            raise ValueError('search implementation or calibration changed')
        result = dict(model=config['model'], batch=config['batch'], config=str(path.resolve()),
            config_sha256=sha(path), candidates=selected, search_receipt=str(search_path),
            search_receipt_sha256=sha(search_path), evaluated=search['evaluated'],
            search_ms=search['search_ms'], converged=search['converged'],
            budget_exhausted=search['budget_exhausted'], purpose='E-3 development ranking; not E-4 or final M')
        write(directory / 'cohort.json', result)
        print(json.dumps({k: result[k] for k in ('model', 'batch', 'evaluated', 'search_ms', 'converged')}), flush=True)
        return str(directory / 'cohort.json')

    with ThreadPoolExecutor(max_workers=2) as pool:
        cohorts = list(pool.map(prepare, a.config))
    write(a.out / 'manifest.json', dict(policy=policy, cohorts=cohorts))


if __name__ == '__main__':
    main()
