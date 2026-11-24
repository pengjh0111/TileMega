"""Build phase-1 stage probes and emit three-process queue manifests."""

import argparse
import concurrent.futures
import os
from pathlib import Path

from calibration_queue import CALIB
from guarded import ROOT, read, run, sha, write


def build(suite, bound):
    source = Path(suite['source'])
    binary = Path(suite['binary'])
    prefix = source.parent / 'build_receipts' / suite['id']
    parents = sorted(prefix.glob('attempt*'))
    for prior in parents:
        file = prior / 'build.json'
        if file.exists():
            info = read(file)
            assert sha(binary) == info['binary_sha256']
            assert sha(source) == info['source_sha256'] and bound == info['header_sha256']
            return info
    attempt = prefix / f'attempt{len(parents):04d}'
    attempt.mkdir(parents=True)
    temporary = attempt / 'probe'
    command = ['nvcc', '-std=c++17', '-O3', '-gencode', 'arch=compute_90a,code=sm_90a',
               '--expt-relaxed-constexpr', '--split-compile', '4', '-Iinclude',
               '-Ithird_party/cutlass/include', '-Iexperiments/search_quality_v2',
               source, '-lcuda', '-o', temporary]
    if run(command, attempt, dict(os.environ, CUDA_VISIBLE_DEVICES='')):
        raise RuntimeError(f'probe build failed at {attempt}')
    # An earlier manual compile was not part of the calibrated cohort.
    if binary.exists():
        binary.rename(attempt / 'preexisting-probe')
    temporary.rename(binary)
    info = {'source_sha256': sha(source), 'binary_sha256': sha(binary),
            'header_sha256': bound, 'build_attempt': str(attempt)}
    write(attempt / 'build.json', info)
    print(suite['id'], 'built', flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['gemm', 'attention'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    directory = CALIB / args.kind
    suites = read(directory / 'suites.json')
    bound = read(CALIB / 'body_binding.json')['header_sha256']
    bound.update({
        f'experiments/search_quality_v2/{name}': sha(ROOT / 'experiments/search_quality_v2' / name)
        for name in ('gemm_probe.cuh', 'attention_probe.cuh')
    })
    for path, digest in bound.items():
        assert sha(ROOT / path) == digest, path
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        builds = list(pool.map(lambda suite: build(suite, bound), suites))
    jobs = []
    for repeat in range(3):
        for suite, info in zip(suites, builds, strict=True):
            jobs.append({'id': f'{suite["id"]}-f{repeat}', 'command': [suite['binary']],
                         'binary_sha256': info['binary_sha256'], 'geometry_ids': suite['geometry_ids'],
                         'build_attempt': info['build_attempt'], 'process_round': repeat})
    write(directory / 'queue.json', jobs)


if __name__ == '__main__':
    main()
