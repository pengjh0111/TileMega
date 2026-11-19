#!/usr/bin/env python3
"""Attach five strict SM80 MMA measurements to a new calibrated target."""
import argparse
import math
from pathlib import Path
import statistics

from collect_sm90r3_constants import process_rows
from sm90d1_common import ROOT, read, sha, write


SHAPE = ('tile_n', 'tile_k', 'warps_n', 'register_stages')
PROCESSES = 5


def collect(run, target, audit, base_hash):
    profile = target['sm80_native_constants']
    device = target['device_provenance']
    if (target['arch_tag'] != 'sm_80' or 'A800' not in device['name']
            or profile['gpu_uuid'] != device['uuid'] or 'mma_mainloop' in profile):
        raise ValueError('provide an A800 base calibration without an MMA table')
    jobs = read(run/'queue/queue_calibration.json')
    expected_names = {f'sm80-mainloop-f{i}' for i in range(PROCESSES)}
    if {job['name'] for job in jobs} != expected_names or any(
            job['artifact_sha256'] != audit['binary_sha256'] or job['target_sha256'] != base_hash
            or job['expected_uuid'] != device['uuid'] for job in jobs):
        raise ValueError('mainloop allocation does not match the audited binary, target and card')
    grid = int(target['resources']['num_sms'])
    shared = int(target['resources']['max_dynamic_smem_per_cta']) // 256 * 256
    expected = {tuple(row[key] for key in SHAPE) for row in audit['kernels']
                if row['production_compatible'] and 2 * (16 + row['tile_n']) * row['tile_k'] <= shared}
    records = {}
    for rows in process_rows(read(run/'results/state.json'), 'sm80-mainloop',
                             PROCESSES, run/'results', device['uuid']):
        fresh = {}
        for row in rows:
            shape = tuple(int(row[k]) for k in SHAPE)
            median, p99 = float(row['median']), float(row['p99'])
            if (row['metric'] != 'native_ldmatrix_mma_k16' or row['unit'] != 'ns'
                    or int(row['grid']) != grid or int(row['warps_k']) * shape[2] != 4
                    or int(row['samples']) <= 0 or not 0 < median <= p99 < math.inf
                    or shape in fresh):
                raise ValueError(f'invalid MMA calibration row for {shape}')
            fresh[shape] = median, p99
        if not expected or fresh.keys() != expected:
            raise ValueError('MMA process did not measure every audited production shape')
        for shape, sample in fresh.items():
            records.setdefault(shape, []).append(sample)
    return {'median_ns': {':'.join(map(str, (grid, *shape))): statistics.median(x[0] for x in values)
                          for shape, values in sorted(records.items())},
            'p99_ns': {':'.join(map(str, (grid, *shape))): statistics.median(x[1] for x in values)
                       for shape, values in sorted(records.items())},
            'processes': PROCESSES, 'gpu_uuid': device['uuid'], 'base_target_sha256': base_hash,
            'binary_sha256': audit['binary_sha256'], 'evidence': str(run.relative_to(ROOT)),
            'method': 'fresh process median, per K16 and split-K warp group, one CTA per SM'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('use a new output profile path')
    target = read(a.target)
    target['sm80_native_constants']['mma_mainloop'] = collect(
        a.run.resolve(), target, read(a.audit), sha(a.target))
    write(a.out, target)


if __name__ == '__main__':
    main()
