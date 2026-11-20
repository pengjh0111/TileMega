#!/usr/bin/env python3
"""Attach guarded SM80 MMA measurements to a new calibrated target."""
import argparse
import math
from pathlib import Path
import statistics

from collect_sm90r3_constants import process_rows
from sm90d1_common import ROOT, read, sha, write


SHAPE = ('tile_n', 'tile_k', 'warps_n', 'register_stages')
def collect(run, target, audit, base_hash, processes=5):
    device = target['device_provenance']
    if target['arch_tag'] != 'sm_80' or 'A800' not in device['name']:
        raise ValueError('require an A800 calibration target')
    jobs = read(run/'queue/queue_calibration.json')
    expected_names = {f'sm80-mainloop-f{i}' for i in range(processes)}
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
                             processes, run/'results', device['uuid']):
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
            'processes': processes, 'gpu_uuid': device['uuid'], 'base_target_sha256': base_hash,
            'binary_sha256': audit['binary_sha256'], 'evidence': str(run.relative_to(ROOT)),
            'method': 'fresh process median, per K16 and split-K warp group, one CTA per SM'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--allocation-target', type=Path,
                   help='supplement disjoint shapes measured on this guarded A800')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repetitions', type=int, choices=(1, 5), default=5)
    a = p.parse_args()
    if a.out.exists():
        p.error('use a new output profile path')
    target = read(a.target)
    profile = target['sm80_native_constants']
    if profile['gpu_uuid'] != target['device_provenance']['uuid']:
        raise ValueError('base target lost its original device provenance')
    if a.allocation_target is None:
        if 'mma_mainloop' in profile:
            raise ValueError('the base target already contains a mainloop table')
        profile['mma_mainloop'] = collect(a.run.resolve(), target, read(a.audit), sha(a.target), a.repetitions)
    else:
        probe = read(a.allocation_target)
        keys = ('name', 'memory_mib', 'capability', 'mig', 'driver')
        if (target['resources'] != probe['resources'] or
                any(target['device_provenance'][k] != probe['device_provenance'][k] for k in keys)):
            raise ValueError('mainloop supplement requires equivalent A800 resources and driver')
        table = profile['mma_mainloop']
        extra = collect(a.run.resolve(), probe, read(a.audit), sha(a.allocation_target), a.repetitions)
        if table['median_ns'].keys() & extra['median_ns'].keys():
            raise ValueError('supplement overlaps an existing measured shape')
        record = {k: v for k, v in extra.items() if k not in ('median_ns', 'p99_ns')}
        record['shape_keys'] = sorted(extra['median_ns'])
        record['parent_profile_sha256'] = sha(a.target)
        table.setdefault('supplements', []).append(record)
        for name in ('median_ns', 'p99_ns'):
            table[name].update(extra[name])
    write(a.out, target)


if __name__ == '__main__':
    main()
