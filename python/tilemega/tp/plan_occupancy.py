"""Query the same dense TP kernel on every guarded device without launching it."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path

from tilemega.mx5_plan_occupancy import Library, Runtime, query, record, validate_inputs
from tilemega.serving.measure_tp import require_group


def group_evidence():
    devices = require_group()
    ancestors = {}
    pid = os.getppid()
    while pid > 1 and pid not in ancestors:
        base = Path('/proc') / str(pid)
        ancestors[pid] = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        pid = int((base / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    locks = Path('/proc/locks').read_text().splitlines()
    owners = []
    for gpu in devices:
        path = Path(f'/tmp/tilemega_mx_gpu{gpu}.lock')
        info = path.stat()
        device = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
        matches = []
        for line in locks:
            fields = line.split()
            if (len(fields) >= 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE']
                    and fields[5] == device and int(fields[4]) in ancestors
                    and 'tilemega.bench.gpu_guard' in ancestors[int(fields[4])]):
                matches.append(dict(pid=int(fields[4]), lock=str(path), evidence=line))
        if len(matches) != 1:
            raise ValueError('group guard does not own the physical GPU lock')
        owners.append(matches[0])
    if len({owner['pid'] for owner in owners}) != 1:
        raise ValueError('one guard must own the entire TP group')
    return dict(gpu_indices=devices, owners=owners)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('preserve prior occupancy evidence')
    guard = group_evidence()
    validated = validate_inputs(json.loads(args.inputs.read_text()))
    runtime, library = Runtime(), Library(validated['paths']['binary'])
    world = library.handle.tm_plan_tp_world
    world.argtypes, world.restype = [], C.c_uint
    if world() != len(guard['gpu_indices']):
        raise ValueError('binary TP degree differs from the guarded group')
    set_device = runtime.handle.wcudaSetDevice
    set_device.argtypes, set_device.restype = [C.c_int], C.c_int
    rows = []
    for rank, gpu in enumerate(guard['gpu_indices']):
        if set_device(rank):
            raise RuntimeError('cudaSetDevice failed in occupancy query')
        row = query(validated, runtime, library, guard)
        row.update(rank=rank, physical_gpu=gpu)
        rows.append(row)
    result = dict(input_manifest=record(args.inputs), ranks=rows,
                  qualified=all(row['query_status'] == 0 and row['resident_ctas'] == 1
                                and row['attributes']['localSizeBytes'] == 0
                                and row['attributes']['sharedSizeBytes'] == 0 for row in rows))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
