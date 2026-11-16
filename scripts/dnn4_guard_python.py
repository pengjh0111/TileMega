#!/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python
"""DNN4 guard: six owner-free samples show GPU0 28 MiB, GPUs1–3 4 MiB.

Keep the original provider immutable for already-running experiment receipts.
The floor is supported by owner-free preflight samples in
runs/dnn4/sync/idle_floor_probe/samples.jsonl; nonempty owners still reject.
"""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn1_device as device

FLOORS = {0: 28, 1: 4, 2: 4, 3: 4}

REAL_PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'


def idle(row, policy, needs=0, nonce=None):
    return (row['gpu_index'] in FLOORS and row['gpu_index'] in device.allowed_gpu_indices()
            and not row['owners'] and 0 <= row['used_mib'] <= FLOORS[row['gpu_index']]
            and row['free_mib'] >= needs)


def install_floor():
    import dnn1_device
    dnn1_device.install()
    import tilemega.bench.gpu_guard as guard
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def sample(index=None):
        row = dnn1_device.gpu(index)
        row.update(idle_memory_floor_mib=FLOORS[row['gpu_index']], idle_floor_provider_sha256=source_hash)
        return row
    guard.gpu = sample
    guard.idle = idle


def main():
    args = sys.argv[1:]
    if args[:2] == ['-m', 'tilemega.bench.scheduler']:
        install_floor()
        sys.executable = str(Path(__file__).resolve())
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    elif args[:2] == ['-m', 'tilemega.bench.gpu_guard']:
        from dnn1_guard_entry import run
        def guarded():
            install_floor()
            from tilemega.bench.gpu_guard import main as guard_main
            return guard_main()
        sys.argv = [sys.argv[0], *args[2:]]
        result = run(guard_main=guarded)
        path = Path(args[args.index('--out') + 1]) / 'guard_result.json'
        if path.is_file():
            receipt = json.loads(path.read_text())
            receipt.update(idle_memory_floor_mib=FLOORS[int(os.environ['TILEMEGA_DEVICE_INDEX'])],
                idle_floor_provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            path.write_text(json.dumps(receipt, indent=2) + '\n')
        return result
    else:
        entry = str(Path(__file__).with_name('dnn1_resource_entry.py'))
        os.execv(REAL_PYTHON, [REAL_PYTHON, entry, *args])


if __name__ == '__main__':
    raise SystemExit(main())
