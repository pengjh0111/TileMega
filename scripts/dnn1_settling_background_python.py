#!/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python
"""Isolated bounded preflight waiting; shared guard/provider files stay intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import dnn1_background_python as original

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'python/tilemega/bench/gpu_guard.py': 'd3dd77988770010a585ecbfdc02f1ecac904e14ffba030852c165ed98f56a84b',
    'scripts/dnn1_preflight_settle_guard.patch': 'da62e633441e65489305afdf520e16f1fa040116eadc79500232a582603a41ca',
    'scripts/dnn1_preflight_settle.py': 'b7fecbc137df195f72aa7711a2391feffc3100901eeee5a3d22158d6526d4a5d',
    'scripts/dnn1_background_python.py': '7b3476f22480f15dd0a2556b49d9a885373e631e2c64946c6ef20ab2d64928e6',
}
PATCHED_SHA = '0405675175d910c5077a736d2ff315410951ebf774f1fb55365fad0107e943e2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_guard(out):
    for relative, digest in SOURCES.items():
        if sha(ROOT / relative) != digest:
            raise ValueError('preflight dependency changed: ' + relative)
    folder = Path(out) / 'preflight_source'
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / 'gpu_guard.py'
    subprocess.run(['patch', '--silent', '--output', str(source),
        str(ROOT / 'python/tilemega/bench/gpu_guard.py'),
        str(ROOT / 'scripts/dnn1_preflight_settle_guard.patch')], check=True)
    if sha(source) != PATCHED_SHA:
        raise ValueError('isolated preflight guard differs from CPU-validated source')
    manifest = folder / 'identity.json'
    manifest.write_text(json.dumps(dict(schema='bounded-preflight-settle-v1',
        source=str(source), source_sha256=PATCHED_SHA, dependencies=SOURCES,
        memory_floor_mib=1, required_samples='original policy.samples=6',
        interval_s='original policy.interval_s=5', timeout_s=120,
        owners='any owner yields before launch',
        reservations='any new native reservation yields before launch',
        observations='all retained; high memory resets; no final reread',
        shared_files_modified=False, power_monitoring=False), indent=2) + '\n')
    return source, manifest


def main():
    args = sys.argv[1:]
    manifest = None
    if args[:2] == ['-m', 'tilemega.bench.gpu_guard']:
        from tilemega.bench import gpu_guard
        if Path(gpu_guard.__file__).resolve() != ROOT / 'python/tilemega/bench/gpu_guard.py':
            raise ValueError('guard imported from an unexpected source root')
        out = Path(args[args.index('--out') + 1])
        source, manifest = render_guard(out)
        exec(compile(source.read_text(), str(source), 'exec'), vars(gpu_guard))
    # The unchanged scheduler propagates this entry to each guarded retry.
    original.__file__ = str(Path(__file__).resolve())
    result = original.main()
    if manifest is not None:
        path = out / 'guard_result.json'
        if path.exists():
            value = json.loads(path.read_text())
            value.update(preflight_protocol='bounded-preflight-settle-v1',
                         preflight_source_identity=str(manifest),
                         preflight_source_identity_sha256=sha(manifest))
            record = out / 'preflight_settle.json'
            if record.exists():
                value.update(preflight_observations=str(record),
                             preflight_observations_sha256=sha(record))
            path.write_text(json.dumps(value, indent=2) + '\n')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
