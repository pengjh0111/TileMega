#!/usr/bin/env python3
"""Copy completed physical variant probes, excluding executables and locks.

The cache is shared across arms and may be warm. This archive preserves the
original command/source/log/resources; it does not count copies as compiles.
Run again after all solves finish to include probes created during the search.
"""
import argparse
import fcntl
import hashlib
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
FILES = ('command.json', 'probe.cu', 'build.log', 'resources.json')


def archive(cache, output):
    manifest = []
    output.mkdir(parents=True, exist_ok=True)
    for entry in sorted(cache.iterdir()):
        if not entry.is_dir() or not (entry/'resources.json').exists():
            continue
        with (entry/'compile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            resource = json.loads((entry/'resources.json').read_text())
            if any(field not in resource for field in ('registers', 'shared_bytes', 'threads')):
                raise ValueError('incomplete resource result: '+str(entry))
            destination = output/entry.name
            destination.mkdir(exist_ok=True)
            for name in FILES:
                source = entry/name
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                shutil.copy2(source, destination/name)
                if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('resource archive differs: '+str(source))
                manifest.append(dict(source=str(source), archived=str((destination/name).relative_to(output)), sha256=digest))
    (output/'manifest.json').write_text(json.dumps(dict(cache=str(cache), files=manifest), indent=2)+'\n')
    print(f'RESOURCE_ARCHIVE probes={len(manifest)//len(FILES)} files={len(manifest)} output={output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('/root/r9_work/variant_resources'))
    parser.add_argument('--out', type=pathlib.Path, default=HERE/'variant_resources')
    args = parser.parse_args()
    archive(args.cache, args.out)
