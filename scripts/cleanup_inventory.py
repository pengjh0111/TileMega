#!/usr/bin/env python3
"""Read-only project inventory. This command never removes files."""
import argparse
import csv
import datetime
import json
import os
import pwd
import shutil
import subprocess
from pathlib import Path


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout


def size(path):
    """Count allocated files without following links to another user's tree."""
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, subdirs, files in os.walk(path, followlinks=False):
        subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except OSError:
                pass
    return total


def repository_audit(tree, status):
    """Include detached HEADs and ignored sources in the deletion review."""
    entries = iter(status.split(b'\0'))
    changes = []
    for entry in entries:
        if not entry:
            continue
        code, name = entry[:2], os.fsdecode(entry[3:])
        if code not in (b'??', b'!!'):
            changes.append(dict(status=code.decode(), path=str(tree / name)))
        if b'R' in code or b'C' in code:
            original = next(entries, b'')
            changes.append(dict(status='rename-source', path=str(tree / os.fsdecode(original))))
    suffixes = {'.py', '.cpp', '.h', '.cuh', '.cu', '.sh', '.cmake', '.td', '.mlir'}
    source_files, ignored_sources = [], []
    for ignored, target in ((False, source_files), (True, ignored_sources)):
        args = ['ls-files', '--others', '--exclude-standard', '-z']
        if ignored:
            args.append('--ignored')
        for name in git(tree, *args).split(b'\0'):
            if name and Path(os.fsdecode(name)).suffix in suffixes:
                target.append(str(tree / os.fsdecode(name)))
    return dict(root=str(tree),
        head=git(tree, 'rev-parse', 'HEAD').decode().strip(),
        stash=git(tree, 'stash', 'list').decode().splitlines(),
        # --branches alone misses a detached worktree's unique unpublished tip.
        unpushed=git(tree, 'log', 'HEAD', '--branches', '--not', '--remotes',
                     '--format=%H %s').decode().splitlines(),
        tracked_changes=changes, untracked_source=source_files, ignored_source=ignored_sources)


def collect(root, out):
    root, out = root.resolve(), out.resolve()
    if not out.is_relative_to(root / 'runs'):
        raise ValueError('Inventory output must be inside the ignored runs/ directory')
    out.mkdir(parents=True, exist_ok=True)
    rows, seen, audits = [], {}, []

    def add(path, category, action='review'):
        path = path.absolute()
        if not path.exists():
            return
        if str(path) in seen:
            # An ignored build/cache directory first appears in category A;
            # its more specific owner must still be represented.
            existing = seen[str(path)]
            priority = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'B': 4, 'E': 5}
            if priority[category] > priority[existing['category']]:
                existing.update(category=category, action=action)
            return
        stat = path.lstat()
        try:
            owner = pwd.getpwuid(stat.st_uid).pw_name
        except KeyError:
            owner = str(stat.st_uid)
        rows.append(dict(path=str(path), category=category, bytes=size(path), owner=owner,
            modified_utc=datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
            action=action))
        seen[str(path)] = rows[-1]

    worktrees = [Path(line[9:]) for line in git(root, 'worktree', 'list', '--porcelain').decode().splitlines()
                 if line.startswith('worktree ')]
    # These clones historically contain project work even when not registered
    # in the main repository's worktree metadata.
    for parent in (Path('/root/r10_work'), Path('/root/r11_work'), Path('/root/r12_work'),
                   Path('/root/r13_work'), Path('/root/r14_work'), Path('/root/dm1_work')):
        if parent.exists():
            add(parent, 'D')
            for marker in parent.glob('*/.git'):
                if marker.parent not in worktrees:
                    worktrees.append(marker.parent)
    for tree in worktrees:
        if not tree.exists():
            continue
        status = git(tree, 'status', '--porcelain=v1', '--ignored', '-z')
        (out / f'worktree_{len(audits)}_status.bin').write_bytes(status)
        entries = iter(status.split(b'\0'))
        for entry in entries:
            if not entry:
                continue
            code, name = entry[:2], os.fsdecode(entry[3:])
            if b'R' in code or b'C' in code:
                next(entries, None)
            if code in (b'??', b'!!'):
                path = tree / name
                if path == out or out.is_relative_to(path):
                    continue
                add(path, 'A')
        audits.append(repository_audit(tree, status))
        for name in ('build', 'build-dm'):
            add(tree / name, 'B')
        for path in tree.glob('build-*'):
            add(path, 'B')
        add(tree / 'runs', 'D')
        if tree != root:
            add(tree, 'D')
    for name in ('.cache/tilemega', '.cache/tilemega-dm', '.cache/torch', '.triton',
                 '.nv/ComputeCache', '.cache/vllm', '.cache/pip'):
        add(Path.home() / name, 'C')
    for path in Path('/tmp').glob('tilemega*'):
        if path.lstat().st_uid == os.getuid():
            add(path, 'C')
    for name in ('models', '.cache/huggingface', 'datasets', 'data/imagenetv2', 'data/SIDD', 'data/wikitext'):
        add(Path.home() / name, 'E', 'keep')
    device = root.stat().st_dev
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = [name for name in subdirs if name != '.git'
                      and not (Path(directory) / name).is_symlink()
                      and (Path(directory) / name).stat().st_dev == device]
        for name in files:
            path = Path(directory) / name
            try:
                if path.lstat().st_size > 1024 ** 3:
                    add(path, 'F')
                elif name == 'core' or name.startswith('core.'):
                    add(path, 'C')
            except OSError:
                pass
    tracked = git(root, 'ls-files', '-z').split(b'\0')
    tracked_bytes = 0
    count = 0
    for name in tracked:
        if not name:
            continue
        count += 1
        try:
            tracked_bytes += (root / os.fsdecode(name)).lstat().st_size
        except FileNotFoundError:
            pass
    with (out / 'cleanup_manifest.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('path', 'category', 'bytes', 'owner', 'modified_utc', 'action'), delimiter='\t')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['category'], r['path'])))
    # Categories can overlap; totals describe inventory entries, not reclaimed
    # disk space. A future deletion manifest must first remove nested entries.
    summary = dict(mode='read-only', overlapping_categories=True, repository=str(root),
        tracked_files=count, tracked_bytes=tracked_bytes, disk_usage=shutil.disk_usage(root)._asdict(),
        categories={c: dict(entries=sum(r['category'] == c for r in rows),
            bytes=sum(r['bytes'] for r in rows if r['category'] == c)) for c in 'ABCDEF'},
        largest_50=sorted(rows, key=lambda r: r['bytes'], reverse=True)[:50], worktrees=audits)
    (out / 'inventory.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('mode', 'tracked_files', 'tracked_bytes', 'categories')}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    collect(args.root, args.out or args.root / 'runs/r15')


if __name__ == '__main__':
    main()
