#!/usr/bin/env python3
"""Remove only unchanged, completed, regenerable files registered by AS-1."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import stat

AS1_ROOT = Path('/home/d50064638/junhuip')
DEFAULT_RUN = AS1_ROOT / 'TileMega/runs/as1'
DEFAULT_WORK = AS1_ROOT / 'tilemega-as1'
PROTECTED = {'inventory', 'probe', 'gate', 'result', 'failure', 'source', 'config'}


def project_paths(project):
    if project not in {'as1', 'as2'}:
        raise ValueError('Unknown controlled project')
    runs = AS1_ROOT / ('TileMega/runs/' + project)
    work = AS1_ROOT / ('tilemega-' + project)
    roots = [runs, *(work / n for n in ('build', 'cache', 'tmp'))]
    if project == 'as2':
        roots.append(work / 'weights')
    return runs, work, roots


def plain_path(path, roots):
    path = Path(os.path.abspath(path))
    root = next((p for p in roots if path.is_relative_to(p) and path != p), None)
    if root is None:
        raise ValueError('Outside controlled subtree: ' + str(path))
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError('Symlink path is protected: ' + str(path))
    if path.resolve() != path:
        raise ValueError('Noncanonical path: ' + str(path))
    return path


def file_stamp(path):
    s = path.stat(follow_symlinks=False)
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def process_alive(ref):
    try:
        text = Path('/proc', str(int(ref['pid'])), 'stat').read_text()
    except FileNotFoundError:
        return False
    except (PermissionError, KeyError, ValueError):
        return True
    fields = text[text.rfind(')')+2:].split()
    if fields[0] == 'Z':
        return False
    return 'start_ticks' not in ref or int(fields[19]) == int(ref['start_ticks'])


def plan(entries, roots, keep_recent=2):
    latest = {str(e['path']): e for e in entries}
    builds = {}
    for e in latest.values():
        if e.get('category') == 'build' and e.get('completed') and e.get('build_id'):
            builds[e['build_id']] = max(builds.get(e['build_id'], ''), e.get('created_at', ''))
    recent = set(sorted(builds, key=lambda b: (builds[b], b), reverse=True)[:keep_recent])
    result = []
    for raw, e in latest.items():
        item = {'path': raw, 'action': 'keep', 'bytes': 0}
        try:
            p = plain_path(Path(raw), roots)
            if p.name in {'manifest.jsonl', 'cleanup_history.jsonl'}:
                raise ValueError('Audit file')
            if e.get('keep', True) or e.get('selected') or e.get('category') in PROTECTED:
                raise ValueError('Retained evidence or selected artifact')
            if not e.get('completed') or not e.get('regenerable'):
                raise ValueError('Incomplete or not regenerable')
            if e.get('build_id') in recent:
                raise ValueError('One of the most recent complete builds')
            if 'active_refs' not in e or any(process_alive(r) for r in e['active_refs']):
                raise ValueError('Active or unknown process references')
            if not p.exists():
                raise ValueError('Already absent')
            s = p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                raise ValueError('Only ordinary single-link files may be removed')
            if e.get('stamp') != file_stamp(p):
                raise ValueError('File changed since registration')
            item.update(action='remove', bytes=s.st_size, stamp=e['stamp'])
        except (ValueError, OSError) as error:
            item['reason'] = str(error)
        result.append(item)
    return result


def apply_item(item, roots):
    p = plain_path(Path(item['path']), roots)
    # Open the parent without following symlinks and recheck immediately before unlink.
    parent_fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        s = os.stat(p.name, dir_fd=parent_fd, follow_symlinks=False)
        current = dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or current != item['stamp']:
            raise ValueError('File changed before unlink: ' + str(p))
        os.unlink(p.name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def main(project='as1'):
    default_run, default_work, roots = project_paths(project)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    parser.add_argument('--run-root', type=Path, default=default_run)
    parser.add_argument('--work-root', type=Path, default=default_work)
    parser.add_argument('--keep-recent', type=int, default=2)
    parser.add_argument('--report-json', type=Path)
    args = parser.parse_args()
    if args.run_root.absolute() != default_run or args.work_root.absolute() != default_work:
        parser.error('Roots must match the selected controlled personal project')
    if args.keep_recent < 2:
        parser.error('At least two complete builds must be retained')
    for root in roots:
        for parent in [root, *root.parents]:
            if parent.is_symlink():
                parser.error('Controlled roots must not traverse symlinks')
    manifest = args.run_root / 'manifest.jsonl'
    entries = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    items = plan(entries, roots, args.keep_recent)
    report = {'time': dt.datetime.now(dt.timezone.utc).isoformat(),
              'mode': 'apply' if args.apply else 'dry-run', 'items': items,
              'eligible_bytes': sum(i['bytes'] for i in items if i['action'] == 'remove'),
              'removed_files': 0, 'released_bytes': 0}
    if args.apply:
        # Re-evaluate all eligibility conditions, including PID references.
        check = {i['path']: i for i in plan(entries, roots, args.keep_recent)}
        for item in items:
            if item['action'] == 'remove' and check[item['path']]['action'] == 'remove':
                apply_item(item, roots)
                report['removed_files'] += 1
                report['released_bytes'] += item['bytes']
        with (args.run_root / 'cleanup_history.jsonl').open('a') as history:
            history.write(json.dumps(report)+'\n')
    if args.report_json:
        out = plain_path(args.report_json, [args.run_root, args.work_root / 'reports'])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'items'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
