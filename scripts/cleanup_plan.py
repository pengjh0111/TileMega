#!/usr/bin/env python3
"""Prepare a conservative local cleanup dry-run; never delete any path."""
import argparse
import csv
import json
import os
import pwd
from pathlib import Path


def overlaps(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def plan(inventory, rows, *, active_jobs=(), owner=None):
    owner = pwd.getpwuid(os.getuid()).pw_name if owner is None else owner
    root = Path(inventory['repository'])
    protected = [(Path(row['path']), 'model-or-dataset') for row in rows if row['category'] == 'E']
    protected.extend([(root, 'repository-root'), (root / '.git', 'git-metadata')])
    for audit in inventory['worktrees']:
        tree = Path(audit['root'])
        if audit.get('stash') or audit.get('unpushed') or 'tracked_changes' not in audit:
            protected.append((tree, 'unpublished-code-or-incomplete-audit'))
        protected.extend((Path(change['path']), 'tracked-edit') for change in audit.get('tracked_changes', []))
        for key in ('untracked_source', 'ignored_source'):
            protected.extend((Path(path), 'untracked-source') for path in audit.get(key, []))
    output, candidates = [], []
    for row in sorted(rows, key=lambda r: (len(Path(r['path']).parts), r['path'])):
        item = dict(row)
        path = Path(row['path'])
        reason = None
        if row['category'] == 'E':
            reason = 'model-or-dataset'
        elif row['owner'] != owner:
            reason = 'different-owner'
        elif active_jobs:
            reason = 'project-jobs-active'
        elif path.is_symlink() or path.resolve() != path:
            reason = 'symlink-review'
        else:
            for protected_path, why in protected:
                # The main checkout stays; its expendable descendants may be
                # candidates. Other protected trees exclude all descendants.
                if why == 'repository-root':
                    blocked = path == protected_path or protected_path.is_relative_to(path)
                else:
                    blocked = overlaps(path, protected_path)
                if blocked:
                    reason = why
                    break
        if reason:
            item.update(action='keep' if row['category'] == 'E' else 'blocked', reason=reason)
        else:
            parent = next((p for p in candidates if path.is_relative_to(p)), None)
            item.update(action='covered' if parent else 'candidate',
                        reason=f'covered-by:{parent}' if parent else 'review-before-deletion')
            if parent is None:
                candidates.append(path)
        output.append(item)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--scheduler-state', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text())
    root = Path(inventory['repository']).resolve()
    if not args.out.resolve().is_relative_to(root / 'runs'):
        raise ValueError('Dry-run output must be inside the ignored runs/ directory')
    with args.inventory.with_name('cleanup_manifest.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    active = []
    for path in args.scheduler_state:
        state = json.loads(path.read_text())
        active.extend(name for name, row in state.items() if row['status'] == 'running')
    result = plan(inventory, rows, active_jobs=active)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'cleanup_manifest.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], 'reason'] if rows else ['path', 'action', 'reason'], delimiter='\t')
        writer.writeheader()
        writer.writerows(result)
    candidates = [r for r in result if r['action'] == 'candidate']
    summary = dict(mode='dry-run-only', active_jobs=active, inventory=str(args.inventory.resolve()),
        candidates=len(candidates), candidate_bytes=sum(int(r['bytes']) for r in candidates),
        actions={action: sum(r['action'] == action for r in result)
                 for action in ('keep', 'blocked', 'covered', 'candidate')},
        limitations=['No path is deleted. Candidates still require ownership and project-scope review.',
                    'Repeat inventory, scheduler and unpublished-code checks immediately before deletion.'])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
