#!/usr/bin/env python3
"""Inventory documentation dependencies without moving or removing inputs."""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
import json
from pathlib import Path
import re
import subprocess


SCOPES = ('CMakeLists.txt', 'cmake', 'test', 'tools', 'python', 'scripts',
          'configs', 'include', 'lib')
KEPT = {'docs/DEPENDENCIES.md', 'docs/BUILD_MLIR.md', 'docs/TOOLS.md'}
DIRECT = re.compile(r'(?:/[\w.-]+)*/docs/[^\s\"\'`);,]+|docs/[^\s\"\'`);,]+')
LITERAL = re.compile(r"(['\"])([^'\"\r\n]*)\1")


def git(root, *args, allow_empty=False):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if result.returncode and not (allow_empty and result.returncode == 1):
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout


def references(line):
    """Recognize literal paths and joined literals; leave expressions for review."""
    found = []
    for match in DIRECT.finditer(line):
        value = match.group().rstrip('.:')
        rest = line[match.end():]
        dynamic = (value.endswith('/') or bool(re.search(r'[$*?{}<>\[\]]', value))
                   or bool(re.match(r"['\"]\s*\+", rest)))
        found.append((value, dynamic))
    literals = list(LITERAL.finditer(line))
    for index, token in enumerate(literals):
        if token.group(2) != 'docs':
            continue
        parts, end = ['docs'], token.end()
        for following in literals[index + 1:]:
            if not re.fullmatch(r'\s*[/,+]\s*', line[end:following.start()]):
                break
            parts.append(following.group(2).strip('/'))
            end = following.end()
        rest = line[end:]
        dynamic = len(parts) == 1 or bool(re.match(r'\s*(?:\)\s*)?[/,+]', rest))
        found.append(('/'.join(parts) + ('/' if len(parts) == 1 else ''), dynamic))
    return list(dict.fromkeys(found))


def classify(source, line, reference):
    if line.lstrip().startswith(('#', '//', '/*', '*')):
        return 'comment', 'rewrite citation'
    if source.startswith('test/') and re.search(r'\.(mkdir|write_text|write_bytes)\(', line):
        return 'synthetic_test_path', 'review synthetic fixture'
    if source.endswith('.json') and re.search(r'"(?:source|method|note|provenance)"\s*:', line):
        return 'provenance', 'preserve provenance without a runtime dependency'
    if reference in KEPT or reference.startswith('docs/results/'):
        return 'retained_document', 'keep'
    if re.search(r'(?:see |recorded in )[^\n]*docs/', line):
        return 'diagnostic_link', 'update help link'
    if '/requirements-' in reference:
        return 'requirements', 'requirements/'
    if '/prompts/' in reference:
        return 'prompts', 'configs/prompts/'
    if 'measurement_policy' in reference or 'guard_policy' in reference:
        return 'measurement_policy', 'configs/bench/'
    if '/MODELS/sources/' in reference:
        return 'model_config', 'configs/models/'
    if (reference.endswith('.so') or '.so.' in reference
            or (source.startswith('configs/e2e/')
                and reference.endswith(('/plan.json', '/classes.tsv')))):
        return 'pinned_artifact', 'remove historical pin or replace with a build input'
    if reference.endswith('.json') and ('target' in reference.lower() or '/fit/' in reference):
        return 'target_fixture', 'test/fixtures/targets/ or configs/targets/'
    if reference.endswith(('.cpp', '.cu', '.cuh', '.h', '.py', '.sh', '.cmake', '/CMakeLists.txt')):
        return 'tool_source', 'review usage before migrating to source tree or retiring'
    if (source.startswith('test/') or source == 'CMakeLists.txt'
            or source == 'scripts/check_full_checkpoint.py'):
        return 'test_fixture', 'test/fixtures/<topic>/'
    return 'unclassified', 'manual review'


def inspect(root, tracked, source, number, line, reference, dynamic):
    # C++ often appends "/docs/..." to the repository root. An absolute
    # path with another parent is kept distinct and never opened by the audit.
    outside = False
    if reference.startswith(str(root) + '/docs/'):
        relative = reference[len(str(root)) + 1:]
    elif reference.startswith('/docs/'):
        relative = reference[1:]
    elif reference.startswith('/'):
        relative, outside = reference, True
    else:
        relative = reference
    kind, suggested = classify(source, line, relative)
    base = re.split(r'[$*?{}<>\[\]]', relative, maxsplit=1)[0]
    if base != relative and not base.endswith('/'):
        base = str(Path(base).parent)
    path = root / base
    if not outside and not path.resolve().is_relative_to(root):
        outside = True
    exists = False if outside else path.exists()
    prefix = base.rstrip('/') + '/'
    children = bisect_left(tracked, prefix + '\uffff') - bisect_left(tracked, prefix)
    exact_tracked = relative in tracked
    if outside:
        status = 'external_path'
    elif dynamic:
        status = 'dynamic_review'
    elif not exists:
        status = 'missing'
    elif path.is_dir():
        status = 'directory_review'
    elif exact_tracked:
        status = 'tracked_file'
    else:
        status = 'untracked_file'
    return dict(source=source, line=number, reference=reference, relative=relative,
        status=status, kind=kind, suggested=suggested, exists=exists,
        tracked=exact_tracked, tracked_children=children,
        bytes=path.stat().st_size if exists and path.is_file() else None,
        expression=line.strip())


def collect(root, out):
    root, out = root.resolve(), out.resolve()
    if not out.is_relative_to(root / 'runs'):
        raise ValueError('Audit output must be inside the ignored runs/ directory')
    tracked = sorted(p.decode(errors='surrogateescape') for p in
        git(root, 'ls-files', '-z', '--', 'docs').split(b'\0') if p)
    raw = git(root, 'grep', '-n', '-I', '-z', '-F', '-e', 'docs/', '-e', "'docs'",
              '-e', '"docs"', '--', *SCOPES, allow_empty=True)
    rows, matched_lines = [], 0
    for record in raw.splitlines():
        filename, number, content = record.split(b'\0', 2)
        source, line = filename.decode(), content.decode(errors='replace')
        matched_lines += 1
        for reference, dynamic in references(line):
            rows.append(inspect(root, tracked, source, int(number), line, reference, dynamic))
    summary = dict(mode='read-only', root=str(root),
        head=git(root, 'rev-parse', 'HEAD').decode().strip(),
        matched_lines=matched_lines, references=len(rows),
        source_files=len({r['source'] for r in rows}),
        statuses=dict(sorted(Counter(r['status'] for r in rows).items())),
        kinds=dict(sorted(Counter(r['kind'] for r in rows).items())),
        limitations=['Textual scan of tracked files only; no semantic dependency proof.',
            'Dynamic expressions and directories require manual review.',
            'Multiline joins may appear only as an unresolved docs/ prefix.',
            'Migration suggestions are categories, not approved moves or deletions.'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dependencies.json').write_text(json.dumps(dict(summary=summary, references=rows), indent=2) + '\n')
    with (out / 'dependencies.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
            ['source', 'line', 'reference', 'status'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    collect(args.root, args.out or args.root / 'runs/r15/cleanup_dependencies')


if __name__ == '__main__':
    main()
