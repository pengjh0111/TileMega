"""Strict source/resource/SASS regression checks for template unification."""
import argparse
import difflib
import json
import os
from pathlib import Path
import re
import subprocess

from tilemega.build.identity import parse_resources, sha


def parse_sass(text):
    functions, name = {}, None
    for line in text.splitlines():
        match = re.search(r'Function\s*:\s*(\S+)', line)
        if match:
            name = match[1]
            if name in functions:
                raise ValueError('ambiguous SASS function; select one architecture')
            functions[name] = []
        elif name:
            # Remove instruction addresses and encodings, retaining opcodes,
            # registers, predicates and operands (including branch targets).
            instruction = re.sub(r'/\*.*?\*/', '', line).strip()
            if instruction and not instruction.startswith(('.', 'Fatbin', 'code for')):
                functions[name].append(re.sub(r'\s+', ' ', instruction))
    if not functions or not any(functions.values()):
        raise ValueError('no SASS instructions')
    return functions


def compare(reference, candidate, reference_sass, candidate_sass):
    reference, candidate = Path(reference), Path(candidate)
    old, new = parse_sass(reference_sass), parse_sass(candidate_sass)
    old_resources = parse_resources(Path(str(reference) + '.ptxas.log').read_text())
    new_resources = parse_resources(Path(str(candidate) + '.ptxas.log').read_text())
    if not old_resources or not new_resources:
        raise ValueError('no entry-kernel resource records')
    source_old, source_new = Path(str(reference) + '.cu'), Path(str(candidate) + '.cu')
    if not source_old.stat().st_size or not source_new.stat().st_size:
        raise ValueError('empty generated source')
    differences = {name: list(difflib.unified_diff(lines, new.get(name, [])))
                   for name, lines in old.items() if lines != new.get(name)}
    # parse_resources includes the called functions under each entry context;
    # an extra new kernel must not replace a callee in an existing kernel.
    resource_differences = {name: dict(reference=row, candidate=new_resources.get(name))
                            for name, row in old_resources.items()
                            if row != new_resources.get(name)}
    report = dict(source_equal=source_old.read_bytes() == source_new.read_bytes(),
                  sass_equal=not differences, resources_equal=not resource_differences,
                  reference_sha256=sha(reference), candidate_sha256=sha(candidate),
                  reference_source_sha256=sha(source_old), candidate_source_sha256=sha(source_new),
                  existing_functions=sorted(old), new_functions=sorted(set(new) - set(old)))
    report['pass'] = all(report[key] for key in ('source_equal', 'sass_equal', 'resources_equal'))
    return report, differences, resource_differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    dumps = []
    for label, so in [('reference', args.reference), ('candidate', args.candidate)]:
        text = subprocess.check_output([os.getenv('CUOBJDUMP', '/usr/local/cuda/bin/cuobjdump'),
                                        '--dump-sass', str(so)], text=True, timeout=300)
        (args.out / (label + '.sass')).write_text(text)
        dumps.append(text)
    report, instructions, resources = compare(args.reference, args.candidate, *dumps)
    for name, value in [('comparison', report), ('sass_differences', instructions),
                         ('resource_differences', resources)]:
        (args.out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
