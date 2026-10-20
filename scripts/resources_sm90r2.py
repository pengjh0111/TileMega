#!/usr/bin/env python3
"""Report native static-entry and outlined-function SASS resource facts."""
import argparse
import csv
import re
import subprocess
from pathlib import Path

from sm90d1_common import read, sha


def inspect(binary):
    binary = binary.resolve()
    identity = read(str(binary) + '.identity.json')
    if sha(binary) != identity['binary_sha256']:
        raise ValueError('library identity changed')
    directory = binary.parent / 'resource-audit' / identity['binary_sha256'][:16]
    directory.mkdir(parents=True, exist_ok=True)
    assembly = directory / 'sass.txt'
    if not assembly.exists():
        with assembly.open('w') as log:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)],
                           stdout=log, check=True, timeout=120)
    sections = {}
    for text in re.split(r'Function : ', assembly.read_text())[1:]:
        name = text.splitlines()[0].strip()
        sections[name] = [(int(m[1], 16), m[2]) for m in
                          re.finditer(r'/\*([0-9a-f]+)\*/([^\n]*);', text)]
    cubins = list(directory.glob('*.cubin'))
    if not cubins:
        subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--extract-elf', 'all', str(binary)],
                       cwd=directory, check=True, stdout=subprocess.DEVNULL, timeout=120)
        cubins = list(directory.glob('*.cubin'))
    if not cubins:
        raise ValueError('the library contains no cubin')
    symbols = '\n'.join(subprocess.check_output(['readelf', '-sW', str(cubin)], text=True)
                        for cubin in cubins)
    rows = []
    for entry, properties in identity['resources'].items():
        if 'sm90_dense_' not in entry:
            continue
        functions = [(entry, 0, None, properties, 'entry')]
        for line in symbols.splitlines():
            words = line.split()
            if len(words) != 8 or words[3] != 'FUNC' or not words[7].startswith('$' + entry + '$'):
                continue
            name = words[7][len(entry) + 2:]
            functions.append((name, int(words[1], 16), int(words[2]),
                              properties.get('callees', {}).get(name, {}), 'body'))
        for name, begin, size, facts, kind in functions:
            instructions = [text for address, text in sections[entry]
                            if address >= begin and (size is None or address < begin + size)]
            regs = [int(x) for text in instructions for x in re.findall(r'\bR(\d+)\b', text)]
            text = '\n'.join(instructions)
            rows.append(dict(model_batch=binary.parent.name, source_head=identity['source']['head'],
                binary_sha256=identity['binary_sha256'], entry=entry, kind=kind, function=name,
                ptxas_registers=facts.get('registers', ''),
                sass_highest_register_plus_one=max(regs) + 1 if regs else 0,
                stack_bytes=facts.get('stack_bytes', ''),
                spill_store_bytes=facts.get('spill_store_bytes', ''),
                spill_load_bytes=facts.get('spill_load_bytes', ''),
                shared_bytes=identity['shared_memory_bytes'] if kind == 'entry' else '',
                ldl_stl=len(re.findall(r'\b(?:LDL|STL)\b', text)),
                traps=len(re.findall(r'\bBPT\.TRAP\b', text)), instructions=len(instructions)))
    if not rows or any(row['kind'] == 'body' and row['instructions'] == 0 for row in rows):
        raise ValueError('missing native function boundaries')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = [row for binary in a.input for row in inspect(binary)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} entry/function records -> {a.out}')


if __name__ == '__main__':
    main()
