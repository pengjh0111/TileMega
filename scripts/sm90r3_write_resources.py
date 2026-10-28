#!/usr/bin/env python3
"""T8: per-artifact resources of every shipped SM90R3 native program.

For each artifact: the entry kernel's linked registers, stack, spill and
static shared memory (from its identity), the dynamic shared memory it
launches with, the entry's SASS instruction count (cuobjdump, CPU only), and
the noinline bodies ptxas reports beside the entry with their own registers,
stack and spill. Nothing here touches a GPU.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CUOBJDUMP = '/usr/local/cuda/bin/cuobjdump'


def bodies(log):
    """Per-function ptxas properties: name -> registers, stack, spill."""
    result, name = {}, None
    for line in Path(log).read_text(errors='replace').splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            name = m[1]
            result.setdefault(name, {})
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and name:
            result[name].update(stack=int(m[1]), spill_store=int(m[2]), spill_load=int(m[3]))
            continue
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            name = m[1]
            result.setdefault(name, {})
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and name:
            result[name]['registers'] = int(m[1])
    return result


def instructions(binary, entry):
    out = subprocess.run([CUOBJDUMP, '-sass', '-fun', entry, str(binary)],
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if re.match(r'\s+/\*[0-9a-f]{4,}\*/\s+\S', line))


def row(kind, model, batch, binary):
    binary = Path(binary)
    identity = json.loads(Path(str(binary) + '.identity.json').read_text())
    entry_name, entry = next(iter(identity['resources'].items()))
    functions = bodies(str(binary) + '.ptxas.log')
    callees = {k: v for k, v in functions.items() if k != entry_name and 'registers' in v}
    worst = max(callees.values(), key=lambda v: v['registers'], default=None)
    return dict(kind=kind, model=model, batch=batch, artifact=str(binary),
                artifact_id=identity.get('artifact_id'),
                entry_registers=entry['registers'], entry_stack_bytes=entry['stack_bytes'],
                entry_spill_store_bytes=entry['spill_store_bytes'],
                entry_spill_load_bytes=entry['spill_load_bytes'],
                static_smem_bytes=entry.get('static_smem_bytes'),
                dynamic_smem_bytes=identity.get('shared_memory_bytes'),
                entry_sass_instructions=instructions(binary, entry_name),
                noinline_bodies=len(callees),
                max_body_registers=worst['registers'] if worst else None,
                max_body_stack_bytes=max((v.get('stack', 0) for v in callees.values()), default=0),
                body_spill_bytes=sum(v.get('spill_store', 0) + v.get('spill_load', 0)
                                     for v in callees.values()),
                allocation_source=entry.get('allocation_source', 'ptxas'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dnn', nargs='*', default=[], help='model=<plan.so>')
    p.add_argument('--out', type=Path, default=ROOT / 'docs/results/SM90R3_resources.csv')
    a = p.parse_args()
    rows = []
    for path in sorted((ROOT / 'configs/serving/sm90r3').glob('*.json')):
        cell = json.loads(path.read_text())
        binary = cell['solve_command'][3]
        if Path(binary).is_file():
            rows.append(row('llm', cell['model'], cell['batch'], binary))
    for item in a.dnn:
        model, _, binary = item.partition('=')
        rows.append(row('dnn', model, 1, binary))
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(table=str(a.out), rows=len(rows),
                          spilling=[r['model'] + f"-B{r['batch']}" for r in rows
                                    if r['entry_spill_store_bytes'] or r['body_spill_bytes']]), indent=1))


if __name__ == '__main__':
    main()
