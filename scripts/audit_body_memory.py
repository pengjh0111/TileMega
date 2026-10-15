#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Count source-attributed SASS memory operations without executing a GPU."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

INSTRUCTION = re.compile(r'/\*([0-9a-fA-F]+)\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b')
MEMORY_SPACES = {'LDG': 'global', 'STG': 'global', 'LDS': 'shared',
                 'STS': 'shared', 'LDL': 'local', 'STL': 'local',
                 'LD': 'generic', 'ST': 'generic'}


def memory_space(opcode):
    # LDGDEPBAR synchronizes async copies; it does not load global memory.
    return MEMORY_SPACES.get(opcode.split('.')[0])


def attribute(dumps, bodies):
    instructions = {}
    for _, lines in dumps:
        function = ''; frames = []; pending = []; was_frame = False
        for line in lines:
            if '.section\t.text.' in line:
                function = line.strip(); frames = []; pending = []; was_frame = False
            if '//## File ' in line:
                if not was_frame:
                    pending = []
                pending.extend(re.findall(r'"([^"]+)"', line)); was_frame = True
                continue
            match = INSTRUCTION.search(line)
            if not match:
                continue
            if was_frame:
                frames = pending; was_frame = False
            opcode = match[2]
            if memory_space(opcode) is None:
                continue
            key = (function, match[1], opcode)
            row = instructions.setdefault(key, dict(opcode=opcode, owners=set()))
            owners = {body for body in bodies if any(Path(frame).stem == body for frame in frames)}
            row['owners'].update(owners)
    counts = {body: collections.Counter() for body in bodies}
    for row in instructions.values():
        for owner in row['owners']:
            counts[owner][row['opcode']] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sass', type=Path, action='append', required=True)
    parser.add_argument('--generated-cu', type=Path, required=True)
    parser.add_argument('--body-stage', action='append', required=True, metavar='BODY=KIND')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mapping = dict(item.split('=', 1) for item in args.body_stage)
    source = args.generated_cu.read_bytes()
    stages = collections.Counter(re.findall(r'\{TaskKind::(\w+),', source.decode()))
    hashes = {}; streams = []
    try:
        for path in args.sass:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            streams.append((path, gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()))
        counts = attribute(streams, mapping)
    finally:
        for _, stream in streams:
            stream.close()
    rows = []
    for body, count in counts.items():
        present = stages.get(mapping[body], 0)
        rows.append(dict(body=body, declared_stages=present,
            memory_opcodes=dict(count),
            vector128_global_shared_instructions=sum(value for opcode, value in count.items()
                if memory_space(opcode) in ('global', 'shared') and '128' in opcode.split('.')),
            local_instructions=sum(value for opcode, value in count.items()
                if memory_space(opcode) == 'local'),
            production_coverage='declared stage; execution/elision unknown' if present else
                'absent stage; generic dispatcher instructions do not establish production use'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(evidence='verified static source attribution only',
        generated_source=str(args.generated_cu),
        generated_source_sha256=hashlib.sha256(source).hexdigest(), sass_sha256=hashes,
        stage_counts=dict(stages), bodies=rows, gpu_execution=False,
        limits=['Static instruction counts do not measure performance or dynamic vector coverage.',
                'Binary equivalence and source line provenance require separate receipts.',
                'Local 128-bit accesses are not counted as global/shared vectorization.']), indent=2)+'\n')


if __name__ == '__main__':
    main()
