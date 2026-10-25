"""Attribute one actual inlined MX-5 program by compiler source provenance.

This never derives program counts from isolated bodies. A copied debug build
must reproduce the original executable text, allocated ELF sections, resource
record and normalized final-MIR instruction stream before its provenance is
mapped onto the original. Full debug stacks and residual buckets are retained.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from tilemega.build.identity import parse_resources, verify
from tilemega.mx5_resources import device_elfs, kernel_instructions

STAGES = {'Qkv': 'qkv', 'Attn': 'attention', 'O': 'o', 'GateUp': 'gate_up',
          'Down': 'down', 'DownLast': 'down', 'Head': 'head'}
OPCODE = re.compile(r'^\s+(?:[^=\n]*=\s+)?(?:(?:contract|nuw|nsw|exact|nnan|ninf|nsz|arcp|afn|reassoc|disjoint)\s+)*([A-Z][A-Z0-9_]+)')
LOCATION = re.compile(r'([^\s@\[\]]+):(\d+):(\d+)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def elf_sections(elf):
    if elf[:6] != b'\x7fELF\x02\x01':
        raise ValueError('ELF64 little-endian device image required')
    offset, = struct.unpack_from('<Q', elf, 40)
    stride, count, strings = struct.unpack_from('<HHH', elf, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', elf, offset + i * stride) for i in range(count)]
    names = elf[rows[strings][4]:rows[strings][4] + rows[strings][5]]
    result = []
    for row in rows:
        name = names[row[0]:names.index(b'\0', row[0])].decode()
        result.append(dict(name=name, type=row[1], flags=row[2], address=row[3],
                           offset=row[4], size=row[5], link=row[6], entry_size=row[9],
                           data=elf[row[4]:row[4] + row[5]] if row[1] != 8 else b''))
    return result


def kernel_symbol(elf, sections):
    matches = set()
    for section in sections:
        if section['type'] not in (2, 11):
            continue
        strings = sections[section['link']]['data']
        for offset in range(0, section['size'], section['entry_size']):
            name, info, other, index, value, size = struct.unpack_from('<IBBHQQ', section['data'], offset)
            name = strings[name:strings.index(b'\0', name)].decode()
            if info & 15 == 2 and 'mx_decode_kernel_v2' in name and '__device_stub__' not in name:
                matches.add((name, value, size))
    if len(matches) != 1:
        raise ValueError('one actual v2 kernel symbol required')
    name, start, size = matches.pop()
    return dict(name=name, begin=start, end=start + size, size=size)


def normalize_warnings(text):
    counts = sorted(set(re.findall(r'^(\d+) warnings generated when compiling for [^\n]+\.', text, re.M)))
    pattern = r'(?:' + '|'.join(map(re.escape, counts or ['-1'])) + r') warnings generated when compiling for [^\n]+\.\n'
    return re.sub(pattern, '', text)


def mir_rows(text, name):
    marker = '# Machine code for function ' + name + ':'
    begin = text.index(marker)
    end = text.index('# End machine code', begin)
    rows, block = [], None
    first_line = text[:begin].count('\n') + 1
    for index, line in enumerate(text[begin:end].splitlines()):
        match = re.match(r'^bb\.(\d+)', line)
        if match:
            block = int(match[1])
        opcode = OPCODE.match(line) if block is not None else None
        if opcode is None:
            continue
        location = line.split('debug-location', 1)[1].split(';', 1)[1] if 'debug-location' in line and ';' in line.split('debug-location', 1)[1] else ''
        frames = [dict(FileName=m[1], Line=int(m[2]), Column=int(m[3])) for m in LOCATION.finditer(location)]
        # Memory descriptions and metadata IDs are compiler annotations, not encoded operands.
        canonical = re.split(r'(?:,\s*|\s+)debug-location', line, maxsplit=1)[0].split(' ::', 1)[0].strip()
        canonical = re.sub(r'!\d+', '!MD', canonical)
        rows.append(dict(index=len(rows), mir_line=first_line + index, block=block,
                         opcode=opcode[1], canonical=canonical, text=line, inline_stack=frames))
    return rows


def attach_raw_mir_lines(rows, raw, normalized):
    counts = sorted(set(re.findall(r'^(\d+) warnings generated when compiling for [^\n]+\.', raw, re.M)))
    pattern = r'(?:' + '|'.join(map(re.escape, counts or ['-1'])) + r') warnings generated when compiling for [^\n]+\.\n'
    removed, size = [], 0
    for match in re.finditer(pattern, raw):
        removed.append((match.start() - size, len(match[0])))
        size += len(match[0])
    starts = [m.start() for m in re.finditer(r'^', normalized, re.M)]
    newlines = [m.start() for m in re.finditer('\n', raw)]
    for row in rows:
        row['normalized_mir_line'] = row['mir_line']
        offset = starts[row['mir_line'] - 1]
        raw_offset = offset + sum(length for position, length in removed if position <= offset)
        row['mir_line'] = bisect_right(newlines, raw_offset) + 1


class Classifier:
    def __init__(self, generated, include_root):
        self.generated = generated.resolve()
        self.calls = {}
        self.entries = set()
        for number, line in enumerate(generated.read_text().splitlines(), 1):
            match = re.search(r'RunStageV2<(\w+)>', line)
            if match:
                following = re.search(r'\[&\].*?return\s+(\w+)::Prefetch', line)
                if match[1] not in STAGES or following is None or following[1] not in STAGES:
                    raise ValueError('unsupported generated stage/callback syntax')
                self.calls[number] = dict(stage=STAGES[match[1]], body_type=match[1],
                                          next=STAGES[following[1]], callback_begin=line.index('[&]') + 1,
                                          callback_end=line.rindex('}') + 1, source=line.strip())
            elif re.search(r'return\s+Qkv::Prefetch', line):
                self.entries.add(number)
        program = include_root / 'tilemega/Codegen/tasks/mx/StageProgram.cuh'
        self.program = program.resolve()
        self.body_return = {i for i, line in enumerate(program.read_text().splitlines(), 1)
                            if 'return Body::Run(' in line}
        if not self.calls or len(self.body_return) != 1:
            raise ValueError('unrecognized stage wrapper source')

    def classify(self, frames):
        usable = [f for f in frames if f.get('FileName')]
        if not usable:
            return 'unattributed', 'no compiler source location'
        generated = [f for f in usable if Path(f['FileName']).resolve() == self.generated]
        sites = {f['Line'] for f in generated if f['Line'] in self.calls}
        if len(sites) > 1 or any(f['Line'] == 0 for f in generated):
            return 'shared', 'multiple or merged generated call-site locations'
        if sites:
            site = self.calls[next(iter(sites))]
            nearest = next(f for f in generated if f['Line'] in sites)
            if site['callback_begin'] <= nearest.get('Column', 0) <= site['callback_end']:
                return 'prefetch_edge:' + site['stage'] + '->' + site['next'], 'nested next-prefetch callback context'
            body = any(Path(f['FileName']).resolve() == self.program and f['Line'] in self.body_return for f in usable)
            if body:
                return 'body:' + site['stage'], 'unique current-body inline call-site context'
            return 'setup:' + site['stage'], 'operand/control setup at generated stage call'
        if any(f['Line'] in self.entries for f in generated):
            return 'entry_prefetch:qkv', 'entry weight prefetch and its operand setup'
        body_headers = ('DecodeGemmV2.h', 'DecodeAttentionV2.h', 'DecodeHeadV2.h', 'ServingGemmMma.h')
        if any(Path(f['FileName']).name in body_headers for f in usable):
            return 'shared', 'body/header provenance without a unique generated caller'
        if any(f.get('Line', 0) for f in usable):
            return 'control', 'program, barrier, loop or other non-body provenance'
        return 'unattributed', 'debug stack has no nonzero source line'


def run_readelf(elf, out, kind):
    command = ['readelf', '--debug-dump=' + kind, str(elf)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    (out / ('readelf_' + kind + '.txt')).write_text(result.stdout)
    if result.stderr:
        (out / ('readelf_' + kind + '.stderr')).write_text(result.stderr)
    return result.stdout


def pc_boundaries(elf, kernel, out):
    lines = run_readelf(elf, out, 'decodedline')
    ranges = run_readelf(elf, out, 'Ranges')
    info = run_readelf(elf, out, 'info')
    versions = re.findall(r'^\s*Version:\s*(\d+)', info, re.M)
    if not versions or set(versions) != {'4'}:
        raise ValueError('this audited range-boundary reader requires DWARF4; retain dumps and extend explicitly')
    points = {kernel['begin'], kernel['end']}
    for match in re.finditer(r'\s(0x[0-9a-f]+)(?:\s|$)', lines, re.M):
        points.add(int(match[1], 16))
    for match in re.finditer(r'^\s*[0-9a-f]+\s+([0-9a-f]{8,16})\s+([0-9a-f]{8,16})\s*$', ranges, re.M):
        points.update(int(value, 16) for value in match.groups())
    # Include both forms of high_pc (absolute address or offset). Extra cuts
    # only subdivide intervals; they never assign bytes to a different caller.
    low = None
    for line in info.splitlines():
        if 'Abbrev Number:' in line:
            low = None
        match = re.search(r'DW_AT_low_pc\s*:\s*(0x[0-9a-f]+)', line)
        if match:
            low = int(match[1], 16)
            points.add(low)
        match = re.search(r'DW_AT_high_pc\s*:\s*(0x[0-9a-f]+)', line)
        if match:
            high = int(match[1], 16)
            points.add(high)
            if low is not None:
                points.add(low + high)
    return sorted(p for p in points if kernel['begin'] <= p <= kernel['end'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-so', 'debug-so', 'original-mir', 'debug-mir', 'generated',
                 'original-generated', 'original-source-manifest', 'include-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--symbolizer', type=Path, default=Path('/opt/maca/mxgpu_llvm/bin/llvm-symbolizer'))
    parser.add_argument('--original-identity', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a new attribution directory')
    args.out.mkdir(parents=True)
    original_images = device_elfs(args.original_so.read_bytes())
    debug_images = device_elfs(args.debug_so.read_bytes())
    if len(original_images) != 1 or len(debug_images) != 1 or original_images[0][0] != debug_images[0][0]:
        parser.error('one corresponding device image is required')
    original, debug = original_images[0][1], debug_images[0][1]
    original_sections, debug_sections = elf_sections(original), elf_sections(debug)
    def content(sections, allocated=False):
        return {s['name']: (s['address'], s['size'], s['data']) for s in sections
                if (s['flags'] & 2 if allocated else s['name'] == '.text' or s['name'].startswith('.text.'))}
    kernel = kernel_symbol(debug, debug_sections)
    if kernel != kernel_symbol(original, original_sections):
        parser.error('kernel symbol/range changed')
    original_raw, debug_raw = args.original_mir.read_text(), args.debug_mir.read_text()
    original_text, debug_text = normalize_warnings(original_raw), normalize_warnings(debug_raw)
    original_rows, debug_rows = mir_rows(original_text, kernel['name']), mir_rows(debug_text, kernel['name'])
    attach_raw_mir_lines(original_rows, original_raw, original_text)
    attach_raw_mir_lines(debug_rows, debug_raw, debug_text)
    if len(original_rows) != kernel_instructions(original_text)[kernel['name']] or len(debug_rows) != kernel_instructions(debug_text)[kernel['name']]:
        parser.error('MIR parser does not cover the existing resource-counter total')
    original_resource = parse_resources(original_text)[kernel['name']]
    debug_resource = parse_resources(debug_text)[kernel['name']]
    source_manifest = json.loads(args.original_source_manifest.read_text())
    original_identity_path = args.original_identity or Path(str(args.original_so) + '.identity.json')
    verified_identity = verify(args.original_so)
    original_identity = json.loads(original_identity_path.read_text())
    if original_identity != verified_identity:
        parser.error('explicit original identity differs from the verified artifact sidecar')
    original_bound = (sha(args.original_so.read_bytes()) == original_identity['binary_sha256'] and
                      sha(args.original_generated.read_bytes()) == original_identity['generated_source_sha256'] and
                      source_manifest['source_files'] == original_identity['source']['source_files'])
    header_equal = all(sha((args.include_root / path.removeprefix('include/')).read_bytes()) == expected
                       for path, expected in source_manifest['source_files'].items() if path.startswith('include/'))
    threads_match = re.search(r'kThreads\s*=\s*(512|576)', args.generated.read_text())
    if threads_match is None:
        parser.error('generated thread shape is not explicit')
    cap = 255 if int(threads_match[1]) == 512 else 227
    contract = (debug_resource['registers'] <= cap and debug_resource['stack_bytes'] == 0 and
                debug_resource['static_smem_bytes'] == 0)
    proof = dict(original_device_elf_sha256=sha(original), debug_device_elf_sha256=sha(debug),
                 original_artifact_identity_bound=original_bound,
                 original_mir_identity_bound=sha(args.original_mir.read_bytes()) == original_identity['resources_sha256'],
                 original_artifact_id=original_identity.get('artifact_id'),
                 strict_resource_contract_pass=contract, register_cap=cap,
                 device_elf_equal=original == debug,
                 executable_text_identical=content(original_sections) == content(debug_sections),
                 allocated_sections_identical=content(original_sections, True) == content(debug_sections, True),
                 original_text=[dict(section=s['name'], address=s['address'], bytes=s['size'], sha256=sha(s['data']))
                                for s in original_sections if s['name'] == '.text' or s['name'].startswith('.text.')],
                 resources_identical=original_resource == debug_resource,
                 resources=debug_resource, generated_source_identical=args.generated.read_bytes() == args.original_generated.read_bytes(),
                 frozen_include_files_identical=header_equal,
                 normalized_final_mir_identical=[r['canonical'] for r in original_rows] == [r['canonical'] for r in debug_rows],
                 original_mir_instructions=len(original_rows), debug_mir_instructions=len(debug_rows))
    save(args.out / 'identity_proof.json', proof)
    required = ('original_artifact_identity_bound', 'original_mir_identity_bound', 'strict_resource_contract_pass',
                'executable_text_identical', 'allocated_sections_identical', 'resources_identical',
                'generated_source_identical', 'frozen_include_files_identical', 'normalized_final_mir_identical')
    if not all(proof[k] for k in required):
        parser.error('debug build cannot attribute the original program; comparison retained, no counts transferred')
    debug_elf = args.out / 'debug_device.elf'
    debug_elf.write_bytes(debug)
    classifier = Classifier(args.generated, args.include_root)
    mir_counts = Counter()
    with (args.out / 'mir_instruction_stacks.jsonl').open('w') as stream:
        for original_row, row in zip(original_rows, debug_rows):
            bucket, reason = classifier.classify(row['inline_stack'])
            mir_counts[bucket] += 1
            stream.write(json.dumps(dict(original_mir_line=original_row['mir_line'], bucket=bucket,
                                         reason=reason, **row)) + '\n')
    boundaries = pc_boundaries(debug_elf, kernel, args.out)
    command = [str(args.symbolizer), '--obj=' + str(debug_elf), '--inlining', '--output-style=JSON']
    result = subprocess.run(command, input=''.join(hex(p) + '\n' for p in boundaries[:-1]),
                            capture_output=True, text=True, check=True)
    if result.stderr:
        (args.out / 'symbolizer.stderr').write_text(result.stderr)
    symbols = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(symbols) != len(boundaries) - 1:
        parser.error('symbolizer coverage mismatch')
    pc_counts = Counter()
    with (args.out / 'pc_interval_stacks.jsonl').open('w') as stream:
        for begin, end, record in zip(boundaries, boundaries[1:], symbols):
            if int(record['Address'], 16) != begin:
                parser.error('symbolizer changed PC order')
            bucket, reason = classifier.classify(record['Symbol'])
            pc_counts[bucket] += end - begin
            stream.write(json.dumps(dict(begin=begin, end=end, bytes=end-begin,
                                         bucket=bucket, reason=reason, inline_stack=record['Symbol'])) + '\n')
    assert sum(mir_counts.values()) == len(original_rows)
    assert sum(pc_counts.values()) == kernel['size']
    for counts in (mir_counts, pc_counts):
        for bucket in ('control', 'shared', 'unattributed'):
            counts.setdefault(bucket, 0)
        for stage in set(STAGES.values()):
            counts.setdefault('body:' + stage, 0)
            counts.setdefault('setup:' + stage, 0)
    text_bytes = sum(s['size'] for s in debug_sections if s['name'] == '.text' or s['name'].startswith('.text.'))
    report = dict(schema=1, status='original_program_text_and_resources_verified', identity=proof, kernel=kernel,
                  definition='Static compiler source-provenance attribution. Body requires a unique generated current-stage caller through Body::Run. Nested callback work belongs to an explicit current->next prefetch edge, including callback operand construction. Stage argument setup and entry prefetch are separate. Control/shared/unattributed work is never distributed to stages.',
                  callsites=classifier.calls, entry_prefetch_lines=sorted(classifier.entries),
                  final_mir=dict(total=len(original_rows), buckets=dict(sorted(mir_counts.items())),
                                 unit='Final MIR instruction records, using the existing resource-report counter; includes pseudos/empty asm and is not a hardware disassembly count.'),
                  executable_pc_bytes=dict(kernel_bytes=kernel['size'], all_text_bytes=text_bytes,
                                           outside_kernel_functions_and_alignment_bytes=text_bytes-kernel['size'],
                                           interval_count=len(symbols), buckets=dict(sorted(pc_counts.items())),
                                           boundaries='Union of DWARF decoded-line addresses, inline/lexical range boundaries, low/high PC boundaries, and the actual kernel symbol range.'),
                  coverage_assertions=dict(mir_buckets_sum_to_original_total=True, pc_buckets_sum_to_kernel_symbol_size=True,
                                           pc_plus_outside_kernel_sum_to_text_size=True),
                  limitations=['Compiler debug provenance can choose one representative location for CSE/shared work; this is not proof of exclusive semantic ownership. Visible ambiguity is shared; missing metadata is unattributed.',
                               'DWARF PC coverage can inherit a preceding source location for compiler-generated waits or padding. It is line/inline-range provenance, not an instruction-by-instruction semantic ownership claim.',
                               'Final MIR instruction counts and executable PC byte coverage are independent; no fixed bytes-per-instruction ratio is assumed.',
                               'These are static counts, without multiplying layer or runtime loop counts.',
                               'Every other actual plan needs its own copied debug build and all identity checks; this representative does not establish any other plan.'],
                  evidence=dict(mir_stacks='mir_instruction_stacks.jsonl', pc_stacks='pc_interval_stacks.jsonl',
                                symbolizer_command=command,
                                input_sha256={str(p): sha(p.read_bytes()) for p in
                                              (args.original_so, args.debug_so, args.original_mir, args.debug_mir,
                                               args.generated, args.original_generated, args.original_source_manifest,
                                               original_identity_path, Path(__file__), args.symbolizer)}))
    save(args.out / 'attribution.json', report)
    print(json.dumps(dict(status=report['status'], mir=report['final_mir'], executable_pc_bytes=report['executable_pc_bytes'])))


if __name__ == '__main__':
    main()
