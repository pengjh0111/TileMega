"""Attribute native stage code using a separately compiled, byte-identical image."""
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import re
import shlex
import signal
import subprocess
import time

from .build.identity import snapshot, validate_snapshot_inputs, sha
from .mx5_resources import device_elfs, inspect, kernel_instructions, text_sections

NAMES = dict(Qkv='qkv', Attn='attention', O='o', Router='router', GateUp='gate_up',
             Down='down', DownLast='down', Head='head')
OPCODE = re.compile(r'^\s+(?:[^=\n]*=\s+)?(?:(?:contract|nuw|nsw|exact|nnan|ninf|nsz|arcp|afn|reassoc|disjoint)\s+)*([A-Z][A-Z0-9_]+)')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def partition_ranges(size, ranges):
    events = defaultdict(list)
    for owner, lo, hi in ranges:
        if not 0 <= lo < hi <= size:
            raise ValueError('inlined PC range is outside the actual text section')
        events[lo].append((owner, 1)); events[hi].append((owner, -1))
    events[0]; events[size]
    active, result, previous = Counter(), Counter(), 0
    for address, updates in sorted(events.items()):
        owners = [name for name, count in active.items() if count]
        bucket = owners[0] if len(owners) == 1 else 'shared' if owners else 'control'
        result[bucket] += address - previous
        for name, delta in updates:
            active[name] += delta
            if active[name] < 0:
                raise ValueError('unbalanced inline ranges')
        previous = address
    if any(active.values()) or sum(result.values()) != size:
        raise ValueError('inline byte partition does not conserve actual code size')
    return dict(result)


def profile(normal, annotated, normal_log, annotated_log, source, root, kernel):
    from elftools.elf.elffile import ELFFile
    from elftools.dwarf.ranges import BaseAddressEntry
    normal_images, annotated_images = (device_elfs(Path(p).read_bytes()) for p in (normal, annotated))
    if len(normal_images) != 1 or len(annotated_images) != 1:
        raise ValueError('source attribution requires one native device image')
    sections = text_sections(normal_images[0][1])
    if sections != text_sections(annotated_images[0][1]):
        raise ValueError('annotated device code differs; preserve both artifacts')
    if len(sections) != 1 or sections[0]['name'] != '.text':
        raise ValueError('source attribution requires the exact single text section')
    source, root = Path(source).resolve(), Path(root).resolve()
    calls = {}
    for line, text in enumerate(source.read_text().splitlines(), 1):
        match = re.search(r'mx::RunStageV3<(\w+)>', text)
        if match:
            calls[line] = NAMES[match[1]]
    if not calls:
        raise ValueError('generated source has no concrete V3 stage calls')
    elf = ELFFile(BytesIO(annotated_images[0][1]))
    base = elf.get_section_by_name('.text')['sh_addr']
    dwarf, ranges, sites = elf.get_dwarf_info(), [], []
    source_spellings = {str(source), os.path.relpath(source, root)}
    for cu in dwarf.iter_CUs():
        if cu['version'] != 4:
            raise ValueError('this MACA attribution protocol requires DWARF4')
        program = dwarf.line_program_for_CU(cu)
        for die in cu.iter_DIEs():
            if die.tag != 'DW_TAG_inlined_subroutine':
                continue
            origin = die.get_DIE_from_attribute('DW_AT_abstract_origin')
            if not origin.attributes['DW_AT_name'].value.startswith(b'RunStageV3<'):
                continue
            line = die.attributes['DW_AT_call_line'].value
            file = program['file_entry'][die.attributes['DW_AT_call_file'].value-1]
            comp_dir = cu.get_top_DIE().attributes['DW_AT_comp_dir'].value.decode()
            directory = program['include_directory'][file.dir_index-1].decode() if file.dir_index else comp_dir
            resolved = dwarf_file_path(file.name.decode(), directory, comp_dir)
            if resolved != source or line not in calls:
                raise ValueError('inline stage call does not bind the generated source')
            source_spellings.add(file.name.decode())
            owner, intervals = calls[line], []
            if 'DW_AT_ranges' in die.attributes:
                address = cu.get_top_DIE().attributes['DW_AT_low_pc'].value
                entries = dwarf.range_lists().get_range_list_at_offset(die.attributes['DW_AT_ranges'].value, cu=cu)
                for entry in entries:
                    if isinstance(entry, BaseAddressEntry):
                        address = entry.base_address
                    else:
                        delta = 0 if entry.is_absolute else address
                        intervals.append((entry.begin_offset+delta-base, entry.end_offset+delta-base))
            else:
                lo, hi = die.attributes['DW_AT_low_pc'].value, die.attributes['DW_AT_high_pc']
                intervals = [(lo-base, hi.value-base if hi.form == 'DW_FORM_addr' else lo+hi.value-base)]
            ranges.extend((owner, lo, hi) for lo, hi in intervals)
            sites.append(dict(line=line, owner=owner, ranges=intervals))
    code = partition_ranges(sections[0]['bytes'], ranges)
    text = Path(annotated_log).read_text()
    starts = list(re.finditer(r'^# Machine code for function (\S+?):', text, re.M))
    matches = [m for m in starts if kernel in m[1] and '__device_stub__' not in m[1]]
    if len(matches) != 1:
        raise ValueError('expected one annotated native kernel')
    start = matches[0]; end = text.find('# End machine code', start.end())
    if end < 0:
        raise ValueError('unterminated annotated MIR')
    paths = sorted(source_spellings)
    location = re.compile(r'(?:'+'|'.join(re.escape(p) for p in paths)+r'):(\d+):\d+')
    counts, in_block = Counter(), False
    for line in text[start.end():end].splitlines():
        if re.match(r'^bb\.\d+', line):
            in_block = True
        if in_block and OPCODE.match(line):
            owners = {calls[int(v)] for v in location.findall(line) if int(v) in calls}
            owner = next(iter(owners)) if len(owners) == 1 else 'shared' if owners else 'control'
            counts[owner] += 1
    total = sum(kernel_instructions(Path(normal_log).read_text(), kernel).values())
    if total != sum(counts.values()) or any(not code.get(name) or not counts.get(name) for name in set(calls.values())):
        raise ValueError('annotated stage/MIR partition is incomplete')
    return dict(body_instruction_status='verified_same_text_source_attribution', body_instructions=dict(counts),
        body_code_bytes=code, program_instruction_count=total, device_text=sections, sites=sites,
        body_definition='lexical RunStageV3 call, including outgoing prefetch and arrival; shared/control are separate',
        instruction_definition='static MIR opcodes, including MIR pseudo instructions; not dynamic hardware counts',
        byte_definition='union of inlined DWARF PC ranges; uncovered text includes control and alignment',
        annotated_device_text_bitwise_equal=True, timing_binary_changed=False)


def dwarf_file_path(name, directory, comp_dir):
    path = Path(name)
    if not path.is_absolute():path = Path(directory) / path
    if not path.is_absolute():path = Path(comp_dir) / path
    return path.resolve()


def annotate(so, log, kernel):
    so, log = Path(so).resolve(), Path(log).resolve()
    source = Path(str(so)+'.cu')
    context = json.loads(Path(str(so)+'.source.json').read_text())
    root = Path(context['root'])
    args = shlex.split(Path(str(so)+'.build_command.txt').read_text())
    if args.count('-o') != 1 or Path(args[args.index('-o')+1]).resolve() != so or str(source) not in args:
        raise ValueError('attribution command does not bind the normal artifact and source')
    version = importlib.metadata.version('pyelftools')
    if version != '0.32':
        raise ValueError('install experiments/mx6/requirements.txt for attribution')
    auxiliary = Path(str(so)+'.attribution.so'); auxiliary_log = Path(str(auxiliary)+'.ptxas.log')
    receipt = Path(str(so)+'.attribution.json')
    for path in (auxiliary, auxiliary_log, receipt):
        if path.exists():
            raise ValueError('preserve existing attribution artifact: '+str(path))
    before = [reference(p) for p in (so, log, source, str(so)+'.build_command.txt', str(so)+'.source.json')]
    captured = snapshot(root, str(auxiliary)+'.source.json', compiler=args[0])
    args[args.index('-o')+1] = str(auxiliary)
    args.append('--generate-line-info')
    start = time.monotonic()
    with auxiliary_log.open('x') as output:
        process = subprocess.Popen(args, cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = process.wait(timeout=1800)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            raise
    build = dict(argv=args, returncode=rc, compile_s=time.monotonic()-start,
                 primary_inputs=before, helper=reference(__file__), pyelftools=version)
    Path(str(auxiliary)+'.compile.json').write_text(json.dumps(build, indent=2)+'\n')
    if rc:
        raise ValueError('annotated compilation failed; artifacts retained')
    validate_snapshot_inputs(root, captured)
    if before != [reference(p['path']) for p in before]:
        raise ValueError('normal artifact inputs changed during attribution')
    result = profile(so, auxiliary, log, auxiliary_log, source, root, kernel)
    result.update(build, annotated=reference(auxiliary), annotated_log=reference(auxiliary_log))
    with receipt.open('x') as output:
        json.dump(result, output, indent=2); output.write('\n')
    return result, reference(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('so', 'log', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--threads', type=int, choices=[512], required=True)
    parser.add_argument('--kernel', choices=['mx_moe_kernel_v3'], required=True)
    args = parser.parse_args()
    report = inspect(args.so, args.log, args.threads, args.kernel)
    if not report['violations']:
        result, receipt = annotate(args.so, args.log, args.kernel)
        for key in ('body_instruction_status', 'body_instructions', 'body_code_bytes', 'program_instruction_count'):
            report[key] = result[key]
        report['body_attribution'] = receipt
        report['attribution_compile_s'] = result['compile_s']
    with args.out.open('x') as output:
        json.dump(report, output, indent=2); output.write('\n')
    return bool(report['violations'])


if __name__ == '__main__':
    raise SystemExit(main())
