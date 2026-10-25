"""Read MX-5 resources and device code from one actual cucc artifact, without CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from .build.identity import parse_resources


def device_elfs(data):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    base = data.find(magic)
    if base < 0:
        raise ValueError('artifact lacks a clang offload bundle')
    cursor = base + len(magic)
    count, = struct.unpack_from('<Q', data, cursor)
    cursor += 8
    if not 1 <= count <= 256:
        raise ValueError('invalid offload bundle count')
    result = []
    for _ in range(count):
        offset, size, length = struct.unpack_from('<QQQ', data, cursor)
        cursor += 24
        if length > len(data) - cursor:
            raise ValueError('truncated offload target name')
        name = data[cursor:cursor + length].decode()
        cursor += length
        if offset > len(data) - base or size > len(data) - base - offset:
            raise ValueError('truncated offload image')
        body = data[base + offset:base + offset + size]
        if body.startswith(b'\x7fELF') and name.startswith('maca-'):
            result.append((name, body))
    if not result:
        raise ValueError('offload bundle lacks a device ELF')
    return result


def text_sections(elf):
    if elf[:6] != b'\x7fELF\x02\x01':
        raise ValueError('device image must be ELF64 little endian')
    section_offset, = struct.unpack_from('<Q', elf, 40)
    entry_size, count, strings_index = struct.unpack_from('<HHH', elf, 58)
    if entry_size < 64 or not count or strings_index >= count:
        raise ValueError('unsupported device ELF section table')
    if section_offset > len(elf) or count * entry_size > len(elf) - section_offset:
        raise ValueError('truncated device ELF section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', elf, section_offset + i * entry_size) for i in range(count)]
    strings = sections[strings_index]
    if strings[4] > len(elf) or strings[5] > len(elf) - strings[4]:
        raise ValueError('truncated device ELF string table')
    names = elf[strings[4]:strings[4] + strings[5]]
    result = []
    for section in sections:
        end = names.find(b'\0', section[0])
        if end < 0:
            raise ValueError('invalid device ELF section name')
        name = names[section[0]:end].decode()
        if name == '.text' or name.startswith('.text.'):
            if section[4] > len(elf) or section[5] > len(elf) - section[4]:
                raise ValueError('truncated device code section')
            result.append({'name': name, 'bytes': section[5],
                           'sha256': hashlib.sha256(elf[section[4]:section[4] + section[5]]).hexdigest()})
    if not result:
        raise ValueError('device ELF lacks .text sections')
    return result


def kernel_instructions(log):
    starts = list(re.finditer(r'^# Machine code for function (\S+?):', log, re.M))
    opcode = re.compile(r'^\s+(?:[^=\n]*=\s+)?(?:(?:contract|nuw|nsw|exact|nnan|ninf|nsz|arcp|afn|reassoc|disjoint)\s+)*([A-Z][A-Z0-9_]+)')
    result = {}
    for index, start in enumerate(starts):
        if 'mx_decode_kernel_v2' not in start[1] or '__device_stub__' in start[1]:
            continue
        end = log.find('# End machine code', start.end())
        if end < 0:
            raise ValueError('incomplete kernel MIR dump')
        if index + 1 < len(starts) and end > starts[index + 1].start():
            raise ValueError('unterminated kernel MIR dump')
        body = log[start.end():end]
        in_block = False
        total = 0
        for line in body.splitlines():
            if re.match(r'^bb\.\d+', line):
                in_block = True
            if in_block and opcode.match(line):
                total += 1
        if start[1] in result or total == 0:
            raise ValueError('duplicate or empty kernel MIR dump')
        result[start[1]] = total
    if len(result) != 1:
        raise ValueError('expected exactly one v2 kernel MIR dump')
    return result


def inspect(so, log, threads):
    if threads not in (512, 576):
        raise ValueError('invalid thread shape')
    data = Path(so).read_bytes()
    text = Path(log).read_text(errors='replace')
    resources = [v for k, v in parse_resources(text).items() if 'mx_decode_kernel_v2' in k and '__device_stub__' not in k]
    if len(resources) != 1:
        raise ValueError('expected exactly one v2 kernel resource record')
    kernel = resources[0]
    for key in ('registers', 'scalar_registers', 'stack_bytes', 'static_smem_bytes'):
        if key not in kernel:
            raise ValueError('incomplete v2 kernel resource record: ' + key)
    images = []
    for name, body in device_elfs(data):
        sections = text_sections(body)
        images.append({'target': name, 'elf_sha256': hashlib.sha256(body).hexdigest(),
                       'text_sections': sections, 'code_bytes': sum(s['bytes'] for s in sections)})
    cap = 227 if threads == 576 else 255
    violations = []
    if kernel['registers'] > cap:
        violations.append(f'MTreg {kernel["registers"]} exceeds {cap}')
    if kernel['stack_bytes']:
        violations.append(f'stack frame is {kernel["stack_bytes"]} bytes')
    if kernel['static_smem_bytes']:
        violations.append(f'static shared memory is {kernel["static_smem_bytes"]} bytes')
    instructions = kernel_instructions(text)
    return {'kernel': kernel, 'register_cap': cap, 'violations': violations,
            'artifact_bytes': len(data), 'code_bytes': sum(i['code_bytes'] for i in images),
            'device_images': images, 'kernel_instructions': instructions,
            'mir_capture': 'same_compile_print_after_stack_frame_layout',
            'body_instruction_status': 'pending_inlined_body_attribution',
            'body_instructions': {name: None for name in ('qkv', 'attention', 'o', 'gate_up', 'down', 'head')},
            'body_code_bytes': {name: None for name in ('qkv', 'attention', 'o', 'gate_up', 'down', 'head')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--so', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.so, args.log, args.threads)
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
