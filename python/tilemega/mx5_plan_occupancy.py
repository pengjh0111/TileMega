"""Query one frozen dense MX5 plan's actual occupancy without creating or launching it."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

KERNEL = '_ZN8tilemega7codegen2mx19mx_decode_kernel_v2IN10tm_mx_plan4PlanEEEvNT_7BindingENS1_13StageLaunchV2E'
STUB = KERNEL.replace('19mx_decode_kernel_v2', '34__device_stub__mx_decode_kernel_v2')
RUNTIME = Path('/opt/maca/lib/libruntime_cu.so')


class Attributes(C.Structure):
    # MACA mc_runtime_types.h cudaFuncAttributes ABI; not NVIDIA's field layout.
    _fields_ = [('binaryVersion', C.c_int), ('cacheModeCA', C.c_int),
                ('constSizeBytes', C.c_size_t), ('localSizeBytes', C.c_size_t),
                ('maxDynamicSharedSizeBytes', C.c_int), ('maxThreadsPerBlock', C.c_int),
                ('numRegs', C.c_int), ('preferredShmemCarveout', C.c_int),
                ('ptxVersion', C.c_int), ('sharedSizeBytes', C.c_size_t)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(item):
    path = Path(item['path']).resolve()
    if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or sha(path) != item['sha256']:
        raise ValueError('bound input changed: ' + str(path))
    return path


def run_text(argv):
    return subprocess.check_output(argv, text=True)


def exported_stub(nm):
    rows = [line.split() for line in nm.splitlines() if line.split()]
    names = [row[-1] for row in rows]
    candidates = [name for name in names if 'mx_decode_kernel_v2' in name]
    if candidates != [STUB] or names.count('tm_plan_shared_bytes') != 1:
        raise ValueError('expected exactly the supported dense registered stub and shared query')
    for symbol in (STUB, 'tm_plan_shared_bytes'):
        entry = next(row for row in rows if row[-1] == symbol)
        if len(entry) != 3 or entry[-2] not in ('T', 'W') or not re.fullmatch('[0-9a-fA-F]+', entry[0]):
            raise ValueError('expected an exported host code symbol: ' + symbol)
    return STUB


def shared_constant(disassembly):
    # Only the current x86-64 constant-return host query is supported.
    section = re.search(r'<tm_plan_shared_bytes>:\n(.*?)(?:\n\n|\Z)', disassembly, re.S)
    if not section:
        raise ValueError('missing host shared query disassembly')
    instructions = []
    for line in section[1].splitlines():
        match = re.match(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*(.*?)\s*$', line)
        if match:
            instructions.append(match[1])
        elif line.strip():
            raise ValueError('unparsed host shared query instruction')
    if len(instructions) != 2 or instructions[1] not in ('ret', 'retq'):
        raise ValueError('shared query is not a supported pure host constant return')
    move = re.fullmatch(r'mov(?:abs)?\s+\$0x([0-9a-f]+),%(?:eax|rax)', instructions[0])
    if not move or not 0 < int(move[1], 16) <= 65536:
        raise ValueError('invalid dynamic shared constant')
    return int(move[1], 16)


def validate_inputs(spec, inspect_resource=None, command=run_text):
    if spec.get('kind') != 'mx5_plan_occupancy_inputs' or spec.get('schema_version') != 1:
        raise ValueError('unsupported input schema')
    threads = spec.get('threads')
    if type(threads) is not int or threads not in (512, 576):
        raise ValueError('unsupported thread shape')
    required = ('binary', 'generated_source', 'resources', 'log', 'context', 'source_snapshot')
    paths = {name: checked(spec[name]) for name in required}
    if inspect_resource is None:
        from .mx5_resources import inspect as inspect_resource
    report = json.loads(paths['resources'].read_text())
    if report != inspect_resource(paths['binary'], paths['log'], threads):
        raise ValueError('resource report does not match actual binary and same-compile MIR')
    if (report.get('kernel', {}).get('name') != KERNEL
            or set(report.get('kernel_instructions', {})) != {KERNEL}
            or len(report.get('device_images', [])) != 1):
        raise ValueError('only the exact current single dense kernel is supported')
    source = paths['generated_source'].read_text()
    values = re.findall(r'\bkThreads\s*=\s*(\d+)\b', source)
    constexpr = re.findall(r'\bstatic\s+constexpr\s+int\s+[^;{}]*\bkThreads\s*=\s*(\d+)\b', source)
    if values != [str(threads)] or constexpr != [str(threads)] or not re.search(r'\bnamespace\s+tm_mx_plan\s*\{', source) or not re.search(r'\bstruct\s+Plan\s*\{', source):
        raise ValueError('generated Plan constexpr kThreads does not match input')
    if 'DecodeProgramV2.cuh' not in source or re.search(r'(?i)moe|cudnn|dnn', source):
        raise ValueError('only dense DecodeProgramV2 is supported')
    context = json.loads(paths['context'].read_text())
    if spec.get('context_kind') == 'native_identity':
        manifest = checked(spec['manifest'])
        paths['manifest'] = manifest
        for field, name in (('binary_sha256', 'binary'), ('generated_source_sha256', 'generated_source'),
                            ('resources_sha256', 'log'), ('manifest_sha256', 'manifest')):
            if context.get(field) != spec[name]['sha256']:
                raise ValueError('native identity mismatch: ' + field)
        if context.get('resources', {}).get(KERNEL) != report['kernel']:
            raise ValueError('native identity resources do not match the report kernel')
        if context.get('source') != json.loads(paths['source_snapshot'].read_text()):
            raise ValueError('native source snapshot mismatch')
        manifest_value = json.loads(manifest.read_text())
        if manifest_value.get('executor') != 'mx_static' or manifest_value.get('threads') != threads or manifest_value.get('kThreads') != threads:
            raise ValueError('native manifest does not describe this dense thread shape')
    elif spec.get('context_kind') == 'compile_receipt':
        for name in ('binary', 'generated_source', 'log', 'source_snapshot'):
            if context.get(name + '_sha256') != spec[name]['sha256']:
                raise ValueError('compile receipt mismatch: ' + name)
        if context.get('returncode') != 0 or context.get('same_compile_mir') != spec['log']['sha256']:
            raise ValueError('compile receipt does not bind a completed actual compile')
    else:
        raise ValueError('unsupported identity/failure-snapshot context')
    nm = command(['nm', '-D', '--defined-only', str(paths['binary'])])
    stub = exported_stub(nm)
    assembly = command(['objdump', '-d', '--disassemble=tm_plan_shared_bytes', str(paths['binary'])])
    shared = shared_constant(assembly)
    return dict(spec=spec, paths=paths, report=report, threads=threads, stub=stub,
                expected_shared_bytes=shared, nm_stdout=nm, shared_query_disassembly=assembly)


def validate_guard(env, lock_stat, locks_text, ancestors):
    gpu = env.get('TILEMEGA_DEVICE_INDEX')
    if (gpu not in ('4', '5', '6', '7') or env.get('TILEMEGA_GPU_LOCK_HELD') != '1'
            or any(env.get(name) != gpu for name in ('CUDA_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES'))):
        raise ValueError('require guarded consistent single-card environment')
    lock = Path('/tmp/tilemega_mx_gpu' + gpu + '.lock')
    if env.get('TILEMEGA_GPU_LOCK') != str(lock):
        raise ValueError('unexpected guard lock path')
    device = f'{os.major(lock_stat.st_dev):02x}:{os.minor(lock_stat.st_dev):02x}:{lock_stat.st_ino}'
    owners = []
    for line in locks_text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and fields[5] == device:
            owner = int(fields[4])
            if owner in ancestors and 'tilemega.bench.gpu_guard' in ancestors[owner]:
                owners.append(dict(pid=owner, command=ancestors[owner], proc_locks_line=line))
    if len(owners) != 1:
        raise ValueError('no unique gpu_guard ancestor owns the actual kernel flock')
    return dict(physical_gpu=gpu, lock_path=str(lock), owner=owners[0],
                environment={name: env[name] for name in ('TILEMEGA_DEVICE_INDEX', 'CUDA_VISIBLE_DEVICES',
                             'MACA_VISIBLE_DEVICES', 'TILEMEGA_GPU_LOCK', 'TILEMEGA_GPU_LOCK_HELD')})


def guard_evidence(env):
    path = Path(env.get('TILEMEGA_GPU_LOCK', '/nonexistent'))
    ancestors = {}
    pid = os.getppid()
    while pid > 1 and pid not in ancestors:
        base = Path('/proc') / str(pid)
        ancestors[pid] = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        pid = int((base / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    return validate_guard(env, path.stat(), Path('/proc/locks').read_text(), ancestors)


class Runtime:
    def __init__(self):
        if C.sizeof(Attributes) != 56:
            raise ValueError('unsupported MACA attributes ABI')
        self.handle = C.CDLL(str(RUNTIME))
        self.attributes = self.handle.wcudaFuncGetAttributes
        self.attributes.argtypes, self.attributes.restype = [C.POINTER(Attributes), C.c_void_p], C.c_int
        self.set_shared = self.handle.wcudaFuncSetAttribute
        self.set_shared.argtypes, self.set_shared.restype = [C.c_void_p, C.c_int, C.c_int], C.c_int
        self.occupancy = self.handle.wcudaOccupancyMaxActiveBlocksPerMultiprocessor
        self.occupancy.argtypes, self.occupancy.restype = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t], C.c_int


class Library:
    def __init__(self, path):
        self.handle = C.CDLL(str(path), mode=C.RTLD_LOCAL)

    def shared_bytes(self):
        query = self.handle.tm_plan_shared_bytes
        query.argtypes, query.restype = [], C.c_uint64
        return query()

    def stub_address(self, name):
        return C.cast(getattr(self.handle, name), C.c_void_p)


def query(validated, runtime, library, guard):
    spec, report = validated['spec'], validated['report']
    row = dict(kind='mx5_plan_occupancy', schema_version=1, binary_sha256=spec['binary']['sha256'],
               threads=validated['threads'], query_status=1, resident_ctas=None,
               query_only=True, iterations=0, no_launch=True, launches=0, pid=os.getpid(),
               guard=guard, bound_inputs=spec, registered_host_stub=validated['stub'],
               symbol_evidence=dict(nm_stdout=validated['nm_stdout'], shared_query_disassembly=validated['shared_query_disassembly']),
               execution_authorized=False, grants_512_exception=False)
    try:
        shared = library.shared_bytes()
        row.update(dynamic_shared_bytes=shared, dynamic_shared_source='tm_plan_shared_bytes():uint64')
        if shared != validated['expected_shared_bytes']:
            raise ValueError('host shared query disagrees with verified constant disassembly')
        address = library.stub_address(validated['stub'])
        row['stub_address'] = address.value
        attributes = Attributes()
        attr_rc = int(runtime.attributes(C.byref(attributes), address))
        row.update(attributes_returncode=attr_rc, attributes_raw_hex=bytes(attributes).hex(),
                   attributes={name: getattr(attributes, name) for name, _ in Attributes._fields_})
        if attr_rc:
            row.update(query_status=attr_rc, query_status_source='attributes_returncode')
            return row
        if attributes.maxThreadsPerBlock != validated['threads']:
            raise ValueError('attribute maxThreads does not match generated/input kThreads')
        expected = report['kernel']
        if (attributes.numRegs != expected['registers'] or attributes.localSizeBytes != expected['stack_bytes']
                or attributes.sharedSizeBytes != expected['static_smem_bytes']):
            raise ValueError('runtime attributes do not match bound device resource report')
        set_rc = int(runtime.set_shared(address, 8, shared))
        row['set_dynamic_shared_returncode'] = set_rc
        if set_rc:
            row.update(query_status=set_rc, query_status_source='set_dynamic_shared_returncode')
            return row
        occupancy = C.c_int(-1)
        occupancy_rc = int(runtime.occupancy(C.byref(occupancy), address, validated['threads'], shared))
        row.update(occupancy_returncode=occupancy_rc, occupancy_raw_value=occupancy.value,
                   resident_ctas=occupancy.value if occupancy_rc == 0 else None,
                   query_status=occupancy_rc, query_status_source='occupancy_returncode')
    except Exception as error:
        row.update(query_status=1, query_status_source='helper_validation_or_exception', exception=repr(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--query-only', action='store_true', required=True)
    parser.add_argument('--iterations', type=int, choices=[0], required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('output exists; preserve historical evidence')
    guard = guard_evidence(os.environ)
    inputs_record = record(args.inputs)
    validated = validate_inputs(json.loads(checked(inputs_record).read_text()))
    # Reserve the output before loading any GPU runtime/library; no overwrite on races.
    with args.out.open('x') as output:
        runtime_record, row = None, None
        try:
            runtime_record = record(RUNTIME)
            row = query(validated, Runtime(), Library(validated['paths']['binary']), guard)
            for item in [inputs_record, runtime_record, *(spec_item for spec_item in validated['spec'].values()
                                          if isinstance(spec_item, dict) and 'path' in spec_item and 'sha256' in spec_item)]:
                checked(item)
        except Exception as error:
            if row is None:
                row = dict(kind='mx5_plan_occupancy', schema_version=1, binary_sha256=validated['spec']['binary']['sha256'],
                           threads=validated['threads'], resident_ctas=None, query_only=True, iterations=0,
                           no_launch=True, launches=0, pid=os.getpid(), guard=guard, bound_inputs=validated['spec'],
                           registered_host_stub=validated['stub'], execution_authorized=False, grants_512_exception=False)
            row.update(query_status=1, exception=repr(error), query_status_source='loader_or_identity_error')
        row.update(inputs_manifest=inputs_record, runtime_library=runtime_record, helper_source=record(Path(__file__)))
        output.write(json.dumps(row, indent=2) + '\n')
    return 0 if row['query_status'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
