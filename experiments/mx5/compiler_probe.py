"""Validate an unexecuted Pw0 attribute probe, never an executable cost row."""
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

FIELDS = ('mtreg', 'local_bytes', 'static_smem_bytes', 'resident_ctas')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def check_file(item):
    path = Path(item['path']).resolve()
    if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or sha(path) != item['sha256']:
        raise ValueError('compiler probe file identity mismatch: ' + str(path))
    return path


def attributes(value):
    result = {}
    for name in FIELDS:
        number = value.get(name)
        if type(number) is not int or number < (1 if name == 'mtreg' else 0):
            raise ValueError('compiler probe needs complete valid attributes: ' + name)
        result[name] = number
    return result


def validate_normal_command(kind, key, command, binary):
    """Bind output before ignoring the one permitted MIR observer, wherever placed."""
    if command.count('-o') != 1:
        raise ValueError('normal bench command must have exactly one output')
    output_index=command.index('-o')+1
    if output_index >= len(command) or Path(command[output_index]).resolve() != binary:
        raise ValueError('normal bench command output differs from actual binary')
    output=command[output_index]
    normal=[command[0], '-std=c++17', '-O3', '-shared', '-Xcompiler', '-fPIC,-resource-usage',
            '-fno-strict-aliasing', '-DTILEMEGA_SYNC_V3=1', '-DTILEMEGA_WATCHDOG=0',
            '-DMX5_TEST_THREADS='+str(key['threads'])]
    if kind=='attention':
        normal += ['-DMX5_TEST_DIM='+str(key['dim']), '-DMX5_TEST_Q='+str(key['q']),
                   '-DMX5_BENCH_PREFETCH_KIND='+str(key['prefetch_kind'])]
    else:
        normal += ['-DMX5_TEST_TN='+str(key['tn']), '-DMX5_TEST_DEPTH='+str(key['depth']),
                   '-DMX5_BENCH_PW='+str(key['prefetch'])]
    source='experiments/mx5/'+('stage_bench.cu' if kind=='gemm' else kind+'_stage_bench.cu')
    normal += ['-Iinclude', source, '-o', output]
    observed=list(command)
    if '-print-after=stack-frame-layout' in observed:
        i=observed.index('-print-after=stack-frame-layout')
        if i < 1 or observed[i-1] != '-mllvm':
            raise ValueError('invalid MIR observation argument')
        del observed[i-1:i+1]
    if observed != normal:
        raise ValueError('diagnostic/forced-register flags do not match normal bench build policy')


def validate(reference, *, primary=False, normal_policy=False):
    if reference.get('primary_resource_rejection' if primary else 'compiler_probe_only') is not True:
        raise ValueError('primary rejection requires primary_resource_rejection=true' if primary else 'reference requires compiler_probe_only=true')
    query = reference.get('probe', {})
    if (query.get('query_only') is not True or type(query.get('iterations')) is not int
            or query['iterations'] != 0 or type(query.get('query_status')) is not int or query['query_status'] != 0):
        raise ValueError('compiler probe requires genuine query_only/iterations=0/status=0 evidence')
    kind, key, provenance = reference['kind'], reference['key'], reference['provenance']
    if (kind not in ('gemm','head','attention') or provenance.get('executor') != 'mx_static_stage_bench'
            or provenance.get('implementation') != 'mx5_' + kind + '_v2'
            or not isinstance(provenance.get('base_head'), str) or not provenance['base_head']):
        raise ValueError('invalid compiler probe executor/implementation provenance')
    for name in ('source_sha256','binary_sha256'):
        if not isinstance(provenance.get(name), str) or not re.fullmatch('[0-9a-f]{64}', provenance[name]):
            raise ValueError('invalid compiler probe provenance ' + name)
    field = 'prefetch_kind' if kind == 'attention' else 'prefetch'
    if type(key.get(field)) is not int or (not primary and key[field] != 0):
        raise ValueError('compiler probe must describe exact Pw0/PF0 coordinates')
    resources = attributes(reference['resources'])
    threads = key['threads']
    if threads not in (512, 576):
        raise ValueError('invalid compiler probe thread shape')
    fits = resources['mtreg'] <= (227 if threads == 576 else 255) and resources['local_bytes'] == 0 and resources['static_smem_bytes'] == 0 and resources['resident_ctas'] == 1
    if type(reference.get('fits')) is not bool or reference['fits'] != fits:
        raise ValueError('compiler probe diagnostic fits disagrees with observed attributes')
    binary = check_file(query['binary']); metadata_path = check_file(query['build_metadata'])
    archive = check_file(query['source_archive']); metadata = json.loads(metadata_path.read_text())
    for name in ('source_sha256', 'binary_sha256', 'base_head'):
        if metadata.get(name) != provenance.get(name):
            raise ValueError('compiler probe build/provenance mismatch: ' + name)
    if query['binary']['sha256'] != provenance.get('binary_sha256'):
        raise ValueError('compiler probe binary/provenance mismatch')
    if (metadata.get('body') != kind or metadata.get('bench') is not True
            or type(metadata.get('returncode')) is not int or metadata['returncode'] != 0):
        raise ValueError('compiler probe requires a successful matching bench build')
    expected = dict(threads=threads)
    macros = dict(MX5_TEST_THREADS='threads')
    if kind == 'attention':
        expected.update(dim=key['dim'], query=key['q'], prefetch_kind=key[field] if primary else 0)
        macros.update(MX5_TEST_DIM='dim', MX5_TEST_Q='query', MX5_BENCH_PREFETCH_KIND='prefetch_kind')
    else:
        expected.update(tn=key['tn'], depth=key['depth'], pw=key[field] if primary else 0)
        macros.update(MX5_TEST_TN='tn', MX5_TEST_DEPTH='depth', MX5_BENCH_PW='pw')
    for name, number in expected.items():
        if type(metadata.get(name)) is not int or metadata[name] != number:
            raise ValueError('compiler probe build coordinate mismatch: ' + name)
    definitions = {}; command = metadata.get('command', [])
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError('compiler probe build command is missing')
    arguments = iter(command)
    for argument in arguments:
        if argument == '-D':
            argument = next(arguments, '')
        elif argument.startswith('-D'):
            argument = argument[2:]
        else:
            continue
        macro, _, raw = argument.partition('=')
        if macro in macros:
            if macros[macro] in definitions or not re.fullmatch('[0-9]+', raw):
                raise ValueError('invalid or duplicate compiler probe build macro')
            definitions[macros[macro]] = int(raw)
    if definitions != expected:
        raise ValueError('compiler probe build macros differ from exact coordinates')
    filename = 'stage_bench.cu' if kind == 'gemm' else kind + '_stage_bench.cu'
    source_name = 'experiments/mx5/' + filename
    digest = hashlib.sha256()
    try:
        with tarfile.open(archive, 'r:gz') as source:
            members = [item for item in source.getmembers() if item.isfile()]
            names = [item.name for item in members]
            if len(names) != len(set(names)) or source_name not in names or source_name not in command:
                raise ValueError('compiler probe archive/source command mismatch')
            for member in sorted(members, key=lambda item: item.name):
                digest.update(member.name.encode() + b'\0' + source.extractfile(member).read() + b'\0')
    except tarfile.TarError as error:
        raise ValueError('compiler probe invalid source archive: ' + str(error)) from error
    if digest.hexdigest() != provenance.get('source_sha256'):
        raise ValueError('compiler probe archived source digest mismatch')
    if primary or normal_policy:
        validate_normal_command(kind,key,command,binary)
    if primary:
        if fits:
            raise ValueError('primary rejection actually fits executable resource gates')
        compiler = check_file(query['compiler'])
        if (reference.get('compiler_binding') != 'retained_executable_at_qualification'
                or reference.get('original_compiler_identity_recorded') is not False
                or 'original_build_compiler_sha256' not in reference or reference['original_build_compiler_sha256'] is not None):
            raise ValueError('primary query must explicitly distinguish qualification-time compiler identity')
        if (not command or not Path(command[0]).is_absolute() or Path(command[0]).resolve() != compiler
                or not os.access(compiler, os.X_OK)):
            raise ValueError('primary query lacks retained executable/command binding')
        if reference.get('build_policy') != 'mx5_build_bodies_default_v1':
            raise ValueError('primary rejection requires the normal stage-bench build policy')
        descriptor=query['raw_query']; raw_text=check_file(descriptor).read_text()
        if 'line' in descriptor:
            line=descriptor['line']
            if type(line) is not int or not 1 <= line <= len(raw_text.splitlines()):
                raise ValueError('invalid primary raw query line')
            raw_text=raw_text.splitlines()[line-1]
        raw = json.loads(raw_text)
        if not isinstance(raw.get('library'),str) or Path(raw['library']).resolve() != binary:
            raise ValueError('primary raw query library differs from compiled binary')
        for name in ('kind', 'key', 'resources', 'provenance', 'fits'):
            if raw.get(name) != reference[name]:
                raise ValueError('primary raw query differs from derived rejection: ' + name)
        if (raw.get('query_only') is not True or type(raw.get('iterations')) is not int or raw['iterations'] != 0
                or type(raw.get('query_status')) is not int or raw['query_status'] != 0):
            raise ValueError('primary raw query must be successful zero-execution attributes')
        if any(field in raw for field in ('median_ns', 'p90_ns', 'rounds', 'spread', 'processes')):
            raise ValueError('primary rejection raw query contains execution costs')
    return resources


def from_query(kind, key, result):
    if (result.get('query_only') is not True or type(result.get('iterations')) is not int or result['iterations'] != 0
            or type(result.get('query_status')) is not int or result['query_status'] != 0):
        raise ValueError('compiler probe requires genuine query_only/iterations=0/status=0 evidence')
    values = attributes(result.get('resources') or {})
    reference = dict(kind=kind, key=dict(key), resources=values, provenance=dict(result['provenance']),
        compiler_probe_only=True, fits=result['fits'], probe=dict(query_only=result['query_only'],
            iterations=result['iterations'], query_status=result['query_status'],
            binary=record(result['library']), build_metadata=record(Path(result['library']).with_suffix('.json')),
            source_archive=record(Path(result['library']).parent / 'sources.tar.gz')))
    validate(reference)
    return reference
