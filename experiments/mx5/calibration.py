#!/usr/bin/env python3
"""Enumerate native body keys and query immutable bench resources."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MODELS = ['qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b']


def inventory(domain, models_root, threads):
    rows = {}
    for model in MODELS:
        config = json.loads((models_root / (model + '_rand') / 'config.json').read_text())
        hidden, intermediate = config['hidden_size'], config['intermediate_size']
        heads, hkv = config['num_attention_heads'], config['num_key_value_heads']
        dim = config.get('head_dim', hidden // heads)
        shapes = [('qkv', (heads + 2 * hkv) * dim, hidden, 0, 1),
                  ('o', hidden, heads * dim, 1, 0),
                  ('gate_up', 2 * intermediate, hidden, 2, 1),
                  ('down', hidden, intermediate, 1, 0),
                  ('head', config['vocab_size'], hidden, 0, 1)]
        for batch in (1, 2, 4, 8, 16):
            for count in threads:
                for name, n, k, op, norm in shapes:
                    kind = 'head' if name == 'head' else 'gemm'
                    command = [str(domain), '--kind', kind, '--n', str(n), '--k', str(k),
                               '--batch', str(batch), '--threads', str(count)]
                    if kind == 'gemm':
                        command.extend(['--op', str(op)])
                    result = subprocess.run(command, text=True, capture_output=True, check=True)
                    candidates = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
                    if not candidates:
                        raise RuntimeError('empty native candidate domain: ' + ' '.join(command))
                    for candidate in candidates:
                        key = dict(n=n, k=k, b=batch, tn=candidate['tn'], wn=candidate['wn'],
                                   depth=candidate['depth'], threads=count)
                        if kind == 'gemm':
                            key.update(op=op, norm=norm, part=candidate['part'])
                        add(rows, kind, key, model, name, first_chunks=candidate['first_chunks'])
                for length in (65, 256, 512, 768, 1088):
                    for bmin in (1, 2, 4, 8):
                        key = dict(dim=dim, q=heads // hkv, hkv=hkv, b=batch, length=length,
                                   b_min=bmin, qk_norm=int(config['model_type'].startswith('qwen3')),
                                   capacity=1088, threads=count)
                        add(rows, 'attention', key, model, 'attention')
    return sorted(rows.values(), key=lambda row: (row['kind'], json.dumps(row['key'], sort_keys=True)))


def add(rows, kind, key, model, stage, **extra):
    index = kind, json.dumps(key, sort_keys=True)
    if index not in rows:
        rows[index] = dict(kind=kind, key=key, uses=[], **extra)
    elif any(rows[index].get(name) != value for name, value in extra.items()):
        raise RuntimeError('native geometry disagrees for an identical body key')
    rows[index]['uses'].append(dict(model=model, stage=stage))


def module(kind):
    filename = 'stage_bench.py' if kind == 'gemm' else kind + '_stage_bench.py'
    spec = importlib.util.spec_from_file_location('mx5_inspect_' + kind, ROOT / 'experiments/mx5' / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def library_key(kind, key):
    return ((kind, key['threads'], key['dim'], key['q']) if kind == 'attention'
            else (kind, key['threads'], key['tn'], key['depth']))


def baseline_libraries(root):
    result = {}
    roots = [root] if isinstance(root, Path) else root
    records = sorted(record for directory in roots for record in directory.rglob('*.json'))
    for record in records:
        metadata = json.loads(record.read_text())
        kind = metadata.get('body')
        if kind not in ('gemm', 'head', 'attention') or metadata.get('bench') is not True:
            continue
        if kind == 'attention':
            if metadata.get('prefetch_kind') != 0:
                continue
            key = (kind, metadata['threads'], metadata['dim'], metadata['query'])
        else:
            if metadata.get('pw') != 0:
                continue
            key = (kind, metadata['threads'], metadata['tn'], metadata['depth'])
        if key in result:
            raise RuntimeError('ambiguous unprefetched library key: ' + str(key))
        result[key] = record.with_suffix('.so').resolve()
    return result


def prefetch(kind, key):
    if kind == 'attention':
        return key.get('prefetch_kind', 0)
    if 'prefetch' in key and 'pw' in key and key['prefetch'] != key['pw']:
        raise RuntimeError('conflicting key prefetch and pw')
    return key.get('prefetch', key.get('pw', 0))


def validate_coordinates(path, kind, key, metadata, bench, lib=None, prefix=None):
    width = prefetch(kind, key)
    if metadata.get('threads') != key['threads']:
        raise RuntimeError('baseline threads mismatch: ' + str(path))
    if kind == 'attention':
        fields = dict(dim=key['dim'], query=key['q'], prefetch_kind=width)
        expected = [key['dim'], key['q'], width, key['threads']]
        macros = dict(MX5_TEST_DIM='dim', MX5_TEST_Q='query',
                      MX5_BENCH_PREFETCH_KIND='prefetch_kind', MX5_TEST_THREADS='threads')
    else:
        fields = dict(tn=key['tn'], depth=key['depth'], pw=width)
        expected = [key['tn'], key['depth'], width, key['threads']]
        macros = dict(MX5_TEST_TN='tn', MX5_TEST_DEPTH='depth',
                      MX5_BENCH_PW='pw', MX5_TEST_THREADS='threads')
    compiled = dict(metadata)
    if kind != 'attention' and compiled.get('pw') is None:
        compiled['pw'] = compiled.get('depth')
    if any(compiled.get(field) != value for field, value in fields.items()):
        raise RuntimeError('baseline ' + kind + ' coordinates mismatch: ' + str(path))
    command = metadata.get('command')
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise RuntimeError('missing build command: ' + str(path))
    definitions = {}
    arguments = iter(command)
    for argument in arguments:
        if argument == '-D':
            argument = next(arguments, '')
        elif argument.startswith('-D'):
            argument = argument[2:]
        else:
            continue
        macro, _, value = argument.partition('=')
        if macro not in macros:
            continue
        field = macros[macro]
        if field in definitions:
            raise RuntimeError('duplicate build definition: ' + macro)
        try:
            definitions[field] = int(value)
        except ValueError as error:
            raise RuntimeError('invalid build definition: ' + macro) from error
    if kind != 'attention':
        definitions.setdefault('pw', definitions.get('depth'))
    for field, value in dict(fields, threads=key['threads']).items():
        if definitions.get(field) != value:
            raise RuntimeError('build metadata/command mismatch for ' + field + ': ' + str(path))
    if kind == 'gemm':
        bench.validate_metadata(dict(key, pw=width), metadata)
    elif lib is not None:
        coordinates = (C.c_int * 4)()
        query = getattr(lib, prefix + '_coordinates')
        query.argtypes, query.restype = [C.POINTER(C.c_int)], None
        query(coordinates)
        if list(coordinates) != expected:
            raise RuntimeError('baseline exported coordinates mismatch: ' + str(path))


def validate_source(path, kind, metadata):
    archive = path.parent / 'sources.tar.gz'
    if not archive.is_file():
        raise RuntimeError('missing source archive: ' + str(archive))
    digest = hashlib.sha256()
    with tarfile.open(archive, 'r:gz') as source:
        members = [member for member in source.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError('duplicate archived source: ' + str(archive))
        filename = 'stage_bench.cu' if kind == 'gemm' else kind + '_stage_bench.cu'
        source_path = 'experiments/mx5/' + filename
        if source_path not in names or source_path not in metadata['command']:
            raise RuntimeError('bench source/command mismatch: ' + str(path))
        for member in sorted(members, key=lambda member: member.name):
            digest.update(member.name.encode() + b'\0' + source.extractfile(member).read() + b'\0')
    if digest.hexdigest() != metadata.get('source_sha256'):
        raise RuntimeError('baseline source identity mismatch: ' + str(path))


def load_library(path, kind, key, bench):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('missing bench library: ' + str(path))
    if not path.with_suffix('.json').is_file():
        raise RuntimeError('missing bench metadata: ' + str(path))
    metadata = json.loads(path.with_suffix('.json').read_text())
    binary = hashlib.sha256(path.read_bytes()).hexdigest()
    if (metadata.get('body') != kind or metadata.get('bench') is not True
            or metadata.get('returncode') != 0 or metadata.get('binary_sha256') != binary
            or metadata.get('threads') != key['threads']):
        raise RuntimeError('baseline identity mismatch: ' + str(path))
    validate_coordinates(path, kind, key, metadata, bench)
    validate_source(path, kind, metadata)
    lib = C.CDLL(str(path))
    prefix = 'mx5_bench' if kind == 'gemm' else 'mx5_' + kind + '_bench'
    size = getattr(lib, prefix + '_args_size')
    size.argtypes, size.restype = [], C.c_int
    if size() != C.sizeof(bench.Args):
        raise RuntimeError('baseline ABI mismatch: ' + str(path))
    validate_coordinates(path, kind, key, metadata, bench, lib, prefix)
    return lib, metadata, prefix


def require_guard():
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('resource queries must run through scripts/mx_gpu_job.sh')


def initialize_cuda():
    require_guard()
    import torch
    torch.cuda.set_device(0)
    torch.cuda.init()


def query_one(kind, key, path, modules, libraries, expected_source=None):
    require_guard()
    if kind not in modules:
        modules[kind] = module(kind)
    bench = modules[kind]
    path = Path(path).resolve()
    if path not in libraries:
        libraries[path] = load_library(path, kind, key, bench)
    lib, metadata, prefix = libraries[path]
    # A shared library serves many runtime keys; never let caching skip key validation.
    if metadata.get('body') != kind:
        raise RuntimeError('cached library body mismatch: ' + str(path))
    validate_coordinates(path, kind, key, metadata, bench, lib, prefix)
    if expected_source is not None and metadata.get('source_sha256') != expected_source:
        raise RuntimeError('R_next and selected kernel have different body sources: ' + str(path))
    args = bench.Args()
    values = (C.c_int * 4)()
    args.resources, args.iterations, args.copies = values, 0, 1
    args.b = key['b']
    launch = getattr(lib, prefix)
    launch.restype = C.c_int
    if kind == 'gemm':
        args.n, args.k = key['n'], key['k']
        launch.argtypes = [C.POINTER(bench.Args), C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
        status = launch(C.byref(args), key['wn'], key['op'], key['norm'], key['part'], None)
    elif kind == 'head':
        args.n, args.k = key['n'], key['k']
        args.head.capacity, args.head.past = 1088, 64
        launch.argtypes = [C.POINTER(bench.Args), C.c_int, C.c_void_p]
        status = launch(C.byref(args), key['wn'], None)
    else:
        args.hkv = key['hkv']
        args.attention.capacity, args.attention.past = key['capacity'], key['length'] - 1
        launch.argtypes = [C.POINTER(bench.Args), C.c_int, C.c_int, C.c_void_p]
        status = launch(C.byref(args), key['b_min'], key['qk_norm'], None)
    resource = {name: int(value) if status == 0 else None for name, value in
                zip(('mtreg', 'local_bytes', 'static_smem_bytes', 'resident_ctas'), values)}
    cap = 227 if key['threads'] == 576 else 255
    fit = status == 0 and 0 < values[0] <= cap and values[1] == 0 and values[2] == 0 and values[3] == 1
    provenance = {field: metadata[field] for field in ('source_sha256', 'binary_sha256', 'base_head')}
    provenance.update(executor='mx_static_stage_bench', implementation='mx5_' + kind + '_v2')
    return dict(query_status=status, resources=resource, fits=fit, library=str(path),
                provenance=provenance, card=os.environ.get('TILEMEGA_DEVICE_INDEX'), query_only=True)


def inspect(rows, root):
    initialize_cuda()
    modules, libraries = {}, {}
    catalogue = baseline_libraries(root)
    for row in rows:
        kind, key = row['kind'], row['key']
        index = library_key(kind, key)
        if index not in catalogue:
            raise RuntimeError('missing unprefetched library: ' + str(index))
        reference_key = dict(key, **{'prefetch_kind' if kind == 'attention' else 'prefetch': 0})
        result = query_one(kind, reference_key, catalogue[index], modules, libraries)
        reference = dict(kind=kind, key=reference_key, resources=result['resources'],
                         provenance=result['provenance'])
        yield dict(row, **{field: result[field] for field in
                          ('query_status', 'fits', 'library', 'card', 'query_only')},
                   r_next_reference=reference)


def check(rows):
    initialize_cuda()
    modules, libraries = {}, {}
    for row in rows:
        if row.get('disposition') != 'derived':
            yield row
            continue
        kind, key = row['kind'], row['key']
        artifacts = row.get('artifacts')
        if not artifacts or artifacts[0].get('resource_step') != 0:
            raise RuntimeError('missing primary resource artifact for key: ' + str(key))
        artifact = artifacts[0]
        if artifact.get('prefetch') != prefetch(kind, key):
            raise RuntimeError('primary artifact/key prefetch mismatch for key: ' + str(key))
        result = query_one(kind, key, artifact['library'], modules, libraries)
        yield dict(row, **result, selected_resource_step=0)


def selection_reference(kind, key, reference):
    if not isinstance(reference, dict):
        raise RuntimeError('missing R_next reference for selection')
    expected = dict(key)
    width_field = 'prefetch_kind' if kind == 'attention' else 'prefetch'
    expected.pop('pw', None)
    expected[width_field] = 0
    actual = dict(reference.get('key', {}))
    if 'pw' in actual:
        if 'prefetch' in actual and actual['prefetch'] != actual['pw']:
            raise RuntimeError('conflicting R_next prefetch coordinates')
        actual['prefetch'] = actual.pop('pw')
    if reference.get('kind') != kind or actual != expected:
        raise RuntimeError('R_next reference coordinates do not match the selected body key')
    resources = reference.get('resources', {})
    cap = 227 if key['threads'] == 576 else 255
    if (type(resources.get('mtreg')) is not int or not 0 < resources['mtreg'] <= cap
            or any(type(resources.get(field)) is not int or resources[field] != value for field, value in
                   (('local_bytes', 0), ('static_smem_bytes', 0), ('resident_ctas', 1)))):
        raise RuntimeError('R_next reference does not have passing unprefetched resources')
    provenance = reference.get('provenance', {})
    if (provenance.get('executor') != 'mx_static_stage_bench'
            or provenance.get('implementation') != 'mx5_' + kind + '_v2'
            or not isinstance(provenance.get('base_head'), str) or not provenance['base_head']):
        raise RuntimeError('invalid R_next reference provenance')
    for field in ('source_sha256', 'binary_sha256'):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('invalid R_next reference ' + field)
    return provenance['source_sha256']


def select(rows, record=None):
    initialize_cuda()
    modules, libraries = {}, {}
    for row_index, row in enumerate(rows):
        if row.get('disposition') == 'unprefetched_resource_failure':
            yield dict(row, selection_status='unresolved_unprefetched_resource_failure',
                       selected_resource_step=None, resource_attempts=[])
            continue
        if row.get('disposition') != 'derived':
            raise RuntimeError('selection requires a derived or unprefetched-resource-failure row')
        kind, original = row['kind'], row['key']
        source = selection_reference(kind, original, row.get('r_next_reference'))
        artifacts = row.get('artifacts')
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 4:
            raise RuntimeError('selection requires primary and at most three reduction artifacts')
        attempts = []
        previous = prefetch(kind, original)
        for step, artifact in enumerate(artifacts):
            width = artifact.get('prefetch')
            key = dict(original)
            key['prefetch_kind' if kind == 'attention' else 'prefetch'] = width
            if 'pw' in key:
                key['pw'] = width
            event = dict(row_index=row_index, kind=kind, original_key=dict(original),
                         resource_step=step, prefetch=width, key=key, library=artifact.get('library'))
            try:
                if type(artifact.get('resource_step')) is not int or artifact['resource_step'] != step:
                    raise RuntimeError('resource artifacts must have contiguous steps starting at zero')
                if type(width) is not int or width < 0:
                    raise RuntimeError('invalid resource artifact prefetch width')
                if step == 0:
                    if width != previous:
                        raise RuntimeError('primary artifact/key prefetch mismatch for key: ' + str(original))
                elif (kind == 'attention' or width != max(original['depth'], previous // 2)
                      or width >= previous):
                    raise RuntimeError('resource artifact is not the next prescribed prefetch halving')
                result = query_one(kind, key, artifact['library'], modules, libraries, expected_source=source)
            except Exception as error:
                if record is not None:
                    record(dict(event, event='query_error', error=str(error), error_type=type(error).__name__))
                raise
            attempt = dict(resource_step=step, prefetch=width, key=key, **result)
            if record is not None:
                record(dict(event, **result, event='resource_query'))
            attempts.append(attempt)
            if result['fits']:
                yield dict(row, key=key, **result, selected_resource_step=step,
                           selection_status='selected', resource_attempts=attempts)
                break
            previous = width
        else:
            yield dict(row, key=key, **result, selected_resource_step=None,
                       selection_status='unresolved_resource_failure', resource_attempts=attempts)


def plan(rows, domain, output_root):
    cache = {}
    catalogue = {}
    references = []
    planned = []
    for row in rows:
        kind, key = row['kind'], dict(row['key'])
        if not row['fits']:
            planned.append(dict(row, disposition='unprefetched_resource_failure'))
            continue
        rnext = row['r_next_reference']['resources']['mtreg']
        if kind == 'attention':
            budget = min(64, (220 if key['threads'] == 576 else 248) - rnext)
            width = 2 if budget >= 64 else 1 if budget >= 32 else 0
            variants = [width]
            key['prefetch_kind'] = width
        else:
            request = (kind, key['n'], key['k'], key['b'], key['threads'], key.get('op'), rnext)
            if request not in cache:
                command = [str(domain), '--kind', kind, '--n', str(key['n']), '--k', str(key['k']),
                           '--batch', str(key['b']), '--threads', str(key['threads']), '--r-next', str(rnext)]
                if kind == 'gemm':
                    command.extend(['--op', str(key['op'])])
                result = subprocess.run(command, text=True, capture_output=True, check=True)
                cache[request] = {(c['tn'], c['wn'], c['depth'], c['part']): c
                                  for c in map(json.loads, result.stdout.splitlines())}
            coordinate = key['tn'], key['wn'], key['depth'], key.get('part', 1)
            candidate = cache[request].get(coordinate)
            if candidate is None or candidate['first_chunks'] != row['first_chunks']:
                raise RuntimeError('inspected key disagrees with the production geometry')
            width = candidate['prefetch']
            variants = [width]
            for _ in range(3):
                reduced = max(key['depth'], variants[-1] // 2)
                if reduced == variants[-1]:
                    break
                variants.append(reduced)
            key['prefetch'] = width
        artifacts = []
        for step, variant in enumerate(variants):
            if kind == 'attention' and variant == 0:
                library = Path(row['library'])
            else:
                stem = (f'd{key["dim"]}_q{key["q"]}' if kind == 'attention'
                        else f'tn{key["tn"]}_d{key["depth"]}')
                directory = output_root / f'{kind}_t{key["threads"]}_pf{variant}_{stem}'
                library = directory / (stem + '.so')
                command = ['python', 'experiments/mx5/build_bodies.py', '--bench', '--body', kind,
                           '--threads', str(key['threads']), '--out', str(directory), '--jobs', '2']
                if kind == 'attention':
                    command.extend(['--dim', str(key['dim']), '--query', str(key['q']),
                                    '--prefetch-kind', str(variant)])
                else:
                    command.extend(['--tn', str(key['tn']), '--depth', str(key['depth']), '--pw', str(variant)])
                index = kind, key['threads'], stem, variant
                catalogue[index] = dict(kind=kind, library=str(library), command=command)
            artifacts.append(dict(resource_step=step, prefetch=variant, library=str(library)))
        references.append(row['r_next_reference'])
        planned.append(dict(row, key=key, disposition='derived', artifacts=artifacts))
    return planned, list(catalogue.values()), references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('inventory')
    create.add_argument('--domain', type=Path, required=True)
    create.add_argument('--models-root', type=Path, default=Path('/root/models'))
    create.add_argument('--threads', type=int, nargs='+', choices=(512, 576), default=[576, 512])
    derive = sub.add_parser('plan')
    derive.add_argument('--keys', type=Path, required=True)
    derive.add_argument('--domain', type=Path, required=True)
    derive.add_argument('--library-root', type=Path, required=True)
    query = sub.add_parser('inspect')
    query.add_argument('--keys', type=Path, required=True)
    query.add_argument('--libraries', type=Path, nargs='+', required=True)
    verify = sub.add_parser('check')
    verify.add_argument('--keys', type=Path, required=True)
    choose = sub.add_parser('select')
    choose.add_argument('--keys', type=Path, required=True)
    for command in (create, query, derive, verify, choose):
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sidecars = {name: Path(str(args.out) + suffix) for name, suffix in
                [('tool', '.tool.py'), ('identity', '.identity.json')]}
    if args.command == 'plan':
        sidecars.update({name: Path(str(args.out) + '.' + name + '.json')
                         for name in ('builds', 'references')})
    if args.command == 'select':
        sidecars['attempts'] = Path(str(args.out) + '.attempts.jsonl')
    for path in [args.out, *sidecars.values()]:
        if path.exists():
            parser.error('output already exists; preserve historical records: ' + str(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tool_source = Path(__file__).read_bytes()
    with sidecars['tool'].open('xb') as stream:
        stream.write(tool_source)
    provenance = dict(command=args.command, rows=0,
                      tool_sha256=hashlib.sha256(tool_source).hexdigest(),
                      tool_source=str(sidecars['tool'].resolve()), run_status='failed')
    count = 0
    attempt_log, attempt_count = None, 0
    attempt_digest = hashlib.sha256()

    def record_attempt(event):
        nonlocal attempt_count
        line = json.dumps(event, sort_keys=True) + '\n'
        attempt_log.write(line)
        attempt_log.flush()
        os.fsync(attempt_log.fileno())
        attempt_digest.update(line.encode())
        attempt_count += 1

    try:
        if args.command == 'select':
            attempt_log = sidecars['attempts'].open('x', encoding='utf-8')
            provenance['attempt_log'] = str(sidecars['attempts'].resolve())
        provenance['head'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if args.command == 'inventory':
            provenance['domain_sha256'] = hashlib.sha256(args.domain.read_bytes()).hexdigest()
            provenance['threads'] = args.threads
            provenance['models'] = {model: hashlib.sha256(
                (args.models_root / (model + '_rand') / 'config.json').read_bytes()).hexdigest()
                for model in MODELS}
            rows = inventory(args.domain.resolve(), args.models_root, args.threads)
        else:
            keys_source = args.keys.read_bytes()
            provenance['keys_sha256'] = hashlib.sha256(keys_source).hexdigest()
            provenance['card'] = os.environ.get('TILEMEGA_DEVICE_INDEX')
            keys = [json.loads(line) for line in keys_source.decode().splitlines() if line.strip()]
            if args.command == 'inspect':
                rows = inspect(keys, args.libraries)
            elif args.command == 'check':
                rows = check(keys)
            elif args.command == 'select':
                rows = select(keys, record=record_attempt)
            else:
                provenance['domain_sha256'] = hashlib.sha256(args.domain.read_bytes()).hexdigest()
                rows, builds, references = plan(keys, args.domain.resolve(), args.library_root.resolve())
                for name, document in [('builds', builds), ('references', dict(references=references))]:
                    with sidecars[name].open('x') as stream:
                        stream.write(json.dumps(document, indent=2) + '\n')
        with args.out.open('x') as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + '\n')
                output.flush()
                count += 1
        provenance['run_status'] = 'complete'
    except Exception as error:
        provenance['error'] = str(error)
        raise
    finally:
        if attempt_log is not None:
            attempt_log.close()
            provenance.update(attempt_log_sha256=attempt_digest.hexdigest(), attempt_events=attempt_count)
        provenance['rows'] = count
        with sidecars['identity'].open('x') as stream:
            stream.write(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(command=args.command, rows=count, out=str(args.out))))


if __name__ == '__main__':
    main()
