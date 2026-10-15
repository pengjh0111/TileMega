"""Standalone provenance gate for R15 backend with repaired frontend CG input."""
import hashlib
import json
from pathlib import Path

R15_HEAD = '243fe9d06f5f8c55297c0c175295812f0843f329'
R15_COMPILER_SHA = 'bcbd4d58b60e66b0e499f454f38787e0bb7044a4690c9bf04a2866e20d11621c'
FRONTEND_HEAD = 'e718333c525fce8c6e1cd803fa880226945ab0ea'
FRONTEND_COMPILER_SHA = '9852bae10543b88d51cd96a8d62d7d4769e3b5f7ecb38fba2e3902b3b2a2e434'
FRONTEND_SNAPSHOT = '20261015T190947Z-prefill-prefix-forward-seed'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bound_json(path, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError('TM-R15-IR receipt SHA differs: ' + str(path))
    return json.loads(path.read_text())


def validate_frontend_ir(identity, root):
    """Validate actual build inputs, preserving the entire old source snapshot.

    Call the ordinary pristine-R15 gate for *both* phase identities as well.
    This additional gate is mandatory for the TM-R15-IR prefill identity.
    """
    root = Path(root).resolve()
    source = identity['source']
    provenance = source.get('frontend_ir_provenance')
    if not isinstance(provenance, dict) or provenance.get('schema') != 1:
        raise ValueError('TM-R15-IR prefill requires identified frontend IR provenance')
    if provenance.get('label') != 'TM-R15-IR' or identity['plan'].get('phase') != 'prefill':
        raise ValueError('TM-R15-IR provenance belongs to another label or phase')
    compiler = root / 'runs/r16/build-r15/tools/tilemega'
    frontend = root / 'runs/r16/build-snapshot' / FRONTEND_SNAPSHOT / 'tools/tilemega'
    if (source.get('head') != R15_HEAD or source.get('worktree_diff_sha256') != hashlib.sha256(b'').hexdigest()
            or Path(source.get('root', '')).resolve() != root / 'runs/r16/tm-r15'
            or Path(source.get('compiler', '')).resolve() != compiler
            or source.get('compiler_sha256') != R15_COMPILER_SHA or sha(compiler) != R15_COMPILER_SHA):
        raise ValueError('TM-R15-IR backend is not the unchanged R15 compiler/source')
    if (provenance.get('frontend_source_head') != FRONTEND_HEAD
            or Path(provenance.get('frontend_compiler', '')).resolve() != frontend
            or provenance.get('frontend_compiler_sha256') != FRONTEND_COMPILER_SHA
            or sha(frontend) != FRONTEND_COMPILER_SHA):
        raise ValueError('TM-R15-IR frontend compiler/source origin differs')
    binary = Path(identity['binary_path']).resolve()
    snapshot = Path(str(binary) + '.source.json')
    if provenance.get('backend_source_snapshot_sha256') != sha(snapshot):
        raise ValueError('TM-R15-IR original source snapshot SHA differs')
    original = {key: value for key, value in source.items() if key != 'frontend_ir_provenance'}
    if original != json.loads(snapshot.read_text()):
        raise ValueError('TM-R15-IR altered old source_files/HEAD/source snapshot')
    cg = Path(provenance['input_CG_path']).resolve()
    if not cg.is_relative_to(root / 'runs/r16') or sha(cg) != provenance['input_CG_sha256']:
        raise ValueError('TM-R15-IR input CG SHA/path differs')
    emit = bound_json(provenance['frontend_emit_receipt'], provenance['frontend_emit_receipt_sha256'])
    build = bound_json(provenance['backend_build_receipt'], provenance['backend_build_receipt_sha256'])
    for receipt in (emit, build):
        if (type(receipt.get('returncode')) is not int or receipt['returncode'] != 0 or receipt.get('phase') != 'prefill'
                or Path(receipt['input_CG_path']).resolve() != cg
                or receipt['input_CG_sha256'] != provenance['input_CG_sha256']):
            raise ValueError('TM-R15-IR frontend/backend receipt input differs')
    if (emit['frontend_source_head'] != FRONTEND_HEAD or emit['frontend_compiler_sha256'] != FRONTEND_COMPILER_SHA
            or Path(emit['frontend_compiler']).resolve() != frontend
            or emit['command'][0] != str(frontend) or emit['command'][1] != 'compile'
            or '--dump-cg' not in emit['command']
            or Path(emit['command'][emit['command'].index('--dump-cg') + 1]).resolve() != cg
            or sha(emit['input_export']) != emit['input_export_sha256']):
        raise ValueError('TM-R15-IR frontend emit command or original export differs')
    if (Path(build['frontend_emit_receipt']).resolve() != Path(provenance['frontend_emit_receipt']).resolve()
            or build['frontend_emit_receipt_sha256'] != provenance['frontend_emit_receipt_sha256']
            or build.get('CG_unchanged_after') is not True or build['pristine_R15_head'] != R15_HEAD
            or build['pristine_R15_compiler_sha256'] != R15_COMPILER_SHA
            or build['command'][:2] != [str(compiler), 'compile']
            or Path(build['command'][2]).resolve() != cg or Path(build['command'][3]).resolve() != binary
            or build['binary_sha256'] != identity['binary_sha256']
            or build['generated_source_sha256'] != identity['generated_source_sha256']
            or build['manifest_sha256'] != identity['manifest_sha256']
            or build['source_snapshot_sha256'] != provenance['backend_source_snapshot_sha256']):
        raise ValueError('TM-R15-IR official old build command/output differs')
    for suffix, field in (('', 'binary_sha256'), ('.cu', 'generated_source_sha256'), ('.plan.json', 'manifest_sha256')):
        if sha(str(binary) + suffix) != identity[field]:
            raise ValueError('TM-R15-IR actual backend output SHA differs')
    if identity['plan'].get('batch_hi') != emit['batch'] or emit['batch'] != build['batch']:
        raise ValueError('TM-R15-IR frontend/backend batch differs')
    return provenance


def attach_frontend_ir(binary, frontend_receipt, backend_receipt, root):
    """Add provenance metadata only; retain original complete identity separately."""
    binary = Path(binary).resolve()
    path = Path(str(binary) + '.identity.json')
    before = path.read_bytes()
    identity = json.loads(before)
    if identity.get('schema') != 1 or identity.get('artifact_id') != digest({key: value for key, value in identity.items() if key != 'artifact_id'}):
        raise ValueError('TM-R15-IR original identity digest differs')
    if 'frontend_ir_provenance' in identity['source']:
        raise ValueError('TM-R15-IR provenance already attached')
    emit = json.loads(Path(frontend_receipt).read_text())
    identity['source']['frontend_ir_provenance'] = dict(schema=1, label='TM-R15-IR',
        input_CG_path=emit['input_CG_path'], input_CG_sha256=emit['input_CG_sha256'],
        frontend_source_head=emit['frontend_source_head'], frontend_compiler=emit['frontend_compiler'],
        frontend_compiler_sha256=emit['frontend_compiler_sha256'],
        frontend_emit_receipt=str(Path(frontend_receipt).resolve()), frontend_emit_receipt_sha256=sha(frontend_receipt),
        backend_build_receipt=str(Path(backend_receipt).resolve()), backend_build_receipt_sha256=sha(backend_receipt),
        backend_source_snapshot_sha256=sha(str(binary) + '.source.json'))
    validate_frontend_ir(identity, root)
    original = Path(str(binary) + '.identity.original.json')
    if original.exists():
        raise ValueError('TM-R15-IR original identity backup already exists')
    original.write_bytes(before)
    identity['artifact_id'] = digest({key: value for key, value in identity.items() if key != 'artifact_id'})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(identity, indent=2) + '\n')
    temporary.replace(path)
    return identity
