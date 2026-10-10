# SPDX-License-Identifier: BSD-3-Clause
"""FX-25 fields for DM-1 builds until the upstream identity API is merged."""
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from .compiler_macros import capture as capture_macros


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(root, compiler):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)
    paths = git('ls-files', '--cached', '--others', '--exclude-standard', '-z',
                'include', 'lib', 'python', 'tools', 'CMakeLists.txt', 'cmake').decode().split('\0')
    files = {p: sha(Path(root) / p) for p in paths if p and '__pycache__' not in p
             and not p.endswith('.pyc') and (Path(root) / p).is_file()}
    return dict(root=str(Path(root).resolve()),
                head=git('rev-parse', 'HEAD').decode().strip(),
                diff_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
                source_files=files, compiler_path=str(Path(compiler).resolve()),
                compiler_binary_sha256=sha(compiler),
                compiler_version=json.loads(subprocess.check_output(
                    [str(compiler), 'version', '--json'], text=True)))


def resources(text):
    entries = {}
    current = None
    for line in text.splitlines():
        found = re.search(r"Compiling entry function '([^']+)'", line)
        if found:
            current = found[1]
            entries[current] = dict(registers=None, stack_bytes=0,
                                    spill_store_bytes=0, spill_load_bytes=0,
                                    smem_bytes=0, resource_lines=[])
        if current is None:
            continue
        row = entries[current]
        if re.search(r'(Used \d+ registers|stack frame|spill stores|spill loads|bytes smem)', line):
            row['resource_lines'].append(line.strip())
        for pattern, key in [(r'Used (\d+) registers', 'registers'),
                             (r'(\d+) bytes stack frame', 'stack_bytes'),
                             (r'(\d+) bytes spill stores', 'spill_store_bytes'),
                             (r'(\d+) bytes spill loads', 'spill_load_bytes'),
                             (r'(\d+) bytes smem', 'smem_bytes')]:
            match = re.search(pattern, line)
            if match:
                row[key] = max(row[key] or 0, int(match[1]))
    if not entries or any(v['registers'] is None for v in entries.values()):
        raise ValueError('missing ptxas kernel resources')
    for row in entries.values():
        row['spill'] = bool(row['spill_store_bytes'] or row['spill_load_bytes'])
    return entries


def validate_snapshot_inputs(source):
    root = Path(source['root'])
    if not source.get('source_files'):
        raise ValueError('build snapshot lacks source input hashes')
    for name, expected in source['source_files'].items():
        if sha(root / name) != expected:
            raise ValueError('build source changed after its before-build snapshot: ' + name)
    compiler = Path(source.get('compiler_path', root / 'build-dm/tools/tilemega'))
    if sha(compiler) != source['compiler_binary_sha256']:
        raise ValueError('plan compiler changed after its before-build snapshot')


def generate(so, source, executor='L1', loop=False):
    so = Path(so)
    command = shlex.split(Path(str(so) + '.build_command.txt').read_text())
    cu = Path(str(so) + '.cu')
    manifest = Path(str(so) + '.plan.json')
    plan = json.loads(manifest.read_text())
    validate_snapshot_inputs(source)
    complete = capture_macros(command, Path(str(so) + '.macro_capture'))
    validate_snapshot_inputs(source)
    definitions = dict(re.findall(r'^#define\s+(\w+)\s+([^\n]+)', cu.read_text(), re.M))
    for arg in command:
        if arg.startswith('-D'):
            name, _, value = arg[2:].partition('=')
            definitions[name] = value or '1'
    trace = any((name.startswith('TILEMEGA_TRACE') or name == 'TILEMEGA_PAGE_TRACE')
                and value != '0'
                for name, value in definitions.items())
    arch = next((v.split('=', 1)[1] for v in command if v.startswith('-arch=')), None)
    if arch is None:
        raise ValueError('compiler command lacks an explicit architecture')
    kernels = resources(Path(str(so) + '.ptxas.log').read_text())
    vllm_python = os.environ.get('TILEMEGA_VLLM_PYTHON', '/root/venv_vllm/bin/python')
    vllm_version = None
    if Path(vllm_python).is_file():
        vllm_version = subprocess.check_output(
            [vllm_python, '-c', 'from importlib.metadata import version; print(version("vllm"))'],
            text=True).strip()
    # All callable launch entries are retained; measurements identify their
    # executed entry from this map rather than borrowing another mode's row.
    identity = dict(schema='tilemega.dm1.identity.v1', source=source,
                    cu_sha256=sha(cu), so_sha256=sha(so), binary_sha256=sha(so),
                    manifest_sha256=sha(manifest),
                    nvcc_version=subprocess.check_output([command[0], '--version'], text=True),
                    baselines=dict(vllm_version=vllm_version, vllm_python=vllm_python),
                    compiler_command=command, macros=definitions, complete_macros=complete,
                    arch=arch, trace=trace,
                    execution=dict(pg=plan['pg'], executor=executor, loop=loop,
                                   pdl=plan.get('pdl'), phase=plan['phase']),
                    implementations=dict(gemm='ServingGemmTaskBody',
                                         paged_gemm='PagedGemmTaskBody' if plan['pg'] == 'pages' else None,
                                         attention_qperkv=definitions.get('TILEMEGA_SERVING_QPERKV'),
                                         attention_kv_block=plan['attention_kv_block'],
                                         attention_query_rows=plan['attention_query_rows'],
                                         gemms=plan['gemms'], frontend=plan.get('frontend'),
                                         reuse=plan.get('reuse', 'none'),
                                         deferred_ln=plan.get('deferred_ln'),
                                         dwpw_fuse=plan.get('dwpw_fuse'),
                                         moe_dynamic=plan.get('moe_dynamic', False),
                                         moe_opaque=plan.get('moe_opaque', False),
                                         moe_gemv=plan.get('moe_gemv', False),
                                         shared_weight_layout_sha256=plan.get('shared_weight_layout_sha256'),
                                         dm_pool_la=plan.get('dm_pool_la', False),
                                         dm_moe_la=plan.get('dm_moe_la', False),
                                         moe_binding=plan.get('moe_binding'),
                                         moe_bm=plan.get('moe_bm'),
                                         memory_arena_bytes=plan.get('memory_arena_bytes', 0)),
                    placement={k: plan[k] for k in ('kappa', 'grid', 'residency', 'pages')},
                    kernels=kernels, spill=any(r['spill'] for r in kernels.values()))
    for key in ('dm_reduction_mask', 'dm_reduction_proved_stages'):
        if key in plan:
            identity['implementations'][key] = plan[key]
    if 'dnn_structure' in plan:
        identity['implementations']['dnn_structure'] = plan['dnn_structure']
        identity['implementations']['dnn_structure_search_sha256'] = plan['dnn_structure_search_sha256']
    if 'moe_structure' in plan:
        identity['implementations']['moe_structure'] = plan['moe_structure']
        identity['implementations']['moe_structure_search_sha256'] = plan['moe_structure_search_sha256']
    if 'routing_profile' in plan:
        identity['implementations']['routing_profile'] = plan['routing_profile']
    if 'moe_profile_pricing' in plan:
        identity['implementations']['moe_profile_pricing'] = plan['moe_profile_pricing']
    identity['artifact_id'] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    Path(str(so) + '.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    return identity


def verify(so):
    identity = json.loads(Path(str(so) + '.identity.json').read_text())
    claimed = identity.pop('artifact_id')
    computed = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    if computed != claimed or sha(so) != identity['so_sha256']:
        raise ValueError('artifact identity or binary changed')
    if sha(str(so) + '.cu') != identity['cu_sha256'] or \
            sha(str(so) + '.plan.json') != identity['manifest_sha256']:
        raise ValueError('source or manifest changed')
    complete = identity.get('complete_macros')
    if complete is not None:
        if complete['compiler_command'] != identity['compiler_command']:
            raise ValueError('macro capture compiler command differs from the build')
        for unit in complete['translation_units']:
            if sha(unit['macros_file']) != unit['macros_sha256']:
                raise ValueError('complete preprocessor macro table changed')
    identity['artifact_id'] = claimed
    return identity


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--compiler',type=Path)
    parser.add_argument('--so',type=Path)
    parser.add_argument('--check-reuse',type=Path)
    args=parser.parse_args()
    if args.check_reuse:
        previous=verify(args.check_reuse)['source']
        current=json.loads(args.snapshot.read_text())
        if any(previous.get(k)!=current.get(k) for k in ('source_files','compiler_binary_sha256')):
            raise ValueError('DM reuse source/compiler identity mismatch')
        return
    if args.capture:
        if args.compiler is None:parser.error('--capture requires --compiler')
        args.snapshot.write_text(json.dumps(source_snapshot(args.root,args.compiler),indent=2)+'\n')
    else:
        if args.so is None:parser.error('--so is required after compilation')
        generate(args.so,json.loads(args.snapshot.read_text()))

if __name__=='__main__':main()
