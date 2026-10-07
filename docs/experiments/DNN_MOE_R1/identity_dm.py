#!/usr/bin/env python3
"""FX-25 fields for DM-1 builds until the upstream identity API is merged."""
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(root, compiler):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)
    paths = git('ls-files', '-z', 'include', 'lib', 'python', 'tools',
                'CMakeLists.txt', 'cmake').decode().split('\0')
    files = {p: sha(Path(root) / p) for p in paths if p and (Path(root) / p).is_file()}
    return dict(head=git('rev-parse', 'HEAD').decode().strip(),
                diff_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
                source_files=files, compiler_binary_sha256=sha(compiler),
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


def generate(so, source, executor='L1', loop=False):
    so = Path(so)
    command = shlex.split(Path(str(so) + '.build_command.txt').read_text())
    cu = Path(str(so) + '.cu')
    manifest = Path(str(so) + '.plan.json')
    plan = json.loads(manifest.read_text())
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
    vllm_version = subprocess.check_output(
        [vllm_python, '-c', 'from importlib.metadata import version; print(version("vllm"))'],
        text=True).strip()
    # All callable launch entries are retained; measurements identify their
    # executed entry from this map rather than borrowing another mode's row.
    identity = dict(schema='tilemega.dm1.identity.v1', source=source,
                    cu_sha256=sha(cu), so_sha256=sha(so), manifest_sha256=sha(manifest),
                    nvcc_version=subprocess.check_output([command[0], '--version'], text=True),
                    baselines=dict(vllm_version=vllm_version, vllm_python=vllm_python),
                    compiler_command=command, macros=definitions, arch=arch, trace=trace,
                    execution=dict(pg=plan['pg'], executor=executor, loop=loop,
                                   pdl=plan.get('pdl'), phase=plan['phase']),
                    implementations=dict(gemm='ServingGemmTaskBody',
                                         paged_gemm='PagedGemmTaskBody' if plan['pg'] == 'pages' else None,
                                         attention_qperkv=definitions.get('TILEMEGA_SERVING_QPERKV'),
                                         attention_kv_block=plan['attention_kv_block'],
                                         attention_query_rows=plan['attention_query_rows'],
                                         gemms=plan['gemms']),
                    placement={k: plan[k] for k in ('kappa', 'grid', 'residency', 'pages')},
                    kernels=kernels, spill=any(r['spill'] for r in kernels.values()))
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
    identity['artifact_id'] = claimed
    return identity
