#!/usr/bin/env python3
"""Compile the scope probe for both architectures; never launch sm_120 code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / 'cluster_compile'


def main():
    RAW.mkdir(exist_ok=True)
    free = shutil.disk_usage(RAW).free // 2**20
    print(f'DISK NEED_MIB=512 FREE_MIB={free}', flush=True)
    if free < 512:
        raise RuntimeError('insufficient disk before compilation')
    commands = []
    for arch in ('sm_89', 'sm_120'):
        for kind in ('cubin', 'ptx'):
            cmd = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', f'-arch={arch}',
                   f'-I{REPO}/include', f'-I{REPO}/third_party/cutlass/include', f'-{kind}',
                   str(REPO / 'test/litmus/sync_v3_cluster_compile.cu'), '-o', str(RAW / f'probe_{arch}.{kind}')]
            with (RAW / f'{arch}_{kind}.log').open('w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            commands.append(dict(command=cmd, exit_code=result.returncode))
            (RAW / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            result.check_returncode()
    ptx = (RAW / 'probe_sm_120.ptx').read_text()
    if 'atom.acq_rel.cluster.shared::cluster.add.u64' not in ptx:
        raise ValueError('cluster instruction is missing from sm_120 PTX')
    if 'atom.acq_rel.cluster.shared::cluster.add.u64' in (RAW / 'probe_sm_89.ptx').read_text():
        raise ValueError('sm_89 did not discard the cluster branch')
    inputs = [REPO / 'test/litmus/sync_v3_cluster_compile.cu',
              REPO / 'include/tilemega/Codegen/tasks/ClusterSync.cuh',
              REPO / 'include/tilemega/Target/ArchDispatch.h']
    (RAW / 'inputs.json').write_text(json.dumps({str(p.relative_to(REPO)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)+'\n')
    print('COMPILE sm_89 fallback and sm_120 cluster atomic PASS; sm_120 not executed', flush=True)


if __name__ == '__main__':
    main()
