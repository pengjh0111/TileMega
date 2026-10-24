"""Read allocated kernel resources from the final CUDA ELF, including RDC."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from .identity import sha


def kernel_resources(text: str, arch: str, kernels: list[str]) -> dict:
    """Select exact kernel names and reject ambiguous code objects."""
    if not arch or not kernels or len(set(kernels)) != len(kernels):
        raise ValueError('one architecture and distinct kernel names are required')
    records = []
    current_arch = None
    current = None

    def finish():
        nonlocal current
        if current is not None:
            records.append(current)
            current = None

    for line in text.splitlines():
        if line.strip().startswith('Fatbin '):
            finish()
            current_arch = None
        match = re.fullmatch(r'\s*arch\s*=\s*(sm_\w+)\s*', line)
        if match:
            finish()
            current_arch = match[1]
            continue
        match = re.fullmatch(r'\s*Function\s+(.+):\s*', line)
        if match:
            finish()
            if current_arch is None:
                raise ValueError('function resources lack an architecture')
            current = dict(arch=current_arch, name=match[1], values={})
            continue
        if current is not None:
            for key, raw in re.findall(r'\b([A-Z]+(?:\[\d+\])?):(-?\d+)\b', line):
                if key in current['values'] or int(raw) < 0:
                    raise ValueError('duplicate or negative linked resource field')
                current['values'][key] = int(raw)
    finish()

    selected = {}
    for name in kernels:
        matches = [r['values'] for r in records if r['arch'] == arch and r['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'expected one final {arch} resource record for kernel {name}')
        row = matches[0]
        if not {'REG', 'STACK', 'SHARED', 'LOCAL'} <= row.keys() or row['REG'] <= 0:
            raise ValueError('incomplete allocated kernel resources')
        # RDC device functions commonly report REG:0; they cannot supply a
        # kernel allocation or be substituted for the requested entry.
        selected[name] = dict(registers=row['REG'], stack_bytes=row['STACK'],
            static_smem_bytes=row['SHARED'], local_bytes=row['LOCAL'], raw=row)
    return selected


def inspect(binary: str | Path, arch: str, kernels: list[str],
            cuobjdump: str | Path = '/usr/local/cuda/bin/cuobjdump') -> dict:
    """Inspect a binary without loading CUDA or creating a device context."""
    binary = Path(binary).resolve()
    executable = Path(cuobjdump).resolve(strict=True)
    before = sha(binary)
    command = [str(executable), '--dump-resource-usage', str(binary)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    resources = kernel_resources(result.stdout, arch, kernels)
    if sha(binary) != before:
        raise ValueError('CUDA binary changed during linked resource inspection')
    return dict(binary_sha256=before, arch=arch, resources=resources,
        command=command, tool_sha256=sha(executable), stdout=result.stdout,
        stderr=result.stderr, scope='final linked ELF kernel allocation; no GPU query')
