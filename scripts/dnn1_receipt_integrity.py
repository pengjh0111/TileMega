"""Verify the official post-process device annotation against original bytes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import dnn1_baselines as base

FIELDS = ('physical_device_identity', 'physical_device_identity_file',
          'physical_device_identity_sha256')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(process, identity_file, wrapper_file, gpu):
    path = Path(process['receipt'])
    current = sha(path)
    expected = process['receipt_sha256']
    if current == expected:
        return dict(mode='unchanged bytes', original_sha256=expected, current_sha256=current)
    value = json.loads(path.read_text())
    if any(key not in value for key in FIELDS):
        raise ValueError('changed receipt lacks the complete official device annotation')
    original = dict(value)
    for key in FIELDS:
        del original[key]
    recovered = hashlib.sha256((json.dumps(original, indent=2) + '\n').encode()).hexdigest()
    if recovered != expected:
        raise ValueError('stripping official annotation does not recover original byte SHA')
    identity_file, wrapper_file = Path(identity_file), Path(wrapper_file)
    if Path(value['physical_device_identity_file']).resolve() != identity_file.resolve():
        raise ValueError('annotation points outside the original outer guard identity file')
    identity = json.loads(identity_file.read_text())
    if (value['physical_device_identity'] != identity or
            value['physical_device_identity_sha256'] != sha(identity_file)):
        raise ValueError('device annotation differs from the original identity contents/hash')
    canonical = lambda x: str(x).removeprefix('GPU-').lower()
    if (identity.get('wrapper_sha256') != sha(wrapper_file) or
            identity.get('gpu_index') != gpu or value.get('gpu_index') != gpu or
            identity.get('cuda_pci_bus_id', '').lower() != base.PCI[gpu] or
            identity.get('numa_node') != (0 if gpu < 4 else 1) or
            identity.get('guard_policy_sha256') != sha(base.POLICY) or
            identity.get('outer_guard_holds_percard_lock') is not True or
            identity.get('probe_exit_code') != 0 or
            canonical(identity.get('cuda_uuid')) != canonical(identity.get('nvml_uuid'))):
        raise ValueError('official wrapper SHA, GPU identity or guarded probe differs')
    diag = value.get('configuration', {}).get('tvm_diag', {})
    uuid = diag.get('cache_identity', {}).get('gpu_uuid')
    if not uuid or canonical(uuid) != canonical(identity['cuda_uuid']):
        raise ValueError('compiled VM and annotated physical GPU UUID differ')
    return dict(mode='exact official annotation only', original_sha256=expected,
        recovered_original_sha256=recovered, current_sha256=current,
        annotation_fields=list(FIELDS), identity_file=str(identity_file),
        identity_file_sha256=sha(identity_file), wrapper_file=str(wrapper_file),
        wrapper_sha256=sha(wrapper_file), gpu=gpu, cuda_uuid=identity['cuda_uuid'])
