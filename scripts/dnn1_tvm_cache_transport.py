#!/usr/bin/env python3
"""Copy a completed VM to an absent, hardware-equivalent GPU cache identity.

This CPU-only transport never compiles or loads TVM/CUDA. It preserves the
original tuning device and requires fresh correctness/timing/guard evidence.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

VERSION = 'tvm-cache-transport-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def rename_absent(source, destination):
    # Linux renameat2 is atomic and refuses even an empty destination directory.
    libc = ctypes.CDLL(None, use_errno=True)
    operation = libc.renameat2
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def validate(source, identity):
    receipt = json.loads((source/'compiled.json').read_text())
    original = receipt['identity']
    if original.get('recipe', {}).get('schema') != 6:
        raise ValueError('only frozen recipe6 is supported')
    left, right = copy.deepcopy(original), copy.deepcopy(identity)
    old_gpu, new_gpu = left.pop('gpu_uuid'), right.pop('gpu_uuid')
    if not old_gpu or not new_gpu or old_gpu == new_gpu or left != right:
        raise ValueError('identities must differ only in nonempty GPU UUID')
    if original['compute_capability'] != [9, 0] or original['target_configuration']['arch'] != 'sm_90a':
        raise ValueError('transport is limited to the recorded SM90a target')
    if source.name != key(original) or sha(source/'model.so') != receipt['model_sha256']:
        raise ValueError('source cache identity or VM hash differs')
    files = dict(original['libraries'])
    files.update(original['tvm_source_identity']['modified_sources'])
    files.update(original['tvm_source_identity']['patches'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('frozen source/library differs: ' + path)
    return receipt


def transport(source, identity, destination_parent):
    source, parent = Path(source).resolve(), Path(destination_parent).resolve()
    destination = parent/key(identity)
    if destination.exists():
        raise FileExistsError(errno.EEXIST, 'refuse existing destination, including partial cache', str(destination))
    with (source/'cache.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = validate(source, identity)
        source_digest = sha(source/'compiled.json')
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.transport-', dir=parent))
        try:
            # Preserve IR, generated CUDA and tuning database provenance as well as VM.
            for item in source.iterdir():
                if item.name == 'cache.lock':
                    continue
                if item.is_symlink():
                    raise ValueError('source cache symlink unsupported: ' + str(item))
                if item.is_dir():
                    if any(path.is_symlink() for path in item.rglob('*')):
                        raise ValueError('source cache contains nested symlink: ' + str(item))
                    shutil.copytree(item, staging/item.name)
                else:
                    shutil.copy2(item, staging/item.name)
            output = copy.deepcopy(receipt)
            origin_gpu = receipt['identity']['gpu_uuid']
            # A second hop retains the first tuning GPU, never relabels its trial provenance.
            prior = receipt.get('diagnostics', {}).get('cache_transport', {})
            provenance = dict(version=VERSION, time=time.time(), source_cache=str(source),
                source_compiled_sha256=source_digest, source_vm_sha256=receipt['model_sha256'],
                tuning_gpu_uuid=prior.get('tuning_gpu_uuid', origin_gpu),
                source_gpu_uuid=origin_gpu, execution_gpu_uuid=identity['gpu_uuid'],
                source_identity=receipt['identity'], destination_identity=identity,
                compiler_or_model_changed=False, retuned=False,
                acceptance='unvalidated transport; fresh P-CORR5, P-DNN3 and strict PID audit required')
            output['identity'] = identity
            output['diagnostics'].update(cache_directory=str(destination), cache_key=key(identity),
                gpu_uuid=identity['gpu_uuid'], cache_identity=identity,
                model_library=str(destination/'model.so'), cache_transport=provenance)
            (staging/'compiled.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
            (staging/'transport.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
            if sha(source/'compiled.json') != source_digest or sha(staging/'model.so') != receipt['model_sha256']:
                raise ValueError('source changed during transport or VM copy differs')
            rename_absent(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dict(destination=str(destination), compiled_sha256=sha(destination/'compiled.json'),
                transport_sha256=sha(destination/'transport.json'), **provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--destination-parent', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(transport(args.source, json.loads(args.identity.read_text()), args.destination_parent), indent=2))


if __name__ == '__main__':
    main()
