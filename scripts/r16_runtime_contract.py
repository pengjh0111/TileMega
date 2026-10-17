# SPDX-License-Identifier: BSD-3-Clause
"""Bind a candidate's Python runtime independently of its compiled artifacts."""
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source(source):
    root=Path(source['root']).resolve()
    manifest=Path(source['manifest']).resolve()
    if digest(manifest)!=source['manifest_sha256']:
        raise ValueError('runtime manifest changed')
    value=json.loads(manifest.read_text())
    if value['root']!=str(root) or value['head']!=source['head']:
        raise ValueError('runtime manifest source differs')
    files=value['files']
    actual={str(p.relative_to(root/'python')) for p in (root/'python/tilemega').rglob('*.py')}
    if set(files)!=actual:
        raise ValueError('runtime Python manifest is incomplete')
    for name,sha in files.items():
        path=(root/'python'/name).resolve()
        if not path.is_relative_to(root/'python') or digest(path)!=sha:
            raise ValueError('runtime Python source changed: '+name)
    for key,name in [('runner_sha256','r16_run.py'),('hf_checker_sha256','hf_check.py')]:
        if source[key]!=files['tilemega/serving/'+name]:
            raise ValueError('runtime entry SHA differs from complete manifest')
    return value


def attest(source):
    manifest=validate_source(source)
    base=Path(source['root']).resolve()/'python'
    modules={}
    for name,module in list(sys.modules.items()):
        if name!='tilemega' and not name.startswith('tilemega.'):
            continue
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(base):
            raise ValueError('TileMega module imported outside frozen runtime: '+name)
        relative=str(path.relative_to(base))
        sha=digest(path)
        if manifest['files'].get(relative)!=sha:
            raise ValueError('imported TileMega module differs from frozen runtime: '+name)
        modules[name]=dict(path=str(path),sha256=sha)
    required={'tilemega.serving.r16_run','tilemega.serving.engine','tilemega.serving.plan',
              'tilemega.serving.baseline_common'}
    if not required.issubset(modules):
        raise ValueError('native runtime imports are incomplete')
    return dict(pid=os.getpid(),source=source,modules=modules)


def validate_receipt(candidate,value,directory):
    source=candidate.get('runtime_source')
    if source is None:return
    manifest=validate_source(source)
    directory=Path(directory)
    receipt=json.loads((directory/'measurements/runtime-execution.json').read_text())
    if receipt['pid']!=value['pid'] or receipt['source']!=source:
        raise ValueError('runtime execution receipt belongs to another process/source')
    modules=receipt['modules']
    required={'tilemega.serving.r16_run','tilemega.serving.engine','tilemega.serving.plan',
              'tilemega.serving.baseline_common'}
    if not required.issubset(modules):raise ValueError('actual runtime imports are incomplete')
    base=Path(source['root']).resolve()/'python'
    for name,row in modules.items():
        path=Path(row['path']).resolve()
        if (not name.startswith('tilemega') or not path.is_relative_to(base) or
            manifest['files'].get(str(path.relative_to(base)))!=row['sha256'] or
            digest(path)!=row['sha256']):
            raise ValueError('actual runtime import evidence differs: '+name)
    config=value['configuration']
    if any(config.get(k)!=source[k] for k in ('runner_sha256','hf_checker_sha256')):
        raise ValueError('measured runtime entry hashes differ')
    return receipt
