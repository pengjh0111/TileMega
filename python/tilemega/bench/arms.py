"""Register immutable binaries without loading a CUDA context."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def register(arm):
    row = dict(arm)
    if row['kind'] not in ('tm', 'vllm'):
        raise ValueError('supported arms are tm and vllm')
    if row['kind'] == 'vllm':
        return row
    binaries = {}
    for phase in ('prefill', 'decode'):
        path = Path(row[phase]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        record = {'path': str(path), 'sha256': sha256(path)}
        for suffix, key in (('.plan.json', 'manifest'), ('.identity.json', 'identity')):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                record[key] = json.loads(sidecar.read_text())
                record[key + '_sha256'] = sha256(sidecar)
        binaries[phase] = record
        row[phase] = str(path)
    row['binaries'] = binaries
    return row


def check_binaries(arm):
    for phase, record in arm.get('binaries', {}).items():
        path = Path(arm[phase])
        if sha256(path) != record['sha256']:
            raise ValueError('binary changed after registration: ' + str(path))
        for suffix, key in (('.plan.json', 'manifest'), ('.identity.json', 'identity')):
            if key + '_sha256' in record and sha256(str(path) + suffix) != record[key + '_sha256']:
                raise ValueError('metadata changed after registration: ' + str(path))
