"""Validate physical GPU routing and guarded CUDA recovery diagnostics."""
import json
import re


def pci_address(value):
    match = re.fullmatch(r'([0-9a-f]{4,8}):([0-9a-f]{2}):([0-9a-f]{2}\.[0-7])',
                         value.lower())
    if match is None:
        raise ValueError('invalid GPU PCI address')
    return f'{int(match[1], 16):04x}:{match[2]}:{match[3]}'


def inventory(text):
    devices = {}
    for line in text.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) != 3:
            raise ValueError('GPU inventory requires index, UUID and PCI address')
        index, uuid, pci = fields
        if not index.isdecimal() or not uuid.startswith('GPU-'):
            raise ValueError('invalid GPU inventory identity')
        index = int(index)
        if index in devices or any(row['uuid'] == uuid for row in devices.values()):
            raise ValueError('duplicate GPU inventory identity')
        devices[index] = dict(uuid=uuid, pci=pci_address(pci))
    return devices


def matches(expected, observed):
    """Require the original four physical devices before enabling fallback."""
    wanted = {int(index): dict(uuid=row['uuid'], pci=pci_address(row['pci']))
              for index, row in expected['devices'].items()}
    return wanted == observed


def require_cuda(result, expected_uuid, index, guard):
    """A successful diagnostic wrapper does not imply CUDA initialized."""
    if guard.get('code') != 0 or not result.get('diagnostic_only'):
        raise ValueError('CUDA recovery probe lacks an accepted diagnostic guard')
    rows = result.get('rows', [])
    if len(rows) != 2 or {row['mask'] for row in rows} != {str(index), expected_uuid}:
        raise ValueError('CUDA recovery probe changed its physical device')
    pids = set()
    for row in rows:
        if row['exit_code'] or row['stderr']:
            raise ValueError('CUDA recovery child failed')
        probe = json.loads(row['stdout'])
        if (probe['mask'] != row['mask'] or probe['cuInit'] != 0
                or probe['cuDeviceGetCount'] != 0 or probe['count'] < 1):
            raise ValueError('CUDA initialization has not recovered')
        if not isinstance(probe['pid'], int) or probe['pid'] <= 0:
            raise ValueError('CUDA recovery child lacks a valid PID')
        pids.add(probe['pid'])
    if len(pids) != 2:
        raise ValueError('CUDA recovery requires two fresh initialization processes')
