#!/usr/bin/env python3
"""Reuse TileMega's exclusive GPU guard with this experiment's physical map."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.bench import gpu_guard


def install():
    config = json.loads((ROOT / 'configs/moe_profile/devices.json').read_text())
    original = gpu_guard.gpu

    def sample(index=None):
        row = original(index)
        actual = gpu_guard.subprocess.check_output([
            'nvidia-smi', '-i', str(row['gpu_index']), '--query-gpu=uuid,pci.bus_id',
            '--format=csv,noheader'], text=True).strip().split(',')
        uuid, pci = [value.strip() for value in actual]
        domain, bus, slot = pci.lower().split(':')
        pci = f'{int(domain, 16):04x}:{bus}:{slot}'
        expected = config['devices'][str(row['gpu_index'])]
        if (uuid, pci) != (expected['uuid'], expected['pci']):
            raise ValueError('physical GPU identity changed since experiment setup')
        row.update(gpu_uuid=uuid, pci_bus_id=pci,
                   resource_policy='moe-profile-physical-lock-owner-memory-v1')
        return row

    def idle(row, policy, needs=0, nonce=None):
        return (not row['owners'] and row['used_mib'] <= policy['max_idle_used_mib']
                and row['free_mib'] >= needs and not gpu_guard.throttle_invalid(row))

    gpu_guard.gpu = sample
    gpu_guard.idle = idle


def main():
    install()
    return gpu_guard.main()


if __name__ == '__main__':
    raise SystemExit(main())
