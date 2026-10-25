"""Explicit v2 pins must reject malformed coordinates before any device build."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('TILEMEGA_BIN', ROOT / 'build/tools/tilemega'))


def pin():
    cls = dict(tn=16, wn=1, d=1, part='kTileRange', prefetch=1)
    return dict(mx_decode_v2={**{name: dict(cls) for name in ('qkv', 'o', 'gate_up', 'down', 'head')},
                            'threads': 576, 'b_min': 1, 'attn_prefetch': 2})


def invoke(tmp_path, doc):
    if not BINARY.is_file():
        pytest.skip('tilemega host compiler is not built')
    config = tmp_path / 'pin.json'
    config.write_text(json.dumps(doc))
    result = subprocess.run([str(BINARY), 'compile', str(tmp_path / 'missing_bridge.json'),
        str(tmp_path / 'output/plan.so'), '--serving', 'decode', '--emit', 'serving',
        '--frontend', 'decoder', '--evaluate-configs', str(config), '--selection', 'predicted'],
        capture_output=True, text=True, env={**os.environ, 'CUDACXX': '/no/device/compiler/allowed'}, timeout=30)
    assert result.returncode != 0
    assert not (tmp_path / 'output/plan.so.build_command.txt').exists()
    return result.stderr


def test_v2_route_parses_part_without_attempting_legacy_or_device_build(tmp_path):
    doc = pin()
    doc['mx_decode_v2']['qkv']['part'] = 'unknown'
    assert 'MX-5 pin has invalid part' in invoke(tmp_path, doc)


def test_v2_coordinates_require_integers(tmp_path):
    doc = pin()
    doc['mx_decode_v2']['gate_up']['wn'] = 1.5
    assert 'MX-5 pin requires integer wn' in invoke(tmp_path, doc)


def test_v2_manual_pin_cannot_claim_solver_selection(tmp_path):
    doc = pin()
    doc['mx_decode_v2']['purpose'] = 'solver_selected'
    assert 'purpose manual or resource_probe' in invoke(tmp_path, doc)


def test_v2_missing_coordinates_stay_errors(tmp_path):
    doc = pin()
    del doc['mx_decode_v2']['head']['prefetch']
    assert 'prefetch' in invoke(tmp_path, doc)


def test_legacy_pin_does_not_enter_v2_route(tmp_path):
    assert 'missing_bridge.json' in invoke(tmp_path, {'mx_decode': {}})
