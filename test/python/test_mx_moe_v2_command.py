"""CPU-only native MoE route checks against an actual exported bridge."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('TILEMEGA_BIN', ROOT / 'build/tools/tilemega'))
BRIDGE = Path(os.environ.get('TILEMEGA_MOE_BRIDGE', ROOT / 'runs/mx1/export/qwen3_30b_a3b_l24_decode/bridge.json'))


def pin():
    dense = dict(tn=64, wn=1, d=1, part='kTileRange', prefetch=1)
    return {'mx_moe_v2': dict(qkv=dict(dense), o=dict(dense),
        router=dict(tn=32, wn=1, d=1, part='kStreamK', prefetch=1),
        gate_up=dict(tn=64, wn=2, d=1, part='kTileRange', prefetch=0),
        down=dict(tn=64, wn=2, d=1, part='kTileRange', prefetch=1),
        head=dict(tn=32, wn=8, d=1, part='kTileRange', prefetch=1),
        threads=576, b_min=4, attn_prefetch=0, purpose='manual')}


def reject(tmp_path, doc=None, extra=(), phase='decode', batch=1):
    if not BINARY.is_file() or not BRIDGE.is_file():
        pytest.skip('host compiler and actual MoE export are required')
    out = tmp_path / 'plan.so'
    argv = [str(BINARY), 'compile', str(BRIDGE), str(out), '--serving', phase,
        '--emit', 'serving', '--frontend', 'decoder', '--batch', str(batch), '--selection', 'predicted']
    if doc is not None:
        path = tmp_path / 'pin.json'
        path.write_text(json.dumps(doc))
        argv.extend(['--evaluate-configs', str(path)])
    result = subprocess.run(argv + list(extra), capture_output=True, text=True, timeout=60,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MACA_VISIBLE_DEVICES': '', 'CUDACXX': '/no/device/compiler/allowed'})
    assert result.returncode != 0
    assert not list(tmp_path.glob('plan.so*'))
    return result.stderr


def test_moe_without_measured_configuration_rejects_legacy_lowering(tmp_path):
    assert 'explicit diagnostic mx_moe_v2 pin' in reject(tmp_path)


@pytest.mark.parametrize('batch', [1, 16])
def test_moe_rejects_interpreted_executor_flags(tmp_path, batch):
    assert 'requires native stages' in reject(tmp_path, pin(), ('--pg', 'l2'), batch=batch)


@pytest.mark.parametrize('phase,extra', [('decode', ('--mx-prompt', '1')),
    ('prefill', ('--mx-prompt', '1', '--prompt-len', '32')),
    ('prefill', ('--mx-prompt', '2'))])
def test_prompt_route_requires_64_token_prefill(tmp_path, phase, extra):
    assert '--mx-prompt' in reject(tmp_path, pin(), extra, phase)


def test_prompt_uses_native_analysis_and_diagnostic_configuration(tmp_path):
    assert 'explicit diagnostic mx_moe_v2 pin' in reject(tmp_path, extra=('--mx-prompt', '1'), phase='prefill')


@pytest.mark.parametrize('batch', [1, 16])
def test_grouped_streamk_is_rejected_before_device_build(tmp_path, batch):
    doc = pin()
    doc['mx_moe_v2']['gate_up']['part'] = 'kStreamK'
    assert 'grouped units require TileRange' in reject(tmp_path, doc, batch=batch)


def test_manual_pin_cannot_claim_model_selection(tmp_path):
    doc = pin()
    doc['mx_moe_v2']['purpose'] = 'solver_selected'
    assert 'purpose manual or resource_probe' in reject(tmp_path, doc)


def test_moe_coordinates_require_integral_values(tmp_path):
    doc = pin()
    doc['mx_moe_v2']['router']['wn'] = 1.5
    assert 'integer wn' in reject(tmp_path, doc)


def test_prompt_cannot_enter_dense_v2_route(tmp_path):
    assert 'mx_moe_v2 configuration' in reject(tmp_path, {'mx_decode_v2': {}}, ('--mx-prompt', '1'), 'prefill')
