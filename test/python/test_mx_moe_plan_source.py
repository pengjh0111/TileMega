"""Compare actual generated programs and recipes with their physical pins."""
import copy
import json
from pathlib import Path

import pytest

from tilemega.moe.plan_source import validate_body_snapshot, validate_geometry, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / 'runs/mx5/moe_rank_whole_program_candidate_v1'
CLOSURE = ROOT / 'runs/mx5/moe_rank_adopted_body_source_closure_cpu_v2/ACTUAL_FIVE_BODY_HEADER_CLOSURES_MATCH_ADOPTED_CPU.json'
pytestmark = pytest.mark.skipif(not PLANS.is_dir() or not CLOSURE.is_file(),
    reason='requires the existing native manual programs and frozen measured dependency closure; no GPU')


def read(batch=1, watchdog=0, prompt=False):
    binary = PLANS / ('B%d_w%d_p%d' % (batch, watchdog, int(prompt))) / 'plan.so'
    source = Path(str(binary) + '.cu').read_text()
    pin = json.loads(Path(str(binary) + '.mx5_pin.json').read_text())
    manifest = json.loads(Path(str(binary) + '.plan.json').read_text())
    snapshot = json.loads(Path(str(binary) + '.source.json').read_text())
    return source, pin, manifest, snapshot


@pytest.mark.parametrize('batch', (1, 16))
@pytest.mark.parametrize('watchdog', (0, 1))
@pytest.mark.parametrize('prompt', (False, True))
def test_actual_generated_body_shapes_packed_weights_and_measured_headers(batch, watchdog, prompt):
    source, pin, manifest, snapshot = read(batch, watchdog, prompt)
    assert validate_geometry(source, pin, batch, prompt)['classes'] == 6
    assert validate_manifest(manifest, pin)['packed_weight_buffers'] == 121
    assert validate_body_snapshot(snapshot, json.loads(CLOSURE.read_text()))['compiled_measured_header_matches'] > 0


def test_routing_dtype_must_match_both_dispatch_and_combine():
    source, pin, _, _ = read()
    bf16 = source.replace('kRoutingFp32=true', 'kRoutingFp32=false')
    with pytest.raises(ValueError, match='body alias differs: Down'):
        validate_geometry(bf16, pin, 1, False, routing_weights_fp32=False)
    bf16 = bf16.replace('MxMoeDownCompletionV2<true>', 'MxMoeDownCompletionV2<false>')
    assert validate_geometry(bf16, pin, 1, False, routing_weights_fp32=False)['routing_weights_fp32'] is False
    with pytest.raises(ValueError, match='constant differs: kRoutingFp32'):
        validate_geometry(bf16, pin, 1, False)


@pytest.mark.parametrize('batch', (1, 16))
@pytest.mark.parametrize('prompt', (False, True))
def test_actual_rejected_576_program_geometry_does_not_grant_resource_acceptance(batch, prompt):
    binary = ROOT / ('runs/mx5/moe_rank_adopted_min576_programs_cpu_v1/B%d_p%d/plan.so' % (batch, int(prompt)))
    if not binary.is_file():
        pytest.skip('requires the existing rejected minimum576 program')
    source = Path(str(binary) + '.cu').read_text()
    pin = json.loads(Path(str(binary) + '.mx5_pin.json').read_text())
    resource = json.loads(Path(str(binary) + '.resources.json').read_text())
    result = validate_geometry(source, pin, batch, prompt)
    assert result['threads'] == 576 and result['not_resource_or_numerical_or_model_acceptance']
    assert resource['violations'] and resource['kernel']['stack_bytes'] > 0


@pytest.mark.parametrize('change', ('threads', 'batch', 'phase', 'qkv', 'o', 'router',
    'gate_up', 'down', 'head', 'attention', 'dispatch', 'down_last', 'shape', 'routing_dtype'))
def test_changed_actual_code_or_pin_is_rejected(change):
    source, pin, _, _ = read()
    if change == 'threads':
        pin['mx_moe_v2']['threads'] = 576
    elif change == 'batch':
        source = source.replace('kBatch=1,', 'kBatch=2,')
    elif change == 'phase':
        source = source.replace('kPrompt=false,', 'kPrompt=true,')
    elif change in ('qkv', 'o', 'router', 'gate_up', 'down', 'head'):
        pin['mx_moe_v2'][change]['prefetch'] += 1
    elif change == 'attention':
        pin['mx_moe_v2']['b_min'] = 8
    elif change == 'dispatch':
        source = source.replace('kIntermediate,64,64,kRoutingFp32', 'kIntermediate,32,64,kRoutingFp32')
    elif change == 'down_last':
        source = source.replace('using DownLast =', 'using WrongDownLast =')
    elif change == 'routing_dtype':
        source = source.replace('kRoutingFp32=true', 'kRoutingFp32=false')
    else:
        source = source.replace('MxStaticShapeV2<5120,2048,kBatch', 'MxStaticShapeV2<4096,2048,kBatch')
    with pytest.raises(ValueError):
        validate_geometry(source, pin, 1, False)


@pytest.mark.parametrize('change', ('tile_n', 'tile_k', 'layout', 'missing_layer', 'class', 'header'))
def test_stale_recipe_or_compiled_header_is_rejected(change):
    _, pin, manifest, snapshot = read()
    if change == 'header':
        closure = json.loads(CLOSURE.read_text())
        path = closure['families'][0]['headers'][0]['adopted']['path']
        name = 'include/' + path.split('/include/', 1)[1]
        snapshot['source_files'][name] = '0' * 64
        with pytest.raises(ValueError):
            validate_body_snapshot(snapshot, closure)
        return
    row = next(r for r in manifest['buffers'] if r.get('recipe', {}).get('kind') == 'tile_pages')
    if change in ('tile_n', 'tile_k'):
        row['recipe'][change] *= 2
    elif change == 'layout':
        row['recipe']['layout'] = 'stale_layout'
    elif change == 'missing_layer':
        manifest['buffers'].remove(row)
    else:
        manifest['classes']['qkv'] = copy.deepcopy(manifest['classes']['qkv'])
        manifest['classes']['qkv']['tn'] *= 2
    with pytest.raises(ValueError):
        validate_manifest(manifest, pin)
