"""CPU-only CLI integration using synthetic costs and the actual MoE bridge."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(os.environ.get('TILEMEGA_BIN', ROOT / 'build/tools/tilemega'))
DOMAIN = ROOT / 'build/mx_moe_selection_v2'
BRIDGE = ROOT / 'runs/mx1/export/qwen3_30b_a3b_l24_decode/bridge.json'
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in (TOOL, DOMAIN, BRIDGE)),
    reason='requires native host tools and the existing MoE export; no GPU')


def synthetic_table():
    result = subprocess.run([str(DOMAIN), '--dump-domain'], cwd=ROOT,
        capture_output=True, text=True, check=True)
    rows = []

    def append(kind, key, dispatch=None):
        implementation = 'moe_grouped' if kind in ('moe_gate_up', 'moe_down_combine') else kind
        row = dict(kind=kind, key=key, median_ns=100. + len(rows), p90_ns=101. + len(rows),
            spread=.01, processes=3, r_next=240 if kind == 'attention' else 120,
            rounds=[dict(round=i) for i in (1, 2, 3)],
            resources=dict(mtreg=160, local_bytes=0, static_smem_bytes=0, resident_ctas=1),
            provenance=dict(executor='mx_static_stage_bench', implementation='mx5_' + implementation + '_v2',
                source_sha256='a' * 64, binary_sha256='b' * 64, base_head='synthetic_cpu_contract'))
        if dispatch:
            row['dispatch_context'] = dispatch
        rows.append(row)

    for line in result.stdout.splitlines():
        values = line.split(',')
        if len(values) != 12:
            continue
        stage = values[0]
        threads, batch, n, k, active, tn, wn, depth, part, op, norm = map(int, values[1:])
        if threads != 512 or batch != 1:
            continue
        kind = dict(qkv='gemm', o='gemm', router='moe_router', gate_up='moe_gate_up',
            down='moe_down_combine', head='head')[stage]
        key = dict(n=n, k=k, b=batch, threads=threads, tn=tn, wn=wn, depth=depth,
            prefetch=0 if stage == 'gate_up' else depth)
        if stage != 'head':
            key.update(part=part, op=op, norm=norm)
        if active:
            key['n_active'] = active
        if stage == 'router':
            for gu in (32, 64, 128):
                for dn in (16, 32, 64, 128):
                    append(kind, key, dict(gate_up_tn=gu, down_tn=dn))
        else:
            append(kind, key)
    for bmin in (1, 2, 4, 8):
        for length in (65, 256, 512, 768, 1088):
            append('attention', dict(dim=128, q=8, hkv=4, b=1, length=length,
                b_min=bmin, qk_norm=1, capacity=1088, threads=512, prefetch_kind=0))
    return dict(schema_version=1, table_kind='mx5_moe_geometry_outcomes', rows=rows,
        rejections=[], synthetic_fixture=True, qualified=False)


def invoke(table, output, *extra, bridge=BRIDGE):
    return subprocess.run([str(TOOL), 'compile', str(bridge), str(output),
        '--serving', 'decode', '--emit', 'serving', '--frontend', 'decoder',
        '--batch', '1', '--capacity', '1088', '--past-range', '64:1086',
        '--runtime-target', str(ROOT / 'configs/targets/xcore1000.json'),
        '--mx-decode-v2-solve', str(table), '--mx-decode-v2-threads', '512',
        '--selection', 'predicted', *extra], cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES=''))


def test_actual_moe_bridge_routes_to_cpu_top4_without_execution_or_overwrite(tmp_path):
    table = tmp_path / 'synthetic_geometry.json'
    table.write_text(json.dumps(synthetic_table()))
    output = tmp_path / 'plan.so'
    result = invoke(table, output)
    assert result.returncode == 0, result.stderr
    preview = Path(str(output) + '.mx5_candidates.json')
    original = preview.read_bytes()
    doc = json.loads(original)
    audit = subprocess.run([str(DOMAIN), '--audit-rows', str(table), '1', '512'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    coverage = json.loads(audit.stdout)
    assert coverage['complete_geometry'] and coverage['missing'] == []
    assert coverage['measured'] == len(json.loads(table.read_text())['rows'])
    assert coverage['stage_sizes'][6] == 4 and not coverage['execution_authorized']
    assert (doc['layers'], doc['batch'], doc['stages_per_step']) == (24, 1, 145)
    assert len(doc['candidates']) == 4 and doc['coverage']['missing'] == 0
    assert not any(doc[k] for k in ('execution_authorized', 'file_qualification_implied',
        'solver_selected', 'final', 'default_enabled'))
    assert not output.exists() and not Path(str(output) + '.cu').exists()
    assert doc['input']['path'] == str(BRIDGE)
    assert doc['stage_table']['path'] == str(table)
    for candidate in doc['candidates']:
        assert candidate['prompt_pair_required']
        assert len(candidate['evidence_row_indices']['attention']) == 5
        costs = candidate['predicted_stage_ns']
        assert candidate['predicted_tpot_ns'] == pytest.approx(
            costs['head'] + 24 * sum(v for k, v in costs.items() if k != 'head'), rel=1e-12)
    retry = invoke(table, output)
    assert retry.returncode != 0 and 'existing preview is preserved' in retry.stderr
    assert preview.read_bytes() == original


def test_moe_table_cannot_be_applied_to_a_dense_bridge(tmp_path):
    bridge = ROOT / 'runs/mx1/export/qwen3_0_6b_decode/bridge.json'
    if not bridge.is_file():
        pytest.skip('requires the existing dense export')
    table = tmp_path / 'synthetic_geometry.json'
    table.write_text(json.dumps(synthetic_table()))
    output = tmp_path / 'plan.so'
    result = invoke(table, output, bridge=bridge)
    assert result.returncode != 0 and 'requires a MoE decode bridge' in result.stderr
    assert not output.exists() and not Path(str(output) + '.mx5_candidates.json').exists()


@pytest.mark.parametrize('change', ('missing_router', 'partial_attention', 'results', 'interpreted', 'wrong_schema'))
def test_incomplete_or_unsupported_moe_request_never_writes_preview(tmp_path, change):
    doc = synthetic_table()
    extra = []
    if change == 'missing_router':
        doc['rows'] = [r for r in doc['rows'] if r['kind'] != 'moe_router']
    elif change == 'partial_attention':
        doc['rows'].pop()
    elif change == 'results':
        extra = ['--mx-decode-v2-results', str(tmp_path / 'unused_results.json')]
    elif change == 'interpreted':
        extra = ['--pg', 'off']
    else:
        doc['table_kind'] = 'legacy_costs'
    table = tmp_path / 'synthetic_geometry.json'
    table.write_text(json.dumps(doc))
    output = tmp_path / 'plan.so'
    result = invoke(table, output, *extra)
    assert result.returncode != 0, result.stdout
    assert not output.exists() and not Path(str(output) + '.mx5_candidates.json').exists()
