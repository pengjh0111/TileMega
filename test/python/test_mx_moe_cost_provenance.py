"""CPU checks that raw evidence cannot silently change a selected cost row."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tilemega.moe.cost_provenance import Files, validate_complete_scope, validate_outcomes, validate_row


def reference(path, **extra):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)


@pytest.fixture
def row(tmp_path):
    library = tmp_path / 'body.so'
    library.write_bytes(b'unit fixture, not a GPU binary')
    key = dict(n=128, k=2048, b=1, threads=512, tn=32, wn=1, depth=1,
        prefetch=1, op=0, norm=1, part=1)
    resource = dict(mtreg=120, local_bytes=0, static_smem_bytes=0, resident_ctas=1)
    provenance = dict(executor='mx_static_stage_bench', implementation='mx5_moe_router_v2',
        source_sha256='a' * 64, binary_sha256=reference(library)['sha256'])
    result = dict(kind='moe_router', key=key, provenance=provenance, resources=resource,
        dispatch_context=dict(gate_up_tn=64, down_tn=32), processes=3,
        median_ns=102, p90_ns=112, spread=round(2 / 102, 4), rounds=[])
    for rnd in (1, 2, 3):
        folder = tmp_path / str(rnd)
        folder.mkdir()
        (folder / 'pgid').write_text(str(100 + rnd))
        process = folder / 'process.json'
        guard = folder / 'guard_result.json'
        process.write_text(json.dumps(dict(returncode=0)))
        guard.write_text(json.dumps(dict(code=0)))
        actual = dict(key, library=str(library), binary_sha256=provenance['binary_sha256'],
            dispatch_gate_tn=64, dispatch_down_tn=32)
        sample = dict(key=actual, round=rnd, median_ns=100 + rnd, p90_ns=110 + rnd,
            executor=provenance['executor'], implementation=provenance['implementation'],
            source_digest=provenance['source_sha256'], iterations=128,
            source_bound=True, cost_protocol=True, repeat_bitwise=True,
            FP32_reference_passed=True, legacy_topk_weights_bitwise=True, dispatch_verified=True,
            addressed_rotating_weight_bytes=256 * 1024**2,
            resources=dict(MTreg=120, stack=0, static_smem=0, resident=1))
        raw = folder / 'raw.jsonl'
        raw.write_text(json.dumps(sample) + '\n')
        result['rounds'].append(dict(round=rnd, PID=100 + rnd, input=reference(raw, line=1),
            process=reference(process), guard=reference(guard), measurement=sample))
    return result


def test_three_bound_rounds_and_actual_resources(row):
    assert validate_row(row) == dict(rounds=3, actual_guard_pids=3, legacy_head_command_records=0)


@pytest.mark.parametrize('change', ('raw', 'aggregate', 'source', 'resources', 'coordinate',
    'dispatch', 'binary', 'working_set', 'pid', 'guard', 'zero_iterations', 'missing_round', 'body_check'))
def test_tampered_or_unqualified_evidence_is_rejected(row, change):
    r = copy.deepcopy(row)
    sample = r['rounds'][0]['measurement']
    if change == 'raw':
        Path(r['rounds'][0]['input']['path']).write_text('{}\n')
    elif change == 'aggregate':
        r['median_ns'] += 1
    elif change == 'source':
        r['provenance']['source_sha256'] = 'f' * 64
    elif change == 'resources':
        r['resources']['local_bytes'] = 16
    elif change == 'coordinate':
        r['key']['tn'] = 64
    elif change == 'dispatch':
        r['dispatch_context']['gate_up_tn'] = 128
    elif change == 'binary':
        Path(sample['key']['library']).write_bytes(b'changed ELF')
    elif change == 'working_set':
        sample['addressed_rotating_weight_bytes'] = 1024
    elif change == 'pid':
        r['rounds'][0]['PID'] = r['rounds'][1]['PID']
    elif change == 'guard':
        path = Path(r['rounds'][0]['guard']['path'])
        path.write_text(json.dumps(dict(code=1)))
        r['rounds'][0]['guard'] = reference(path)
    elif change == 'zero_iterations':
        sample['iterations'] = 0
    elif change == 'body_check':
        sample['legacy_topk_weights_bitwise'] = False
    else:
        r['rounds'].pop()
    if change in ('working_set', 'zero_iterations', 'body_check'):
        path = Path(r['rounds'][0]['input']['path'])
        path.write_text(json.dumps(sample) + '\n')
        r['rounds'][0]['input'] = reference(path, line=1)
    with pytest.raises(ValueError):
        validate_row(r)


def test_stat_cache_rechecks_a_file_that_changes(row):
    files = Files()
    ref = row['rounds'][0]['input']
    files.check(ref)
    Path(ref['path']).write_text('changed\n')
    with pytest.raises(ValueError):
        files.check(ref)


def test_partial_file_validation_grants_no_geometry_or_model_acceptance(row, tmp_path):
    path = tmp_path / 'outcomes.json'
    doc = dict(schema_version=1, table_kind='mx5_moe_geometry_outcomes', rows=[row], rejections=[])
    path.write_text(json.dumps(doc))
    result = validate_outcomes(path)
    assert result['measured'] == 1 and result['rejected'] == 0
    assert all(result[name] is False for name in ('geometry_completeness_implied',
        'model_acceptance_implied', 'solver_selected', 'final', 'default_enabled'))
    doc['synthetic_fixture'] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='synthetic'):
        validate_outcomes(path)


def test_resource_rejections_cannot_supply_timing_or_hide_fitting_resources(row, tmp_path):
    path = tmp_path / 'rejections.json'
    rejected = dict(kind='moe_router', key=row['key'], resources=row['resources'])
    doc = dict(schema_version=1, table_kind='mx5_moe_geometry_outcomes', rows=[], rejections=[rejected])
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='fitting resources'):
        validate_outcomes(path)
    rejected['resources'] = dict(row['resources'], local_bytes=16)
    rejected['median_ns'] = 100
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='contains a measured cost'):
        validate_outcomes(path)


@pytest.mark.parametrize('change', ('none', 'partial', 'other_batch', 'body', 'source', 'coverage', 'policy'))
def test_complete_scope_binds_body_source_request_and_native_audit(row, tmp_path, monkeypatch, change):
    frozen, adopted = tmp_path / 'frozen.h', tmp_path / 'adopted.h'
    frozen.write_text('measured GPU body\n')
    adopted.write_text('changed GPU body\n' if change == 'body' else frozen.read_text())
    closure = tmp_path / 'closure.json'
    families = []
    for name in ('gemm', 'head', 'attention', 'grouped_v2', 'router_contexts'):
        families.append(dict(family=name, source=reference(frozen), process=reference(frozen),
            dependencies=reference(frozen), headers=[dict(frozen=reference(frozen), adopted=reference(adopted))]))
    closure.write_text(json.dumps(dict(families=families)))
    tool = tmp_path / 'domain_tool'
    tool.write_bytes(b'CPU audit tool fixture')
    table = tmp_path / 'scope.json'
    doc = dict(schema_version=1, table_kind='mx5_moe_geometry_outcomes', rows=[row], rejections=[],
        complete_scope=change != 'partial', batch=2 if change == 'other_batch' else 1, threads=512,
        body_header_closure=reference(closure), source_epochs=dict(moe_router='a' * 64),
        final=change == 'policy')
    if change == 'source':
        doc['source_epochs']['moe_router'] = 'c' * 64
    table.write_text(json.dumps(doc))
    calls = []

    def audit(command, **kwargs):
        calls.append(command)
        coverage = dict(complete_geometry=change != 'coverage', missing=[], measured=1,
            rejected=0, expected=1)
        return subprocess.CompletedProcess(command, 0, json.dumps(coverage), '')

    monkeypatch.setattr('tilemega.moe.cost_provenance.subprocess.run', audit)
    if change != 'none':
        with pytest.raises(ValueError):
            validate_complete_scope(table, tool)
    else:
        result = validate_complete_scope(table, tool)
        assert result['geometry_completeness_implied'] and not result['model_acceptance_implied']
        assert calls == [[str(tool), '--audit-rows', str(table), '1', '512']]
