"""CPU validation of existing MoE programs; these tests never load CUDA."""
import copy
import json
from pathlib import Path

import pytest

from tilemega.moe import plan_occupancy as api

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / 'runs/mx5/moe_rank_whole_program_candidate_v1'


def inputs(so, batch, context='compile_receipt'):
    ref = api.common.record
    return dict(kind='mx5_moe_plan_occupancy_inputs', schema_version=1, threads=512, batch=batch,
        binary=ref(so), generated_source=ref(str(so) + '.cu'), resources=ref(str(so) + '.resources.json'),
        log=ref(str(so) + '.ptxas.log'), source_snapshot=ref(str(so) + '.source.json'),
        context_kind=context, context=ref(str(so) + ('.compile_receipt.json' if context == 'compile_receipt' else '.identity.json')),
        **({} if context == 'compile_receipt' else dict(manifest=ref(str(so) + '.plan.json'))))


@pytest.fixture
def real_spec():
    so = PLANS / 'B1_w0_p0/plan.so'
    if not so.is_file():
        pytest.skip('requires an existing native MoE manual program, no GPU')
    return inputs(so, 1)


@pytest.mark.parametrize('batch', (1, 16))
@pytest.mark.parametrize('watchdog,prompt', ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_actual_eight_moe_programs_validate_without_runtime(batch, watchdog, prompt):
    so = PLANS / f'B{batch}_w{watchdog}_p{prompt}/plan.so'
    if not so.is_file():
        pytest.skip('requires existing native manual programs, no GPU')
    for context in ('compile_receipt', 'native_identity'):
        value = api.validate_inputs(inputs(so, batch, context))
        assert value['stub'] == api.STUB and value['threads'] == 512
        assert value['expected_shared_bytes'] == json.loads(Path(str(so) + '.plan.json').read_text())['smem']


@pytest.mark.parametrize('change', ('dense_kind', 'threads', 'batch', 'context', 'log', 'kernel', 'source'))
def test_moe_query_rejects_wrong_scope_or_same_compile_binding(real_spec, tmp_path, change):
    spec = copy.deepcopy(real_spec)
    if change == 'dense_kind':
        spec['kind'] = 'mx5_plan_occupancy_inputs'
    elif change == 'threads':
        spec['threads'] = 576
    elif change == 'batch':
        spec['batch'] = 2
    elif change == 'context':
        spec['context_kind'] = 'unchecked_binary'
    elif change == 'log':
        path = tmp_path / 'bad_receipt.json'
        receipt = json.loads(Path(spec['context']['path']).read_text())
        receipt['same_compile_mir'] = '0' * 64
        path.write_text(json.dumps(receipt))
        spec['context'] = api.common.record(path)
    elif change == 'kernel':
        path = tmp_path / 'bad_resources.json'
        value = json.loads(Path(spec['resources']['path']).read_text())
        value['kernel']['name'] = api.common.KERNEL
        path.write_text(json.dumps(value))
        spec['resources'] = api.common.record(path)
    else:
        path = tmp_path / 'bad_source.cu'
        source = Path(spec['generated_source']['path']).read_text()
        assert 'kHidden=2048' in source
        path.write_text(source.replace('kHidden=2048', 'kHidden=1024'))
        spec['generated_source'] = api.common.record(path)
    with pytest.raises(ValueError):
        api.validate_inputs(spec)


def test_dense_and_moe_stub_names_are_not_interchangeable():
    valid = f'00000001 W {api.STUB}\n00000002 T tm_plan_shared_bytes\n'
    assert api.exported_stub(valid) == api.STUB
    for value in (valid + f'00000003 W {api.STUB}\n', valid.replace(api.STUB, api.common.STUB),
            valid.replace(' W ', ' U '), valid + f'00000003 W {api.common.STUB}\n'):
        with pytest.raises(ValueError):
            api.exported_stub(value)


def test_runtime_loader_failure_is_recorded_without_launch_or_overwrite(real_spec, tmp_path, monkeypatch):
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps(real_spec))
    output = tmp_path / 'failed_query.json'
    monkeypatch.setattr(api.common, 'guard_evidence', lambda env: dict(CPU_synthetic_guard=True))

    def fail_runtime():
        raise RuntimeError('CPU injected runtime loader failure; no CUDA loaded')

    monkeypatch.setattr(api.common, 'Runtime', fail_runtime)
    argv = ['--inputs', str(path), '--out', str(output), '--query-only', '--iterations', '0']
    assert api.main(argv) == 1
    raw = output.read_bytes()
    value = json.loads(raw)
    assert value['program_kind'] == 'moe' and value['query_status'] == 1
    assert value['no_launch'] and value['launches'] == value['iterations'] == 0
    assert not value['execution_authorized'] and not value['grants_512_exception']
    assert 'CPU injected runtime loader failure' in value['exception']
    with pytest.raises(SystemExit):
        api.main(argv)
    assert output.read_bytes() == raw
