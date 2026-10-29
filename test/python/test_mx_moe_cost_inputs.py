"""Verify actual cost receipts, including tamper rejection, without CUDA."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def scope():
    directory = Path(os.environ.get('MX5_MOE_COST_SCOPE', ROOT / 'runs/mx5/moe_grouped_cost_scope_v1'))
    if not (directory / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json').is_file():
        pytest.skip('requires the actual frozen MoE grouped cost epoch')
    spec = importlib.util.spec_from_file_location('mx5_cost_actual', directory / 'measure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return directory, module


def test_actual_cost_batches_validate_without_loading_cuda(scope):
    directory, module = scope
    report = json.loads((directory / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json').read_text())
    assert report['total_keys'] == 772
    for job in report['jobs']:
        module.validate_cost_inputs(directory / 'epoch.json', Path(job['keys']['path']))


@pytest.mark.parametrize('field,value', [('tn', 128), ('r_next', 999), ('b', 32), ('binary_sha256', '0' * 64)])
def test_changed_key_coordinates_or_binary_stop_before_cuda(scope, tmp_path, field, value):
    directory, module = scope
    rows = json.loads((directory / 'chunks/keys000.json').read_text())
    assert rows[0][field] != value
    rows[0][field] = value
    keys = tmp_path / 'bad_keys.json'
    keys.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        module.validate_cost_inputs(directory / 'epoch.json', keys)


@pytest.mark.parametrize('change', ['register_cap', 'source_hash', 'missing_query'])
def test_budget_source_and_device_query_are_required(scope, tmp_path, change):
    directory, module = scope
    doc = json.loads((directory / 'epoch.json').read_text())
    if change == 'register_cap':
        doc['R_cap']['t576'] = 160
    elif change == 'source_hash':
        doc['inputs'][next(iter(doc['inputs']))] = '0' * 64
    else:
        doc['eligible_keys'] = []
    epoch = tmp_path / 'bad_epoch.json'
    epoch.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        module.validate_cost_inputs(epoch, directory / 'chunks/keys000.json')
