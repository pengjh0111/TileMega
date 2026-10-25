"""Reject equal-sized KV buffers whose prefill and decode byte layouts differ."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tilemega.serving import engine


class CompatiblePlans(Exception):
    pass


@pytest.mark.parametrize('prefill_storage,decode_storage', [
    (None, 'block_fragments'), ('block_fragments', None),
    ('linear', 'block_fragments'), ('block_fragments', 'linear'),
])
def test_incompatible_storage_rejected_before_binding(tmp_path, monkeypatch,
                                                      prefill_storage, decode_storage):
    plans, execution, weights, state = prepare(tmp_path, monkeypatch, prefill_storage, decode_storage)
    with pytest.raises(ValueError, match='prefill kv_storage .* differs from decode kv_storage'):
        engine.ServingEngine(tmp_path, *plans, batch=1)
    execution.assert_not_called()
    weights.assert_not_called()
    state.assert_not_called()


@pytest.mark.parametrize('prefill_storage,decode_storage', [
    (None, None), ('linear', 'linear'), (None, 'linear'), ('linear', None),
    ('block_fragments', 'block_fragments'),
])
def test_matching_storage_reaches_execution_setup(tmp_path, monkeypatch,
                                                 prefill_storage, decode_storage):
    plans, execution, _, _ = prepare(tmp_path, monkeypatch, prefill_storage, decode_storage)
    with pytest.raises(CompatiblePlans):
        engine.ServingEngine(tmp_path, *plans, batch=1)
    execution.assert_called_once()


def prepare(tmp_path, monkeypatch, prefill_storage, decode_storage):
    plans = [tmp_path / 'prefill.so', tmp_path / 'decode.so']
    for path, storage in zip(plans, (prefill_storage, decode_storage)):
        manifest = dict(kv_layout='vt')
        if storage is not None:
            manifest['kv_storage'] = storage
        path.with_suffix('.so.plan.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(engine.torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(engine.torch.cuda, 'init', Mock())
    libraries = [SimpleNamespace(info=SimpleNamespace(seq=seq, capacity=1088)) for seq in (64, 1)]
    monkeypatch.setattr(engine, 'PlanLibrary', Mock(side_effect=libraries))
    execution = Mock(side_effect=CompatiblePlans)
    weights, state = Mock(), Mock()
    monkeypatch.setattr(engine, 'resolve_execution', execution)
    monkeypatch.setattr(engine, 'load_weights', weights)
    monkeypatch.setattr(engine, 'allocate_state', state)
    return plans, execution, weights, state
