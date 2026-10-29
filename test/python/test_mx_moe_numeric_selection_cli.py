"""The native solver's numeric handoff never qualifies process/model evidence."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'build/mx_moe_selection_v2'
pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason='requires the native MoE host solver helper')


def invoke(tmp_path, predicted, pairs):
    path = tmp_path / 'numeric_pairs.json'
    path.write_text(json.dumps(dict(predicted_ns=predicted, process_tpot_ns=pairs)))
    return subprocess.run([str(TOOL), '--select-pairs', str(path)], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES=''),
        capture_output=True, text=True)


@pytest.mark.parametrize('pairs,winner', (
    ([[99, 101], [98.2, 100], [110, 110], [120, 120]], 0),
    ([[99, 101], [98, 100], [110, 110], [120, 120]], 1),
    ([[99, 101], [98, 100], [99, 99], [120, 120]], 1),
    ([[99, 101], [100, 100], [100, 100], [90, 90]], 3),
))
def test_native_one_percent_anchor_and_tie_order(tmp_path, pairs, winner):
    result = invoke(tmp_path, [100, 101, 102, 103], pairs)
    assert result.returncode == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc['selected_index'] == winner
    assert doc['measured_ns'] == pytest.approx([sum(p) / 2 for p in pairs])
    assert all(doc[k] is False for k in
        ('file_qualification_implied', 'model_acceptance_implied', 'final', 'default_enabled'))


@pytest.mark.parametrize('change', ('predictions', 'unsorted', 'missing_pair', 'one_sample',
    'zero', 'negative', 'nan', 'boolean'))
def test_invalid_numeric_handoff_is_rejected(tmp_path, change):
    predicted = [100, 101, 102, 103]
    pairs = [[100, 100], [100, 100], [100, 100], [100, 100]]
    if change == 'predictions':
        predicted.pop()
    elif change == 'unsorted':
        predicted[0], predicted[1] = predicted[1], predicted[0]
    elif change == 'missing_pair':
        pairs.pop()
    elif change == 'one_sample':
        pairs[0].pop()
    else:
        pairs[0][0] = dict(zero=0, negative=-1, nan=float('nan'), boolean=True)[change]
    result = invoke(tmp_path, predicted, pairs)
    assert result.returncode == 1
    assert not result.stdout.strip()
