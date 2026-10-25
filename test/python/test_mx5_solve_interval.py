"""CPU-only request validation before reading a solver table or invoking cucc."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(os.environ.get('TILEMEGA_TEST_ROOT', Path(__file__).resolve().parents[2]))
BINARY = Path(os.environ.get('TILEMEGA_BIN', ROOT / 'build/tools/tilemega'))
pytestmark = pytest.mark.skipif(not BINARY.is_file(), reason='native host tool unavailable')


def request(tmp_path, upper):
    bridge = tmp_path / 'bridge.json'
    bridge.write_text('{}\n')
    target = tmp_path / 'target.json'
    target.write_text('{}\n')
    missing_table = tmp_path / 'missing_table.json'
    output = tmp_path / 'plan.so'
    command = [str(BINARY), 'compile', str(bridge), str(output), '--serving', 'decode',
               '--emit', 'serving', '--frontend', 'decoder', '--batch', '1', '--capacity', '1088',
               '--past-range', '64:' + str(upper), '--selection', 'predicted',
               '--runtime-target', str(target), '--mx-decode-v2-solve', str(missing_table),
               '--mx-decode-v2-threads', '512']
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MACA_VISIBLE_DEVICES': ''})
    assert not list(tmp_path.glob('plan.so*'))
    return result, missing_table


def test_inclusive_1087_rejected_before_table_read(tmp_path):
    result, _ = request(tmp_path, 1087)
    assert result.returncode != 0
    assert 'invalid decode past interval' in result.stdout


def test_inclusive_1086_passes_interval_validation(tmp_path):
    result, missing_table = request(tmp_path, 1086)
    # Passing the range check reaches this deliberately missing CPU fixture.
    assert result.returncode != 0
    assert 'invalid decode past interval' not in result.stdout
    assert str(missing_table) in result.stdout
