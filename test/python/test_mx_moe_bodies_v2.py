"""Native MoE body fixtures against CPU FP32 and actual legacy CUDA bodies."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.environ.get('MX5_MOE_BODIES_DIR'),
                    reason='requires GPU guard and MX5_MOE_BODIES_DIR')
@pytest.mark.parametrize('fixture,cases', [('router_combine', 100), ('router_gemm', 8), ('grouped', 12)])
def test_native_moe_fixture(tmp_path, fixture, cases):
    directory = Path(os.environ['MX5_MOE_BODIES_DIR']).resolve()
    out = tmp_path / fixture
    subprocess.run([sys.executable, str(ROOT / 'test/unit/mx' / f'moe_check_{fixture}.py'),
        '--libraries', str(directory), '--out', str(out)], check=True, timeout=600)
    result = json.loads((out / 'result.json').read_text())
    assert result['passed'] and result['zero_skips'] and len(result['rows']) == cases
