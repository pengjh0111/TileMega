"""H800 decode is native only: a legacy MoE (DM) decode build is refused."""
import lzma
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

COMPILER = os.environ.get('TILEMEGA_BIN')
ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'configs/serving/sm90d1/inputs/qwen3_30b_a3b/decode.export.json.xz'


@unittest.skipUnless(COMPILER, 'set TILEMEGA_BIN to the compiler under test')
class NativeDecodeOnlyTest(unittest.TestCase):
    def test_legacy_moe_decode_is_refused_on_h800(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = Path(folder) / 'decode.export.json'
            bridge.write_bytes(lzma.decompress(EXPORT.read_bytes()))
            result = subprocess.run(
                [COMPILER, 'compile', str(bridge), str(Path(folder) / 'decode.so'), '--serving', 'decode',
                 '--emit', 'serving', '--batch', '1', '--runtime-target',
                 str(ROOT / 'configs/targets/sm_90_h800.json')],
                cwd=ROOT, capture_output=True, text=True, timeout=1200,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('H800 decode requires native tmexec coordinates', result.stderr + result.stdout)


if __name__ == '__main__':
    unittest.main()
