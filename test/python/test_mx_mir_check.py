"""MIR acceptance must follow CFG cycles, not backward layout branches."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MirCheck(unittest.TestCase):
    def check(self, blocks):
        script = Path(__file__).resolve().parents[2] / 'experiments/mx5/mir_check.py'
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.mir', Path(directory) / 'output.json'
            source.write_text('# Machine code for function kernel:\n' + blocks + '\n# End machine code\n')
            result = subprocess.run([sys.executable, str(script), str(source), '--out', str(output)],
                                    capture_output=True, text=True)
            return result.returncode, json.loads(output.read_text())

    def test_backward_layout_without_cycle(self):
        code, report = self.check('''bb.0:
  successors: %bb.2
bb.1:
  successors: %bb.3
  ARRIVE 4160
  MMA_16X16X16BF16
bb.2:
  successors: %bb.1
  LDG_B128
bb.3:
  RET
''')
        self.assertEqual(code, 0)
        self.assertEqual(report['functions'][0]['rings'], [])

    def test_real_ring_wait_and_generic_load_fail(self):
        for instruction in ['ARRIVE 4160', 'LDX_B128']:
            with self.subTest(instruction=instruction):
                code, report = self.check('''bb.0:
  successors: %bb.1
bb.1:
  successors: %bb.2
  LDG_B128
  MMA_16X16X16BF16
  ''' + instruction + '''
bb.2:
  successors: %bb.1, %bb.3
bb.3:
  RET
''')
                self.assertEqual(code, 1)
                self.assertEqual(len(report['failures']), 1)


if __name__ == '__main__': unittest.main()
