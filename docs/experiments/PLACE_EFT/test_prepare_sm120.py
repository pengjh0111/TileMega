#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_sm120


class PreparationTest(unittest.TestCase):
    def test_measured_inputs_and_target_are_local(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if Path(command[0]).name == 'tilemega-calibrate':
                Path(command[command.index('--out') + 1]).write_text(
                    '{"arch_tag":"sm_120","calibration_by_dtype":{"bf16":{"calibrated":true}}}')
            return subprocess.CompletedProcess(command, 0,
                'RESULT status=PASS\nE2E_RESOURCE grid=340\nE2E_PLACE_BASE pos=0\n', '')

        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)
            with patch.object(sys, 'argv', ['prepare', '--out', str(out)]), \
                    patch.object(subprocess, 'run', side_effect=run):
                prepare_sm120.main()
            with (out / 'manifest.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(Path(row['out_prefix']).parent, out / 'prepare')
                self.assertEqual(row['trace_dir'], '-')
            solver = calls[-1][0]
            self.assertTrue(solver[solver.index('--target') + 1].endswith('/sm_120.json'))
            self.assertTrue(solver[solver.index('--hop') + 1].endswith('/raw_sm120/hop_ns.tsv'))
            probes = [kwargs for command, kwargs in calls if 'env' in kwargs]
            self.assertEqual(len(probes), 4)
            for kwargs in probes:
                self.assertEqual(kwargs['env']['TILEMEGA_PLACEMENT_BASE_DUMP'], '1')
            self.assertFalse((out / 'plan/gqa2_s4_eft.cu').exists())

    def test_old_output_is_rejected(self):
        old = Path(prepare_sm120.__file__).resolve().parent / 'raw'
        with patch.object(sys, 'argv', ['prepare', '--out', str(old)]), \
                patch.object(subprocess, 'run') as run:
            with self.assertRaises(SystemExit):
                prepare_sm120.main()
            run.assert_not_called()

    def test_unaccepted_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / 'sm_120.json'
            target.write_text('{"arch_tag":"sm_120","calibration_by_dtype":{"bf16":{"calibrated":false}}}')
            result = subprocess.CompletedProcess([], 0,
                'RESULT status=PASS\nE2E_RESOURCE grid=340\nE2E_PLACE_BASE pos=0\n', '')
            with patch.object(sys, 'argv', ['prepare', '--out', scratch]), \
                    patch.dict(os.environ, {'TARGET_JSON': str(target)}), \
                    patch.object(subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'not accepted'):
                    prepare_sm120.main()


if __name__ == '__main__':
    unittest.main()
