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

CALIBRATED = ('{"arch_tag":"sm_120",'
              '"calibration_by_dtype":{"bf16":{"calibrated":true}}}')


class PreparationTest(unittest.TestCase):
    def probe(self, calls, chain_plan=b'chain\n', shared_plan=b'shared\n'):
        """Stand in for nvcc, the control binaries and both placement tools."""
        def run(command, **kwargs):
            calls.append((command, kwargs))
            name = Path(command[0]).name
            if name == 'tilemega-calibrate':
                Path(command[command.index('--out') + 1]).write_text(CALIBRATED)
            elif name.startswith('tilemega-place-'):
                tool = name.rsplit('-', 1)[1]
                plan = Path(command[3]) / 'plan'
                plan.mkdir(parents=True, exist_ok=True)
                # The three candidates both tools emit must match byte for byte;
                # each tool's own candidate is its alone.
                (plan / 'gqa2_s4_legacy_grid_stride.cu').write_bytes(shared_plan)
                own = 'eft' if tool == 'eft' else 'chain'
                (plan / f'gqa2_s4_{own}.cu').write_bytes(
                    shared_plan if tool == 'eft' else chain_plan)
            return subprocess.CompletedProcess(command, 0,
                'RESULT status=PASS\nE2E_RESOURCE grid=340\nE2E_PLACE_BASE pos=0\n', '')
        return run

    def test_both_tools_are_solved_from_local_sm120_inputs(self):
        calls = []
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)
            with patch.object(sys, 'argv',
                              ['prepare', '--out', str(out), '--realwidth', '0']), \
                    patch.object(subprocess, 'run', side_effect=self.probe(calls)):
                prepare_sm120.main()
            with (out / 'manifest.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row['trace_dir'], '-')
                self.assertEqual(Path(row['out_prefix']).parent, out / 'prepare')
            solvers = [command for command, _ in calls
                       if Path(command[0]).name.startswith('tilemega-place-')]
            self.assertEqual([Path(c[0]).name for c in solvers],
                             ['tilemega-place-eft', 'tilemega-place-chain'])
            for solver in solvers:
                self.assertTrue(
                    solver[solver.index('--target') + 1].endswith('/sm_120.json'))
                self.assertTrue(
                    solver[solver.index('--hop') + 1].endswith('/raw_sm120/hop_ns.tsv'))
            probes = [kwargs for command, kwargs in calls if 'env' in kwargs]
            self.assertEqual(len(probes), 4)
            for kwargs in probes:
                self.assertEqual(kwargs['env']['TILEMEGA_PLACEMENT_BASE_DUMP'], '1')

    def test_plans_are_merged_and_agreement_recorded(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)
            with patch.object(sys, 'argv',
                              ['prepare', '--out', str(out), '--realwidth', '0']), \
                    patch.object(subprocess, 'run', side_effect=self.probe([])):
                prepare_sm120.main()
            merged = {p.name for p in (out / 'plan').glob('*.cu')}
            self.assertEqual(merged, {'gqa2_s4_legacy_grid_stride.cu',
                                      'gqa2_s4_eft.cu', 'gqa2_s4_chain.cu'})
            with (out / 'tool_agreement.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertTrue(rows)
            self.assertTrue(
                all(row['identical_between_tools'] == '1' for row in rows))

    def test_disagreeing_tools_are_refused_before_the_merge(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)
            # The shared candidate differs between the tools: merging would
            # silently keep whichever was copied second.
            def run(command, **kwargs):
                name = Path(command[0]).name
                if name == 'tilemega-calibrate':
                    Path(command[command.index('--out') + 1]).write_text(CALIBRATED)
                elif name.startswith('tilemega-place-'):
                    tool = name.rsplit('-', 1)[1]
                    plan = Path(command[3]) / 'plan'
                    plan.mkdir(parents=True, exist_ok=True)
                    (plan / 'gqa2_s4_legacy_grid_stride.cu').write_bytes(
                        b'eft\n' if tool == 'eft' else b'chain\n')
                return subprocess.CompletedProcess(command, 0,
                    'RESULT status=PASS\nE2E_RESOURCE grid=340\nE2E_PLACE_BASE pos=0\n', '')
            with patch.object(sys, 'argv',
                              ['prepare', '--out', str(out), '--realwidth', '0']), \
                    patch.object(subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'differs between'):
                    prepare_sm120.main()

    def test_real_width_adds_its_own_cells(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)
            with patch.object(sys, 'argv',
                              ['prepare', '--out', str(out), '--realwidth', '1']), \
                    patch.object(subprocess, 'run', side_effect=self.probe([])):
                prepare_sm120.main()
            with (out / 'manifest.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(rows), 6)
            self.assertEqual(sum(row['model'] == 'real' for row in rows), 2)

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
            target.write_text('{"arch_tag":"sm_120",'
                              '"calibration_by_dtype":{"bf16":{"calibrated":false}}}')
            result = subprocess.CompletedProcess([], 0,
                'RESULT status=PASS\nE2E_RESOURCE grid=340\nE2E_PLACE_BASE pos=0\n', '')
            with patch.object(sys, 'argv',
                              ['prepare', '--out', scratch, '--realwidth', '0']), \
                    patch.dict(os.environ, {'TARGET_JSON': str(target)}), \
                    patch.object(subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'not accepted'):
                    prepare_sm120.main()


if __name__ == '__main__':
    unittest.main()
