#!/usr/bin/env python3
"""CPU-only scientific-receipt integrity checks; all fixture times are synthetic.

No CUDA API or probe executable runs here. Temporary fake measurement trees
are removed at test completion and never become DNN1 reported measurements.
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('dnn1_constants', ROOT / 'scripts/dnn1_constants.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstantsIntegrity(unittest.TestCase):
    def test_target_append_preserves_every_existing_value_and_bytes(self):
        # Exercise the append transition even after the shipped target has
        # measured forward constants. The existing-section refusal is tested below.
        base = json.loads((ROOT / 'configs/targets/sm_90_h800.json').read_text())
        base.pop('sm90_forward_constants', None)
        original = json.dumps(base, indent=2) + '\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'target.json'
            path.write_text(original)
            MODULE.append_forward_section(path, {'schema': 0, 'cpu_fixture_only': True})
            after = json.loads(path.read_text())
            before = json.loads(original)
            self.assertTrue(path.read_text().startswith(original[:len(original.rstrip())-1].rstrip()))
            self.assertEqual({k: after[k] for k in before}, before)
            self.assertEqual(after['sm90_native_constants'], before['sm90_native_constants'])
            frozen = path.read_bytes()
            with self.assertRaises(ValueError):
                MODULE.append_forward_section(path, {'schema': 0})
            self.assertEqual(path.read_bytes(), frozen)

    def test_unguarded_execution_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.require_session(0)

    def test_parallel_session_requires_trial_actual_gpu_lock_and_cpu_subset(self):
        env = dict(TILEMEGA_GPU_LOCK_HELD='1', TILEMEGA_TIMING='1',
            TILEMEGA_DEVICE_INDEX='0', CUDA_VISIBLE_DEVICES='0',
            TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu0.lock',
            TILEMEGA_NUMA_CONCURRENCY_TRIAL='1')
        spec = dict(parallel=True, cpu_affinity={'0': [32, 33, 34, 35]})
        with patch.dict(MODULE.os.environ, env, clear=True), \
                patch.object(MODULE, 'held', return_value=True), \
                patch.object(MODULE.os, 'sched_getaffinity', return_value={32, 33}):
            MODULE.require_session(0, spec)
            with patch.object(MODULE.os, 'sched_getaffinity', return_value={0, 32}):
                with self.assertRaises(ValueError): MODULE.require_session(0, spec)
            with patch.object(MODULE, 'held', return_value=False):
                with self.assertRaises(ValueError): MODULE.require_session(0, spec)
            with patch.dict(MODULE.os.environ, {'TILEMEGA_NUMA_CONCURRENCY_VERIFIED': '1'}):
                with self.assertRaises(ValueError): MODULE.require_session(0, spec)

    def test_collector_requires_five_accepted_unchanged_process_receipts(self):
        # Real parsing is exercised against explicit synthetic CPU fixtures.
        # verify_build is replaced only to avoid depending on a local CUDA
        # compiler installation when this integrity test runs elsewhere.
        with tempfile.TemporaryDirectory(prefix='dnn1_cpu_synthetic_') as directory:
            root = Path(directory)
            case = dict(id='cpu_fixture', metric='epilogue_step_ns', key='fixture_only', binary='direct')
            build = dict(variant='cpu_fixture', original_sources={}, source_sha256='cpu_fixture',
                clock_overlay={}, binaries={'direct': {'sha256': 'cpu_fixture'}})
            spec = dict(build='cpu_fixture', gpus=[0, 5, 7], cases=[case], samples=24,
                pending=[], omitted_ready=[], uuid_by_gpu={str(g): 'cpu_fixture' for g in (0, 5, 7)},
                policy_sha256='cpu_fixture')
            MODULE.write(root / 'queue-spec.json', spec)
            results = []
            for gpu in spec['gpus']:
                state = {}
                for round_ in range(5):
                    name = f'dnn1-constants-g{gpu}-r{round_}'
                    attempt = root / f'gpu{gpu}/raw' / name / 'attempt1'
                    job = attempt / 'job'
                    row_path = job / 'cases/cpu_fixture/result.json'
                    row_path.parent.mkdir(parents=True)
                    files = {}
                    for key in ('raw', 'stdout', 'stderr'):
                        path = row_path.parent / (key + '.txt')
                        path.write_text('CPU SYNTHETIC TEST FIXTURE; not a measurement\n')
                        files[key] = str(path)
                        files[key + '_sha256'] = MODULE.sha(path)
                    result = dict(case=case, valid=True, fresh_executable_process=True,
                        process_pid=1000 + gpu*10 + round_, binary_sha256='cpu_fixture',
                        before={'sm_clock_mhz': 1}, after={'sm_clock_mhz': 1},
                        rows=[dict(metric=case['metric'], key=case['key'], samples='16',
                            median_ns='1', p99_ns='2', unit='ns', method='CPU_SYNTHETIC_FIXTURE')], **files)
                    MODULE.write(row_path, result)
                    results.append(row_path)
                    MODULE.write(job / 'constants_session.json', dict(status='complete', gpu=gpu,
                        round=round_, variant='cpu_fixture', cases=[str(row_path)]))
                    MODULE.write(attempt / 'guard_result.json', {'code': 0, 'cpu_fixture_only': True})
                    MODULE.write(job / 'physical-device-identity.json', dict(gpu_index=gpu,
                        nvml_uuid='cpu_fixture', guard_policy_sha256='cpu_fixture',
                        numa_node=0 if gpu < 4 else 1, cpu_fixture_only=True))
                    state[name] = dict(status='done', out=str(attempt))
                MODULE.write(root / f'gpu{gpu}/scheduler/state.json', state)
            original = MODULE.verify_build
            MODULE.verify_build = lambda _: (build, [case])
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.collect(argparse.Namespace(run_root=root, out=root / 'complete', target=None))
                with (root / 'complete/DNN1_constants.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(row['processes'] == '5' for row in rows))
                # Four valid rounds are insufficient, even though all numeric
                # rows are present. Guard rejection cannot be log-scraped away.
                state_path = root / 'gpu0/scheduler/state.json'
                state = MODULE.read(state_path)
                state['dnn1-constants-g0-r4']['status'] = 'failed'
                MODULE.write(state_path, state)
                with self.assertRaisesRegex(ValueError, 'missing accepted'):
                    MODULE.collect(argparse.Namespace(run_root=root, out=root / 'four', target=None))
                state['dnn1-constants-g0-r4']['status'] = 'done'
                MODULE.write(state_path, state)
                # Identical summaries do not legitimize modified raw samples.
                result = MODULE.read(results[0])
                Path(result['raw']).write_text('CHANGED CPU FIXTURE\n')
                with self.assertRaisesRegex(ValueError, 'raw measurement changed'):
                    MODULE.collect(argparse.Namespace(run_root=root, out=root / 'changed', target=None))
            finally:
                MODULE.verify_build = original


if __name__ == '__main__':
    unittest.main()
