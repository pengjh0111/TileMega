"""CPU-only P-SEL protocol examples and inherited-guard orchestration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn1_select', ROOT / 'scripts/dnn1_select.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example(samples, predictions=None):
    predictions = predictions or list(range(len(samples)))
    candidates = [dict(rank=i, predicted_ns=100 + predictions[i], binary_sha256=str(i))
                  for i in range(len(samples))]
    order = list(range(len(samples))) + list(reversed(range(len(samples))))
    used = {i: 0 for i in range(len(samples))}
    visits = []
    for pid, rank in enumerate(order, start=100):
        value = samples[rank][used[rank]]
        used[rank] += 1
        visits.append(dict(rank=rank, pid=pid, median_ms=value, p90_ms=value * 1.1))
    return candidates, visits


class Protocol(unittest.TestCase):
    def test_clear_measured_winner_overrides_prediction(self):
        c, v = example([[1, 1], [1.02, 1.02]], predictions=[1, 0])
        self.assertEqual(module.summarize(c, v)['winner_rank'], 0)

    def test_half_percent_floor_uses_prediction(self):
        c, v = example([[1, 1], [1.004, 1.004]], predictions=[1, 0])
        result = module.summarize(c, v)
        self.assertEqual(result['winner_rank'], 1)
        self.assertTrue(result['tie_broken_by_prediction'])
        self.assertEqual(result['threshold_pct'], 0.5)

    def test_observed_process_spread_expands_tie_threshold(self):
        c, v = example([[1, 1], [1.01, 1.03]], predictions=[1, 0])
        result = module.summarize(c, v)
        self.assertEqual(result['winner_rank'], 1)
        self.assertGreater(result['threshold_pct'], 3.9)

    def test_fresh_process_and_sequence_are_required(self):
        c, v = example([[1, 1], [2, 2]])
        v[1]['pid'] = v[0]['pid']
        with self.assertRaisesRegex(ValueError, 'fresh process'):
            module.summarize(c, v)
        v[1]['pid'] = 400
        v[1], v[2] = v[2], v[1]
        v[0], v[1] = v[1], v[0]
        with self.assertRaisesRegex(ValueError, 'forward then reverse'):
            module.summarize(c, v)

    def test_nan_is_not_a_winner(self):
        c, v = example([[float('nan'), 1], [2, 2]])
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            module.summarize(c, v)

    def test_single_resource_survivor_still_gets_two_processes(self):
        c, v = example([[2, 4]])
        result = module.summarize(c, v)
        self.assertEqual(result['winner_rank'], 0)
        self.assertEqual(result['arms'][0]['median_ms'], 3)
        self.assertIsNone(result['threshold_pct'])


class Orchestration(unittest.TestCase):
    def exercise(self, child_failure=False, parallel=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export = directory / 'export'
            (export / 'checkpoint').mkdir(parents=True)
            for path in (export / 'manifest.json', export / 'exported_program.pt2',
                         export / 'checkpoint/model.safetensors', directory / 'bridge.json',
                         directory / 'inputs_b2.safetensors'):
                path.write_text('frozen test input')
            candidates, identities = [], {}
            for rank in range(4):
                binary = directory / f'candidate-{rank}.so'
                binary.write_text(str(rank))
                Path(str(binary) + '.identity.json').write_text('{}')
                digest = module.sha(binary)
                candidates.append(dict(status='compiled', rank=rank, predicted_ns=rank + 10,
                                       binary=str(binary), binary_sha256=digest))
                identities[str(binary)] = dict(binary_sha256=digest, artifact_id=str(rank), trace=False,
                    source=dict(head='test-head'), plan=dict(runtime_batch=2,
                        export_sha256=module.sha(directory / 'bridge.json')))
            source = directory / 'candidates.json'
            source.write_text(json.dumps(candidates))
            argv = ['dnn1_select.py', '--candidates', str(source), '--out', str(directory / 'out'),
                    '--export', str(export), '--bridge', str(directory / 'bridge.json'),
                    '--inputs', str(directory / 'inputs_b{batch}.safetensors')]
            if parallel:
                argv += ['--cpu-affinity', '32,33,34,35']
            commands, sessions = [], []
            def child(command, **kwargs):
                commands.append(command)
                self.assertEqual(Path(command[0]).resolve(),
                                 (module.ROOT / 'scripts/dnn1_idle_floor_python.py').resolve())
                self.assertEqual(command[1:3], ['-m', 'tilemega.dnn.benchmark_timing_only'])
                self.assertNotIn('tilemega.bench.scheduler', command)
                self.assertNotIn('tilemega.bench.gpu_guard', command)
                if child_failure:
                    return subprocess.CompletedProcess(command, 17)
                out = Path(command[command.index('--out') + 1]) / 'measurements'
                out.mkdir(parents=True)
                library = command[command.index('--library') + 1]
                identity = identities[library]
                row = dict(gpu_index=0, batch=2, protocol='P-SEL',
                    input_sha256=module.sha(directory / 'inputs_b2.safetensors'),
                    round_index=int(command[command.index('--round-index') + 1]),
                    identity=identity, pid=1000 + len(commands),
                    median_ms=int(identity['artifact_id']) + 1, p90_ms=int(identity['artifact_id']) + 1.1,
                    guard_log=str(out / 'guard.jsonl'))
                (out / 'measurements.json').write_text(json.dumps(row))
                return subprocess.CompletedProcess(command, 0)
            baseline = types.SimpleNamespace(require_session=lambda cell: sessions.append(cell))
            identity = types.SimpleNamespace(verify=lambda path: identities[str(path)])
            with patch.object(sys, 'argv', argv), patch.object(module.subprocess, 'run', child), \
                    patch.dict(os.environ, {'TILEMEGA_DEVICE_INDEX': '0'}, clear=True), \
                    patch.dict(sys.modules, {'dnn1_baselines': baseline, 'tilemega.build.identity': identity}):
                if child_failure:
                    with self.assertRaisesRegex(RuntimeError, 'P-SEL child failed'):
                        module.main()
                    self.assertEqual(len(commands), 1)
                    self.assertEqual(json.loads((directory / 'out/failure.json').read_text())['returncode'], 17)
                    self.assertFalse((directory / 'out/result.json').exists())
                else:
                    module.main()
                    result = json.loads((directory / 'out/result.json').read_text())
                    self.assertEqual([v['rank'] for v in result['visits']], [0, 1, 2, 3, 3, 2, 1, 0])
                    self.assertEqual(result['winner_rank'], 0)
                    self.assertEqual(result['winner_binary_sha256'], candidates[0]['binary_sha256'])
                    self.assertEqual(len(sessions), 9)
                    expected = dict(gpu=0, parallel=True, cpu_affinity=[32,33,34,35]) if parallel else dict(gpu=0)
                    self.assertTrue(all(session == expected for session in sessions))
                    self.assertEqual(len(commands), 8)
                    wrapper = str((module.ROOT / 'scripts/dnn1_idle_floor_python.py').resolve())
                    self.assertEqual(result['source_sha256'][wrapper], module.sha(wrapper))

    def test_eight_children_reuse_the_outer_guard(self):
        self.exercise()

    def test_explicit_cpu_trial_reaches_each_session_check(self):
        self.exercise(parallel=True)

    def test_first_child_failure_stops_selection(self):
        self.exercise(child_failure=True)


if __name__ == '__main__':
    unittest.main()
