"""Independent guards preserve valid starts without hiding numerical failures."""
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from tilemega.tp.process_cohorts import guarded_command, output_job, ref, sha, validate_run, seal_and_attach, wait_idle, validate_checks


class FakeCommands:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self._cohort_root = Path('/unused')
        self._unsegmented_command = self.command
        self.run_command = MethodType(guarded_command, self)

    def command(self, folder, argv, guarded=False, cap=900):
        self.calls.append((folder, argv))
        code, reason, child, numerical_failure = self.responses.pop(0)
        folder.mkdir(parents=True)
        guard = folder / 'guard'
        guard.mkdir()
        (guard / 'guard_result.json').write_text(json.dumps(dict(code=code, reason=reason, gpu_indices=['1', '3'])))
        if child is not None:
            (guard / 'pgid').write_text(str(child))
        result = dict(returncode=code, PID=900 + len(self.calls), argv=list(map(str, argv)))
        (folder / 'PROCESS.json').write_text(json.dumps(result))
        if numerical_failure:
            output = Path(argv[-1])
            output.mkdir()
            (output / 'check.json').write_text(json.dumps({'pass': False, 'rank_tokens_identical': True}))
        return result


class ProcessCohortTests(unittest.TestCase):
    def test_startup_scheduling_waits_without_changing_guard_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'configs/bench'
            config.mkdir(parents=True)
            policy = config / 'guard_policy_c550.json'
            original = json.dumps(dict(max_hidden_mib=1024, max_idle_wait_s=20, interval_s=0))
            policy.write_text(original)
            cell = SimpleNamespace(a=SimpleNamespace(group='1,3', startup_hidden_mib=64))
            samples = [dict(hidden_mib=n) for n in (500, 0, 0, 0)]
            with patch('tilemega.bench.gpu_guard.gpu', side_effect=samples), patch('tilemega.bench.gpu_guard.idle', return_value=True), patch('tilemega.tp.process_cohorts.time.sleep'):
                wait_idle(cell, root / 'execution', root)
            rows = [json.loads(line) for line in (root / 'execution_readiness.jsonl').read_text().splitlines()]
            self.assertEqual([r['ready'] for r in rows], [False, True])
            self.assertEqual(policy.read_text(), original)
            cell.a.startup_hidden_mib = 2048
            with self.assertRaisesRegex(ValueError, 'only narrow'):
                wait_idle(cell, root / 'rejected', root)

    @patch('tilemega.tp.process_cohorts.wait_idle')
    def test_coherent_wrong_gpu_group_is_rejected(self, wait):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = FakeCommands([(0, 'child exit 0', 100 + i, False) for i in range(4)])
            runs = [dict(manifest=cell.run_command(root / f'exec{i}', ['dummy'], guarded=True)['guard_manifest']) for i in range(4)]
            binary = root / 'fixture_binary'
            binary.write_bytes(b'CPU metadata fixture; never loaded as a device library')
            def record(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return ref(path)
            native = record('native.json', dict(PID=100, steps=512, rank_tokens_identical=True,
                            execution_identity={'artifact_id': 'fixture'}))
            hf = record('hf.json', {'pass': True})
            checks = []
            for i in range(2):
                tokens = record(f'c2_{i}/tokens.json', [[7] * 1024])
                checks.append(record(f'c2_{i}/check.json', dict(PID=102+i, steps=1024, tp=2, batch=1,
                    rank_tokens_identical=True, token_sha256=tokens['sha256'],
                    execution_identity={'artifact_id': 'fixture'}, **{'pass': True})))
            data = dict(kind='mx7_individual_check_cohort', fresh_processes=2, producer=validate_run(runs[0]['manifest'])['producer'],
                binary=ref(binary), native_C1=native, C1=hf, C2=checks, runs=runs,
                tp=2, batch=1, group='1,3', artifact_id='fixture')
            proof = record('qualification.json', data)
            validate_checks(proof)
            manifest_path = Path(runs[-1]['manifest']['path'])
            manifest = json.loads(manifest_path.read_text())
            guard_path = Path(manifest['attempts'][-1]['guard']['path'])
            guard = json.loads(guard_path.read_text())
            guard['gpu_indices'] = ['1', '4']
            guard_path.write_text(json.dumps(guard))
            manifest['attempts'][-1]['guard'] = ref(guard_path)
            manifest_path.write_text(json.dumps(manifest))
            data['runs'][-1]['manifest'] = ref(manifest_path)
            proof = record('wrong_group.json', data)
            with self.assertRaisesRegex(ValueError, 'different physical GPU group'):
                validate_checks(proof)

    def test_incomplete_telemetry_waits_without_starting_a_gpu_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'configs/bench'
            config.mkdir(parents=True)
            (config / 'guard_policy_c550.json').write_text(json.dumps(dict(max_idle_wait_s=20, interval_s=0)))
            cell = SimpleNamespace(a=SimpleNamespace(group='1,3'))
            with patch('tilemega.bench.gpu_guard.gpu', side_effect=[KeyError(('Memory', 'vram total')), {}, {}, {}]), patch('tilemega.bench.gpu_guard.idle', return_value=True), patch('tilemega.tp.process_cohorts.time.sleep'):
                wait_idle(cell, root / 'execution', root)
            rows = [json.loads(line) for line in (root / 'execution_readiness.jsonl').read_text().splitlines()]
            self.assertFalse(rows[0]['ready'])
            self.assertTrue(rows[0]['telemetry_errors'])
            self.assertTrue(rows[1]['ready'])

    def test_process_executes_its_sealed_cohort_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = FakeCommands([(0, 'child exit 0', 101, False)])
            cell.run_command = cell.command
            archive = root / 'cohorts.py'
            seal_and_attach(archive, cell, root)
            cell.run_command.__func__.__globals__['wait_idle'] = lambda *args: None
            result = cell.run_command(root / 'execution', ['dummy'], guarded=True)
            data = validate_run(result['guard_manifest'])
            self.assertEqual(Path(data['producer']['path']), archive)
            self.assertEqual(data['producer']['sha256'], sha(archive))

    @patch('tilemega.tp.process_cohorts.wait_idle')
    def test_preflight_refusal_does_not_consume_a_fresh_start(self, wait):
        with tempfile.TemporaryDirectory() as directory:
            cell = FakeCommands([(75, 'preflight occupied GPU3', None, False),
                                 (0, 'child exit 0', 101, False)])
            result = cell.run_command(Path(directory) / 'execution', ['dummy'], guarded=True)
            data = validate_run(result['guard_manifest'])
            self.assertEqual([item['child_pid'] for item in data['attempts']], [None, 101])
            self.assertEqual(wait.call_count, 2)

    @patch('tilemega.tp.process_cohorts.wait_idle')
    def test_later_interruption_preserves_an_earlier_valid_start(self, wait):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = FakeCommands([(0, 'child exit 0', 101, False),
                                 (75, 'running interference GPU3', 102, False),
                                 (0, 'child exit 0', 103, False)])
            _, first = output_job(cell, root, 'first', root / 'first_out', lambda out: ['dummy', out])
            original = sha(first['manifest']['path'])
            _, second = output_job(cell, root, 'second', root / 'second_out', lambda out: ['dummy', out])
            self.assertEqual(sha(first['manifest']['path']), original)
            self.assertEqual(first['PID'], 101)
            self.assertEqual(second['PID'], 103)
            self.assertEqual(len(second['failed_runs']), 1)
            self.assertEqual(validate_run(first['manifest'])['attempts'][-1]['child_pid'], 101)
            _, retained = output_job(cell, root, 'first', root / 'first_out', lambda out: ['dummy', out])
            self.assertEqual(retained['PID'], 101)
            self.assertEqual(len(cell.calls), 3)

    @patch('tilemega.tp.process_cohorts.wait_idle')
    def test_numerical_failure_is_not_retried_as_interference(self, wait):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = FakeCommands([(75, 'running interference GPU3', 101, True)])
            with self.assertRaisesRegex(RuntimeError, 'numerical/token failure'):
                output_job(cell, root, 'native', root / 'tokens', lambda out: ['dummy', out])
            self.assertEqual(len(cell.calls), 1)

    @patch('tilemega.tp.process_cohorts.wait_idle')
    def test_started_failure_cannot_be_relabelled_as_preflight(self, wait):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = FakeCommands([(75, 'preflight occupied GPU3', None, False),
                                 (0, 'child exit 0', 102, False)])
            result = cell.run_command(root / 'execution', ['dummy'], guarded=True)
            path = Path(result['guard_manifest']['path'])
            value = json.loads(path.read_text())
            value['attempts'][0]['child_pid'] = 101
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'unlaunched preflight'):
                validate_run(ref(path))


if __name__ == '__main__':
    unittest.main()
