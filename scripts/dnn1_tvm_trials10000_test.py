#!/usr/bin/env python3
"""CPU-only contract checks; synthetic files never become measurement evidence."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn1_tvm_trials10000_sample as sample
import dnn1_tvm_trials10000_collect as collect


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_worker_budget_and_immutable_receipt(self):
        directory = self.root / 'fresh/cache/tune'
        cell = dict(batch=16, model='bert', cache=str(directory.parent))
        calls = []
        def original(*args):
            calls.append(args)
            directory.mkdir(parents=True)
            sample.save(directory / 'request.json', dict(max_trials_global=args[4]))
            result = dict(maximum_trials=args[4], timeout_s=args[5], committed_trials=42)
            sample.save(directory / 'receipt.json', result)
            return 'database', result
        database, result = sample.tuning_callback(original, cell)(
            'fixed.pt2', 'bf16', 16, directory, 2000, 1800, True)
        self.assertEqual(calls[0][4:], (10000, 9000, True))
        self.assertEqual(database, 'database')
        self.assertEqual(result['num_trials_per_iter'], 64)
        self.assertIsNone(result['max_trials_per_task'])
        self.assertEqual(result['initial_database_records'], 0)
        self.assertEqual(sample.read(result['actual_worker_request'])['max_trials_global'], 10000)
        self.assertEqual(sample.sha(directory / 'receipt.json'), result['original_tuning_receipt_sha256'])
        self.assertNotIn('initial_database_records', sample.read(directory / 'receipt.json'))

    def test_prior_tune_directory_refused_before_worker(self):
        old = self.root / 'cache/tune'
        old.mkdir(parents=True)
        marker = old / 'database_tuning_record.json'
        marker.write_bytes(b'old database bytes\n')
        digest = sample.sha(marker)
        original = Mock()
        callback = sample.tuning_callback(original, dict(batch=1, model='resnet18', cache=str(old.parent)))
        with self.assertRaisesRegex(ValueError, 'retain existing'):
            callback('fixed.pt2', 'bf16', 1, old)
        original.assert_not_called()
        self.assertEqual(sample.sha(marker), digest)

    def test_wrong_actual_budget_is_rejected(self):
        callback = sample.tuning_callback(Mock(return_value=(None, dict(maximum_trials=2000, timeout_s=9000))),
            dict(batch=1, model='bert', cache=str(self.root)))
        with self.assertRaisesRegex(ValueError, 'actual tuning callback'):
            callback('fixed.pt2', 'bf16', 1, self.root / 'tune')

    def test_recipe_is_epoch_specific_without_round_variants(self):
        old = dict(schema=6, tuning=dict(max_trials_global=2000), target='unchanged')
        first = sample.recipe(old, dict(epoch_manifest_sha256='first'))
        self.assertEqual(first, sample.recipe(old, dict(epoch_manifest_sha256='first')))
        self.assertNotEqual(first, sample.recipe(old, dict(epoch_manifest_sha256='second')))
        self.assertEqual(first['target'], 'unchanged')
        self.assertEqual(first['tuning'], sample.BUDGET)
        self.assertEqual(first['compatibility_entry']['max_trials_global'], 2000)
        self.assertEqual(old['schema'], 6)

    def test_pinned_tvm_defaults_and_unchanged_worker(self):
        path = collect.ROOT / 'runs/dnn1/resume_tvm/typed_runner_repair_v1/tvm/source/python/tvm/s_tir/meta_schedule/relax_integration.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'tune_relax')
        defaults = dict(zip([a.arg for a in function.args.args][-len(function.args.defaults):],
                            map(ast.literal_eval, function.args.defaults)))
        defaults.update({a.arg: ast.literal_eval(value) for a, value in
            zip(function.args.kwonlyargs, function.args.kw_defaults) if value is not None})
        self.assertIsNone(defaults['max_trials_per_task'])
        self.assertEqual(defaults['num_trials_per_iter'], 64)
        adapter = ast.parse((collect.ROOT / 'python/tilemega/dnn/tvm_baseline.py').read_text())
        calls = [n for n in ast.walk(adapter) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and n.func.attr == 'tune_relax']
        self.assertEqual(len(calls), 1)
        self.assertFalse({'max_trials_per_task', 'num_trials_per_iter'} & {k.arg for k in calls[0].keywords})

    def test_cards_and_timeouts(self):
        self.assertEqual(collect.gpu_pool('0,1,2,3,4,5,7'), [0, 1, 2, 3, 4, 5, 7])
        for value in ('6', '0,0', ''):
            with self.assertRaises(ValueError):
                collect.gpu_pool(value)
        self.assertGreaterEqual(collect.PROCESS_TIMEOUT, 18000)
        self.assertGreater(collect.CELL_TIMEOUT, 5 * collect.PROCESS_TIMEOUT)

    def test_new_files_excluded_from_ait_source7(self):
        proof = sample.read(collect.ROOT / 'runs/dnn1/resume/aitemplate/runtime_ABI_graph_v7/prepare/prepare.json')
        files = proof['identity']['files']
        installed = collect.ROOT / 'runs/dnn1/resume/aitemplate/python'
        for name in ('sample', 'collect', 'test'):
            path = collect.ROOT / f'scripts/dnn1_tvm_trials10000_{name}.py'
            self.assertNotIn(str(path), files)
            self.assertFalse(path.is_relative_to(installed))

    def test_wait_is_file_only_before_explicit_release(self):
        events = []
        def release_on_sleep(_seconds):
            self.assertEqual(sample.read(self.root / 'waiting.json')['GPU_queries'], 0)
            sample.save(self.root / 'release.json', dict(synthetic=True))
            events.append('release')
        def dispatch(options):
            self.assertTrue((self.root / 'release.json').exists())
            events.append('dispatch')
            return 0
        with patch.object(collect, 'validate_manifest'), \
             patch.object(collect.time, 'sleep', side_effect=release_on_sleep), \
             patch.object(collect, 'dispatch', side_effect=dispatch), \
             patch.object(collect.device, 'gpu') as gpu, patch.object(collect.base, 'held') as held, \
             patch.object(collect.subprocess, 'Popen') as popen:
            self.assertEqual(collect.wait_for_release(SimpleNamespace(epoch=self.root, poll_seconds=30)), 0)
        self.assertEqual(events, ['release', 'dispatch'])
        gpu.assert_not_called()
        held.assert_not_called()
        popen.assert_not_called()

    def test_prepare_full25_without_cache_jobs_or_processes(self):
        exports = self.root / 'runs/dnn1/exports'
        rows = []
        for model, batch in sorted(collect.CELLS):
            export = exports / model
            for name in ('manifest.json', 'exported_program.pt2', 'checkpoint/model.safetensors',
                         f'inputs_b{batch}.safetensors'):
                path = export / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            rows.append(dict(model=model, batch=batch, system='tvm', precision='bf16', status='pass',
                exported_program_sha256=sample.sha(export / 'exported_program.pt2'),
                source_checkpoint_sha256=sample.sha(export / 'checkpoint/model.safetensors'), inputs_sha256='fixed'))
        docs = self.root / 'docs/results'
        docs.mkdir(parents=True)
        csv_path = docs / 'DNN1_baselines.csv'
        with csv_path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (docs / 'DNN1_tvm_diag.csv').write_text('original diagnostics\n')
        old_cache = self.root / 'runs/dnn1/old/cache.bin'
        old_cache.parent.mkdir(parents=True)
        old_cache.write_bytes(b'old immutable cache')
        proof = self.root / 'ait_source7.json'
        sample.save(proof, dict(identity=dict(files={})))
        epoch = self.root / 'runs/dnn1/new_epoch'
        args = SimpleNamespace(epoch=epoch, tvm_root=self.root / 'tvm', ait_source_proof=proof,
            exports=exports, gpus='0,1,2,3,4,5,7', max_parallel=4)
        before = {p: sample.sha(p) for p in (csv_path, old_cache)}
        with patch.object(collect, 'ROOT', self.root), patch.object(collect, 'source_freeze',
             return_value=({'source_diff_sha256': hashlib.sha256(b'').hexdigest()}, {})), \
             patch.object(collect, 'affinities'), patch.object(collect.subprocess, 'Popen') as popen, \
             patch.object(collect.device, 'gpu') as gpu, redirect_stdout(io.StringIO()):
            collect.prepare(args)
        manifest = sample.read(epoch / 'manifest.json')
        self.assertEqual(len(manifest['cells']), 25)
        self.assertEqual(manifest['budget'], sample.BUDGET)
        self.assertEqual(manifest['max_parallel'], 4)
        self.assertFalse((epoch / 'cache').exists())
        self.assertFalse((epoch / 'jobs').exists())
        self.assertEqual(sample.read(epoch / 'progress.json')['GPU_jobs_started'], 0)
        self.assertEqual(before, {p: sample.sha(p) for p in before})
        popen.assert_not_called()
        gpu.assert_not_called()
        with patch.object(collect, 'ROOT', self.root), self.assertRaisesRegex(ValueError, 'absent epoch'):
            collect.prepare(args)

    def fake_cell(self):
        prepared = self.root / 'cell.json'
        sample.save(prepared, dict(epoch=str(self.root), release_sha256='release', model='bert', batch=1,
            gpu=0, cache_root=str(self.root / 'cache/bert_b1'), python='fake-python', process_timeout_s=18000))
        out = self.root / 'raw/attempt1/job'
        out.mkdir(parents=True)
        sample.save(out / 'physical-device-identity.json', dict(wrapper=True))
        return SimpleNamespace(cell=prepared, out=out)

    def drain(self, empty=True):
        row = dict(owners={} if empty else {'foreign': {}}, used_mib=0 if empty else 100)
        return dict(status='empty_observed' if empty else 'timeout',
                    samples=[dict(telemetry=row), dict(telemetry=row)])

    def test_cell_wrapper_directory_roles_timeouts_and_fresh_cache(self):
        args = self.fake_cell()
        calls = []
        def execute(command, folder, env, timeout):
            calls.append((command, timeout))
            payload = Path(command[command.index('--out') + 1])
            sample.save(payload / 'measurements.json', dict(configuration=dict(inputs_sha256='fixed')))
            (payload / 'guard.jsonl').write_text(''.join(json.dumps(dict(label=label,
                telemetry={'clocks.sm': 1980})) + '\n' for label in ('before-measurement', 'after-measurement')))
            return dict(returncode=0, pid=100 + len(calls))
        with patch.object(collect.base, 'require_session'), patch.object(collect, 'validate_release'), \
             patch.object(sample, 'validate_cell'), patch.object(collect, 'idle_boundary', side_effect=lambda c: self.drain()), \
             patch.object(collect.base, 'execute', side_effect=execute):
            self.assertEqual(collect.run_cell(args), 0)
            with self.assertRaisesRegex(ValueError, 'retain the original'):
                collect.run_cell(args)
        self.assertEqual([timeout for _, timeout in calls], [18000] * 5)
        self.assertEqual(['--correctness-only' in command for command, _ in calls], [False] * 3 + [True] * 2)
        result = sample.read(args.out / 'cell-result.json')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['logical_jobs']), 5)
        self.assertTrue(result['cache_initially_absent'])
        runtime = sample.read(args.out / 'runtime-cell.json')
        self.assertEqual(runtime['cache'], str(self.root / 'cache/bert_b1/attempt1'))

    def test_failed_post_boundary_preserves_executed_process(self):
        args = self.fake_cell()
        with patch.object(collect.base, 'require_session'), patch.object(collect, 'validate_release'), \
             patch.object(sample, 'validate_cell'), patch.object(collect, 'idle_boundary',
                 side_effect=[self.drain(), self.drain(False)]), \
             patch.object(collect.base, 'execute', return_value=dict(returncode=1, pid=123)):
            with self.assertRaisesRegex(ValueError, 'idle samples'):
                collect.run_cell(args)
        result = sample.read(args.out / 'cell-result.json')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['logical_jobs'][0]['final']['pid'], 123)
        self.assertEqual(result['logical_jobs'][0]['final']['drain']['status'], 'timeout')

    def test_repair_pending_cannot_be_release(self):
        csv_path = self.root / 'AIT.csv'
        csv_path.write_text('published bytes\n')
        rows = [dict(model=m, batch=str(b), status='pass') for m, b in sorted(collect.CELLS)]
        ledger = self.root / 'AIT_complete.json'
        declaration = dict(schema=1, system='aitemplate', status='final_25_terminal', repair_pending=False,
            csv_sha256=sample.sha(csv_path), cells=[dict(model=r['model'], batch=int(r['batch']),
                status=r['status'], final=True) for r in rows])
        with patch.object(collect, 'ROOT', self.root):
            sample.save(ledger, declaration)
            collect.validate_ait_completion(ledger, csv_path, rows)
            for field, bad in [('repair_pending', True), ('status', 'cohort_drained'), ('csv_sha256', 'wrong')]:
                sample.save(ledger, dict(declaration, **{field: bad}))
                with self.assertRaises(ValueError):
                    collect.validate_ait_completion(ledger, csv_path, rows)
            sample.save(ledger, dict(declaration, cells=declaration['cells'][:-1]))
            with self.assertRaises(ValueError):
                collect.validate_ait_completion(ledger, csv_path, rows)

    def test_terminal_ait_rejects_mixed_inputs_and_pending_failures(self):
        rows = []
        for model, batch in sorted(collect.CELLS):
            attempt = self.root / f'{model}_b{batch}/attempt1'
            paths = []
            for index in range(5):
                path = attempt / f'job/ait/p{index}/measurements.json'
                sample.save(path, dict(system='aitemplate', model=model, batch=batch, precision='bf16',
                    pid=100 + index, correctness_passed=True, process_role='measurement' if index < 3 else 'correctness',
                    correctness={}, inputs_sha256='inputs', library_sha256='library', compiled_receipt_sha256='compiled',
                    identity=dict(exported_program_sha256='export', source_checkpoint_sha256='weights')))
                paths.append(str(path))
            session = attempt / 'job/ait/cell-result.json'
            sample.save(session, dict(status='complete_pending_outer_audit'))
            sample.save(attempt / 'guard_result.json', dict(code=0))
            rows.append(dict(model=model, batch=str(batch), system='aitemplate', precision='bf16', status='pass',
                versions='source-bound', source=json.dumps(paths), correctness_passes='5/5', cell_session=str(session),
                exported_program_sha256='export', source_checkpoint_sha256='weights', inputs_sha256='inputs',
                engine_or_library_sha256=json.dumps(dict(library_sha256='library', compiled_receipt_sha256='compiled'))))
        with patch.object(collect, 'csv_rows', return_value=rows), \
             patch.object(collect.base, 'strict_correctness', return_value=(True, '')):
            selected, evidence = collect.terminal_ait(self.root / 'unused.csv')
            self.assertEqual(len(selected), 25)
            self.assertEqual(len(evidence), 175)
            path = Path(json.loads(rows[0]['source'])[0])
            sample.save(path, dict(sample.read(path), inputs_sha256='different'))
            with self.assertRaisesRegex(ValueError, 'different exports/weights/inputs'):
                collect.terminal_ait(self.root / 'unused.csv')
            rows[0].update(status='N/A', reason='repair pending')
            with self.assertRaisesRegex(ValueError, 'infrastructure/pending'):
                collect.terminal_ait(self.root / 'unused.csv')

    def test_missing_release_refuses_before_resource_or_launch(self):
        with patch.object(collect, 'validate_manifest', return_value={}), \
             patch.object(collect.device, 'gpu') as gpu, patch.object(collect.subprocess, 'Popen') as popen:
            with self.assertRaises(FileNotFoundError):
                collect.dispatch(SimpleNamespace(epoch=self.root))
        gpu.assert_not_called()
        popen.assert_not_called()

    def test_collector_rechecks_actual_budget_db_and_vm(self):
        epoch = self.root
        sample.save(epoch / 'manifest.json', dict(synthetic=True))
        folder = epoch / 'jobs/bert_b1'
        sample.save(folder / 'cell.json', dict(model='bert', batch=1))
        attempt = folder / 'raw/bert_b1/attempt1'
        sample.save(folder / 'scheduler/state.json', dict(bert_b1=dict(status='done', out=str(attempt))))
        cache = epoch / 'cache/bert_b1/attempt1'
        work = cache / 'key/tune'
        sample.save(work / 'request.json', dict(max_trials_global=10000))
        sample.save(work / 'receipt.json', dict(maximum_trials=10000, timeout_s=9000))
        (work / 'database_tuning_record.json').write_text(json.dumps([0, [[], [.001]]]) + '\n')
        (work / 'builder_results.jsonl').write_text(json.dumps(dict(error=None)) + '\n')
        library = cache / 'model.so'
        library.write_bytes(b'synthetic VM identity; never loaded')
        runtime = dict(model='bert', batch=1, cache=str(cache), epoch_manifest_sha256=sample.sha(epoch / 'manifest.json'),
            release_sha256='release', exported_program_sha256='export', source_checkpoint_sha256='weights',
            expected_inputs_sha256='inputs')
        runtime_path = attempt / 'job/runtime-cell.json'
        sample.save(runtime_path, runtime)
        tuning = dict(maximum_trials=10000, timeout_s=9000, initial_database_records=0, work_dir=str(work),
            original_tuning_receipt=str(work / 'receipt.json'), original_tuning_receipt_sha256=sample.sha(work / 'receipt.json'),
            actual_worker_request=str(work / 'request.json'), actual_worker_request_sha256=sample.sha(work / 'request.json'),
            committed_trials=1, successful_trials=1, failed_trials=0, attempted_builds=1, build_errors=0,
            elapsed_s=9000., deadline_hit=True)
        recipe = sample.recipe({}, runtime)
        paths, jobs = [], []
        manifest = dict(epoch=str(epoch), source_identity=dict(frozen='source'))
        for index in range(5):
            path = attempt / f'job/measurements/p{index}/measurements.json'
            paths.append(str(path))
            values = [1.] * 200 if index < 3 else []
            sample.save(path, dict(round_index=index, samples_ms=values, median_ms=1., p90_ms=1.,
                driver_sha256=sample.sha(collect.ROOT / 'scripts/dnn1_tvm_trials10000_sample.py'),
                exported_program_sha256='export', source_checkpoint_sha256='weights',
                configuration=dict(inputs_sha256='inputs', tvm_diag=dict(tuning=tuning,
                    model_library=str(library), model_sha256=sample.sha(library),
                    cache_identity=dict(recipe=recipe, tvm_source_identity=manifest['source_identity'])))))
            jobs.append(dict(final=dict(measurement=str(path))))
        sample.save(attempt / 'job/cell-result.json', dict(cell_spec=str(runtime_path),
            cell_spec_sha256=sample.sha(runtime_path), prepared_cell_sha256=sample.sha(folder / 'cell.json'),
            cache_initially_absent=True, logical_jobs=jobs))
        sample.save(attempt / 'guard_result.json', dict(code=0))
        (attempt / 'guard.jsonl').write_text('original synthetic outer samples\n')
        row = {name: '' for name in collect.base.FIELDS}
        row.update(source=json.dumps(paths), status='pass')
        def finalizer(cell, raw, destination):
            sample.save(destination / 'accepted.json', dict(synthetic=True))
            return dict(row)
        with patch.object(sample, 'validate_cell'), patch.object(collect.finalizer, 'collect', side_effect=finalizer) as audit:
            result = collect.collect_cell(folder, manifest, 'release')
            self.assertEqual(result['row']['requested_trials'], 10000)
            self.assertEqual(result['row']['successful_trials'], 1)
            self.assertTrue(result['row']['deadline_hit'])
            library.write_bytes(b'tampered VM')
            with self.assertRaisesRegex(ValueError, 'compiled VM changed'):
                collect.collect_cell(folder, manifest, 'release')
            self.assertEqual(audit.call_count, 1)

    def test_controller_audit_error_cleans_registered_own_job(self):
        epoch = self.root
        sample.save(epoch / 'release.json', dict(pushed_commit='commit', pushed_branch='branch'))
        manifest = dict(cells=[dict(model='bert', batch=1)], gpus=[0], max_parallel=1)
        owned = SimpleNamespace(nonce='test-nonce', registered=[], cleaned=False)
        class Lifetime:
            def __init__(self, out):
                self.nonce = owned.nonce
            def __enter__(self):
                return self
            def register(self, process):
                owned.registered.append(process)
            def __exit__(self, *_args):
                owned.cleaned = True
        process = SimpleNamespace(pid=123, poll=lambda: None)
        def enqueue(spec, manifest, gpu, digest, lifetime):
            lifetime.register(process)
            return process, io.StringIO(), epoch / 'jobs/bert_b1'
        releases = [(manifest, 'release'), (manifest, 'release'), ValueError('raw audit failed')]
        with patch.object(collect, 'validate_release', side_effect=releases), \
             patch.object(collect.subprocess, 'check_output', return_value='commit refs/heads/branch'), \
             patch.object(collect, 'prior_jobs_live', return_value=False), \
             patch.object(collect, 'OwnedController', Lifetime), \
             patch.object(collect, 'collect_epoch', return_value=({}, {})), \
             patch.object(collect, 'main_reservations', return_value=False), \
             patch.object(collect.base, 'held', return_value=False), \
             patch.object(collect.device, 'gpu', return_value=dict(owners={}, used_mib=0, free_mib=100000)), \
             patch.object(collect, 'enqueue', side_effect=enqueue), patch.object(collect.time, 'sleep'), \
             self.assertRaisesRegex(ValueError, 'raw audit failed'):
            collect.dispatch(SimpleNamespace(epoch=epoch))
        self.assertEqual(owned.registered, [process])
        self.assertTrue(owned.cleaned)
        self.assertEqual(sample.read(epoch / 'progress.json')['status'], 'stopped')
        self.assertTrue(sample.read(epoch / 'controller_error.json')['unfinished_cohorts_invalidated'])

    def test_actual_resource_wrapper_help_without_cuda(self):
        reject = collect.ROOT / 'runs/dnn1/drafts/batch/review_compile_v4/reject_cuda.so'
        self.assertTrue(reject.is_file())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_PRELOAD=str(reject), PYTHONDONTWRITEBYTECODE='1')
        for name in ('sample', 'collect'):
            result = subprocess.run([str(collect.ROOT / 'scripts/dnn1_idle_floor_python.py'),
                str(collect.ROOT / f'scripts/dnn1_tvm_trials10000_{name}.py'), '--help'],
                env=env, cwd=collect.ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
