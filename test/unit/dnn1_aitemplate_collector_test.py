"""CPU fixtures for the independent AIT strict collector; no GPU measurements."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_aitemplate_cell as collector
import dnn1_aitemplate as runner


class ReleaseGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'runs/dnn1/resume_tvm'
        self.directory.mkdir(parents=True)
        evidence = self.directory / 'evidence.json'
        collector.save(evidence, {'fixture': True})
        self.dispatcher = self.directory / 'dispatcher.json'
        collector.save(self.dispatcher, {'status': 'primary_complete', 'workers': {}})
        record = {'path': str(evidence), 'sha256': collector.sha(evidence)}
        self.release = dict(schema=1, primary_cells_complete=125, primary_evidence=record,
            dispatchers=[str(self.dispatcher)], cells=[dict(record, model=model, batch=batch,
                status='validated', source_version='fixture-source-version')
                for model in ('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet')
                for batch in (1, 2, 4, 8, 16)])

    def check(self):
        collector.save(self.directory / 'aitemplate_release.json', self.release)
        with patch.object(runner, 'ROOT', self.root):
            return runner.require_tvm_complete()

    def test_different_version_and_genuine_tuner_failure_release_gate(self):
        self.release['cells'][-1].update(status='genuine_failure', failure_kind='tune',
                                         source_version='fixed-source-version')
        self.assertIn('TVM25', self.check()['policy'])

    def test_infrastructure_rejection_does_not_release_gate(self):
        self.release['cells'][-1].update(status='genuine_failure', failure_kind='infrastructure')
        with self.assertRaises(ValueError):
            self.check()

    def test_incomplete_coverage_does_not_release_gate(self):
        self.release['cells'].pop()
        with self.assertRaises(RuntimeError):
            self.check()

    def test_active_tuner_does_not_release_gate(self):
        collector.save(self.dispatcher, {'status': 'running', 'workers': {'0': {'pid': 123}}})
        with self.assertRaises(RuntimeError):
            self.check()

    def test_changed_evidence_does_not_release_gate(self):
        Path(self.release['primary_evidence']['path']).write_text('changed')
        with self.assertRaises(ValueError):
            self.check()


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.outer = []
        self.session = dict(model='resnet18', batch=1, gpu=0, ended=60,
                            graph_mode=True, status='complete_pending_outer_audit', processes=[])
        for i in range(5):
            directory = self.root / str(i)
            directory.mkdir()
            path = directory / 'measurements.json'
            role = 'measurement' if i < 3 else 'correctness'
            value = dict(system='aitemplate', model='resnet18', batch=1, precision='bf16',
                gpu_index=0, pid=100+i, process_role=role, correctness_passed=True,
                correctness=[dict(cosine_vs_fp32=1., top1_agreement_vs_fp32=1.)],
                identity={'synthetic': 'fixture', 'recipe': {'graph_mode': True}},
                graph_mode={'requested': True}, protocol={'warmup': 50},
                library_sha256='library-fixture', compiled_receipt_sha256='compiled-fixture',
                inputs_sha256='inputs-fixture',
                samples_ms=[float(i+1)]*200 if role == 'measurement' else [],
                median_ms=float(i+1) if role == 'measurement' else None,
                p90_ms=float(i+1) if role == 'measurement' else None)
            collector.save(path, value)
            stamps = [dict(label='before-measurement', time=10*i+2, pids=[1000+i]),
                      dict(label='after-measurement', time=10*i+4, pids=[1000+i])]
            (directory / 'guard.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in stamps))
            self.outer.append(dict(time=10*i+3, owners={str(1000+i): 100}))
            self.session['processes'].append(dict(pid=100+i, receipt=str(path),
                receipt_sha256=collector.sha(path), returncode=0,
                before={'owners': {}, 'used_mib': 0}, after={'owners': {}, 'used_mib': 0},
                before_drain={'status': 'empty_observed', 'samples': [
                    {'telemetry': {'owners': {}, 'used_mib': 0}} for _ in range(2)]},
                drain={'status': 'empty_observed', 'samples': [
                    {'telemetry': {'owners': {}, 'used_mib': 0}} for _ in range(2)]}))
        self.outer.append(dict(time=70, phase='after', exit_code=0, owners={}))

    def tearDown(self):
        self.temp.cleanup()

    def collect(self):
        session = self.root / 'cell-result.json'
        guard = self.root / 'outer.jsonl'
        collector.save(session, self.session)
        guard.write_text(''.join(json.dumps(s)+'\n' for s in self.outer))
        output = self.root / 'collected.json'
        status = collector.collect(argparse.Namespace(session=session, guard=guard, out=output))
        return status, json.loads(output.read_text())

    def test_clean_sampled_windows_aggregate_all_three_processes(self):
        code, result = self.collect()
        self.assertEqual(code, 0)
        self.assertEqual(result['median_ms'], 2.)
        self.assertEqual(result['process_range_pct'], 100.)
        self.assertEqual(result['correctness_passes'], 5)

    def test_extra_owner_quarantines_without_timing(self):
        self.outer[1]['owners']['9999'] = 80
        code, result = self.collect()
        self.assertEqual(code, 1)
        self.assertNotIn('median_ms', result)

    def test_outer_guard_rejection_is_not_salvaged(self):
        self.outer[-1]['exit_code'] = 75
        self.assertEqual(self.collect()[0], 1)

    def test_modified_process_receipt_is_rejected(self):
        path = Path(self.session['processes'][0]['receipt'])
        path.write_text(path.read_text() + '\n')
        self.assertEqual(self.collect()[0], 1)

    def test_five_distinct_processes_are_required(self):
        self.session['processes'][1]['pid'] = self.session['processes'][0]['pid']
        self.assertEqual(self.collect()[0], 1)

    def test_mixed_engine_identity_is_rejected(self):
        item = self.session['processes'][1]
        path = Path(item['receipt'])
        value = json.loads(path.read_text())
        value['identity']['different_recipe'] = True
        collector.save(path, value)
        item['receipt_sha256'] = collector.sha(path)
        self.assertEqual(self.collect()[0], 1)

    def mutate(self, change):
        item = self.session['processes'][0]
        path = Path(item['receipt'])
        value = json.loads(path.read_text())
        change(value)
        collector.save(path, value)
        item['receipt_sha256'] = collector.sha(path)

    def test_graph_reference_cannot_join_graph_cohort(self):
        self.mutate(lambda r: r['graph_mode'].update(requested=False))
        self.assertEqual(self.collect()[0], 1)

    def test_changed_library_cannot_join_cohort(self):
        self.mutate(lambda r: r.update(library_sha256='other-library'))
        self.assertEqual(self.collect()[0], 1)

    def test_pass_flag_does_not_override_original_threshold(self):
        self.mutate(lambda r: r.update(correctness=[dict(cosine_vs_fp32=.99,
                                                        top1_agreement_vs_fp32=1.)]))
        self.assertEqual(self.collect()[0], 1)

    def test_transient_dirty_memory_is_rejected(self):
        self.session['processes'][0]['before']['used_mib'] = 4
        self.assertEqual(self.collect()[0], 1)

    def test_extra_good_reread_cannot_replace_final_boundary(self):
        self.session['processes'][0]['drain']['samples'][-1]['telemetry']['used_mib'] = 1
        self.assertEqual(self.collect()[0], 1)

    def test_queue_matches_scheduler_glob_and_records_guarded_numa_command(self):
        output = self.root / 'queue-run'
        release = self.root / 'classic-release.json'
        release.write_text('{"frozen":true}')
        with patch.object(collector, 'source_identity', return_value={'files': {}}), \
             patch.object(collector.subprocess, 'check_output', return_value='GPU-fixture, 00000000:41:00.0'):
            collector.queue(argparse.Namespace(out=output, gpu=2, model='resnet18', batch=1, release=release))
        queues = list((output/'queue').glob('queue_*.json'))
        self.assertEqual(len(queues), 1)
        job = json.loads(queues[0].read_text())[0]
        self.assertTrue(job['gpu'] and job['timing'])
        self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(job['env']['TILEMEGA_DNN1_AIT_RELEASE'], str(release))
        self.assertEqual(job['env']['TILEMEGA_DNN1_AIT_RELEASE_SHA256'], collector.sha(release))
        self.assertIn('--physcpubind=16,17,18,19,20,21,22,23', job['command'])
        self.assertEqual(job['command'][:2], ['numactl', '--all'])
        self.assertIn('0000:41:00.0', job['command'])

    def test_five_fresh_children_bind_declared_cpus_from_restricted_controller(self):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 2:
            self.skipTest('requires two permitted CPUs')
        target, control = allowed[0], allowed[-1]
        for _ in range(5):
            result = subprocess.run(['taskset', '-c', str(control), 'numactl', '--all',
                f'--physcpubind={target}', sys.executable, '-c',
                'import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],
                check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(result.stdout), [target])


if __name__ == '__main__':
    unittest.main()
