"""CPU checks for eight-card scheduling, immutable resumes and TP8 result scope."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tilemega.bench import gpu_guard, wait_idle
from tilemega.tp import sweep_tp8 as sweep
from tilemega.tp.selection_tp import k_best, pin, stage_lists
from test_mx7_selection import fixture


ROOT = Path(__file__).resolve().parents[2]


def device(**extra):
    return dict(availability='Available', owners={}, utilization_pct=0, hidden_mib=0,
                power_w=95, free_mib=64000, **extra)


POLICY = dict(idle_power_w=100, power_margin_w=80, max_util_pct=5, max_hidden_mib=1024)


class EightCardTests(unittest.TestCase):
    def test_all_eight_cards_share_the_existing_sorted_locks(self):
        with patch.dict(os.environ, {'TILEMEGA_DEVICE_GROUP': '7,6,5,4,3,2,1,0'}, clear=True), \
                patch.object(gpu_guard.fcntl, 'flock') as lock:
            indices = gpu_guard.device_indices()
            with gpu_guard.device_locks(indices):
                self.assertEqual([Path(c.args[0].name).name for c in lock.call_args_list],
                                 [f'tilemega_mx_gpu{i}.lock' for i in range(8)])

    def test_a_single_busy_card_blocks_the_group(self):
        def telemetry(policy, index):
            row = device()
            if int(index) == 2:
                row.update(hidden_mib=8192, utilization_pct=95)
            return row
        with patch.object(gpu_guard, 'gpu', side_effect=telemetry), \
                patch.object(gpu_guard, 'external', return_value=set()), \
                patch.object(wait_idle.fcntl, 'flock'):
            value = wait_idle.sample(list(range(8)), POLICY, 12288)
        self.assertFalse(value['ready'])
        self.assertEqual(set(value['rows']), set(map(str, range(8))))

    def test_idle_wait_rechecks_without_holding_any_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(wait_idle, 'sample', side_effect=[dict(ready=False, rows={}, locked=['4'], errors={}),
                    dict(ready=True, rows={}, locked=[], errors={})]), \
                patch.object(wait_idle.time, 'sleep') as sleep:
            log = Path(tmp) / 'monitor.jsonl'
            self.assertTrue(wait_idle.wait(['0'], POLICY, 12288, log, 60, 1)['ready'])
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(len(log.read_text().splitlines()), 2)

    def test_telemetry_error_never_counts_as_idle(self):
        with patch.object(gpu_guard, 'gpu', side_effect=RuntimeError('mx-smi failed')), \
                patch.object(wait_idle.fcntl, 'flock'):
            value = wait_idle.sample(['0', '1'], POLICY, 12288)
        self.assertFalse(value['ready'])
        self.assertEqual(set(value['errors']), {'0', '1'})

    def test_unlimited_idle_wait_survives_arbitrarily_long_occupancy(self):
        import itertools
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(wait_idle, 'sample', side_effect=[dict(ready=False, rows={}, locked=['4'], errors={})] * 3 +
                    [dict(ready=True, rows={}, locked=[], errors={})]), \
                patch.object(wait_idle.time, 'monotonic', side_effect=itertools.count(0, 1000000)), \
                patch.object(wait_idle.time, 'sleep') as sleep:
            self.assertTrue(wait_idle.wait(['0'], POLICY, 12288, Path(tmp) / 'monitor.jsonl', -1, 1)['ready'])
            self.assertEqual(sleep.call_count, 3)

    def test_tp8_selection_uses_its_own_measured_scope(self):
        rows, constants = fixture()
        for row in rows:
            row['key'].update(model='qwen3_1_7b_tp8', tp=8)
        for values in constants.values():
            for row in values:
                row['key']['tp'] = 8
        lists = stage_lists(rows, 'qwen3_1_7b_tp8', 1, constants)
        choice = next(k_best(lists, 28))
        self.assertEqual(pin(choice, 'a' * 64, 8)['evidence']['tp'], 8)
        with self.assertRaisesRegex(ValueError, 'empty measured'):
            stage_lists(rows, 'qwen3_1_7b_tp4', 1, constants)

    def test_measurement_guard_accepts_eight_physical_devices(self):
        from tilemega.serving.measure_tp import require_group
        group = sweep.GROUP
        with patch.dict(os.environ, dict(TILEMEGA_GPU_LOCK_HELD='1', TILEMEGA_DEVICE_GROUP=group,
                    CUDA_VISIBLE_DEVICES=group, MACA_VISIBLE_DEVICES=group), clear=True):
            self.assertEqual(require_group(), list(range(8)))


class CollectionTests(unittest.TestCase):
    def metrics(self, root, batch=1, native=False):
        rows = []
        for count in (1024, 1):
            for number in range(3):
                path = root / f'N{count}_r{number}.json'
                path.write_text(json.dumps([[0] * count for _ in range(batch)]))
                rows.append(dict(N=count, warmup=number == 0, tokens_file=path.name))
        value = dict(tp=8, batch=batch, gpu_indices=list(range(8)), max_tokens=1024,
                     tpot_seconds=.001, ttft_seconds=.1, e2e_seconds=1.123,
                     output_tokens_per_second=1000)
        if native:
            value.update(runs=rows, timed_tokens_identical=True, rank_tokens_identical=True, timing_eligible=True)
        else:
            value['generation_runs'] = rows
        return value

    def test_reject_short_tokens_and_incomplete_timing_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = self.metrics(root)
            self.assertEqual(len(sweep.validate_measurement(value, 1, 'vllm', root)), 6)
            broken = copy.deepcopy(value)
            broken['generation_runs'].pop()
            with self.assertRaisesRegex(ValueError, 'warmup1'):
                sweep.validate_measurement(broken, 1, 'vllm', root)
            (root / value['generation_runs'][0]['tokens_file']).write_text('[[0]]')
            with self.assertRaisesRegex(ValueError, 'token count'):
                sweep.validate_measurement(value, 1, 'vllm', root)

    def test_reject_wrong_tp_and_failed_rank_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = self.metrics(root, native=True)
            value['rank_tokens_identical'] = False
            with self.assertRaisesRegex(ValueError, 'qualification'):
                sweep.validate_measurement(value, 1, 'tilemega', root)
            value['tp'] = 4
            with self.assertRaisesRegex(ValueError, 'TP8 scope'):
                sweep.validate_measurement(value, 1, 'tilemega', root)

    def test_status_includes_all_twenty_cells_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(sweep.summarize(root, write=False)['pending'], 20)
            self.assertFalse(list(root.iterdir()))

    def test_two_rounds_use_medians_and_na_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number, value in enumerate((.001, .003), 1):
                path = root / f'm{number}.json'
                path.write_text(json.dumps(dict(tpot_seconds=value, ttft_seconds=.1, e2e_seconds=2,
                                                output_tokens_per_second=512)))
                sweep.save(root / 'results' / f'vllm_B1_r{number}.json',
                           dict(status='pass', metrics=sweep.ref(path)))
            sweep.save(root / 'results/sglang_B1_r1.json', dict(status='N/A'))
            state = sweep.summarize(root)
            self.assertEqual((state['passed'], state['unavailable'], state['pending']), (1, 1, 18))
            import csv
            rows = list(csv.DictReader((root / 'TP8_results.csv').open()))
            self.assertEqual(float(next(r for r in rows if r['batch'] == '1' and r['system'] == 'vllm')['tpot_ms']), 2)

    def test_completed_process_is_reused_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = sweep.Campaign.__new__(sweep.Campaign)
            campaign.out = Path(tmp)
            campaign.cpu_env = dict(os.environ)
            campaign.env = campaign.cpu_env
            command = lambda p: [sys.executable, '-c', 'print("one fresh process")']
            folder, first = campaign.job('mock', command)
            with patch.object(sweep.subprocess, 'Popen') as start:
                same, second = campaign.job('mock', command)
            self.assertEqual(folder, same)
            self.assertEqual(first, second)
            start.assert_not_called()
            (folder / 'stdout.log').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'changed'):
                campaign.job('mock', command)

    def test_guard_refusals_retry_until_success_without_a_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = sweep.Campaign.__new__(sweep.Campaign)
            campaign.out = Path(tmp)
            campaign.cpu_env = campaign.env = dict(os.environ)
            codes = iter([75, 75, 75, 0])
            def spawn(argv, **kwargs):
                code = next(codes)
                guard = Path(argv[3]);guard.mkdir()
                (guard / 'guard_result.json').write_text(json.dumps(dict(code=code)))
                return SimpleNamespace(pid=12345, wait=lambda: code)
            with patch.object(sweep.subprocess, 'Popen', side_effect=spawn) as start, \
                    patch.object(sweep.time, 'sleep') as sleep:
                folder, result = campaign.job('guard', lambda p: ['true'], sweep.GROUP, allow_failure=True)
            self.assertEqual(start.call_count, 4)
            self.assertEqual(sleep.call_count, 3)
            self.assertEqual(result['returncode'], 0)
            self.assertEqual(folder.name, 'attempt003')
            self.assertEqual(len(list((Path(tmp) / 'jobs/guard').glob('attempt*/PROCESS.json'))), 4)

    def test_numerical_failure_stops_even_when_guard_reports_interference(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = sweep.Campaign.__new__(sweep.Campaign)
            campaign.out = Path(tmp)
            campaign.cpu_env = campaign.env = dict(os.environ)
            def spawn(argv, **kwargs):
                guard = Path(argv[3]);guard.mkdir()
                (guard / 'guard_result.json').write_text(json.dumps(dict(code=75)))
                (guard.parent / 'check.json').write_text(json.dumps({'pass': False}))
                return SimpleNamespace(pid=12345, wait=lambda: 75)
            with patch.object(sweep.subprocess, 'Popen', side_effect=spawn) as start:
                with self.assertRaisesRegex(RuntimeError, 'numerical failure'):
                    campaign.job('guard', lambda p: ['true'], sweep.GROUP, allow_failure=True)
            self.assertEqual(start.call_count, 1)

    def test_nested_helper_guard_failure_can_be_requeued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'nested/guard_result.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(code=75)))
            self.assertTrue(sweep.interference_failure(root, 1))
            path.write_text(json.dumps(dict(code=1)))
            self.assertFalse(sweep.interference_failure(root, 1))


if __name__ == '__main__':
    unittest.main()
