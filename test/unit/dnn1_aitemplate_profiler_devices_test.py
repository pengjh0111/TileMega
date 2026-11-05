"""Execute pinned upstream launch functions with fake subprocesses, no CUDA."""
import ast
import logging
import os
from pathlib import Path
from queue import Queue
import sys
import time
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_aitemplate as runner


class DeviceTest(unittest.TestCase):
    def test_guard_mismatch_or_excluded_card_cannot_start_profilers(self):
        for card, cvd in [(5, '0'), (6, '6'), (0, '0,1')]:
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=cvd, TILEMEGA_DEVICE_INDEX=str(card)):
                with self.assertRaises(ValueError):
                    runner.guarded_profile_devices(card)

    def test_both_pinned_runners_launch_on_physical_guarded_card(self):
        backend = runner.AIT_ROOT / 'python/aitemplate/backend'
        task_tree = ast.parse((backend / 'task_runner.py').read_text())
        profiler_tree = ast.parse((backend / 'profiler_runner.py').read_text())
        task_node = next(n for n in task_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Task')
        funcs = [n for n in profiler_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('run_task', 'update_inplace')]
        for card in (0, 2, 5, 7):
            calls = []
            def popen(command, **kwargs):
                calls.append(('old-Popen', kwargs['env']['CUDA_VISIBLE_DEVICES']))
                return SimpleNamespace(kill=lambda: None)
            def run(command, **kwargs):
                calls.append(('new-run', kwargs['env']['CUDA_VISIBLE_DEVICES']))
                return SimpleNamespace(stdout='test', stderr='')
            namespace = dict(typing=typing, os=os, time=time,
                subprocess=SimpleNamespace(Popen=popen, run=run, PIPE=-1),
                _LOGGER=logging.getLogger(__name__), PROFILER_RUN_MAX_ATTEMPTS=3,
                PROFILER_RUN_RETRY_DELAY_SECONDS=5, sleep=lambda _: None)
            exec(compile(ast.Module(body=[task_node, *funcs], type_ignores=[]),
                'pinned-upstream-launch-functions', 'exec'), namespace)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=str(card), TILEMEGA_DEVICE_INDEX=str(card)):
                physical = runner.guarded_profile_devices(card)[0]
                task = namespace['Task'](0, ['unused'], 'test', dev_flag='CUDA_VISIBLE_DEVICES')
                task(physical)
                queue = Queue()
                queue.put(str(physical))
                namespace['run_task'](['unused'], queue, 'CUDA_VISIBLE_DEVICES')
                self.assertEqual(queue.get_nowait(), str(card))
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], str(card))
            self.assertEqual(calls, [('old-Popen', str(card)), ('new-run', str(card))])


if __name__ == '__main__':
    unittest.main()
