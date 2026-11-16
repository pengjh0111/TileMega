#!/usr/bin/env python3
"""Exercise independent retries and four workers without using real GPUs."""
import collections
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sm90r6_gpu_items as module


class QueueTest(unittest.TestCase):
    def test_concurrency_retry_dependencies_and_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = module.read(module.dispatch.POLICY)
            inventory = '\n'.join(f'{gpu}, {policy["idle_power_mapping_provenance"]["uuid_by_gpu"][str(gpu)]}'
                                  for gpu in range(4))
            with patch.object(module.subprocess, 'check_output', return_value=inventory):
                queue = module.GPUItems(root / 'queue')
            mutex = threading.Lock()
            barrier = threading.Barrier(4)
            active, held, peak = set(), set(), [0]
            attempts, events = collections.Counter(), []

            class Handle:
                def __init__(self, gpu):
                    self.gpu = gpu
                def close(self):
                    with mutex:
                        held.remove(self.gpu)

            def claim(gpu):
                with mutex:
                    self.assertNotIn(gpu, held)
                    held.add(gpu)
                return Handle(gpu)

            def execute(job, gpu, directory):
                name = job['id']
                with mutex:
                    self.assertNotIn(name, active)
                    active.add(name)
                    peak[0] = max(peak[0], len(active))
                    attempts[name] += 1
                    first_wave = name in 'abcd' and attempts[name] == 1
                    events.append((name, attempts[name]))
                    if name == 'dependent':
                        self.assertEqual(queue.get('a')['status'], 'done')
                if first_wave:
                    barrier.wait(timeout=5)
                time.sleep(.03)
                with mutex:
                    active.remove(name)
                if name == 'a' and attempts[name] == 1:
                    return 75, dict(reason='simulated interference')
                return (1 if name == 'failed' else 0), dict(gpu=gpu)

            for name in ['a', 'b', 'c', 'd', 'fresh', 'failed', 'dependent']:
                queue.add(dict(id=name, kind='mock', after=['a'] if name == 'dependent' else []))
            real_write = module.write

            def write(path, value):
                path = Path(path)
                if path.parent == Path('/tmp'):
                    path = root / 'claims' / path.name
                real_write(path, value)

            with patch.object(queue, 'claim', side_effect=claim), \
                    patch.object(queue, 'execute', side_effect=execute), \
                    patch.object(module, 'write', side_effect=write):
                queue.start()
                deadline = time.monotonic() + 12
                try:
                    while time.monotonic() < deadline:
                        with queue.lock:
                            if all(v['status'] in ('done', 'failed') for v in queue.items.values()):
                                break
                        time.sleep(.02)
                    else:
                        self.fail('queue failed to drain')
                finally:
                    queue.stopping.set()
                    for thread in queue.threads:
                        thread.join(timeout=6)
            self.assertEqual(peak[0], 4)
            self.assertFalse(held)
            self.assertFalse(active)
            self.assertEqual(attempts['a'], 2)
            self.assertEqual(queue.get('failed')['status'], 'failed')
            self.assertTrue(all(queue.get(name)['status'] == 'done'
                                for name in ['a', 'b', 'c', 'd', 'fresh', 'dependent']))
            self.assertLess(events.index(('fresh', 1)), events.index(('a', 2)))
            for name, record in queue.items.items():
                self.assertEqual(len(record['attempts']), attempts[name])
                self.assertTrue(all('ended' in row for row in record['attempts']))


if __name__ == '__main__':
    unittest.main()
