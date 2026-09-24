#!/usr/bin/env python3
"""Exercise resource waiting and prevent recovery from masking numeric errors."""
import json
import pathlib
import tempfile
import unittest

from gpu_admission import required_mib, resource_failure, wait_for_device
from recover_resource_measurement import recoverable,contention_attempt

GOOD = ('E2E_HASH l05=abc l1=abc l2=abc\n'
        'l1_vs_l05_mismatch=0 l2_vs_l1_mismatch=0\n'
        'E2E_TIME l05_ms=1 l1_ms=2 l2_ms=3')


class AdmissionTest(unittest.TestCase):
    def test_memory_and_idle_must_remain_available(self):
        values = iter([(10, 0), (100, 60), (100, 0), (100, 0), (100, 20),
                       (100, 0), (100, 0), (100, 0)])
        def probe():
            free, busy = next(values)
            return dict(free_mib=free, utilization_percent=busy)
        with tempfile.TemporaryDirectory() as temp:
            log = pathlib.Path(temp) / 'admission.jsonl'
            wait_for_device(temp, log, probe=probe, sleep=lambda _: None, needed=100)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r['consecutive_ready'] for r in rows], [0, 0, 1, 2, 0, 1, 2, 3])

    def test_budget_rounds_up_and_retains_headroom(self):
        with tempfile.TemporaryDirectory() as temp:
            (pathlib.Path(temp) / 'weight').write_bytes(b'x')
            self.assertEqual(required_mib(temp), 2049)

    def test_whole_attempt_only_and_never_numeric_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = []
            for i in range(10):
                log = pathlib.Path(temp) / f'process_{i:02}.log'
                log.write_text('out of memory' if i < 4 else GOOD)
                logs.append(log)
            self.assertEqual(recoverable(logs), 4)
            with self.assertRaises(ValueError):
                recoverable(logs[:4])
            for raw in [GOOD.replace('l2=abc', 'l2=bad'), 'timeout',
                        GOOD + '\nout of memory']:
                # The last case is an equal completed output; the other OOMs
                # remain the only retryable records, never this completed one.
                logs[0].write_text(raw)
                if raw.startswith(GOOD):
                    self.assertFalse(resource_failure(raw))
                    self.assertEqual(recoverable(logs), 3)
                else:
                    with self.assertRaises(ValueError):
                        recoverable(logs)

    def test_contention_requires_raw_evidence_and_correctness(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp);logs=[]
            for i in range(10):
                p=root/f'process_{i}.log';p.write_text(GOOD);logs.append(p)
            device=root/'device.log';device.write_text('Utilization\n    GPU : 93 %\n')
            self.assertEqual(contention_attempt(logs,device),93)
            device.write_text('Utilization\n    GPU : 0 %\n')
            with self.assertRaises(ValueError):contention_attempt(logs,device)
            device.write_text('Utilization\n    GPU : 93 %\n')
            logs[0].write_text(GOOD.replace('l2=abc','l2=bad'))
            with self.assertRaises(ValueError):contention_attempt(logs,device)


if __name__ == '__main__':
    unittest.main()
