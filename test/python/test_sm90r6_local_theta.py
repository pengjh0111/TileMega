import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('theta', Path(__file__).resolve().parents[2] / 'scripts/sm90r6_local_theta.py')
theta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theta)


class LocalThetaTests(unittest.TestCase):
    def test_window_precedes_traffic(self):
        rows = [dict(tasks=64, traffic_bytes=1), dict(tasks=66, traffic_bytes=9),
                dict(tasks=264, traffic_bytes=8), dict(tasks=265, traffic_bytes=0)]
        chosen, evidence = theta.traffic_choice(rows, 132)
        self.assertEqual(chosen['tasks'], 264)
        self.assertIsNone(evidence['exception'])

    def test_user_empty_domain_rule(self):
        rows = [dict(tasks=501, traffic_bytes=20), dict(tasks=1002, traffic_bytes=10)]
        chosen, evidence = theta.traffic_choice(rows, 132)
        self.assertEqual(chosen['tasks'], 501)
        self.assertEqual(evidence['window_distance'], 237)
        self.assertEqual(evidence['exception'], 'nearest_task_window_shell')

    def test_equal_traffic_prefers_distance_to_grid(self):
        rows = [dict(tasks=66, traffic_bytes=7), dict(tasks=128, traffic_bytes=7)]
        self.assertEqual(theta.traffic_choice(rows, 132)[0]['tasks'], 128)

    def test_measured_requires_full_domain_and_fresh_processes(self):
        options = [dict(geometry=dict(tile_n=32)), dict(geometry=dict(tile_n=64))]
        records = [dict(family='qkv', geometry=row['geometry'], process_id=i,
                        fresh_process=True, guard_code=0,
                        stage_ns=[5, 1, 3] if j == 0 else [1, 9, 7])
                   for j, row in enumerate(options) for i in range(3)]
        chosen, evidence = theta.measured_choice(options, records, 3)
        self.assertEqual(chosen['geometry']['tile_n'], 32)
        self.assertEqual(evidence['isolated_ns'], 3)
        with self.assertRaises(ValueError):
            theta.measured_choice(options, records[:-1], 3)
        records[-1]['guard_code'] = 75
        with self.assertRaises(ValueError):
            theta.measured_choice(options, records, 3)
