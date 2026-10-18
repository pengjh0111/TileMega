import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tilemega.bench import trace_analysis


class CausalParentTest(unittest.TestCase):
    def analyze(self, serialized=False, causal=True):
        def task(stage, worker, begin, end, count=0):
            return dict(stage=stage, worker=worker, slot=stage, logical_task=0,
                        run_begin=begin, run_end=end, publish_end=end+1,
                        wait_begin=begin-1, ready=begin, dependency_begin=0,
                        dependency_count=count)
        slots = [task(0, 0, 0, 10), task(1, 0 if serialized else 1,
                 20 if serialized else 1, 100), task(2, 2, 101, 110, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'runtime_dependencies.cuh').write_text(
                'constexpr StageDependency kDependencies0[] = {\n'
                '{0u, 2u, StageDependency::Map::kAll, 1u, 0, 0, 1u},\n'
                '{1u, 2u, StageDependency::Map::kAll, 1u, 0, 0, 1u}\n};')
            with patch.object(trace_analysis, 'load', return_value=(
                    {'l2_ms': .000111}, slots, [], [])):
                return trace_analysis.analyze_task_space_trace(
                    root, latest_ready_parent=causal)

    def test_parallel_branch_uses_last_ready_producer(self):
        result, _, links = self.analyze()
        self.assertEqual(result['cp_realized_path'], '1:0,2:0')
        self.assertEqual(sum(row['idle_ns'] for row in links), 0)
        self.assertEqual(sum(row['wall_ns'] for row in links),
                         result['cp_reconstructed_with_publish_ns'])
        historical, _, _ = self.analyze(causal=False)
        self.assertEqual(historical['cp_realized_path'], '0:0,2:0')

    def test_worker_serial_order_is_retained(self):
        result, _, links = self.analyze(serialized=True)
        self.assertEqual(result['cp_realized_path'], '0:0,1:0,2:0')
        self.assertGreater(sum(row['idle_ns'] for row in links), 0)


if __name__ == '__main__':
    unittest.main()
