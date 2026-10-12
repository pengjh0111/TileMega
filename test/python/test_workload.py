"""Request validation and the generalized selection integral."""
import unittest

from tilemega.serving.integrated_selection import integrated_ms, successive_halving
from tilemega.serving.workload import Workload, workload_points


class WorkloadTest(unittest.TestCase):
    def test_default_preserves_historical_objective_and_samples(self):
        point = workload_points({})[0]
        self.assertEqual((point.isl, point.osl, point.capacity), (64, 1024, 1088))
        self.assertEqual(point.pasts, (64, 575, 1000))
        self.assertEqual(point.objective, 'uniform_64_1087')
        samples = [2., 4., 7.]
        total = 0.
        for past in range(64, 1088):
            if past >= 1000:
                value = samples[2]
            else:
                i = 0 if past < 575 else 1
                weight = (past-point.pasts[i])/(point.pasts[i+1]-point.pasts[i])
                value = samples[i]*(1-weight)+samples[i+1]*weight
            total += value
        rows = {str(p): {'mean_ms': ms} for p, ms in zip(point.pasts, samples)}
        self.assertEqual(integrated_ms(rows), total/1024)

    def test_sweep_and_capacity(self):
        first, second = workload_points(dict(points=[[64, 1024], [128, 2048]],
                                            batch=[1, 16], capacity=4096))
        self.assertEqual(first.capacity, 4096)
        self.assertEqual(second.capacity, 4096)
        self.assertEqual(second.batches, (1, 16))
        self.assertEqual(second.pasts, (128, 1151, 2174))
        self.assertEqual(second.objective, 'uniform_128_2175')
        self.assertEqual(Workload(128, 2048).capacity, 2176)

    def test_integral_is_over_requested_interval(self):
        point = Workload(128, 2048)
        rows = {str(p): {'mean_ms': float(p)} for p in point.pasts}
        expected = (sum(range(128, 2175))+2174)/2048
        self.assertEqual(integrated_ms(rows, point), expected)
        for count in (1, 2, 3):
            point = Workload(5, count)
            rows = {str(p): {'mean_ms': 2.} for p in point.pasts}
            self.assertEqual(integrated_ms(rows, point), 2.)

    def test_halving_uses_configured_pasts(self):
        point = Workload(128, 2048)
        def measure(candidate, label):
            return dict(by_past={str(p): {'mean_ms': candidate['id']+1.}
                                 for p in point.pasts},
                        execution_identity={'id': candidate['id']})
        winner, rows = successive_halving([{'id': 0}, {'id': 1}], measure, workload=point)
        self.assertEqual(winner['id'], 0)
        self.assertEqual([row['median_ms'] for row in rows], [1., 2.])

    def test_invalid_workloads_are_rejected(self):
        for kwargs in (dict(isl=0), dict(osl=-1), dict(osl=True), dict(capacity=1087),
                       dict(batches=(1, 1)), dict(batches=(17,)), dict(batches=(False,))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Workload(**kwargs)
        for section in (dict(points=[]), dict(points=[[64]]),
                        dict(points=[[64, 1024], [64, 1024]]),
                        dict(points=[[128, 2048]], capacity=1088), dict(batch=1)):
            with self.subTest(section=section), self.assertRaises(ValueError):
                workload_points(section)


if __name__ == '__main__':
    unittest.main()
