"""CPU math and metadata checks; no GPU measurement or body gate evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fit_dm import fit, fit_observations, sha


def row(case, b, f, ns):
    return dict(case_id=case, bytes=b, flops=f, body_ns=ns)


class FitTests(unittest.TestCase):
    def test_independent_full_rank_fit_and_heldout_residual(self):
        train = [row(str(i), b, f, 20+.02*b+.001*f)
                 for i, (b, f) in enumerate(((100, 1000), (700, 800), (300, 6000), (800, 9000)))]
        held = [row('unseen', 500, 7500, 38.5)]
        model = fit(train, held)
        np.testing.assert_allclose([model['fixed_ns'], model['byte_ns'], model['flop_ns']],
                                   [20, .02, .001], atol=1e-10)
        self.assertTrue(model['coefficients_identifiable'])
        self.assertAlmostEqual(model['heldout_residuals'][0]['residual_ns'], -1, places=9)
        self.assertAlmostEqual(model['heldout_max_relative_error'], 1/38.5, places=9)

    def test_rank_deficiency_is_reported(self):
        train = [row(str(i), i*10, i*20, 7+i*2) for i in range(1, 7)]
        model = fit(train, [row('held', 90, 180, 25)])
        self.assertFalse(model['coefficients_identifiable'])
        self.assertEqual(model['feature_rank'], 2)
        self.assertLess(model['heldout_max_relative_error'], 1e-12)

    def test_boundary_solution_remains_nonnegative(self):
        train = [row(str(i), b, f, 50-b*.01+f*.001)
                 for i, (b, f) in enumerate(((100, 1000), (700, 800), (300, 6000), (800, 9000)))]
        model = fit(train, [row('held', 500, 7500, 52.5)])
        self.assertEqual(model['byte_ns'], 0)
        self.assertGreaterEqual(model['fixed_ns'], 0)
        self.assertGreaterEqual(model['flop_ns'], 0)
        self.assertGreater(model['heldout_max_relative_error'], 0)

    def test_invalid_observations_fail(self):
        for value in (-1, float('nan'), float('inf'), True, 0):
            with self.assertRaises(ValueError):
                fit([row('train', 10, 20, value)], [row('held', 20, 40, 8)])
        with self.assertRaises(ValueError):
            fit([row('train', 10, 20, 8)], [])

    def test_gate_identity_work_and_split_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root/'fixture.bin';binary.write_bytes(b'CPU metadata fixture only')
            identity = dict(schema='tilemega.dm1.native-test.identity.v1',
                            arch='sm_89', binary_sha256=sha(binary))
            identity['artifact_id'] = hashlib.sha256(json.dumps(identity,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            ip = root/'identity.json';ip.write_text(json.dumps(identity))
            case = dict(passed=True, artifact_id=identity['artifact_id'], body_kind='layernorm',
                        synchronization_required=True, fresh_processes=dict(passes=50, attempts=50))
            gp = root/'gate.json'
            gate = dict(passed=True, scope='CPU metadata fixture only', cases={'a':case, 'b':case})
            gp.write_text(json.dumps(gate))
            wp = root/'work.json'
            wp.write_text(json.dumps(dict(source='l_sem', scope='CPU metadata fixture only',
                cases={'a':dict(bytes=10, flops=20), 'b':dict(bytes=20, flops=40)})))
            common = dict(body_kind='layernorm', family='layernorm', section='serving',
                artifact_id=identity['artifact_id'], identity=str(ip), identity_sha256=sha(ip),
                numerical_gate=str(gp), numerical_gate_sha256=sha(gp),
                work=str(wp), work_sha256=sha(wp), binary=str(binary))
            rows = [dict(row('a', 10, 20, 8), split='train', **common),
                    dict(row('b', 20, 40, 11), split='heldout', **common)]
            self.assertEqual(len(fit_observations(rows)), 1)
            with self.assertRaises(ValueError):
                fit_observations(rows + [dict(rows[0], split='heldout')])
            bad = [dict(rows[0], flops=21), rows[1]]
            with self.assertRaises(ValueError):
                fit_observations(bad)
            case['fresh_processes']['passes']=49
            case['fresh_processes']['attempts']=49
            gp.write_text(json.dumps(gate))
            for r in rows:r['numerical_gate_sha256']=sha(gp)
            with self.assertRaises(ValueError):
                fit_observations(rows)
            binary.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                fit_observations(rows)


if __name__ == '__main__':
    unittest.main()
