#!/usr/bin/env python3
"""Keep historical allocation evidence separate from the current GPU limit."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn4_bert_report', ROOT/'scripts/dnn4_bert_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class AllocationTest(unittest.TestCase):
    def test_preserved_transition(self):
        before = dict(allowed_gpu_indices=[0, 1, 2, 3], request='original request', resumed_at=100)
        current = dict(before, allowed_gpu_indices=[0], gpu_restriction_request='将测试进程限定在使用0号卡上')
        result = dict(allowed_gpu_indices=[0])
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            transition = session/'gpu0_transition_1794589079598058614'
            transition.mkdir()
            def write(a, b, c):
                for path, value in [(session/'authorization.json', a),
                                    (transition/'authorization_before.json', b),
                                    (transition/'result.json', c)]:
                    path.write_text(json.dumps(value))
            write(current, before, result)
            self.assertEqual(report.historical_allcards_authorization(session), before)
            self.assertEqual(json.loads((session/'authorization.json').read_text())['allowed_gpu_indices'], [0])
            mutations = [('current', 'allowed_gpu_indices', [1]),
                         ('current', 'request', 'different request'),
                         ('current', 'resumed_at', 200),
                         ('current', 'gpu_restriction_request', 'unknown'),
                         ('before', 'allowed_gpu_indices', [0]),
                         ('result', 'allowed_gpu_indices', [0, 1, 2, 3])]
            for target, field, value in mutations:
                with self.subTest(target=target, field=field):
                    items = dict(current=copy.deepcopy(current), before=copy.deepcopy(before), result=copy.deepcopy(result))
                    items[target][field] = value
                    write(items['current'], items['before'], items['result'])
                    with self.assertRaises(ValueError):
                        report.historical_allcards_authorization(session)
            write(before, before, result)
            self.assertEqual(report.historical_allcards_authorization(session), before)


if __name__ == '__main__':
    unittest.main()
