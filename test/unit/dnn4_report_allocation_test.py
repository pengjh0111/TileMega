#!/usr/bin/env python3
"""Keep historical allocation evidence separate from the current GPU limit."""
import copy
import hashlib
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
    def test_fallback_preserves_both_transitions(self):
        before = dict(allowed_gpu_indices=[0, 1, 2, 3], request='original request', resumed_at=100)
        restricted = dict(before, allowed_gpu_indices=[0], gpu_restriction_at=200,
                          gpu_restriction_request='将测试进程限定在使用0号卡上')
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            old = session/'gpu0_transition_1794589079598058614'
            old.mkdir()
            transition = session/'gpu02_transition_1794626179576967972'
            transition.mkdir()
            expanded = dict(restricted, allowed_gpu_indices=[0, 2], primary_gpu=0,
                            fallback_gpu=2, same_card_pair_required=False,
                            gpu_fallback_at=300, latest_request='fallback and cross-card',
                            authorization_history_transition=str(transition))
            def write(path, value):
                path.write_text(json.dumps(value))
            def setup():
                write(old/'authorization_before.json', before)
                write(old/'result.json', dict(allowed_gpu_indices=[0]))
                write(transition/'authorization_before.json', restricted)
                write(transition/'authorization_after.json', expanded)
                result = dict(allowed_gpu_indices=[0, 2], same_card_pair_required=False,
                              effective_at=300)
                for label in ('before', 'after'):
                    result[label+'_sha256'] = hashlib.sha256(
                        (transition/f'authorization_{label}.json').read_bytes()).hexdigest()
                write(transition/'result.json', result)
                write(session/'authorization.json', expanded)
            setup()
            self.assertEqual(report.historical_allcards_authorization(session), before)
            mutations = [
                (session/'authorization.json', 'fallback_gpu', 1),
                (session/'authorization.json', 'same_card_pair_required', True),
                (session/'authorization.json', 'gpu_restriction_at', 0),
                (transition/'authorization_before.json', 'request', 'changed'),
                (transition/'authorization_after.json', 'latest_request', 'changed'),
                (transition/'result.json', 'before_sha256', '0'*64),
                (transition/'result.json', 'after_sha256', '0'*64),
                (transition/'result.json', 'effective_at', 0),
                (transition/'result.json', 'allowed_gpu_indices', [0, 1, 2, 3]),
                (old/'authorization_before.json', 'request', 'changed'),
                (old/'result.json', 'allowed_gpu_indices', [3]),
            ]
            for path, key, value in mutations:
                with self.subTest(path=path.name, key=key):
                    setup()
                    changed = json.loads(path.read_text())
                    changed[key] = value
                    write(path, changed)
                    with self.assertRaises(ValueError):
                        report.historical_allcards_authorization(session)

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
