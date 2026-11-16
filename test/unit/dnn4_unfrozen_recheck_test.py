"""Check the explicit no-freezing arm before any baseline child can run."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('unfrozen_recheck', ROOT / 'scripts/dnn4_bert_unfrozen_recheck.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChildProbe(Exception):
    pass


class UnfrozenTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.worker_root = self.base / 'worker-root'
        self.authorization = (self.worker_root /
            'runs/dnn4/bert_priority/resume_allcards_20261112_v1/authorization.json')
        self.authorization.parent.mkdir(parents=True)
        self.authorization.write_text(json.dumps(dict(status='active', allowed_gpu_indices=[1])))
        root = patch.object(module, 'ROOT', self.worker_root)
        root.start()
        self.addCleanup(root.stop)
        self.value = dict(model='bert', batch=1, gpu=1,
            roles=['correctness', 'measurement', 'measurement', 'measurement', 'correctness'],
            arm=dict(arm='torchcompile-max-autotune', system='torchcompile', precision='bf16',
                     compile_mode='max-autotune', freezing=False, fullgraph=True),
            frozen={}, export=str(self.base), inductor_cache=str(self.base / 'cache'),
            compile_cpu_set='136-143', cpu_set='16-23')

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, value):
        path = self.base / 'spec.json'
        path.write_text(json.dumps(value))
        with patch('tilemega.serving.baseline_common.require_guard', return_value=('1', {})):
            module.execute(path, self.base / 'out')

    def reject(self, mutate):
        value = copy.deepcopy(self.value)
        mutate(value)
        with patch.object(module.subprocess, 'Popen') as child:
            with self.assertRaises(ValueError):
                self.execute(value)
            child.assert_not_called()

    def test_freezing_true_rejected(self):
        self.reject(lambda value: value['arm'].update(freezing=True))

    def test_frozen_arm_name_rejected(self):
        self.reject(lambda value: value['arm'].update(arm='torchcompile-max-autotune-freezing'))

    def test_undeclared_batch_rejected(self):
        self.reject(lambda value: value.update(batch=3))

    def test_process_roles_rejected(self):
        self.reject(lambda value: value.update(roles=['measurement'] * 3 + ['correctness'] * 2))

    def test_unfrozen_compile_precedes_measurement(self):
        with patch.object(module.subprocess, 'Popen', side_effect=ChildProbe) as child:
            with self.assertRaises(ChildProbe):
                self.execute(self.value)
            command = child.call_args.args[0]
            self.assertEqual(command[:3], ['taskset', '-c', '136-143'])
            self.assertNotIn('--freezing', command)
            self.assertIn('--correctness-only', command)
            self.assertEqual(child.call_args.kwargs['env']['TORCHINDUCTOR_FREEZING'], '0')

    def test_paused_authorization_prevents_a_child(self):
        self.authorization.write_text(json.dumps(dict(status='paused', allowed_gpu_indices=[1])))
        with patch.object(module.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(RuntimeError, 'authorization disallows'):
                self.execute(self.value)
            child.assert_not_called()

    def test_unauthorized_device_prevents_a_child(self):
        self.authorization.write_text(json.dumps(dict(status='active', allowed_gpu_indices=[0])))
        with patch.object(module.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(RuntimeError, 'authorization disallows'):
                self.execute(self.value)
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
