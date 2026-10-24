import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('native_program_key', ROOT / 'scripts/native_sm90r2_program_key.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeProgramKeyTest(unittest.TestCase):
    def key(self, registers=254, stack=8, ptxas=74, lookahead=1, slots=2,
            source='source-a', instructions=None, changed_binary=False):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'test.so'
            plan = dict(native_dense=dict(gemms={'qkv': dict(register_stages=lookahead, slots=slots)}),
                grid=132, runtime_batch=16, capacity=1088, export_sha256='export')
            Path(str(binary) + '.plan.json').write_text(json.dumps(plan))
            identity = dict(trace=False, binary_sha256='binary', shared_memory_bytes=199808,
                nvcc_arch='sm_90a', source=dict(source_digest=source),
                resources={'sm90_dense_native_kernel': dict(registers=ptxas, stack_bytes=0)})
            linked = dict(binary_sha256='changed' if changed_binary else 'binary',
                resources={'sm90_dense_native_kernel': dict(registers=registers,stack_bytes=stack,
                    static_smem_bytes=1024,local_bytes=0)},
                command=['cuobjdump'],tool_sha256='tool',stdout='saved dump',stderr='')
            words = instructions or {'kernel': ['a', 'b']}
            with patch.object(module, 'verify', return_value=identity), \
                    patch.object(module, 'inspect_linked_resources', return_value=linked), \
                    patch.object(module.subprocess, 'check_output', return_value='assembly'), \
                    patch.object(module, 'encodings', return_value=copy.deepcopy(words)):
                return module.program_key(binary)

    def test_final_register_and_stack_change_the_key(self):
        base = self.key()['program_key']
        self.assertNotEqual(base, self.key(registers=252)['program_key'])
        self.assertNotEqual(base, self.key(stack=16)['program_key'])

    def test_prelink_registers_do_not_describe_equivalence(self):
        base = self.key(ptxas=74)
        other = self.key(ptxas=82)
        self.assertEqual(base['program_key'], other['program_key'])
        self.assertEqual(base['signature']['resources']['registers'], 254)
        self.assertEqual(other['ptxas_entry_resources']['registers'], 82)

    def test_lookahead_only_is_dropped_after_compiled_comparison(self):
        base = self.key()['program_key']
        self.assertEqual(base, self.key(lookahead=2)['program_key'])
        self.assertNotEqual(base, self.key(lookahead=2,instructions={'kernel': ['a','c']})['program_key'])
        self.assertNotEqual(base, self.key(slots=3)['program_key'])

    def test_bound_source_prevents_cross_source_aliasing(self):
        self.assertNotEqual(self.key()['program_key'], self.key(source='source-b')['program_key'])
        with self.assertRaises(ValueError):
            self.key(source='')

    def test_changed_binary_is_rejected(self):
        with self.assertRaises(ValueError):
            self.key(changed_binary=True)


if __name__ == '__main__':
    unittest.main()
