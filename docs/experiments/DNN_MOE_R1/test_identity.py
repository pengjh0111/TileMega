import unittest
from identity_dm import resources


class ResourceTests(unittest.TestCase):
    def test_called_function_spill_marks_its_entry(self):
        text = """ptxas info : Compiling entry function 'kernel' for 'sm_89'
ptxas info : Function properties for kernel
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 128 registers, 65536 bytes smem
ptxas info : Function properties for helper
  32 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
ptxas info : Compiling entry function 'other' for 'sm_89'
ptxas info : Used 64 registers
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""
        parsed = resources(text)
        self.assertTrue(parsed['kernel']['spill'])
        self.assertEqual(parsed['kernel']['stack_bytes'], 32)
        self.assertEqual(parsed['kernel']['smem_bytes'], 65536)
        self.assertFalse(parsed['other']['spill'])

    def test_missing_resources_fail_closed(self):
        for value in ('', "Compiling entry function 'kernel'"):
            with self.assertRaises(ValueError):
                resources(value)


if __name__ == '__main__':
    unittest.main()
