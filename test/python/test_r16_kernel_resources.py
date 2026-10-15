import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from r16_kernel_resources import validate_selected_entry


def entry(kernel, spill):
    symbol = '_ZN8tilemega7codegen'+str(len(kernel))+kernel+'Ev'
    return (f"ptxas info : Compiling entry function '{symbol}' for 'sm_90a'\n"
            f'ptxas info : Function properties for {symbol}\n'
            f'    96 bytes stack frame, {spill} bytes spill stores, {spill} bytes spill loads\n'
            'ptxas info : Used 255 registers, 16 bytes smem\n')


class KernelResources(unittest.TestCase):
    def setUp(self):
        self.log = entry('tilemega_l2_kernel', 0)+entry('tilemega_l1_loop_kernel', 8)

    def test_unselected_spill_remains_visible(self):
        receipt = validate_selected_entry(self.log, 'L2', 0)
        self.assertEqual(receipt['selected_entry']['spill_store_bytes'], 0)
        self.assertFalse(receipt['whole_binary_entry_spills_zero'])
        self.assertEqual(len(receipt['all_entries']), 2)

    def test_selected_spill_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selected CUDA entry has compiler spills'):
            validate_selected_entry(self.log, 'L1', 1)

    def test_absent_entry_and_function_property_only_rejected(self):
        with self.assertRaisesRegex(ValueError, 'absent or ambiguous'):
            validate_selected_entry(self.log, 'L1', 0)
        with self.assertRaisesRegex(ValueError, 'absent or ambiguous'):
            validate_selected_entry('Function properties for tilemega_l2_kernel\n', 'L2', 0)

    def test_duplicate_or_incomplete_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous duplicate'):
            validate_selected_entry(self.log+self.log, 'L2', 0)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_selected_entry(self.log.replace('Used 255 registers', 'Used registers'), 'L2', 0)

    def test_fake_prefixed_symbol_rejected(self):
        fake = self.log.replace('_ZN8tilemega7codegen18tilemega_l2_kernelE',
                                'unrelated_tilemega_l2_kernelE')
        with self.assertRaisesRegex(ValueError, 'absent or ambiguous'):
            validate_selected_entry(fake, 'L2', 0)

    def test_non_entry_spill_requires_transitive_attribution(self):
        helper = ('ptxas info : Function properties for helper_function\n'
                  '    8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\n'
                  'ptxas info : Used 32 registers\n')
        with self.assertRaisesRegex(ValueError, 'callgraph attribution'):
            validate_selected_entry(self.log+helper, 'L2', 0)

    def test_invalid_executor_rejected(self):
        for mode, loop in [('L3', 0), ('L2', 2)]:
            with self.assertRaises(ValueError):
                validate_selected_entry(self.log, mode, loop)


if __name__ == '__main__':
    unittest.main()
