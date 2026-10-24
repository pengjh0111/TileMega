import unittest

from tilemega.build.linked_resources import kernel_resources


def output(arch='sm_90a', registers=254, stack=8):
    return f'''Fatbin elf code:
arch = {arch}
Resource usage:
 Common:
  GLOBAL:13146
 Function helper:
  REG:0 STACK:0 SHARED:0 LOCAL:0
 Function kernel:
  REG:{registers} STACK:{stack} SHARED:1024 LOCAL:0 CONSTANT[0]:5176
'''


class LinkedResourcesTest(unittest.TestCase):
    def test_final_allocation_includes_callees(self):
        row = kernel_resources(output(), 'sm_90a', ['kernel'])['kernel']
        self.assertEqual((row['registers'], row['stack_bytes']), (254, 8))
        self.assertEqual(row['static_smem_bytes'], 1024)
        self.assertEqual(row['raw']['CONSTANT[0]'], 5176)

    def test_exact_architecture(self):
        combined = output('sm_89', 167, 0) + output()
        row = kernel_resources(combined, 'sm_90a', ['kernel'])['kernel']
        self.assertEqual(row['registers'], 254)
        with self.assertRaises(ValueError):
            kernel_resources(combined, 'sm_90', ['kernel'])

    def test_duplicate_code_object_is_ambiguous(self):
        with self.assertRaises(ValueError):
            kernel_resources(output() * 2, 'sm_90a', ['kernel'])

    def test_missing_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            kernel_resources(output(), 'sm_90a', ['other_kernel'])

    def test_helper_zero_is_not_kernel_allocation(self):
        with self.assertRaises(ValueError):
            kernel_resources(output(), 'sm_90a', ['helper'])

    def test_incomplete_and_negative_resources(self):
        for bad in [output().replace('STACK:8 ', ''), output(stack=-8)]:
            with self.subTest(text=bad), self.assertRaises(ValueError):
                kernel_resources(bad, 'sm_90a', ['kernel'])

    def test_missing_architecture_and_duplicate_field(self):
        for bad in [output().replace('arch = sm_90a\n', ''), output() + '  REG:254\n']:
            with self.subTest(text=bad), self.assertRaises(ValueError):
                kernel_resources(bad, 'sm_90a', ['kernel'])

    def test_requested_kernel_names_are_distinct(self):
        for names in [[], ['kernel', 'kernel']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                kernel_resources(output(), 'sm_90a', names)


if __name__ == '__main__':
    unittest.main()
