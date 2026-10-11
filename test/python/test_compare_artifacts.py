import tempfile
from pathlib import Path
import unittest

from tilemega.bench.compare_artifacts import compare, parse_sass


SASS = 'Function : old_kernel\n/*0000*/ MOV R0, 0x10; /* 0x1234 */\n'
RESOURCE = """ptxas info : Compiling entry function 'old_kernel' for 'sm_89'
ptxas info : Function properties for old_kernel
 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 32 registers, 4096 bytes smem
ptxas info : Function properties for helper
 0 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
"""


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / name for name in ('reference', 'candidate')]
        for p in self.paths:
            p.write_bytes(b'binary')
            Path(str(p) + '.cu').write_bytes(b'#include <runtime.cuh>\n')
            Path(str(p) + '.ptxas.log').write_text(RESOURCE)

    def test_address_and_encoding_only(self):
        altered = SASS.replace('0000', '0040').replace('0x1234', '0x5678')
        report, _, _ = compare(*self.paths, SASS, altered)
        self.assertTrue(report['pass'])

    def test_new_kernel_allowed_but_missing_reference_rejected(self):
        report, _, _ = compare(*self.paths, SASS, SASS + 'Function : new_kernel\n/*0*/ EXIT;\n')
        self.assertTrue(report['pass'])
        report, _, _ = compare(*self.paths, SASS, SASS.replace('old_kernel', 'new_kernel'))
        self.assertFalse(report['sass_equal'])

    def test_immediate_operand_change_is_not_hidden(self):
        report, _, _ = compare(*self.paths, SASS, SASS.replace('0x10', '0x20'))
        self.assertFalse(report['sass_equal'])

    def test_callee_spill_is_checked_in_entry_context(self):
        Path(str(self.paths[1]) + '.ptxas.log').write_text(RESOURCE.replace('8 bytes spill loads', '16 bytes spill loads'))
        report, _, resources = compare(*self.paths, SASS, SASS)
        self.assertFalse(report['resources_equal'])
        self.assertIn('helper', resources['old_kernel']['reference']['callees'])

    def test_source_is_byte_exact(self):
        Path(str(self.paths[1]) + '.cu').write_bytes(b'#include <runtime.cuh>\r\n')
        report, _, _ = compare(*self.paths, SASS, SASS)
        self.assertFalse(report['source_equal'])

    def test_empty_or_multiple_architecture_dump_rejected(self):
        with self.assertRaises(ValueError):
            parse_sass('')
        with self.assertRaises(ValueError):
            parse_sass(SASS + SASS)


if __name__ == '__main__':
    unittest.main()
