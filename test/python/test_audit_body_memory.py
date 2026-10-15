# SPDX-License-Identifier: BSD-3-Clause
"""Keep static vector attribution separate from barriers and local spills."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from audit_body_memory import attribute, memory_space


class BodyMemoryAuditTest(unittest.TestCase):
    def test_dependencies_and_tensor_transfers_are_not_scalar_loads(self):
        for opcode in ('LDGDEPBAR', 'DEPBAR.LE', 'UTMALDG.2D', 'UBLKCP.128'):
            self.assertIsNone(memory_space(opcode))
        self.assertEqual(memory_space('LDG.E.128'), 'global')
        self.assertEqual(memory_space('LDL.128'), 'local')

    def test_duplicate_inline_and_leaf_dumps_count_one_instruction(self):
        lines=['\t.section\t.text.kernel,"ax"',
               '//## File "/source/ServingRMSNormTaskBody.h", line 7',
               '/*0000*/ @P0 LDG.E.128 R4, [R2] ;',
               '/*0010*/ LDGDEPBAR ;', '/*0020*/ LDL.128 R8, [R1] ;']
        count=attribute([('inline',lines),('leaf',lines)],['ServingRMSNormTaskBody'])
        self.assertEqual(dict(count['ServingRMSNormTaskBody']), {'LDG.E.128':1,'LDL.128':1})

    def test_function_boundaries_clear_source_attribution(self):
        lines=['\t.section\t.text.first,"ax"',
               '//## File "/source/Body.h", line 7', '/*0000*/ STG.E.128 [R2], R4 ;',
               '\t.section\t.text.second,"ax"', '/*0000*/ STG.E.128 [R2], R4 ;']
        count=attribute([('dump',lines)],['Body'])
        self.assertEqual(dict(count['Body']), {'STG.E.128':1})


if __name__ == '__main__':
    unittest.main()
