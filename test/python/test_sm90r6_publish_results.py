import csv
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_publish_results import compact,number,publish


class CompactResults(unittest.TestCase):
    def test_archive_retains_exact_values_and_non_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            path=source/'SM90R6_example.csv'
            original='value,note,raw\n0.123456789123456789,diagnostic only,/raw/trace\n'
            path.write_text(original);result=publish(source,root/'out',root)
            self.assertEqual((root/'out/canonical'/path.name).read_text(),original)
            with (root/'out/compact'/path.name).open() as stream:row=next(csv.DictReader(stream))
            self.assertEqual(row,dict(value='0.123456789',record='2'))
            self.assertEqual(result['files'],1)
            manifest=json.loads((root/'out/SM90R6_result_provenance.json').read_text())
            self.assertEqual(manifest['files'][path.name]['mapping']['archived_columns'],['note','raw'])

    def test_integer_counts_are_not_rounded_through_float(self):
        value='123456789012345678901234567890.0'
        self.assertEqual(number(value),value[:-2])
        for value in ['1.234567890123e-21','999.999999999','-0.00002918283849']:
            before=Decimal(value);after=Decimal(number(value))
            self.assertLessEqual(abs(after-before),abs(before)*Decimal('5e-9'))

    def test_candidate_prefix_collision_and_double_compaction_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'collision'):
            compact([dict(candidate_sha256='a'*16+'b'*48),dict(candidate_sha256='a'*16+'c'*48)],'table.csv',Path('/'))
        with self.assertRaisesRegex(ValueError,'canonical'):
            compact([dict(value='1',record='2')],'table.csv',Path('/'))

    def test_bound_pivot_rejects_a_mismatched_timing_cell(self):
        same=dict(system='TileMega',model='model',tp='1',batch='1',bytes_step='1024',flops_step='2048',
            tpot_ms='1',communication_bound_ms='0',timing_source='/repo/result.csv:2',
            timing_acceptance='',timing_conditions='')
        rows=[dict(same,basis=basis,bound_ms=bound,bound_over_measured=bound) for basis,bound in [('measured','.5'),('nominal','.4')]]
        result,_=compact(rows,'SM90R6_hw_bound.csv',Path('/repo'))
        self.assertEqual(len(result),1);self.assertEqual(result[0]['bound_measured_ms'],'0.5')
        rows[1]['tpot_ms']='2'
        with self.assertRaisesRegex(ValueError,'disagree'):compact(rows,'SM90R6_hw_bound.csv',Path('/repo'))


if __name__=='__main__':unittest.main()
