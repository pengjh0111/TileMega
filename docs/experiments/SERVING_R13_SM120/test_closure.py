import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import closure
import pipeline


class ClosureTests(unittest.TestCase):
    def test_unavailable_is_not_missing_eligible_measurement(self):
        key=('E4a','llama_B1','B0')
        rows=[dict(matrix=key[0],cell=key[1],arm=key[2],round=i) for i in range(3)]
        report=closure.coverage(rows,{key})
        self.assertEqual(report['expected_records'],297)
        self.assertEqual(report['unavailable_records'],294)
        self.assertTrue(report['all_eligible_collected'])
        self.assertFalse(closure.coverage(rows[:2],{key})['all_eligible_collected'])
        rows[-1]['round']=0
        self.assertFalse(closure.coverage(rows,{key})['all_eligible_collected'])

    def test_unavailable_measurement_cannot_be_promoted(self):
        row=dict(matrix='E4a',cell='llama_B1',arm='R13F',round=0)
        self.assertFalse(closure.coverage([row],set())['all_eligible_collected'])

    def test_trace_correction_only_changes_output_and_instrumentation(self):
        command=closure.trace_command('nvcc -arch=sm_120 -o original.so source.cu -DKEEP=1','new.so')
        self.assertIn('original.so','nvcc -arch=sm_120 -o original.so source.cu -DKEEP=1')
        self.assertEqual(command[:6],['nvcc','-arch=sm_120','-o','new.so','source.cu','-DKEEP=1'])
        self.assertNotIn('-DTILEMEGA_TRACE_V2=1',command)
        for flag in ('PAGE_TRACE','TRACE_STAGE','TRACE_STEP'):
            self.assertIn('-DTILEMEGA_'+flag+'=1',command)

    def test_trace_replacement_requires_success_and_clean_outer_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=root/'raw/E5a_llama_B1_r5/trace_results.json'
            pipeline.write(original,[dict(label='B0',exit_code=0,path='base'),
                                     dict(label='PR_L1',exit_code=0,path='old')])
            results=root/'new/results.json';guard=root/'new/guard.json'
            pipeline.write(results,[dict(label='PR_L1',exit_code=0,path='new')])
            pipeline.write(root/'raw/acceptance_04/trace_replacements.json',
                           {'E5a:llama_B1':dict(results=str(results),guard=str(guard))})
            with patch.object(pipeline,'HERE',root):
                self.assertEqual(pipeline.trace_directories('E5a','llama_B1'),[Path('base'),Path('old')])
                pipeline.write(guard,dict(code=75))
                self.assertEqual(pipeline.trace_directories('E5a','llama_B1'),[Path('base'),Path('old')])
                pipeline.write(guard,dict(code=0))
                self.assertEqual(pipeline.trace_directories('E5a','llama_B1'),[Path('base'),Path('new')])
                pipeline.write(results,[dict(label='PR_L1',exit_code=1,path='new')])
                self.assertEqual(pipeline.trace_directories('E5a','llama_B1'),[Path('base'),Path('old')])


if __name__=='__main__':unittest.main()
