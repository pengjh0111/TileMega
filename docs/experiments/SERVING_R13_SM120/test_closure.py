import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import closure
import pipeline
import report


class ClosureTests(unittest.TestCase):
    def page(self,directory,launches=16,past=1000):
        import sys
        sys.path.insert(0,str(pipeline.FRAME))
        root=Path(directory)
        pipeline.write(root/f'past{past}/trace.json',dict(past=past,launches=launches,steps=1))
        page=root/'page_trace.tsv'
        page.write_text('step\tworker\tkernel_begin_ns\tkernel_end_ns\tdependency_wait_ns\tpage_full_ns\tfull_and_wait_ns\tloader_issue_ns\n'
                        f'{past-64}\t0\t100\t200\t160\t320\t80\t160\n')
        return page

    def test_repeated_page_counters_are_not_single_launch_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            page=self.page(directory)
            pipeline.write(Path(directory)/'past575/trace.json',dict(past=575,launches=16,steps=1))
            row=report.page_observations(page,[dict(past=575,dram_ns=100),dict(past=1086,dram_ns=200)],{'native':2})[0]
            self.assertEqual(row['past'],1000)
            self.assertEqual(row['raw_page_full_mean_ns'],320)
            self.assertEqual(row['page_full_mean_ns'],20)
            self.assertEqual(row['counter_launches'],16)
            self.assertIsNone(row['loader_issue_fraction'])
            self.assertEqual(row['page_pasts_not_retained'],[575])
            self.assertAlmostEqual(row['bytes'],(100+100*425/511)*2)

    def test_single_launch_page_fraction_remains_a_matching_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            page=self.page(directory,launches=1,past=575)
            row=report.page_observations(page,[dict(past=575,dram_ns=100)],{'native':2})[0]
            self.assertEqual(row['counter_launches'],1)
            self.assertEqual(row['page_full_mean_ns'],320)
            self.assertEqual(row['loader_issue_fraction'],1.6)
            self.assertEqual(row['span_over_floor_native'],1)

    def test_page_floor_does_not_extrapolate(self):
        with tempfile.TemporaryDirectory() as directory:
            page=self.page(directory,past=1000)
            row=report.page_observations(page,[dict(past=575,dram_ns=100)],{'native':2})[0]
            self.assertIsNone(row['bytes'])
            self.assertNotIn('span_over_floor_native',row)

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
